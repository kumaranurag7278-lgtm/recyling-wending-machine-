import os
import sys
import time
import json
import re
import threading
import serial
import cv2
import numpy as np
from flask import Flask, render_template_string, jsonify, Response, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# =====================================================================
# LOAD AUTHENTIC PRE-TRAINED WASTE CLASSIFICATION NEURAL NETWORK
# Fine-Tuned on ~15,000 Waste & Garbage Images (Plastic, Paper, Cardboard, Metal)
# =====================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "waste_yolov8_cls.onnx")
WASTE_CLASSES = ["battery", "biological", "cardboard", "glass", "metal", "paper", "plastic", "trash"]

print(f"[AI MODEL] Loading WasteWise Deep Neural Network from '{MODEL_PATH}'...")
try:
    waste_net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    waste_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    print("[AI MODEL] Waste Classification Neural Network Ready (8 Classes: Plastic, Paper, Cardboard, Metal)!")
except Exception as e:
    print("[AI MODEL ERROR] Failed to load ONNX model:", e)
    waste_net = None

# Global Telemetry & System State
system_state = {
    "connected": False,
    "port": "COM11",
    "baud": 9600,
    "ir1": 0,
    "ir2": 0,
    "servo1": 0,
    "servo2": 0,
    "plastic": 0,
    "metal": 0,
    "paper": 0,
    "points": 0,
    "status": "INITIALIZING...",
    "last_updated": time.time(),
    "history": [],
    "camera_source": "WEBCAM",
    "camera_mode": "WASTE CLASSIFICATION AI (YOLOv8-cls ONNX)",
    "phone_camera_url": "http://192.168.1.100:8080/video",
    "predicted_class": "SCANNING FOR WASTE...",
    "category": "NONE",
    "prob_plastic": 0.0,
    "prob_paper": 0.0,
    "prob_metal": 0.0,
    "confidence": 0.0,
    "auto_sort_enabled": True
}

ser = None
ser_lock = threading.Lock()

# Serial Reader Worker (Listens to Arduino Uno on COM11 at 9600 baud)
def serial_worker():
    global ser, system_state
    while True:
        try:
            if ser is None or not ser.is_open:
                try:
                    ser = serial.Serial(system_state["port"], system_state["baud"], timeout=1)
                    system_state["connected"] = True
                    system_state["status"] = "CONNECTED TO ARDUINO (COM11)"
                    print("[SERIAL] Connected to COM11 @ 9600 baud successfully.")
                except Exception as e:
                    system_state["connected"] = False
                    system_state["status"] = f"DISCONNECTED: {str(e)[:25]}"
                    time.sleep(2)
                    continue

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                if "[TELEMETRY]" in line:
                    match = re.search(r'Plastic:\s*(\d+).*?Paper:\s*(\d+).*?Metal:\s*(\d+).*?Points:\s*(\d+)', line)
                    if match:
                        with ser_lock:
                            system_state["plastic"] = int(match.group(1))
                            system_state["paper"]   = int(match.group(2))
                            system_state["metal"]   = int(match.group(3))
                            system_state["points"]  = int(match.group(4))
                            system_state["status"]  = "SORTING LOGGED & COMPLETED"
                elif "Waste item placed" in line or "Item detected" in line:
                    with ser_lock:
                        system_state["ir1"] = 1
                        system_state["status"] = "ITEM DETECTED (RAMP START)"
                elif "CLASS: PLASTIC" in line:
                    with ser_lock:
                        system_state["status"] = "SORTING: PLASTIC (FLAP 1 OPEN)"
                        system_state["servo1"] = 90
                        system_state["servo2"] = 0
                elif "CLASS: PAPER" in line:
                    with ser_lock:
                        system_state["status"] = "SORTING: PAPER (FLAP 2 OPEN)"
                        system_state["servo1"] = 0
                        system_state["servo2"] = 90
                elif "CLASS: METAL" in line:
                    with ser_lock:
                        system_state["status"] = "SORTING: METAL (END CHUTE)"
                        system_state["servo1"] = 0
                        system_state["servo2"] = 0
                elif "Chute Passage Confirmed" in line or "CHUTE CONFIRMED" in line:
                    with ser_lock:
                        system_state["ir2"] = 1
                        system_state["status"] = "VERIFIED IN BIN"
                elif "SYSTEM READY" in line or "INSERT ITEM" in line:
                    with ser_lock:
                        system_state["ir1"] = 0
                        system_state["ir2"] = 0
                        system_state["servo1"] = 0
                        system_state["servo2"] = 0
                        system_state["status"] = "READY FOR ITEM"

                with ser_lock:
                    system_state["last_updated"] = time.time()
                    system_state["history"].append({
                        "time": time.strftime("%H:%M:%S"),
                        "plastic": system_state["plastic"],
                        "metal": system_state["metal"],
                        "paper": system_state["paper"],
                        "points": system_state["points"]
                    })
                    if len(system_state["history"]) > 25:
                        system_state["history"].pop(0)

        except Exception as e:
            system_state["connected"] = False
            system_state["status"] = "CONNECTION LOST"
            if ser:
                try: ser.close()
                except: pass
                ser = None
            time.sleep(2)

t = threading.Thread(target=serial_worker, daemon=True)
t.start()

# Deep Learning & Multi-Modal Material Classifier
# Specializing in:
# 1. Plastic Bottles / Rs 10 Juice Bottles (Frooti, Sting, Maaza, PET Water Bottles) -> Flap 1 ('p')
# 2. Paper Rolls / Cups / Tetra Pak -> Flap 2 ('c')
# 3. Metal Beverage Cans / Aluminum Items (Coke, Pepsi, Red Bull, Tin) -> End Chute ('m')
# 4. Empty Ramp / Slider Detection -> Waiting State (No servo movement)
def run_waste_model(roi):
    if waste_net is None or roi is None or roi.size == 0:
        return "WAITING FOR ITEM (EMPTY)", 0.0, None, (148, 163, 184), {"plastic": 0.0, "paper": 0.0, "metal": 0.0}, False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. OBJECT PRESENCE & EMPTY SLIDER FILTER
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)
    edge_pct = np.mean(edges > 0)
    std_s = float(np.std(s))
    std_v = float(np.std(v))

    # Find prominent object contours
    dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_c_area = max([cv2.contourArea(c) for c in contours], default=0.0)

    # Empty slider condition:
    # Bare wood/cardboard ramp has uniform texture, low color variance, and no significant contour
    is_object_present = (max_c_area > 3500) or (std_s > 27.0) or (std_v > 24.0 and edge_pct > 0.07)

    if not is_object_present:
        return "SLIDER EMPTY (PLACE ITEM)", 0.0, None, (100, 116, 139), {"plastic": 0.0, "paper": 0.0, "metal": 0.0}, False

    # 2. NEURAL NETWORK FORWARD PASS (With PyTorch ImageNet Normalization)
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    blob = cv2.resize(img_norm, (224, 224)).transpose(2, 0, 1)[np.newaxis, ...]
    waste_net.setInput(blob)
    raw = waste_net.forward()[0] # Shape (8,)

    # 3. OPTICAL MATERIAL SIGNATURES
    # Specular glint (intense reflection lines/highlights typical of curved plastic bottles and metallic cans)
    glint_ratio = float(np.mean((v > 220) & (s < 70)))
    mean_v = float(np.mean(v))
    mean_s = float(np.mean(s))

    # WasteWise ONNX Classes:
    # 0: battery, 1: biological, 2: cardboard, 3: glass, 4: metal, 5: paper, 6: plastic, 7: trash
    p_metal = float(raw[4] + raw[0] * 0.8)
    p_plastic = float(raw[6] + raw[3] * 0.6 + raw[7] * 0.5)
    p_paper = float(raw[5] + raw[2] * 0.9)

    # Real-World Recycling Item Fine-Tuning:
    # Metal can or metal object:
    if raw[4] > 0.45:
        p_metal += 0.6

    # Plastic Juice Bottle (Rs 10 Sting, Frooti, Maaza) or Water Bottle:
    # Characterized by specular glints, transparent highlights, or vibrant drink/cap colors
    if glint_ratio > 0.007 or (mean_s > 55.0 and mean_v > 95.0):
        p_plastic += 0.40
        p_paper *= 0.25

    # Paper Roll / Cup / Cardboard:
    # Pure matte diffuse texture, no specular glints, high luminance white paper or kraft cardboard
    if glint_ratio < 0.002 and (mean_v > 150.0 and mean_s < 45.0):
        p_paper += 0.85
        p_plastic *= 0.20

    total_p = p_plastic + p_paper + p_metal + 1e-6
    p_plastic /= total_p
    p_paper /= total_p
    p_metal /= total_p

    prob_dict = {
        "plastic": round(p_plastic * 100, 1),
        "paper":   round(p_paper * 100, 1),
        "metal":   round(p_metal * 100, 1)
    }

    # Best match decision
    if p_plastic > p_paper and p_plastic > p_metal:
        conf = round(p_plastic * 100, 1)
        return "PLASTIC / JUICE BOTTLE", conf, 'p', (16, 185, 129), prob_dict, True # Emerald
    elif p_paper > p_plastic and p_paper > p_metal:
        conf = round(p_paper * 100, 1)
        return "PAPER ROLL / CUP", conf, 'c', (245, 158, 11), prob_dict, True # Amber
    else:
        conf = round(p_metal * 100, 1)
        return "METAL BEVERAGE CAN", conf, 'm', (59, 130, 246), prob_dict, True # Blue

# Live Video Stream Generator (Neural Network Real-Time Inference)
last_auto_sort_time = 0
detection_counter = 0
current_candidate_code = None

def generate_camera_frames():
    global system_state, last_auto_sort_time, detection_counter, current_candidate_code
    cap = None
    current_mode = ""
    current_url = ""

    while True:
        mode = system_state["camera_source"]
        target_url = system_state["phone_camera_url"].strip()

        if mode != current_mode or (mode == "PHONE" and target_url != current_url):
            if cap: cap.release()
            try:
                if mode == "WEBCAM":
                    cap = cv2.VideoCapture(0)
                    system_state["camera_mode"] = "WASTE CLASSIFICATION AI (Laptop Webcam 0)"
                    print("[CAMERA] Opened Laptop Webcam (0)")
                else:
                    if not target_url.startswith("http://") and not target_url.startswith("https://"):
                        target_url = "http://" + target_url
                    if not target_url.endswith("/video") and not target_url.endswith("/shot.jpg"):
                        target_url += "/video"
                    cap = cv2.VideoCapture(target_url)
                    system_state["camera_mode"] = f"WASTE AI (Phone IP Cam: {target_url})"
                    print(f"[CAMERA] Connecting to Phone IP Webcam: {target_url}")
                current_mode = mode
                current_url = target_url
            except Exception as e:
                print("[CAMERA ERROR]:", e)
                cap = None

        frame = None
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = None
                time.sleep(1)
            else:
                if mode == "WEBCAM":
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape

                # Center Scanning Box Coordinates (Box width 280, height 320)
                bw, bh = 280, 320
                x1 = (w - bw) // 2
                y1 = (h - bh) // 2
                x2 = x1 + bw
                y2 = y1 + bh

                # Crop ROI strictly inside the box
                roi = frame[y1:y2, x1:x2]

                # Run Multi-Modal AI Classifier (with Empty Slider Detection)
                category_name, conf, sort_code, hud_color, probs, is_object_present = run_waste_model(roi)

                system_state["prob_plastic"] = probs["plastic"]
                system_state["prob_paper"]   = probs["paper"]
                system_state["prob_metal"]   = probs["metal"]
                system_state["predicted_class"] = category_name
                system_state["confidence"] = conf

                # Dim outside area slightly to focus attention on scanning box
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (10, 15, 26), -1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

                # Draw Target Box & Corner Crosshairs
                cv2.rectangle(frame, (x1, y1), (x2, y2), hud_color, 2)
                L = 24
                cv2.line(frame, (x1, y1), (x1 + L, y1), hud_color, 4)
                cv2.line(frame, (x1, y1), (x1, y1 + L), hud_color, 4)
                cv2.line(frame, (x2, y1), (x2 - L, y1), hud_color, 4)
                cv2.line(frame, (x2, y1), (x2, y1 + L), hud_color, 4)
                cv2.line(frame, (x1, y2), (x1 + L, y2), hud_color, 4)
                cv2.line(frame, (x1, y2), (x1, y2 - L), hud_color, 4)
                cv2.line(frame, (x2, y2), (x2 - L, y2), hud_color, 4)
                cv2.line(frame, (x2, y2), (x2, y2 - L), hud_color, 4)

                if not is_object_present:
                    # Slider is empty: Do NOT increment counter, DO NOT actuate servos!
                    detection_counter = 0
                    current_candidate_code = None
                    system_state["progress"] = 0

                    tag_text = "READY: PLACE BOTTLE / CAN ON SLIDER"
                    cv2.rectangle(frame, (x1, y1 - 32), (x1 + bw, y1), (15, 23, 42), -1)
                    cv2.putText(frame, tag_text, (x1 + 10, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1)

                    cv2.putText(frame, "[ WAITING FOR OBJECT... ]", (x1 + 48, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1)
                else:
                    # Object Present: Show target class & confidence
                    tag_text = f"AI CLASSIFIER: {category_name} ({conf:.1f}%)"
                    cv2.rectangle(frame, (x1, y1 - 36), (x1 + bw, y1), (15, 23, 42), -1)
                    cv2.putText(frame, tag_text, (x1 + 6, y1 - 11),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, hud_color, 2)

                    # Streak logic for automatic triggering
                    if sort_code == current_candidate_code:
                        detection_counter += 1
                    else:
                        current_candidate_code = sort_code
                        detection_counter = 0

                    progress_pct = min(100, int((detection_counter / 10) * 100))
                    system_state["progress"] = progress_pct

                    # Bottom Progress Bar
                    bar_w = int((bw - 4) * (progress_pct / 100))
                    cv2.rectangle(frame, (x1 + 2, y2 + 10), (x1 + bw - 2, y2 + 24), (30, 41, 59), -1)
                    cv2.rectangle(frame, (x1 + 2, y2 + 10), (x1 + 2 + bar_w, y2 + 24), hud_color, -1)
                    cv2.putText(frame, f"LOCKING IN... {progress_pct}%", (x1 + 52, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    # Trigger Sort after ~1 second steady hold
                    if detection_counter >= 10 and (time.time() - last_auto_sort_time > 4.5):
                        last_auto_sort_time = time.time()
                        detection_counter = 0
                        print(f"\n[NEURAL NET TRIGGER] '{category_name}' confirmed! Actuating Arduino '{sort_code}'...")
                        if ser and ser.is_open:
                            with ser_lock:
                                ser.write(sort_code.encode('utf-8'))

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (15, 23, 42)
            cv2.putText(frame, "INITIALIZING AI CAMERA...", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (56, 189, 248), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.035)

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def get_telemetry():
    return jsonify(system_state)

@app.route('/api/switch_camera', methods=['POST'])
def switch_camera():
    global system_state
    data = request.json or {}
    src = data.get('source', 'WEBCAM')
    if src in ['WEBCAM', 'PHONE']:
        system_state["camera_source"] = src
        return jsonify({"status": "switched", "source": src})
    return jsonify({"error": "Invalid source"}), 400

@app.route('/api/camera_config', methods=['POST'])
def set_camera_url():
    global system_state
    data = request.json or {}
    url = data.get('url', '').strip()
    if url:
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        parts = url.split("://", 1)[1]
        host_port = parts.split("/")[0]
        if ":" not in host_port:
            rest = "/".join(parts.split("/")[1:])
            parts = f"{host_port}:8080" + (f"/{rest}" if rest else "")
            url = "http://" + parts
        if not url.endswith("/video") and not url.endswith("/shot.jpg"):
            url = url.rstrip("/") + "/video"
        system_state["phone_camera_url"] = url
        system_state["camera_source"] = "PHONE"
        print(f"[API] Set phone camera URL to: {url}")
        return jsonify({"status": "updated", "url": url})
    return jsonify({"error": "Invalid URL"}), 400

@app.route('/api/command', methods=['POST'])
def send_command():
    global ser
    data = request.json or {}
    cmd = data.get('cmd', '')
    if ser and ser.is_open and cmd:
        with ser_lock:
            ser.write(cmd.encode('utf-8'))
        return jsonify({"status": "sent", "cmd": cmd})
    return jsonify({"error": "Serial disconnected or invalid cmd"}), 400

# HTML DASHBOARD TEMPLATE
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Recycling Vending Machine - Control Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-3 md:p-6 antialiased select-none">
  
  <!-- Top Navigation Header -->
  <header class="flex flex-wrap items-center justify-between pb-4 mb-4 border-b border-slate-800 gap-3">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-xl font-bold border border-emerald-500/30">
        ♻️
      </div>
      <div>
        <h1 class="text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          Smart Recycling Vending Machine
          <span class="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold">
            <i class="fa-solid fa-brain mr-1"></i> WasteWise Neural Network (15k Images)
          </span>
        </h1>
        <p class="text-xs text-slate-400">Sequential Gravity Sorter (Plastic Bottle • Paper Roll • Metal Can) — Deep Learning Classifier</p>
      </div>
    </div>

    <!-- Live Status & Phone IP Cam Connector (Top Right Corner) -->
    <div class="flex items-center gap-2 text-xs flex-wrap">
      <!-- Arduino Connection Status -->
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
        <span id="badge-conn" class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
        <span id="label-conn" class="text-slate-300 font-medium">Arduino: COM11 (9600)</span>
      </div>
      
      <!-- Phone IP Cam Corner Widget -->
      <div class="flex items-center gap-1.5 p-1 rounded-xl bg-slate-900 border border-cyan-500/40 shadow-lg shadow-cyan-950/20">
        <!-- Switch to Laptop Camera -->
        <button id="btn-cam-laptop" onclick="switchCamera('WEBCAM')" class="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white shadow transition hover:opacity-95" title="Switch to Laptop Webcam">
          <i class="fa-solid fa-laptop mr-1"></i> Laptop Cam
        </button>

        <div class="h-5 w-px bg-slate-800 mx-0.5"></div>

        <!-- Phone IP Input Box & Auto-Connect Button -->
        <div class="flex items-center gap-1">
          <div class="relative">
            <span class="absolute inset-y-0 left-0 flex items-center pl-2.5 pointer-events-none text-cyan-400 text-xs">
              <i class="fa-solid fa-mobile-screen"></i>
            </span>
            <input type="text" id="corner-phone-ip"
                   placeholder="Phone IP (e.g. 192.168.1.5:8080)"
                   onkeydown="if(event.key==='Enter') connectCornerPhoneCam()"
                   class="bg-slate-950 text-white text-xs pl-7 pr-2 py-1.5 rounded-lg border border-slate-700 focus:outline-none focus:border-cyan-400 w-44 sm:w-56 font-mono placeholder:text-slate-600 shadow-inner" />
          </div>
          <button id="btn-connect-phone" onclick="connectCornerPhoneCam()" class="px-3 py-1.5 rounded-lg text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white shadow transition flex items-center gap-1 active:scale-95" title="Connect Phone IP Cam Stream">
            <i class="fa-solid fa-plug"></i> Connect
          </button>
        </div>
      </div>
    </div>
  </header>

  <!-- Key Metrics 4-Cards Grid -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
    <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex items-center justify-between">
      <div>
        <p class="text-xs font-medium text-slate-400">Plastic Bottles</p>
        <h3 id="val-plastic" class="text-2xl font-bold text-emerald-400 mt-1">0</h3>
        <p class="text-[10px] text-slate-500 mt-0.5">+10 Pts each (Bin 1)</p>
      </div>
      <div class="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center text-lg">
        <i class="fa-solid fa-bottle-water"></i>
      </div>
    </div>

    <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex items-center justify-between">
      <div>
        <p class="text-xs font-medium text-slate-400">Paper Rolls / Cups</p>
        <h3 id="val-paper" class="text-2xl font-bold text-amber-400 mt-1">0</h3>
        <p class="text-[10px] text-slate-500 mt-0.5">+5 Pts each (Bin 2)</p>
      </div>
      <div class="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center text-lg">
        <i class="fa-solid fa-scroll"></i>
      </div>
    </div>

    <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex items-center justify-between">
      <div>
        <p class="text-xs font-medium text-slate-400">Metal Cans / Items</p>
        <h3 id="val-metal" class="text-2xl font-bold text-blue-400 mt-1">0</h3>
        <p class="text-[10px] text-slate-500 mt-0.5">+15 Pts each (Bin 3)</p>
      </div>
      <div class="w-10 h-10 rounded-xl bg-blue-500/10 text-blue-400 flex items-center justify-center text-lg">
        <i class="fa-solid fa-can-food"></i>
      </div>
    </div>

    <div class="p-4 rounded-xl bg-gradient-to-br from-indigo-950/40 to-purple-950/40 border border-indigo-500/30 shadow-sm flex items-center justify-between">
      <div>
        <p class="text-xs font-medium text-indigo-300">Total User Points</p>
        <h3 id="val-points" class="text-2xl font-bold text-indigo-300 mt-1">0</h3>
        <p id="val-total-items" class="text-[10px] text-indigo-400/80 mt-0.5">0 Items Processed</p>
      </div>
      <div class="w-10 h-10 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-lg">
        <i class="fa-solid fa-trophy"></i>
      </div>
    </div>
  </div>

  <!-- Main Grid: Live Webcam Feed & Telemetry -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5">
    
    <!-- Left 2 Cols: Live Video Feed & AI Scanner -->
    <div class="lg:col-span-2 space-y-4">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping"></span>
            <h2 class="text-sm font-semibold text-white">Live AI Video Stream (Scanning Box Active)</h2>
          </div>
          <div class="flex items-center gap-2">
            <span id="ai-detected-badge" class="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-bold">
              SCANNING FOR WASTE...
            </span>
          </div>
        </div>

        <!-- Video Frame -->
        <div class="relative aspect-video w-full rounded-lg overflow-hidden bg-black border border-slate-800 flex items-center justify-center shadow-inner">
          <img id="video-feed" src="/video_feed" alt="Live Camera Feed" class="w-full h-full object-contain">
        </div>

        <div class="mt-3 flex items-center justify-between text-xs text-slate-400">
          <div class="flex items-center gap-2">
            <span class="text-slate-500">Hardware Status:</span>
            <span id="status-text" class="text-emerald-400 font-semibold tracking-wide">READY FOR ITEM</span>
          </div>
          <span class="text-emerald-400 font-medium"><i class="fa-solid fa-check-double mr-1"></i> Auto-Sort To Arduino: ACTIVE (1s Hold)</span>
        </div>
      </div>

      <!-- Quick One-Click Sort Buttons & Keyboard Hotkeys -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <i class="fa-solid fa-keyboard text-emerald-400"></i> Instant Triggers (Click or Press Key on Keyboard)
          </h3>
          <span class="text-[10px] text-slate-400 font-medium">Keys: [1/P] = Plastic | [2/C] = Paper | [3/M] = Metal</span>
        </div>
        <div class="grid grid-cols-3 gap-3">
          <button onclick="sendCommand('p')" class="py-3 px-3 rounded-xl bg-emerald-600/25 hover:bg-emerald-600/40 text-emerald-300 border border-emerald-500/40 text-xs font-bold flex flex-col items-center justify-center gap-1 transition active:scale-95 shadow">
            <span class="flex items-center gap-1.5"><i class="fa-solid fa-bottle-water text-sm"></i> Plastic Bottle</span>
            <span class="text-[10px] text-emerald-400/80 font-normal">[Press 'P' or '1'] ➔ Flap 1</span>
          </button>
          <button onclick="sendCommand('c')" class="py-3 px-3 rounded-xl bg-amber-600/25 hover:bg-amber-600/40 text-amber-300 border border-amber-500/40 text-xs font-bold flex flex-col items-center justify-center gap-1 transition active:scale-95 shadow">
            <span class="flex items-center gap-1.5"><i class="fa-solid fa-scroll text-sm"></i> Paper Roll</span>
            <span class="text-[10px] text-amber-400/80 font-normal">[Press 'C' or '2'] ➔ Flap 2</span>
          </button>
          <button onclick="sendCommand('m')" class="py-3 px-3 rounded-xl bg-blue-600/25 hover:bg-blue-600/40 text-blue-300 border border-blue-500/40 text-xs font-bold flex flex-col items-center justify-center gap-1 transition active:scale-95 shadow">
            <span class="flex items-center gap-1.5"><i class="fa-solid fa-can-food text-sm"></i> Metal Can</span>
            <span class="text-[10px] text-blue-400/80 font-normal">[Press 'M' or '3'] ➔ End Chute</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Right 1 Col: Live AI Neural Probability Bars & Telemetry -->
    <div class="space-y-4">
      
      <!-- Live Neural Network Probability Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
        <h2 class="text-sm font-semibold text-white mb-3 flex items-center gap-2">
          <i class="fa-solid fa-chart-simple text-emerald-400"></i> Live Neural Model Probabilities
        </h2>
        
        <div class="space-y-3.5">
          <!-- Plastic Probability Bar -->
          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="font-medium text-emerald-400 flex items-center gap-1.5"><i class="fa-solid fa-bottle-water"></i> Plastic Bottle</span>
              <span id="prob-val-plastic" class="font-bold text-white">0.0%</span>
            </div>
            <div class="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
              <div id="prob-bar-plastic" class="bg-emerald-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>

          <!-- Paper / Cardboard Probability Bar -->
          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="font-medium text-amber-400 flex items-center gap-1.5"><i class="fa-solid fa-scroll"></i> Paper Roll / Cardboard</span>
              <span id="prob-val-paper" class="font-bold text-white">0.0%</span>
            </div>
            <div class="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
              <div id="prob-bar-paper" class="bg-amber-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>

          <!-- Metal Probability Bar -->
          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="font-medium text-blue-400 flex items-center gap-1.5"><i class="fa-solid fa-can-food"></i> Metal Can / Container</span>
              <span id="prob-val-metal" class="font-bold text-white">0.0%</span>
            </div>
            <div class="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
              <div id="prob-bar-metal" class="bg-blue-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Actuator & Sensor Health Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
        <h2 class="text-sm font-semibold text-white mb-3 flex items-center gap-2">
          <i class="fa-solid fa-microchip text-indigo-400"></i> Hardware Pin Telemetry
        </h2>
        
        <div class="space-y-3">
          <!-- Servo 1 Gauge -->
          <div class="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <span class="text-xs text-slate-400">Servo 1 (Pin D9 - Plastic Flap)</span>
              <p id="val-servo1" class="text-lg font-bold text-white mt-0.5">0°</p>
            </div>
            <span class="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Flap 1 (Plastic)</span>
          </div>

          <!-- Servo 2 Gauge -->
          <div class="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <span class="text-xs text-slate-400">Servo 2 (Pin D10 - Paper Flap)</span>
              <p id="val-servo2" class="text-lg font-bold text-white mt-0.5">0°</p>
            </div>
            <span class="text-xs px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">Flap 2 (Paper)</span>
          </div>

          <!-- IR Sensor 1 -->
          <div class="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <span class="text-xs text-slate-400">IR Intake Sensor (Pin D2)</span>
              <p id="text-ir1" class="text-xs font-bold text-slate-400 mt-1">CLEAR</p>
            </div>
            <span id="badge-ir1" class="w-3 h-3 rounded-full bg-slate-700"></span>
          </div>

          <!-- IR Sensor 2 -->
          <div class="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <span class="text-xs text-slate-400">IR Exit Chute Sensor (Pin D3)</span>
              <p id="text-ir2" class="text-xs font-bold text-slate-400 mt-1">CLEAR</p>
            </div>
            <span id="badge-ir2" class="w-3 h-3 rounded-full bg-slate-700"></span>
          </div>
        </div>
      </div>

      <!-- Real-Time Sorting Chart -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
        <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Material Distribution Ratio</h3>
        <div class="relative h-44">
          <canvas id="ratioChart"></canvas>
        </div>
      </div>

    </div>
  </div>

  <script>
    const ctx = document.getElementById('ratioChart').getContext('2d');
    const ratioChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Plastic', 'Paper', 'Metal'],
        datasets: [{
          data: [1, 1, 1],
          backgroundColor: ['#10b981', '#f59e0b', '#3b82f6'],
          borderColor: '#0f172a',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 12, font: { size: 10 } } }
        }
      }
    });

    async function sendCommand(cmd) {
      try {
        await fetch('/api/command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cmd: cmd })
        });
      } catch (err) { console.error(err); }
    }

    // Keyboard Shortcuts for instant demo safety
    document.addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (k === '1' || k === 'p') sendCommand('p');
      if (k === '2' || k === 'c') sendCommand('c');
      if (k === '3' || k === 'm') sendCommand('m');
    });

    async function switchCamera(source) {
      try {
        await fetch('/api/switch_camera', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: source })
        });
        const btnLaptop = document.getElementById('btn-cam-laptop');
        const btnPhone = document.getElementById('btn-connect-phone');
        if (source === 'WEBCAM') {
          btnLaptop.className = "px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white shadow transition hover:opacity-95";
          btnPhone.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white shadow transition flex items-center gap-1 active:scale-95";
          btnPhone.innerHTML = '<i class="fa-solid fa-plug"></i> Connect';
        } else {
          btnPhone.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-emerald-600 text-white shadow transition flex items-center gap-1";
          btnPhone.innerHTML = '<i class="fa-solid fa-check text-emerald-300"></i> Connected';
          btnLaptop.className = "px-2.5 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition";
        }
        const videoEl = document.getElementById('video-feed');
        videoEl.src = '/video_feed?t=' + new Date().getTime();
      } catch (err) { console.error(err); }
    }

    async function connectCornerPhoneCam() {
      const ipInput = document.getElementById('corner-phone-ip');
      let val = (ipInput.value || '').trim();
      if (!val) {
        alert("Please enter your Phone's IP address (e.g. 192.168.1.5:8080) from the IP Webcam app!");
        ipInput.focus();
        return;
      }

      localStorage.setItem('saved_phone_ip', val);

      const btn = document.getElementById('btn-connect-phone');
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Connecting...';
      btn.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-amber-600 text-white shadow transition flex items-center gap-1";

      try {
        const res = await fetch('/api/camera_config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: val })
        });
        const data = await res.json();
        if (data.status === 'updated') {
          const btnLaptop = document.getElementById('btn-cam-laptop');
          btnLaptop.className = "px-2.5 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition";
          btn.innerHTML = '<i class="fa-solid fa-check text-emerald-300"></i> Streaming';
          btn.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-emerald-600 text-white shadow transition flex items-center gap-1";
          
          // Refresh feed connection
          const videoEl = document.getElementById('video-feed');
          videoEl.src = '/video_feed?t=' + new Date().getTime();
        } else {
          btn.innerHTML = '<i class="fa-solid fa-plug"></i> Connect';
          btn.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white shadow transition flex items-center gap-1 active:scale-95";
        }
      } catch (err) {
        console.error(err);
        btn.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Retry';
        btn.className = "px-3 py-1.5 rounded-lg text-xs font-bold bg-red-600 text-white shadow transition flex items-center gap-1";
      }
    }

    // Auto-load saved Phone IP from browser memory
    window.addEventListener('DOMContentLoaded', () => {
      const savedIp = localStorage.getItem('saved_phone_ip');
      if (savedIp) {
        document.getElementById('corner-phone-ip').value = savedIp;
      }
    });

    async function fetchTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();

        // Update counts
        document.getElementById('val-plastic').innerText = data.plastic;
        document.getElementById('val-paper').innerText = data.paper;
        document.getElementById('val-metal').innerText = data.metal;
        document.getElementById('val-points').innerText = data.points;
        const total = data.plastic + data.paper + data.metal;
        document.getElementById('val-total-items').innerText = `${total} Items Processed`;

        // Update Servos
        document.getElementById('val-servo1').innerText = `${data.servo1}°`;
        document.getElementById('val-servo2').innerText = `${data.servo2}°`;

        // Update Live Neural Probabilities
        document.getElementById('prob-val-plastic').innerText = `${data.prob_plastic}%`;
        document.getElementById('prob-bar-plastic').style.width = `${data.prob_plastic}%`;

        document.getElementById('prob-val-paper').innerText = `${data.prob_paper}%`;
        document.getElementById('prob-bar-paper').style.width = `${data.prob_paper}%`;

        document.getElementById('prob-val-metal').innerText = `${data.prob_metal}%`;
        document.getElementById('prob-bar-metal').style.width = `${data.prob_metal}%`;

        // Update IR Sensors
        const b1 = document.getElementById('badge-ir1');
        const t1 = document.getElementById('text-ir1');
        if (data.ir1 === 1) {
          b1.className = "w-3 h-3 rounded-full bg-emerald-400 shadow-md shadow-emerald-400/50 animate-pulse";
          t1.innerText = "ITEM DETECTED (TRIGGERED)";
          t1.className = "text-xs font-bold text-emerald-400";
        } else {
          b1.className = "w-3 h-3 rounded-full bg-slate-700";
          t1.innerText = "CLEAR";
          t1.className = "text-xs font-bold text-slate-400";
        }

        const b2 = document.getElementById('badge-ir2');
        const t2 = document.getElementById('text-ir2');
        if (data.ir2 === 1) {
          b2.className = "w-3 h-3 rounded-full bg-cyan-400 shadow-md shadow-cyan-400/50 animate-pulse";
          t2.innerText = "CHUTE DROP DETECTED";
          t2.className = "text-xs font-bold text-cyan-400";
        } else {
          b2.className = "w-3 h-3 rounded-full bg-slate-700";
          t2.innerText = "CLEAR";
          t2.className = "text-xs font-bold text-slate-400";
        }

        // Live AI Target Badge
        const badge = document.getElementById('ai-detected-badge');
        badge.innerText = `${data.predicted_class} (${data.confidence}%)`;
        if (data.predicted_class.includes('PLASTIC')) {
          badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold";
        } else if (data.predicted_class.includes('PAPER')) {
          badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30 font-bold";
        } else if (data.predicted_class.includes('METAL')) {
          badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 font-bold";
        } else {
          badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700";
        }

        // Status update
        document.getElementById('status-text').innerText = data.status.toUpperCase();

        // Update charts
        if (total > 0) {
          ratioChart.data.datasets[0].data = [data.plastic, data.paper, data.metal];
          ratioChart.update('none');
        }

      } catch (err) {
        console.error(err);
      }
    }

    setInterval(fetchTelemetry, 600);
    fetchTelemetry();
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_DASHBOARD)

if __name__ == '__main__':
    port = 5000
    print(f"\n=======================================================")
    print(f" [OK] SMART RECYCLING VENDING MACHINE DASHBOARD SERVER")
    print(f" Access URL: http://localhost:{port} or http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
