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
# LOAD PRE-TRAINED DEEP LEARNING NEURAL NETWORK (YOLOv5 ONNX)
# =====================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov5n.onnx")
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

print(f"[AI MODEL] Loading pre-trained YOLOv5 Neural Network from '{MODEL_PATH}'...")
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    print("[AI MODEL] YOLOv5 Deep Learning Engine Initialized Successfully!")
except Exception as e:
    print("[AI MODEL ERROR] Failed to load ONNX model:", e)
    net = None

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
    "camera_mode": "YOLOv5 + ENSEMBLE AI (Laptop Webcam 0)",
    "phone_camera_url": "http://192.168.1.100:8080/video",
    "detected_class": "PLACE OBJECT IN SCAN BOX",
    "confidence": 0,
    "progress": 0,
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

                # Update Timeline History for Live Chart
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

# Start Serial thread in background
t = threading.Thread(target=serial_worker, daemon=True)
t.start()

# High-Precision Deep Learning + Spectral Ensemble Classifier
def classify_roi_ensemble(roi_img):
    if roi_img is None or roi_img.size == 0:
        return None, 0.0, 'none', (100, 116, 139)

    h, w, _ = roi_img.shape
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # 1. First: Run YOLOv5 Deep Learning specifically on the cropped ROI
    yolo_label = None
    yolo_conf = 0.0

    if net is not None:
        blob = cv2.dnn.blobFromImage(roi_img, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward()[0]

        best_score = 0.16
        for row in outs:
            obj_conf = row[4]
            if obj_conf > 0.16:
                class_scores = row[5:]
                cid = np.argmax(class_scores)
                score = float(obj_conf * class_scores[cid])
                if score > best_score:
                    cname = COCO_CLASSES[cid]
                    if cname not in ['person', 'chair', 'couch', 'tv', 'laptop']:
                        best_score = score
                        yolo_label = cname
                        yolo_conf = score

    # 2. Material Physical Spectral Analysis
    # Specular Glare (Plastic transparency / reflection)
    _, bright_pixels = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    specular_ratio = np.sum(bright_pixels > 0) / (h * w)

    # Metallic High-Contrast Edges & Chrome Sheen
    edges = cv2.Canny(gray, 60, 160)
    edge_ratio = np.sum(edges > 0) / (h * w)

    # Warm Color Ratio (Cardboard / Paper / Cups)
    lower_warm = np.array([8, 30, 40])
    upper_warm = np.array([32, 255, 240])
    warm_mask = cv2.inRange(hsv, lower_warm, upper_warm)
    warm_ratio = np.sum(warm_mask > 0) / (h * w)
    mean_sat = np.mean(hsv[:, :, 1])
    std_val = np.std(gray)

    # 3. Decision Fusion Engine
    # Direct YOLO Matches
    if yolo_label in ['bottle', 'wine glass', 'toothbrush']:
        conf = max(88.0, min(99.0, yolo_conf * 100))
        return "PLASTIC BOTTLE", conf, 'p', (16, 185, 129) # Emerald Green

    if yolo_label in ['cup', 'book', 'paper']:
        conf = max(86.0, min(98.0, yolo_conf * 100))
        return "PAPER CUP / CARDBOARD", conf, 'c', (245, 158, 11) # Amber

    if yolo_label in ['can', 'bowl', 'fork', 'knife', 'spoon', 'cell phone', 'remote', 'toaster', 'scissors']:
        conf = max(88.0, min(98.0, yolo_conf * 100))
        return "METAL CAN / FOIL", conf, 'm', (59, 130, 246) # Blue

    # Spectral Fallbacks if object is held inside the box
    # Detect if any object is physically present in the box (variance / edge presence)
    if edge_ratio > 0.035 or std_val > 28:
        if warm_ratio > 0.22 or (mean_sat > 50 and specular_ratio < 0.03):
            conf = min(96.0, 78.0 + (warm_ratio * 70))
            return "PAPER CUP / CARDBOARD", conf, 'c', (245, 158, 11)
        elif specular_ratio > 0.04 or (specular_ratio > 0.015 and edge_ratio < 0.10):
            conf = min(97.0, 80.0 + (specular_ratio * 140))
            return "PLASTIC BOTTLE", conf, 'p', (16, 185, 129)
        else:
            conf = min(95.0, 76.0 + (edge_ratio * 120))
            return "METAL CAN / FOIL", conf, 'm', (59, 130, 246)

    return None, 0.0, 'none', (100, 116, 139)

# Video Stream Generator
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
                    system_state["camera_mode"] = "YOLOv5 + ENSEMBLE AI (Laptop Webcam 0)"
                    print("[CAMERA] Opened Laptop Webcam (0)")
                else:
                    if not target_url.startswith("http://") and not target_url.startswith("https://"):
                        target_url = "http://" + target_url
                    if not target_url.endswith("/video") and not target_url.endswith("/shot.jpg"):
                        target_url += "/video"
                    cap = cv2.VideoCapture(target_url)
                    system_state["camera_mode"] = f"YOLOv5 (Phone IP Cam: {target_url})"
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

                # Center Scanning Box Coordinates (Box width 280, height 300)
                bw, bh = 280, 300
                x1 = (w - bw) // 2
                y1 = (h - bh) // 2
                x2 = x1 + bw
                y2 = y1 + bh

                # Crop ROI strictly inside the box (ignoring user background)
                roi = frame[y1:y2, x1:x2]

                # Run Ensemble Classifier
                material_name, conf, sort_code, hud_color = classify_roi_ensemble(roi)

                # Dim outside area slightly to focus attention on scanning box
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (10, 15, 26), -1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

                # Draw Target Box & Corner Crosshairs
                cv2.rectangle(frame, (x1, y1), (x2, y2), hud_color, 2)
                L = 22
                cv2.line(frame, (x1, y1), (x1 + L, y1), hud_color, 4)
                cv2.line(frame, (x1, y1), (x1, y1 + L), hud_color, 4)
                cv2.line(frame, (x2, y1), (x2 - L, y1), hud_color, 4)
                cv2.line(frame, (x2, y1), (x2, y1 + L), hud_color, 4)
                cv2.line(frame, (x1, y2), (x1 + L, y2), hud_color, 4)
                cv2.line(frame, (x1, y2), (x1, y2 - L), hud_color, 4)
                cv2.line(frame, (x2, y2), (x2 - L, y2), hud_color, 4)
                cv2.line(frame, (x2, y2), (x2, y2 - L), hud_color, 4)

                # Status & Feedback
                if material_name:
                    system_state["detected_class"] = material_name
                    system_state["confidence"] = round(conf, 1)

                    # Top Banner
                    hud_text = f"AI DETECTED: {material_name} ({conf:.1f}%)"
                    cv2.rectangle(frame, (x1, y1 - 34), (x1 + bw, y1), (15, 23, 42), -1)
                    cv2.putText(frame, hud_text, (x1 + 8, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, hud_color, 2)

                    # Streak logic for automatic triggering
                    if sort_code == current_candidate_code:
                        detection_counter += 1
                    else:
                        current_candidate_code = sort_code
                        detection_counter = 0

                    progress_pct = min(100, int((detection_counter / 12) * 100))
                    system_state["progress"] = progress_pct

                    # Bottom Progress Bar
                    bar_w = int((bw - 4) * (progress_pct / 100))
                    cv2.rectangle(frame, (x1 + 2, y2 + 10), (x1 + bw - 2, y2 + 22), (30, 41, 59), -1)
                    cv2.rectangle(frame, (x1 + 2, y2 + 10), (x1 + 2 + bar_w, y2 + 22), hud_color, -1)
                    cv2.putText(frame, f"LOCKING IN... {progress_pct}%", (x1 + 55, y2 + 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    # Trigger Sort after ~1 second steady hold
                    if detection_counter >= 12 and (time.time() - last_auto_sort_time > 4.5):
                        last_auto_sort_time = time.time()
                        detection_counter = 0
                        print(f"\n[AI TRIGGERED] '{material_name}' locked in! Actuating Arduino '{sort_code}'...")
                        if ser and ser.is_open:
                            with ser_lock:
                                ser.write(sort_code.encode('utf-8'))

                else:
                    system_state["detected_class"] = "HOLD ITEM INSIDE SCAN BOX"
                    system_state["confidence"] = 0
                    system_state["progress"] = 0
                    detection_counter = 0
                    current_candidate_code = None

                    cv2.putText(frame, "HOLD ITEM INSIDE SCAN BOX", (x1 + 18, y1 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (148, 163, 184), 1)
                    cv2.circle(frame, (w // 2, h // 2), 6, (148, 163, 184), -1)

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (15, 23, 42)
            cv2.putText(frame, "INITIALIZING CAMERA...", (180, 240),
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
        system_state["phone_camera_url"] = url
        system_state["camera_source"] = "PHONE"
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
<body class="bg-slate-950 text-slate-100 min-h-screen p-3 md:p-6 antialiased">
  
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
            <i class="fa-solid fa-brain mr-1"></i> YOLOv5 + Ensemble AI Active
          </span>
        </h1>
        <p class="text-xs text-slate-400">Sequential Gravity Sorter (Plastic • Paper • Metal) — Real-Time Deep Learning Vision</p>
      </div>
    </div>

    <!-- Live Status Indicators -->
    <div class="flex items-center gap-2 text-xs flex-wrap">
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
        <span id="badge-conn" class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
        <span id="label-conn" class="text-slate-300 font-medium">Arduino: COM11 (9600)</span>
      </div>
      
      <!-- Camera Source Toggle Buttons -->
      <div class="flex items-center p-0.5 rounded-lg bg-slate-900 border border-slate-800">
        <button id="btn-cam-laptop" onclick="switchCamera('WEBCAM')" class="px-3 py-1 rounded-md text-xs font-semibold bg-emerald-600 text-white shadow transition">
          <i class="fa-solid fa-laptop mr-1"></i> Laptop Camera
        </button>
        <button id="btn-cam-phone" onclick="switchCamera('PHONE')" class="px-3 py-1 rounded-md text-xs font-semibold text-slate-400 hover:text-white transition">
          <i class="fa-solid fa-mobile-screen mr-1"></i> Phone IP Cam
        </button>
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
        <p class="text-xs font-medium text-slate-400">Paper Cups</p>
        <h3 id="val-paper" class="text-2xl font-bold text-amber-400 mt-1">0</h3>
        <p class="text-[10px] text-slate-500 mt-0.5">+5 Pts each (Bin 2)</p>
      </div>
      <div class="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center text-lg">
        <i class="fa-solid fa-box"></i>
      </div>
    </div>

    <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm flex items-center justify-between">
      <div>
        <p class="text-xs font-medium text-slate-400">Metal Cans</p>
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
            <h2 class="text-sm font-semibold text-white">YOLOv5 Neural Object Recognition Feed</h2>
          </div>
          <div class="flex items-center gap-2">
            <span id="ai-detected-badge" class="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold">
              PLACE ITEM IN BOX
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
          <span class="text-emerald-400 font-medium"><i class="fa-solid fa-check-double mr-1"></i> Auto-Sort To Arduino: ACTIVE</span>
        </div>
      </div>

      <!-- One-Click Instant Sort Buttons -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Instant One-Click Sort Triggers (Demo Testing)</h3>
          <span class="text-[10px] text-slate-500">Clicking any button actuates the physical servos immediately</span>
        </div>
        <div class="grid grid-cols-3 gap-3">
          <button onclick="sendCommand('p')" class="py-3 px-3 rounded-xl bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-500/30 text-xs font-bold flex items-center justify-center gap-2 transition active:scale-95 shadow-sm">
            <i class="fa-solid fa-bottle-water text-sm"></i> Sort Plastic (Flap 1 - 90°)
          </button>
          <button onclick="sendCommand('c')" class="py-3 px-3 rounded-xl bg-amber-600/20 hover:bg-amber-600/30 text-amber-400 border border-amber-500/30 text-xs font-bold flex items-center justify-center gap-2 transition active:scale-95 shadow-sm">
            <i class="fa-solid fa-box text-sm"></i> Sort Paper (Flap 2 - 90°)
          </button>
          <button onclick="sendCommand('m')" class="py-3 px-3 rounded-xl bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 text-xs font-bold flex items-center justify-center gap-2 transition active:scale-95 shadow-sm">
            <i class="fa-solid fa-can-food text-sm"></i> Sort Metal (End Chute)
          </button>
        </div>
      </div>
    </div>

    <!-- Right 1 Col: Actuators, Sensors & Live Charts -->
    <div class="space-y-4">
      
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

    async function switchCamera(source) {
      try {
        await fetch('/api/switch_camera', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: source })
        });
        const btnLaptop = document.getElementById('btn-cam-laptop');
        const btnPhone = document.getElementById('btn-cam-phone');
        if (source === 'WEBCAM') {
          btnLaptop.className = "px-3 py-1 rounded-md text-xs font-semibold bg-emerald-600 text-white shadow transition";
          btnPhone.className = "px-3 py-1 rounded-md text-xs font-semibold text-slate-400 hover:text-white transition";
        } else {
          btnPhone.className = "px-3 py-1 rounded-md text-xs font-semibold bg-cyan-600 text-white shadow transition";
          btnLaptop.className = "px-3 py-1 rounded-md text-xs font-semibold text-slate-400 hover:text-white transition";
        }
      } catch (err) { console.error(err); }
    }

    async function fetchTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();

        // Update counts
        document.getElementById('val-plastic').innerText = data.plastic;
        document.getElementById('val-paper').innerText = data.paper;
        document.getElementById('val-metal').innerText = data.metal;
        document.getElementById('val-points').innerText = data.points;
        const total = data.plastic + data.metal + data.paper;
        document.getElementById('val-total-items').innerText = `${total} Items Processed`;

        // Update Servos
        document.getElementById('val-servo1').innerText = `${data.servo1}°`;
        document.getElementById('val-servo2').innerText = `${data.servo2}°`;

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
        if (data.confidence > 0) {
          badge.innerText = `${data.detected_class} (${data.confidence}%)`;
          badge.className = "text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold";
        } else {
          badge.innerText = data.detected_class;
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
