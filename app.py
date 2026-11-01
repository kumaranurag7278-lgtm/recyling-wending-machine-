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
    "camera_mode": "SCANNER STANDBY (Awaiting Phone IP Webcam)",
    "phone_camera_url": "http://192.168.1.100:8080/video",
    "camera_connected": False
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
                    system_state["status"] = "CONNECTED TO HARDWARE (COM11)"
                    print("[SERIAL] Connected to COM11 @ 9600 baud successfully.")
                except Exception as e:
                    system_state["connected"] = False
                    system_state["status"] = f"DISCONNECTED: {str(e)[:25]}"
                    time.sleep(2)
                    continue

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                # 1. Parse JSON Telemetry
                if line.startswith('{"telemetry":'):
                    try:
                        data = json.loads(line).get("telemetry", {})
                        with ser_lock:
                            for k in ["ir1", "ir2", "servo1", "servo2", "plastic", "metal", "paper", "points", "status"]:
                                if k in data: system_state[k] = data[k]
                    except: pass

                # 2. Parse Text Telemetry: "[TELEMETRY] Total: 1 | Plastic: 1 | Paper: 0 | Metal: 0 | Points: 10"
                elif "[TELEMETRY]" in line:
                    match = re.search(r'Plastic:\s*(\d+).*?Paper:\s*(\d+).*?Metal:\s*(\d+).*?Points:\s*(\d+)', line)
                    if match:
                        with ser_lock:
                            system_state["plastic"] = int(match.group(1))
                            system_state["paper"]   = int(match.group(2))
                            system_state["metal"]   = int(match.group(3))
                            system_state["points"]  = int(match.group(4))
                            system_state["status"]  = "DEPOSIT COMPLETED & LOGGED"

                # 3. Parse Real-Time Event Notifications
                elif "Waste item placed" in line or "Item detected" in line:
                    with ser_lock:
                        system_state["ir1"] = 1
                        system_state["status"] = "ITEM DETECTED (RAMP START)"
                elif "CLASS: PLASTIC" in line:
                    with ser_lock:
                        system_state["status"] = "CLASSIFIED: PLASTIC (FLAP 1 OPEN)"
                        system_state["servo1"] = 90
                        system_state["servo2"] = 0
                elif "CLASS: PAPER" in line:
                    with ser_lock:
                        system_state["status"] = "CLASSIFIED: PAPER (FLAP 2 OPEN)"
                        system_state["servo1"] = 0
                        system_state["servo2"] = 90
                elif "CLASS: METAL" in line:
                    with ser_lock:
                        system_state["status"] = "CLASSIFIED: METAL (CHUTE EXIT)"
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

# Live Video Stream Generator (Phone IP Webcam + AI HUD Simulation)
def generate_camera_frames():
    global system_state
    cap = None
    current_url = ""

    while True:
        target_url = system_state["phone_camera_url"].strip()
        
        # Normalize IP Webcam URLs (e.g. "192.168.1.5:8080" -> "http://192.168.1.5:8080/video")
        if not target_url.startswith("http://") and not target_url.startswith("https://"):
            target_url = "http://" + target_url
        if not target_url.endswith("/video") and not target_url.endswith("/shot.jpg") and not target_url.endswith(":8080/video"):
            if target_url.endswith(":8080"):
                target_url += "/video"

        # Reopen capture if URL changed
        if current_url != target_url:
            if cap: cap.release()
            try:
                cap = cv2.VideoCapture(target_url)
                current_url = target_url
            except:
                cap = None

        frame = None
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = None
                system_state["camera_connected"] = False
            else:
                system_state["camera_connected"] = True
                system_state["camera_mode"] = f"PHONE IP WEBCAM LIVE ({current_url})"
                
                # Overlay AI Computer Vision Bounding Box & HUD
                h, w, _ = frame.shape
                cx, cy = w // 2, h // 2
                bw, bh = int(w * 0.45), int(h * 0.55)
                x1, y1 = cx - bw // 2, cy - bh // 2
                x2, y2 = cx + bw // 2, cy + bh // 2

                # Corner Reticles
                color = (34, 197, 94) # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 6, color, -1)

                # AI Overlay Label
                label = f"AI DETECTION: ACTIVE | CLASS: {system_state['status'][:18]}"
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + 380, y1), (15, 23, 42), -1)
                cv2.putText(frame, label, (x1 + 8, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 2)

        if frame is None:
            # Fallback high-tech standby scanner screen
            system_state["camera_connected"] = False
            system_state["camera_mode"] = "STANDBY: Enter Phone IP Webcam URL"
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (15, 23, 42)

            # Scanner Grid & Reticle
            cv2.rectangle(frame, (80, 60), (560, 420), (56, 189, 248), 2)
            cv2.circle(frame, (320, 240), 90, (59, 130, 246), 1)
            cv2.circle(frame, (320, 240), 8, (34, 197, 94), -1)

            cv2.putText(frame, "PHONE IP WEBCAM AI SCANNER", (30, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
            cv2.putText(frame, f"URL: {system_state['phone_camera_url']}", (90, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
            cv2.putText(frame, "STATUS: Waiting for phone stream on port 8080...", (90, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 204, 21), 1)
            
            status_txt = system_state["status"]
            cv2.putText(frame, f"SYSTEM: {status_txt}", (30, 455),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            scan_y = int(60 + (time.time() * 140) % 360)
            cv2.line(frame, (80, scan_y), (560, scan_y), (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def get_telemetry():
    return jsonify(system_state)

@app.route('/api/camera_config', methods=['POST'])
def set_camera_url():
    global system_state
    data = request.json or {}
    url = data.get('url', '').strip()
    if url:
        system_state["phone_camera_url"] = url
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
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; }
    .pulse-glow { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .4; } }
  </style>
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
          <span class="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 font-medium">IoT & AI Connected</span>
        </h1>
        <p class="text-xs text-slate-400">Sequential Gravity Sorter (Plastic • Paper • Metal) — Live Telemetry & Control Center</p>
      </div>
    </div>

    <!-- Live Status Indicators -->
    <div class="flex items-center gap-2 text-xs flex-wrap">
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
        <span id="badge-conn" class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
        <span id="label-conn" class="text-slate-300 font-medium">Arduino Uno: COM11 (9600)</span>
      </div>
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
        <i class="fa-solid fa-mobile-screen text-cyan-400"></i>
        <span id="label-cam" class="text-slate-300">Phone IP Webcam</span>
      </div>
    </div>
  </header>

  <!-- Phone Camera URL Config Banner -->
  <div class="mb-5 p-3 rounded-xl bg-slate-900/90 border border-cyan-500/30 flex flex-wrap items-center justify-between gap-3 shadow-lg">
    <div class="flex items-center gap-2.5">
      <div class="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-sm">
        <i class="fa-solid fa-video"></i>
      </div>
      <div>
        <h4 class="text-xs font-bold text-white uppercase tracking-wider">Phone IP Webcam Stream</h4>
        <p class="text-xs text-slate-400">Phone me "IP Webcam" app on karein aur neeche dikhne wala URL enter karein</p>
      </div>
    </div>
    <div class="flex items-center gap-2 flex-grow max-w-lg">
      <input type="text" id="phone-cam-input" value="http://192.168.1.100:8080/video" 
             placeholder="e.g. http://192.168.1.5:8080/video" 
             class="bg-slate-950 border border-slate-700 text-xs text-white rounded-lg px-3 py-2 flex-grow focus:outline-none focus:border-cyan-500">
      <button onclick="updateCameraUrl()" class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-lg shadow-sm transition">
        Connect Stream
      </button>
    </div>
  </div>

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

  <!-- Main Grid: Camera Video Feed & Hardware Telemetry -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5">
    
    <!-- Left 2 Cols: Live Video Feed & System Status -->
    <div class="lg:col-span-2 space-y-4">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-red-500 animate-ping"></span>
            <h2 class="text-sm font-semibold text-white">Live AI Vision Stream</h2>
          </div>
          <span id="camera-mode-badge" class="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
            Connecting...
          </span>
        </div>

        <!-- Video Frame -->
        <div class="relative aspect-video w-full rounded-lg overflow-hidden bg-black border border-slate-800 flex items-center justify-center shadow-inner">
          <img id="video-feed" src="/video_feed" alt="Live Camera Feed" class="w-full h-full object-contain">
        </div>

        <div class="mt-3 flex items-center justify-between text-xs text-slate-400">
          <div class="flex items-center gap-2">
            <span class="text-slate-500">Status:</span>
            <span id="status-text" class="text-emerald-400 font-semibold tracking-wide">READY FOR ITEM</span>
          </div>
          <span class="text-slate-500">Latency: ~40ms</span>
        </div>
      </div>

      <!-- Manual Test Buttons -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
        <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Manual Hardware Override & Test Triggers</h3>
        <div class="grid grid-cols-3 gap-3">
          <button onclick="sendCommand('p')" class="py-2.5 px-3 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-500/30 text-xs font-semibold flex items-center justify-center gap-2 transition active:scale-95">
            <i class="fa-solid fa-bottle-water"></i> Test Plastic (Flap 1)
          </button>
          <button onclick="sendCommand('c')" class="py-2.5 px-3 rounded-lg bg-amber-600/20 hover:bg-amber-600/30 text-amber-400 border border-amber-500/30 text-xs font-semibold flex items-center justify-center gap-2 transition active:scale-95">
            <i class="fa-solid fa-box"></i> Test Paper (Flap 2)
          </button>
          <button onclick="sendCommand('m')" class="py-2.5 px-3 rounded-lg bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 text-xs font-semibold flex items-center justify-center gap-2 transition active:scale-95">
            <i class="fa-solid fa-can-food"></i> Test Metal (End Chute)
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
              <span class="text-xs text-slate-400">Servo 1 (D9 - Plastic Flap)</span>
              <p id="val-servo1" class="text-lg font-bold text-white mt-0.5">0°</p>
            </div>
            <span class="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Bin 1 Diverter</span>
          </div>

          <!-- Servo 2 Gauge -->
          <div class="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <span class="text-xs text-slate-400">Servo 2 (D10 - Paper Flap)</span>
              <p id="val-servo2" class="text-lg font-bold text-white mt-0.5">0°</p>
            </div>
            <span class="text-xs px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">Bin 2 Diverter</span>
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
    // Initialize Chart.js Doughnut
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

    async function updateCameraUrl() {
      const url = document.getElementById('phone-cam-input').value;
      if (!url) return;
      try {
        await fetch('/api/camera_config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url })
        });
        alert('Phone Camera URL updated! Stream connecting...');
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

        // Status & Mode
        document.getElementById('status-text').innerText = data.status.toUpperCase();
        document.getElementById('camera-mode-badge').innerText = data.camera_mode;

        // Chart Update
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
