import os
import sys
import time
import json
import threading
import serial
import cv2
from flask import Flask, render_template_string, jsonify, Response, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global Telemetry & System State
system_state = {
    "connected": False,
    "port": "COM11",
    "baud": 115200,
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
    "camera_mode": "WEBCAM / ESP-SIMULATOR"
}

ser = None
ser_lock = threading.Lock()

# Serial Reader Worker
def serial_worker():
    global ser, system_state
    while True:
        try:
            if ser is None or not ser.is_open:
                try:
                    ser = serial.Serial(system_state["port"], system_state["baud"], timeout=1)
                    system_state["connected"] = True
                    system_state["status"] = "CONNECTED TO HARDWARE (COM11)"
                    print("[SERIAL] Connected to COM11 successfully.")
                except Exception as e:
                    system_state["connected"] = False
                    system_state["status"] = f"DISCONNECTED: {str(e)[:30]}"
                    time.sleep(2)
                    continue

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                if line.startswith('{"telemetry":'):
                    try:
                        data = json.loads(line).get("telemetry", {})
                        with ser_lock:
                            for k in ["ir1", "ir2", "servo1", "servo2", "plastic", "metal", "paper", "points", "status"]:
                                if k in data:
                                    system_state[k] = data[k]
                            system_state["last_updated"] = time.time()
                            
                            # Keep sliding timeline for charts
                            system_state["history"].append({
                                "time": time.strftime("%H:%M:%S"),
                                "plastic": system_state["plastic"],
                                "metal": system_state["metal"],
                                "paper": system_state["paper"],
                                "points": system_state["points"]
                            })
                            if len(system_state["history"]) > 30:
                                system_state["history"].pop(0)
                    except json.JSONDecodeError:
                        pass
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

# Live Video Stream Generator (Connects to ESP32-CAM AP: http://192.168.4.1/stream or fallback)
def generate_camera_frames():
    esp_url = "http://192.168.4.1/stream"
    cap = None
    last_retry = 0

    while True:
        frame = None

        # Attempt to pull from ESP32-CAM stream
        if time.time() - last_retry > 5:
            last_retry = time.time()
            if cap is None or not cap.isOpened():
                try:
                    cap = cv2.VideoCapture(esp_url)
                except:
                    cap = None

        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = None
                frame = None
            else:
                system_state["camera_mode"] = "ESP32-CAM LIVE STREAM (192.168.4.1)"

        if frame is None:
            # Fallback high-tech standby scanner while connecting to ESP32-CAM AP
            system_state["camera_mode"] = "SEARCHING FOR ESP32-CAM AP (192.168.4.1)..."
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (15, 23, 42)

            # Scanner Grid & Reticle
            cv2.rectangle(frame, (80, 60), (560, 420), (56, 189, 248), 2)
            cv2.circle(frame, (320, 240), 90, (59, 130, 246), 1)
            cv2.circle(frame, (320, 240), 8, (34, 197, 94), -1)

            cv2.putText(frame, "AI ESP32-CAM [OV2640] STREAM FEED", (30, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
            cv2.putText(frame, "TARGET AP: 'RecycleMachine-CAM' (192.168.4.1)", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
            
            status_txt = system_state["status"]
            cv2.putText(frame, f"STATUS: {status_txt}", (30, 455),
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
  <title>Smart Recycling Vending Machine - Live Control Dashboard</title>
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
        <p class="text-xs text-slate-400">Automated 3-Stream Material Recognition (Plastic • Metal • Paper) — Live SCADA Center</p>
      </div>
    </div>

    <!-- Live Status Indicators -->
    <div class="flex items-center gap-2 text-xs flex-wrap">
      <div id="conn-pill" class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-semibold">
        <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
        <span id="conn-text">COM11 Connected</span>
      </div>
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
        <i class="fa-solid fa-clock text-slate-400"></i>
        <span id="clock-display">--:--:--</span>
      </div>
    </div>
  </header>

  <!-- Metric Highlights (Row 1) -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 mb-5">
    <!-- Plastic Card -->
    <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-lg hover:border-amber-500/40 transition">
      <div class="flex items-center justify-between text-xs text-amber-400 font-bold uppercase tracking-wider mb-1">
        <span>🟡 Bin 1: Plastic</span>
        <span class="text-slate-400">10 PTS / Item</span>
      </div>
      <div class="flex items-baseline justify-between mt-2">
        <span id="val-plastic" class="text-3xl font-extrabold text-white">0</span>
        <span class="text-xs text-slate-400">Bottles / Wrappers</span>
      </div>
      <div class="w-full bg-slate-950 h-1.5 rounded-full mt-3 overflow-hidden">
        <div id="prog-plastic" class="bg-amber-400 h-full w-0 transition-all duration-500"></div>
      </div>
    </div>

    <!-- Metal Card -->
    <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-lg hover:border-cyan-500/40 transition">
      <div class="flex items-center justify-between text-xs text-cyan-400 font-bold uppercase tracking-wider mb-1">
        <span>⚪ Bin 2: Metal</span>
        <span class="text-slate-400">15 PTS / Item</span>
      </div>
      <div class="flex items-baseline justify-between mt-2">
        <span id="val-metal" class="text-3xl font-extrabold text-white">0</span>
        <span class="text-xs text-slate-400">Cans / Foils / Tins</span>
      </div>
      <div class="w-full bg-slate-950 h-1.5 rounded-full mt-3 overflow-hidden">
        <div id="prog-metal" class="bg-cyan-400 h-full w-0 transition-all duration-500"></div>
      </div>
    </div>

    <!-- Paper Card -->
    <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-lg hover:border-emerald-500/40 transition">
      <div class="flex items-center justify-between text-xs text-emerald-400 font-bold uppercase tracking-wider mb-1">
        <span>🟢 Bin 3: Paper</span>
        <span class="text-slate-400">5 PTS / Item</span>
      </div>
      <div class="flex items-baseline justify-between mt-2">
        <span id="val-paper" class="text-3xl font-extrabold text-white">0</span>
        <span class="text-xs text-slate-400">Cups / Cardboard</span>
      </div>
      <div class="w-full bg-slate-950 h-1.5 rounded-full mt-3 overflow-hidden">
        <div id="prog-paper" class="bg-emerald-400 h-full w-0 transition-all duration-500"></div>
      </div>
    </div>

    <!-- Total Reward Points Card -->
    <div class="bg-gradient-to-br from-purple-900/40 to-slate-900 border border-purple-500/30 rounded-2xl p-4 shadow-lg">
      <div class="flex items-center justify-between text-xs text-purple-300 font-bold uppercase tracking-wider mb-1">
        <span>🏆 Total Rewards</span>
        <span class="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-200">Active</span>
      </div>
      <div class="flex items-baseline justify-between mt-2">
        <span id="val-points" class="text-3xl font-extrabold text-purple-200">0</span>
        <span class="text-xs text-purple-300">Eco Points</span>
      </div>
      <div class="text-[11px] text-purple-400/80 mt-2 flex items-center justify-between">
        <span>Total Processed:</span>
        <span id="val-total-items" class="font-bold text-white">0 Items</span>
      </div>
    </div>
  </div>

  <!-- Main Central Grid (Camera + Live Sensors + Charts) -->
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-5 mb-5">
    
    <!-- Left: ESP32-CAM AI Live Vision Stream (5 cols) -->
    <div class="lg:col-span-5 flex flex-col gap-3">
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col">
        <div class="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
          <div class="flex items-center gap-2">
            <i class="fa-solid fa-video text-emerald-400"></i>
            <h2 class="text-sm font-bold text-white uppercase tracking-wider">ESP32-CAM Live Vision Chamber</h2>
          </div>
          <span class="text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 font-bold animate-pulse">● LIVE AI STREAM</span>
        </div>

        <!-- Video Canvas / Stream -->
        <div class="relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 aspect-video flex items-center justify-center">
          <img src="/video_feed" alt="Live ESP32-CAM Stream" class="w-full h-full object-cover">
          <div class="absolute bottom-2 left-2 bg-slate-950/80 backdrop-blur-md px-2.5 py-1 rounded-md text-[11px] border border-slate-800 text-slate-300">
            Res: 640×480 | AI Model: MobileNetV2
          </div>
        </div>

        <!-- Live Machine Status Bar -->
        <div class="mt-3 p-3 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
          <div class="flex items-center gap-2 text-xs">
            <span class="text-slate-400">Current Action:</span>
            <span id="status-text" class="font-bold text-white text-xs">WAITING FOR DEPOSIT</span>
          </div>
          <span id="status-badge" class="text-[10px] px-2 py-0.5 rounded font-bold bg-blue-500/20 text-blue-300">IDLE</span>
        </div>
      </div>

      <!-- Remote Control & Override Panel -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
        <h3 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
          <i class="fa-solid fa-gamepad text-blue-400"></i> Remote Sort Triggers & Manual Override
        </h3>
        <p class="text-[11px] text-slate-400 mb-3">Manually test mechanical sorting flap actuation from the software:</p>
        
        <div class="grid grid-cols-3 gap-2">
          <button onclick="triggerSort('P')" class="py-2.5 px-3 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 text-xs font-bold transition flex flex-col items-center gap-1">
            <span>🟡 Plastic</span>
            <span class="text-[9px] font-normal text-amber-400/80">Flap 1 ➔ 45°</span>
          </button>
          <button onclick="triggerSort('M')" class="py-2.5 px-3 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-xs font-bold transition flex flex-col items-center gap-1">
            <span>⚪ Metal Can</span>
            <span class="text-[9px] font-normal text-cyan-400/80">Flap 2 ➔ 45°</span>
          </button>
          <button onclick="triggerSort('C')" class="py-2.5 px-3 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 text-xs font-bold transition flex flex-col items-center gap-1">
            <span>🟢 Paper Cup</span>
            <span class="text-[9px] font-normal text-emerald-400/80">Flap 2 ➔ 135°</span>
          </button>
        </div>

        <button onclick="resetStats()" class="w-full mt-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-semibold border border-slate-700 transition">
          🔄 Reset Counter & Stats
        </button>
      </div>
    </div>

    <!-- Center/Right: Live Telemetry, Sensor Status & Analytics (7 cols) -->
    <div class="lg:col-span-7 flex flex-col gap-4">
      
      <!-- Sensor Diagnostics Row -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <!-- IR 1 Intake -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col items-center text-center">
          <span class="text-[10px] text-slate-400 font-bold uppercase">IR 1 (Intake)</span>
          <div id="badge-ir1" class="my-1.5 w-7 h-7 rounded-full flex items-center justify-center bg-slate-800 text-slate-500 text-xs">
            <i class="fa-solid fa-cube"></i>
          </div>
          <span id="text-ir1" class="text-xs font-bold text-slate-400">CLEAR</span>
        </div>

        <!-- IR 2 Chute Exit -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col items-center text-center">
          <span class="text-[10px] text-slate-400 font-bold uppercase">IR 2 (Exit Chute)</span>
          <div id="badge-ir2" class="my-1.5 w-7 h-7 rounded-full flex items-center justify-center bg-slate-800 text-slate-500 text-xs">
            <i class="fa-solid fa-arrow-down"></i>
          </div>
          <span id="text-ir2" class="text-xs font-bold text-slate-400">CLEAR</span>
        </div>

        <!-- Servo 1 Angle -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col items-center text-center">
          <span class="text-[10px] text-slate-400 font-bold uppercase">Servo 1 (Plastic)</span>
          <span id="val-servo1" class="text-xl font-extrabold text-blue-400 my-1">0°</span>
          <span class="text-[10px] text-slate-400">Pin D9 PWM</span>
        </div>

        <!-- Servo 2 Angle -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col items-center text-center">
          <span class="text-[10px] text-slate-400 font-bold uppercase">Servo 2 (Metal/Paper)</span>
          <span id="val-servo2" class="text-xl font-extrabold text-purple-400 my-1">0°</span>
          <span class="text-[10px] text-slate-400">Pin D10 PWM</span>
        </div>
      </div>

      <!-- Real-time Dynamic Charts -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1">
        
        <!-- Doughnut Distribution Chart -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col">
          <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <i class="fa-solid fa-chart-pie text-cyan-400"></i> Material Segregation Ratio
          </h3>
          <div class="relative flex-1 flex items-center justify-center p-2">
            <canvas id="ratioChart" class="max-h-[210px]"></canvas>
          </div>
        </div>

        <!-- Timeline Deposit History Chart -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col">
          <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <i class="fa-solid fa-chart-line text-emerald-400"></i> Live Accumulation Trend
          </h3>
          <div class="relative flex-1 flex items-center justify-center p-2">
            <canvas id="timelineChart" class="max-h-[210px] w-full"></canvas>
          </div>
        </div>

      </div>

      <!-- Vending Machine Operational Log & Features Table -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
        <div class="flex items-center justify-between pb-2 border-b border-slate-800 mb-2">
          <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <i class="fa-solid fa-list-check text-blue-400"></i> Active Vending Subsystems & Protocol
          </h3>
          <span class="text-[10px] text-slate-400 font-mono">Baud: 115200 bps</span>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-2 text-[11px]">
          <div class="p-2 rounded-lg bg-slate-950 border border-slate-800 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span class="text-slate-300">16×2 LCD: Synced</span>
          </div>
          <div class="p-2 rounded-lg bg-slate-950 border border-slate-800 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span class="text-slate-300">Reward Calculator: Active</span>
          </div>
          <div class="p-2 rounded-lg bg-slate-950 border border-slate-800 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span class="text-slate-300">Exit Chute Verification: ON</span>
          </div>
        </div>
      </div>

    </div>
  </div>

  <script>
    // Live Clock
    setInterval(() => {
      document.getElementById('clock-display').innerText = new Date().toLocaleTimeString();
    }, 1000);

    // Chart.js Setup
    const ratioCtx = document.getElementById('ratioChart').getContext('2d');
    const ratioChart = new Chart(ratioCtx, {
      type: 'doughnut',
      data: {
        labels: ['Plastic', 'Metal', 'Paper'],
        datasets: [{
          data: [1, 1, 1],
          backgroundColor: ['#f59e0b', '#06b6d4', '#10b981'],
          borderWidth: 0
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

    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    const timelineChart = new Chart(timelineCtx, {
      type: 'line',
      data: {
        labels: ['00:00'],
        datasets: [
          { label: 'Plastic', data: [0], borderColor: '#f59e0b', tension: 0.3, fill: false, borderWidth: 2 },
          { label: 'Metal', data: [0], borderColor: '#06b6d4', tension: 0.3, fill: false, borderWidth: 2 },
          { label: 'Paper', data: [0], borderColor: '#10b981', tension: 0.3, fill: false, borderWidth: 2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { display: false } },
          y: { ticks: { color: '#64748b', font: { size: 9 }, stepSize: 1 }, grid: { color: '#1e293b' } }
        },
        plugins: {
          legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 10, font: { size: 9 } } }
        }
      }
    });

    // Remote sort trigger
    function triggerSort(cls) {
      fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd: cls })
      });
    }

    function resetStats() {
      fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd: 'r' })
      });
    }

    // Polling Telemetry loop
    async function fetchTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();

        // Update counts & progress
        document.getElementById('val-plastic').innerText = data.plastic;
        document.getElementById('val-metal').innerText = data.metal;
        document.getElementById('val-paper').innerText = data.paper;
        document.getElementById('val-points').innerText = data.points;
        const total = data.plastic + data.metal + data.paper;
        document.getElementById('val-total-items').innerText = `${total} Items`;

        const maxCount = Math.max(total, 1);
        document.getElementById('prog-plastic').style.width = `${(data.plastic / maxCount) * 100}%`;
        document.getElementById('prog-metal').style.width = `${(data.metal / maxCount) * 100}%`;
        document.getElementById('prog-paper').style.width = `${(data.paper / maxCount) * 100}%`;

        // Update Servos
        document.getElementById('val-servo1').innerText = `${data.servo1}°`;
        document.getElementById('val-servo2').innerText = `${data.servo2}°`;

        // Update IR Sensors
        const b1 = document.getElementById('badge-ir1');
        const t1 = document.getElementById('text-ir1');
        if (data.ir1 === 1) {
          b1.className = "my-1.5 w-7 h-7 rounded-full flex items-center justify-center bg-emerald-500/20 text-emerald-400 text-xs shadow-lg shadow-emerald-500/30";
          t1.innerText = "ITEM DETECTED";
          t1.className = "text-xs font-bold text-emerald-400";
        } else {
          b1.className = "my-1.5 w-7 h-7 rounded-full flex items-center justify-center bg-slate-800 text-slate-500 text-xs";
          t1.innerText = "CLEAR";
          t1.className = "text-xs font-bold text-slate-400";
        }

        const b2 = document.getElementById('badge-ir2');
        const t2 = document.getElementById('text-ir2');
        if (data.ir2 === 1) {
          b2.className = "my-1.5 w-7 h-7 rounded-full flex items-center justify-center bg-cyan-500/20 text-cyan-400 text-xs shadow-lg shadow-cyan-500/30";
          t2.innerText = "BIN ENTRY";
          t2.className = "text-xs font-bold text-cyan-400";
        } else {
          b2.className = "my-1.5 w-7 h-7 rounded-full flex items-center justify-center bg-slate-800 text-slate-500 text-xs";
          t2.innerText = "CLEAR";
          t2.className = "text-xs font-bold text-slate-400";
        }

        // Status update
        document.getElementById('status-text').innerText = data.status.toUpperCase();

        // Update charts
        if (total > 0) {
          ratioChart.data.datasets[0].data = [data.plastic, data.metal, data.paper];
          ratioChart.update('none');
        }

        if (data.history && data.history.length > 0) {
          timelineChart.data.labels = data.history.map(h => h.time);
          timelineChart.data.datasets[0].data = data.history.map(h => h.plastic);
          timelineChart.data.datasets[1].data = data.history.map(h => h.metal);
          timelineChart.data.datasets[2].data = data.history.map(h => h.paper);
          timelineChart.update('none');
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
