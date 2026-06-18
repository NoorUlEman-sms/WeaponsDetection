"""
SMS Vision AI — Live Detection Server
Streams annotated frames as MJPEG over HTTP so the browser dashboard
can display real-time weapon detection results.
"""

import cv2
import os
import time
import threading
import json
import os
import time
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION & PERSISTENCE
# ─────────────────────────────────────────────
CONFIG_FILE = "config_detection.json"

def load_config():
    defaults = {
        "model_path": "best.pt",
        "conf_thresh": 0.5,
        "fps_delay": 0.2
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                defaults.update(data)
        except: pass
    return defaults

def save_config(model_path):
    config = load_config()
    config["model_path"] = model_path
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

config = load_config()
MODEL_PATH       = config["model_path"]
DISPLAY_MODEL    = MODEL_PATH
CONF_THRESH      = config["conf_thresh"]
FPS_DELAY        = config["fps_delay"]
PORT             = 8082

# Detections storage
def get_detection_dir(model_name):
    return "cctv_images"

DETECTION_LOG = "gun_detections.log"

# Camera Sources — 192.168.51.241 (Camera 14)
RTSP_SOURCES = [
    # HTTP MJPEG stream (common for Hikvision/Dahua web cameras)
    "http://192.168.51.241/video.cgi",
    "http://admin:Sms786%40sms@192.168.51.241/video.cgi",
    "http://admin:Sms786%40sms@192.168.51.241/mjpeg.cgi",
    "http://admin:Sms786%40sms@192.168.51.241/videostream.cgi",
    # RTSP paths
    "rtsp://admin:Sms786%40sms@192.168.51.241:554/cam/realmonitor?channel=1&subtype=1",
    "rtsp://admin:Sms786%40sms@192.168.51.241:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://admin:Sms786%40sms@192.168.51.241:554/Streaming/Channels/102",
    "rtsp://admin:Sms786%40sms@192.168.51.241:554/Streaming/Channels/101",
    "rtsp://admin:Sms786%40sms@192.168.51.241:554/h264/ch1/sub/av_stream",
    "rtsp://admin:Sms786%40sms@192.168.51.241:554/h264/ch1/main/av_stream",
]

# Available models
AVAILABLE_MODELS = ["models/bestcctv1.pt", "models/best.pt", "models/best2.pt", "best.pt", "best (1).pt", "best (6).pt", "best4.pt", "best(hf).pt", "best11.pt", "best (50).pt"]
model_reload_event = threading.Event()
reload_model_path  = None

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
latest_frame  = None          # JPEG bytes of the last annotated frame
frame_lock    = threading.Lock()

detection_stats = {
    "total_detections": 0,
    "last_detection_time": None,
    "current_detections": [],
    "fps": 0.0,
    "status": "Initializing…",
    "model": DISPLAY_MODEL,
    "uptime_start": datetime.now().isoformat(),
}
stats_lock = threading.Lock()

# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────
def draw_box(frame, box, label, color=(0, 0, 255)):
    x1, y1, x2, y2 = map(int, box)
    # Outer glow (softer, wider)
    cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (color[0]//4, color[1]//4, color[2]//4), 6)
    # Main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Corner brackets
    L, t = 22, 5
    for px, py, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + sx*L, py), color, t)
        cv2.line(frame, (px, py), (px, py + sy*L), color, t)
    # Label background
    font = cv2.FONT_HERSHEY_DUPLEX
    fs   = 0.72
    (tw, th), _ = cv2.getTextSize(label, font, fs, 2)
    label_y = max(y1, th + 12)
    cv2.rectangle(frame, (x1, label_y - th - 10), (x1 + tw + 12, label_y + 2), color, -1)
    cv2.putText(frame, label, (x1 + 6, label_y - 4), font, fs, (255, 255, 255), 2, cv2.LINE_AA)


def overlay_hud(frame, detections, fps, h, w):
    """Draw the HUD overlay on top of the frame."""
    # Top status bar
    bar_h = 60
    overlay = frame.copy()
    if detections:
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame,
                    f"  ⚠  WEAPON DETECTED — {len(detections)} OBJECT(S)",
                    (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.95,
                    (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 30, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, "  MONITORING ACTIVE — No Threat Detected",
                    (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.85,
                    (180, 255, 180), 2, cv2.LINE_AA)

    # Bottom info strip
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    with stats_lock:
        active_model = detection_stats.get("model", MODEL_PATH)
    info = f"  {active_model}  |  Conf >= {CONF_THRESH}  |  {fps:.1f} FPS  |  {ts}"
    cv2.rectangle(frame, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, info, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 180, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# VIDEO STREAM THREAD
# ─────────────────────────────────────────────
class VideoStream:
    def __init__(self, sources):
        self.sources = sources if isinstance(sources, list) else [sources]
        self.cap = None
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.active_source = None
        
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        source_idx = 0
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                url = self.sources[source_idx]
                print(f"📡 VideoStream connecting to camera: {url}")
                with stats_lock:
                    display_addr = url.split('@')[-1] if '@' in url else url.replace('rtsp://', '')
                    detection_stats["status"] = f"Connecting to {display_addr}..."
                
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.active_source = url
                    with stats_lock:
                        display_addr = url.split('@')[-1] if '@' in url else url.replace('rtsp://', '')
                        detection_stats["active_source"] = display_addr
                        detection_stats["status"] = "LIVE"
                    print(f"✅ VideoStream connected successfully to: {url}")
                else:
                    print(f"⚠️ VideoStream failed to connect to: {url}")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    source_idx = (source_idx + 1) % len(self.sources)
                    time.sleep(2.0)
                    continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame.copy()
                    self.ret = True
            else:
                print("⚠️ Camera frame read failed. Reconnecting...")
                with stats_lock:
                    detection_stats["status"] = "Reconnecting..."
                with self.lock:
                    self.ret = False
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1.0)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()


# Thread-safe global detections state
latest_detections = []
detections_lock = threading.Lock()
vstream = VideoStream(RTSP_SOURCES).start()


# ─────────────────────────────────────────────
# DETECTION WORKER THREAD
# ─────────────────────────────────────────────
def detection_worker():
    global latest_frame, detection_stats, latest_detections, vstream

    # Load model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_full = os.path.join(base_dir, MODEL_PATH)

    with stats_lock:
        detection_stats["status"] = f"Loading model: {MODEL_PATH}…"

    if not os.path.exists(model_full):
        with stats_lock:
            detection_stats["status"] = f"ERROR: model not found — {model_full}"
        return

    model = YOLO(model_full)

    fps_counter = 0
    fps_timer   = time.time()
    cur_fps     = 0.0
    last_save_time = 0  # Cooldown for snapshots
    target_fps = 5.0
    frame_interval = 1.0 / target_fps

    # Video Recording State
    RECORDING_DIR = "cctv_recordings"
    if not os.path.exists(RECORDING_DIR):
        os.makedirs(RECORDING_DIR)

    video_writer = None
    recording_active = False
    last_threat_time = 0
    post_record_cooldown = 3.0  # Keep recording for 3 seconds after threat clears

    try:
        while True:
            loop_start = time.time()

            # ── CHECK FOR MODEL RELOAD
            if model_reload_event.is_set():
                new_path = reload_model_path
                model_reload_event.clear()
                if new_path and os.path.exists(new_path):
                    print(f"🔄 Reloading model: {new_path}")
                    with stats_lock:
                        detection_stats["status"] = f"Reloading: {new_path}..."
                        detection_stats["model"] = new_path
                    try:
                        model = YOLO(new_path)
                        save_config(new_path)  # Persist choice
                        print(f"✅ Model reloaded and saved: {new_path}")
                        with stats_lock:
                            detection_stats["status"] = "LIVE"
                    except Exception as e:
                        print(f"❌ Failed to reload model: {e}")
                        with stats_lock:
                            detection_stats["status"] = f"ERROR: Reload failed"

            ret, frame = vstream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # Run inference (use smaller imgsz for speed to reach target FPS)
            results = model(frame, conf=CONF_THRESH, imgsz=320, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    cls  = int(box.cls[0])
                    name = model.names[cls]
                    conf = float(box.conf[0])
                    
                    # FILTER: Only keep 'gun' labels
                    if 'gun' in name.lower() or 'weapon' in name.lower():
                        detections.append({
                            "label": name,
                            "conf":  round(conf, 3),
                            "box":   [round(v, 1) for v in box.xyxy[0].tolist()],
                        })

            # Update global thread-safe detections
            with detections_lock:
                latest_detections = detections.copy()

            # FPS
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                cur_fps     = fps_counter / elapsed
                fps_counter = 0
                fps_timer   = time.time()

            is_alerting = len(detections) > 0

            # Update stats
            with stats_lock:
                detection_stats["fps"] = round(cur_fps, 1)
                detection_stats["current_detections"] = detections
                if detections:
                    detection_stats["total_detections"] += len(detections)
                    ts_now = datetime.now()
                    detection_stats["last_detection_time"] = ts_now.strftime("%H:%M:%S")

                    # SAVE SNAPSHOT & LOG
                    if time.time() - last_save_time > 3.0:  # 3 second cooldown
                        last_save_time = time.time()
                        ts_str = ts_now.strftime("%Y%m%d_%H%M%S")
                        fname = f"gun_{ts_str}.jpg"
                        
                        # Dynamic directory based on model
                        target_dir = get_detection_dir(detection_stats["model"])
                        os.makedirs(target_dir, exist_ok=True)
                        fpath = os.path.join(target_dir, fname)
                        
                        # Create an annotated snapshot
                        snap_frame = frame.copy()
                        for d in detections:
                            color = (0, 0, 255) if ('gun' in d['label'].lower() or 'weapon' in d['label'].lower()) else (0, 165, 255)
                            draw_box(snap_frame, d['box'], f"{d['label'].upper()} {d['conf']:.2f}", color)
                        overlay_hud(snap_frame, detections, cur_fps, h, w)
                        
                        # Save image
                        cv2.imwrite(fpath, snap_frame)
                        
                        # Append to log
                        log_msg = f"[{ts_now.strftime('%Y-%m-%d %H:%M:%S')}] Model: {detection_stats['model']} | Count: {len(detections)} | Snapshot: {fname}\n"
                        with open(DETECTION_LOG, "a") as f:
                            f.write(log_msg)
                        print(f"📸 Saved snapshot: {fname}")

            # Video Recording Event State Machine
            if is_alerting:
                last_threat_time = time.time()
                if not recording_active:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"threat_{ts}.avi"
                    video_path = os.path.join(RECORDING_DIR, video_filename)
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 5.0, (w, h))
                    recording_active = True
                    print(f"🎥 Threat started! Recording to: {video_path}")
                    
            if recording_active and video_writer is not None:
                record_frame = frame.copy()
                for d in detections:
                    color = (0, 0, 255) if ('gun' in d['label'].lower() or 'weapon' in d['label'].lower()) else (0, 165, 255)
                    draw_box(record_frame, d['box'], f"{d['label'].upper()} {d['conf']:.2f}", color)
                overlay_hud(record_frame, detections, cur_fps, h, w)
                
                video_writer.write(record_frame)
                
                # Check if alert cleared and cooldown expired
                if not is_alerting and (time.time() - last_threat_time > post_record_cooldown):
                    video_writer.release()
                    video_writer = None
                    recording_active = False
                    print(f"🎥 Threat cleared. Saved evidence video clip to: {RECORDING_DIR}")

            # Maintain Stable FPS (5 FPS) for AI thread
            elapsed_loop = time.time() - loop_start
            time.sleep(max(0.001, frame_interval - elapsed_loop))
    finally:
        if video_writer is not None:
            video_writer.release()
            print("🎥 Safely released video writer on thread exit.")


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SMS Vision AI — {{ current_model }} Live Detection</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

    :root {
      --bg:       #05080f;
      --panel:    #0b1220;
      --border:   #1a2f50;
      --accent:   #38bdf8;
      --danger:   #ef4444;
      --safe:     #22c55e;
      --text:     #cbd5e1;
      --muted:    #475569;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── HEADER ── */
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 24px;
      background: linear-gradient(135deg, #0d1829, #0a1320);
      border-bottom: 1px solid var(--border);
      box-shadow: 0 4px 30px rgba(0,0,0,.7);
      flex-shrink: 0;
    }
    .logo-block { display:flex; align-items:center; gap:14px; }
    .logo-icon  { font-size:2rem; }
    .logo-text  {
      font-size: 1.45rem; font-weight: 900; letter-spacing: 2px;
      background: linear-gradient(90deg, #38bdf8 0%, #818cf8 60%, #f472b6 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .logo-sub { font-size: .7rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px; }

    .header-badges { display:flex; align-items:center; gap:12px; }
    .badge {
      display: flex; align-items: center; gap: 6px;
      padding: 5px 14px; border-radius: 20px;
      font-size: .78rem; font-weight: 700; letter-spacing: .5px;
    }
    .badge-live {
      background: rgba(239,68,68,.12); border: 1px solid #dc2626; color: #f87171;
      animation: pulse-border 1.8s infinite;
    }
    .badge-model { background: rgba(56,189,248,.1); border: 1px solid #0ea5e9; color: var(--accent); }
    .dot { width:8px; height:8px; border-radius:50%; background:#ef4444; animation: blink 1s infinite; }
    @keyframes blink        { 0%,100%{opacity:1} 50%{opacity:.15} }
    @keyframes pulse-border { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.3)} 50%{box-shadow:0 0 0 6px rgba(239,68,68,0)} }

    /* ── MAIN LAYOUT ── */
    main {
      height: calc(100vh - 130px); /* Constrain height to viewport */
      display: grid;
      grid-template-columns: 1fr 300px;
      gap: 12px;
      padding: 12px;
      overflow: hidden;
    }

    /* ── VIDEO PANEL ── */
    .video-panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
    }
    .video-panel img {
      width: 100%; height: 100%;
      object-fit: contain; /* Ensure full frame is shown */
      display: block;
      background: #000;
      min-height: 0;
    }
    .video-overlay-label {
      position: absolute; top: 12px; left: 12px;
      background: rgba(0,0,0,.6); border: 1px solid var(--border);
      padding: 4px 12px; border-radius: 8px;
      font-size: .72rem; color: var(--accent); letter-spacing: 1px;
      backdrop-filter: blur(4px);
    }
    .no-signal {
      display: none;
      flex-direction: column; align-items: center; justify-content: center;
      gap: 12px; flex: 1; color: var(--muted); font-size: 1rem;
    }
    .no-signal .spinner {
      width: 40px; height: 40px; border: 3px solid var(--border);
      border-top-color: var(--accent); border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── SIDEBAR ── */
    .sidebar {
      display: flex; flex-direction: column; gap: 10px; overflow-y: auto;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
      flex-shrink: 0;
    }
    .card-title {
      font-size: .68rem; text-transform: uppercase; letter-spacing: 1.5px;
      color: var(--muted); margin-bottom: 10px;
    }

    /* ── STATUS CARD ── */
    .status-row { display:flex; align-items:center; justify-content:space-between; margin-bottom: 8px; }
    .status-row:last-child { margin-bottom: 0; }
    .status-key { font-size: .78rem; color: var(--muted); }
    .status-val { font-size: .82rem; font-weight: 600; color: var(--text); }
    .status-val.live   { color: var(--safe); }
    .status-val.danger { color: var(--danger); }
    .status-val.accent { color: var(--accent); }

    /* ── ALERT BOX ── */
    #alert-box {
      border-radius: 10px; padding: 12px 14px;
      transition: background .4s, border-color .4s;
      flex-shrink: 0;
    }
    #alert-box.safe   { background: rgba(34,197,94,.08); border: 1px solid rgba(34,197,94,.3); }
    #alert-box.danger { background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.5); animation: flash-alert .5s ease; }
    @keyframes flash-alert { 0%{transform:scale(1)} 50%{transform:scale(1.02)} 100%{transform:scale(1)} }
    #alert-icon  { font-size: 1.8rem; text-align: center; margin-bottom: 4px; }
    #alert-title { font-size: .95rem; font-weight: 700; text-align: center; }
    #alert-sub   { font-size: .72rem; color: var(--muted); text-align: center; margin-top: 4px; }

    /* ── DETECTIONS LIST ── */
    #det-list { display:flex; flex-direction:column; gap:6px; }
    .det-item {
      background: rgba(239,68,68,.08);
      border: 1px solid rgba(239,68,68,.25);
      border-radius: 8px; padding: 8px 12px;
      display: flex; align-items: center; justify-content: space-between;
      animation: slide-in .25s ease;
    }
    @keyframes slide-in { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:translateY(0)} }
    .det-label { font-weight: 700; font-size: .82rem; color: #fca5a5; }
    .det-conf  { font-size: .75rem; background: rgba(239,68,68,.2); padding: 2px 8px; border-radius: 10px; color: #f87171; }
    .no-det    { font-size: .8rem; color: var(--muted); text-align: center; padding: 10px 0; }

    /* ── STATS ── */
    .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat-box {
      background: rgba(255,255,255,.03); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px; text-align: center;
    }
    .stat-num  { font-size: 1.4rem; font-weight: 900; color: var(--accent); line-height: 1; }
    .stat-label{ font-size: .65rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }

    /* ── FOOTER ── */
    footer {
      background: #070c17; border-top: 1px solid var(--border);
      padding: 8px 24px; display: flex; align-items: center; justify-content: space-between;
      font-size: .7rem; color: var(--muted); flex-shrink: 0;
    }
    footer span { color: var(--accent); }

    /* ── SCROLLBAR ── */
    .sidebar::-webkit-scrollbar { width: 4px; }
    .sidebar::-webkit-scrollbar-track { background: transparent; }
    .sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  </style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="logo-block">
    <div class="logo-icon">🔫</div>
    <div>
      <div class="logo-text">SMS VISION AI</div>
      <div class="logo-sub">Weapon Detection System · Live Monitoring</div>
    </div>
  </div>
  <div class="header-badges">
    <div class="badge badge-model" id="h-model">📦 {{ current_model }}</div>
    <div class="badge badge-live"><div class="dot"></div> LIVE</div>
  </div>
</header>

<!-- MAIN -->
<main>

  <!-- VIDEO FEED -->
  <div class="video-panel">
    <div class="video-overlay-label" id="cam-label">📡 CAM · 192.168.51.242</div>
    <img id="feed" src="/video_feed" alt="Live detection feed"
         onerror="this.style.display='none'; document.getElementById('no-sig').style.display='flex'">
    <div class="no-signal" id="no-sig">
      <div class="spinner"></div>
      <div>Connecting to camera…</div>
    </div>
  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">

    <!-- ALERT BOX -->
    <div id="alert-box" class="card safe">
      <div id="alert-icon">✅</div>
      <div id="alert-title">Area Secure</div>
      <div id="alert-sub">No weapons detected</div>
    </div>

    <!-- STATUS -->
    <div class="card">
      <div class="card-title">System Status</div>
      <div class="status-row"><span class="status-key">Active Model</span><span class="status-val accent" id="s-model">best.pt</span></div>
      <div class="status-row"><span class="status-key">Stream</span><span class="status-val live" id="s-status">—</span></div>
      <div class="status-row"><span class="status-key">Threshold</span><span class="status-val accent">≥ 0.50</span></div>
      <div class="status-row"><span class="status-key">Last Alert</span><span class="status-val" id="s-last">—</span></div>
    </div>

    <!-- MODEL SELECTOR -->
    <div class="card">
      <div class="card-title">Change Model</div>
      <select id="model-select" style="width:100%; background:var(--bg); color:var(--text); border:1px solid var(--border); padding:8px; border-radius:8px; font-family:inherit; outline:none; cursor:pointer;" onchange="changeModel(this.value)">
        {% for m in models %}
        <option value="{{ m }}" {% if m == current_model %}selected{% endif %}>{{ m }}</option>
        {% endfor %}
      </select>
    </div>


    <!-- STATS -->
    <div class="card">
      <div class="card-title">Statistics</div>
      <div class="stats-grid">
        <div class="stat-box"><div class="stat-num" id="s-fps">0</div><div class="stat-label">FPS</div></div>
        <div class="stat-box"><div class="stat-num" id="s-total">0</div><div class="stat-label">Total Alerts</div></div>
      </div>
    </div>

    <!-- DETECTIONS -->
    <div class="card" style="flex:1;">
      <div class="card-title">Current Detections</div>
      <div id="det-list"><div class="no-det">Scanning…</div></div>
    </div>

  </div>
</main>

<!-- FOOTER -->
<footer>
  <div id="footer-info">Camera · <span id="f-addr">rtsp://192.168.51.239</span> · <span id="f-model">{{ current_model }}</span> · Conf ≥ 0.50</div>
  <div id="clock">—</div>
</footer>

<script>
  // ── Clock
  function updateClock() {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString();
  }
  setInterval(updateClock, 1000);
  updateClock();

  // ── Poll stats every 400ms
  async function pollStats() {
    try {
      const r   = await fetch('/stats');
      const d   = await r.json();
      const det = d.current_detections || [];
      const has = det.length > 0;

      // Status
      const ss  = document.getElementById('s-status');
      ss.textContent = d.status;
      ss.className   = 'status-val ' + (d.status === 'LIVE' ? 'live' : d.status.startsWith('ERROR') ? 'danger' : 'accent');

      // FPS & Total
      document.getElementById('s-fps').textContent   = d.fps;
      document.getElementById('s-total').textContent = d.total_detections;
      document.getElementById('s-last').textContent  = d.last_detection_time || '—';

      // Alert box
      const ab    = document.getElementById('alert-box');
      const icon  = document.getElementById('alert-icon');
      const title = document.getElementById('alert-title');
      const sub   = document.getElementById('alert-sub');
      if (has) {
        if (!ab.classList.contains('danger')) {
          ab.className = 'card danger';
          icon.textContent  = '⚠️';
          title.textContent = '🚨 WEAPON DETECTED';
          sub.textContent   = `${det.length} object(s) in frame`;
        }
      } else {
        ab.className = 'card safe';
        icon.textContent  = '✅';
        title.textContent = 'Area Secure';
        sub.textContent   = 'No weapons detected';
      }

      // Detections list
      const list = document.getElementById('det-list');
      if (det.length === 0) {
        list.innerHTML = '<div class="no-det">No active detections</div>';
      } else {
        list.innerHTML = det.map(d => `
          <div class="det-item">
            <span class="det-label">${d.label.toUpperCase()}</span>
            <span class="det-conf">${(d.conf * 100).toFixed(1)}%</span>
          </div>`).join('');
      }

      // Update active model name in UI
      document.getElementById('s-model').textContent = d.model;
      document.getElementById('h-model').textContent = '📦 ' + d.model;
      document.getElementById('f-model').textContent = d.model;
      
      // Update footer address
      if (d.active_source) {
        document.getElementById('f-addr').textContent = 'rtsp://' + d.active_source;
      }
      
    } catch(e) { /* server not yet ready */ }
  }

  async function changeModel(modelName) {
    console.log("Requesting model change:", modelName);
    try {
      const r = await fetch(`/change_model/${modelName}`);
      const res = await r.json();
      if (res.status === 'success') {
        console.log("Model change initiated");
      }
    } catch(e) { console.error("Model change failed", e); }
  }

  setInterval(pollStats, 400);
  pollStats();
</script>

</body>
</html>
"""


@app.route('/')
def index():
    with stats_lock:
        cur = detection_stats["model"]
    return render_template_string(DASHBOARD_HTML, models=AVAILABLE_MODELS, current_model=cur)


def generate_frames():
    """MJPEG generator — renders and yields the latest frame at 25-30 FPS."""
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    target_fps = 25.0
    frame_interval = 1.0 / target_fps
    
    while True:
        loop_start = time.time()
        
        # Read the latest raw frame from VideoStream at full speed
        ret, frame = vstream.read() if vstream is not None else (False, None)
        if not ret or frame is None:
            time.sleep(0.01)
            continue
            
        h, w = frame.shape[:2]
        
        # Fetch the latest thread-safe detections
        with detections_lock:
            active_dets = latest_detections.copy()
            
        # Draw active detections on this latest real-time frame
        for d in active_dets:
            color = (0, 0, 255) if ('gun' in d['label'].lower() or 'weapon' in d['label'].lower()) else (0, 165, 255)
            draw_box(frame, d['box'], f"{d['label'].upper()} {d['conf']:.2f}", color)
            
        # Overlay the HUD on this frame
        with stats_lock:
            fps_val = detection_stats["fps"]
        overlay_hud(frame, active_dets, fps_val, h, w)
        
        # Encode to JPEG on the fly
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield boundary + buf.tobytes() + b'\r\n'
            
        # Sleep to maintain stable 25 FPS
        elapsed = time.time() - loop_start
        time.sleep(max(0.001, frame_interval - elapsed))


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stats')
def stats():
    with stats_lock:
        return jsonify(detection_stats)


@app.route('/change_model/<name>')
def change_model(name):
    global reload_model_path
    if name in AVAILABLE_MODELS:
        reload_model_path = name
        model_reload_event.set()
        return jsonify({"status": "success", "model": name})
    return jsonify({"status": "error", "message": "Model not found"}), 404


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Start detection thread
    t = threading.Thread(target=detection_worker, daemon=True)
    t.start()

    print(f"\n{'='*55}")
    print(f"  🔫  SMS Vision AI — {MODEL_PATH} Live Detection Server")
    print(f"{'='*55}")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Camera  : {RTSP_SOURCES[0]}")
    print(f"  Conf    : {CONF_THRESH}")
    print(f"  Dashboard: http://localhost:{PORT}")
    print(f"{'='*55}\n")

    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
