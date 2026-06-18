"""
SMS Vision AI — Premium ROI Weapon Detection
High-performance Person -> ROI -> Weapon pipeline with responsive UI.
"""

import cv2
import os
import time
import threading
import numpy as np
import re
from datetime import datetime
from tracker import TrackManager
from flask import Flask, Response, render_template_string, jsonify, request
from ultralytics import YOLO
from vidgear.gears import CamGear

# ─────────────────────────────────────────────
# CONFIGURATION LOAD LOGIC
# ─────────────────────────────────────────────
import json

base_dir = os.path.dirname(os.path.abspath(__file__))

# Local config file path
CONFIG_FILE = os.path.join(base_dir, "config_detection.json")

# Default settings
PERSON_MODEL_PATH = "yolov8n.pt"
WEAPON_MODEL_PATH = "models/bestcctv1.pt"
AVAILABLE_MODELS  = ["models/bestcctv1.pt", "models/best2.pt", "models/best.pt"]
CONF_THRESH       = 0.50
MIN_SIZE_RATIO    = 0.01
TEMPORAL_THRESHOLD = 3
EXPANSION_FACTOR  = 0.2
PORT              = 8082
DEFAULT_CAMERA    = "Camera 13"
CAMERAS = {}

# Try to load config from JSON file
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config_data = json.load(f)
            PERSON_MODEL_PATH = config_data.get("person_model_path", PERSON_MODEL_PATH)
            WEAPON_MODEL_PATH = config_data.get("weapon_model_path", WEAPON_MODEL_PATH)
            AVAILABLE_MODELS  = config_data.get("available_models", AVAILABLE_MODELS)
            CONF_THRESH       = config_data.get("conf_thresh", CONF_THRESH)
            MIN_SIZE_RATIO    = config_data.get("min_size_ratio", MIN_SIZE_RATIO)
            TEMPORAL_THRESHOLD = config_data.get("temporal_threshold", TEMPORAL_THRESHOLD)
            EXPANSION_FACTOR  = config_data.get("expansion_factor", EXPANSION_FACTOR)
            PORT              = config_data.get("port", PORT)
            DEFAULT_CAMERA    = config_data.get("default_camera", DEFAULT_CAMERA)
            CAMERAS           = config_data.get("cameras", CAMERAS)
    except Exception as e:
        print(f"⚠️ Error loading configuration: {e}. Using defaults.")

# If JSON load failed or cameras list is empty, use standard fallbacks
if not CAMERAS:
    CAMERAS = {
        "Camera 13": "rtsp://admin:Sms786%40sms@192.168.51.238:554/cam/realmonitor?channel=1&subtype=0",
        "Camera 14": "rtsp://admin:Sms786%40sms@192.168.51.241:554/cam/realmonitor?channel=1&subtype=0",
        "Camera 242": "rtsp://admin:Sms786%40sms@192.168.51.242:554/cam/realmonitor?channel=1&subtype=1"
    }

RTSP_SOURCE = CAMERAS.get(DEFAULT_CAMERA, list(CAMERAS.values())[0])

model_reload_event = threading.Event()
reload_model_path  = None

import sys
if len(sys.argv) > 1:
    custom_source = sys.argv[1]
    if os.path.exists(custom_source) or custom_source.startswith("rtsp://"):
        RTSP_SOURCE = os.path.abspath(custom_source) if os.path.exists(custom_source) else custom_source
        print(f"🎬 Command-line source set: {RTSP_SOURCE}")

SNAPSHOT_DIR = os.path.join(base_dir, "cctv1_images")
UPLOAD_DIR = os.path.join(base_dir, "temp_videos")

# Ensure directories exist
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
latest_raw_frame = None
latest_annotated_frame = None
frame_lock = threading.Lock()
new_frame_event = threading.Event()
ai_history = {}
ai_history_lock = threading.Lock()

stats = {
    "fps": 0.0,
    "detect_fps": 0.0,
    "alerts": 0,
    "status": "Initializing...",
    "alert_active": False,
    "last_alert": None,
    "cam_connected": False,
    "active_camera": "Camera 13",
    "weapon_model": os.path.basename(WEAPON_MODEL_PATH)
}
stats_lock = threading.Lock()

# ─────────────────────────────────────────────
# VIDEO STREAM THREAD
# ─────────────────────────────────────────────
class VideoStream:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.ret = False
        self.frame = None
        self.frame_id = 0
        self.stopped = False
        self.lock = threading.Lock()
        self.is_file = os.path.isfile(source) or source.endswith(('.mp4', '.avi', '.mkv', '.mov'))
        
        # Delayed streaming buffer
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 50
        
        # Connection attempts tracking
        self.connection_attempts = 0
        
    def change_source(self, new_source):
        with self.lock:
            print(f"🔄 Switching camera source to: {new_source}")
            self.source = new_source
            self.is_file = os.path.isfile(new_source) or new_source.endswith(('.mp4', '.avi', '.mkv', '.mov'))
            if self.cap is not None:
                if self.is_file:
                    try: self.cap.release()
                    except: pass
                else:
                    try: self.cap.stop()
                    except: pass
                self.cap = None
            self.ret = False
            self.frame = None
            self.frame_id = 0
            self.connection_attempts = 0
        with self.buffer_lock:
            self.buffer = []
            
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            with self.lock:
                need_connect = (self.cap is None)
                source = self.source
                is_file = self.is_file
            
            if need_connect:
                print(f"📡 VideoStream connecting (is_file={is_file}) to: {source}")
                with stats_lock:
                    stats["status"] = "Connecting to Camera..." if not is_file else "Loading Video..."
                
                if is_file:
                    new_cap = cv2.VideoCapture(source)
                    opened = new_cap.isOpened()
                else:
                    try:
                        # Use CamGear for RTSP to get highly optimized threaded reading
                        options = {"CAP_PROP_BUFFERSIZE": 1}
                        new_cap = CamGear(source=source, logging=True, **options).start()
                        opened = True
                    except Exception as e:
                        print(f"⚠️ CamGear failed to connect: {e}")
                        new_cap = None
                        opened = False
                    
                if opened and new_cap is not None:
                    with self.lock:
                        self.cap = new_cap
                        self.ret = False
                    with stats_lock:
                        stats["cam_connected"] = True
                        stats["status"] = "LIVE" if not is_file else "PLAYING"
                    print(f"✅ VideoStream connected successfully to: {source}")
                    self.connection_attempts = 0
                else:
                    self.connection_attempts += 1
                    if not is_file and self.connection_attempts >= 3:
                        fallback_file = os.path.join(base_dir, "recordings", "detection_20260415_182052.avi")
                        if os.path.exists(fallback_file):
                            print(f"⚠️ RTSP camera stream unreachable. Falling back to local demo video: {fallback_file}")
                            with self.lock:
                                self.source = fallback_file
                                self.is_file = True
                            with stats_lock:
                                stats["active_camera"] = "Camera 13 (Demo Fallback)"
                            self.connection_attempts = 0
                            time.sleep(0.5)
                            continue
                    time.sleep(2.0)
                    continue

            with self.lock:
                if self.cap is not None:
                    if is_file:
                        ret, frame = self.cap.read()
                    else:
                        frame = self.cap.read()
                        ret = frame is not None
                else:
                    ret, frame = False, None

            if ret and frame is not None:
                with self.lock:
                    self.frame = frame.copy()
                    self.ret = True
                    self.frame_id += 1
                with self.buffer_lock:
                    self.buffer.append((self.frame_id, frame.copy()))
                    if len(self.buffer) > self.max_buffer_size:
                        self.buffer.pop(0)
                if is_file:
                    time.sleep(0.04) # Natural 25 FPS speed for file
            else:
                if is_file:
                    with self.lock:
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                    time.sleep(0.1)
                else:
                    print("⚠️ Camera frame read failed/None. Reconnecting...")
                    with stats_lock:
                        stats["status"] = "Reconnecting..."
                    with self.lock:
                        self.ret = False
                        if self.cap:
                            try: self.cap.stop()
                            except: pass
                            self.cap = None
                    time.sleep(1.0)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy(), self.frame_id
            return False, None, self.frame_id

    def read_delayed(self, delay_frames=10):
        with self.buffer_lock:
            if len(self.buffer) > delay_frames:
                return True, self.buffer[-delay_frames][1].copy(), self.buffer[-delay_frames][0]
            elif len(self.buffer) > 0:
                return True, self.buffer[0][1].copy(), self.buffer[0][0]
            return False, None, 0

    def stop(self):
        self.stopped = True
        if self.cap:
            try:
                if self.is_file:
                    self.cap.release()
                else:
                    self.cap.stop()
            except:
                pass


# Initialize non-blocking VideoStream
vstream = VideoStream(RTSP_SOURCE).start()


# ─────────────────────────────────────────────
# PIPELINE UTILS
# ─────────────────────────────────────────────
def expand_bbox(box, frame_w, frame_h, factor=0.2):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1_new = max(0, x1 - factor * w)
    y1_new = max(0, y1 - factor * h)
    x2_new = min(frame_w, x2 + factor * w)
    y2_new = min(frame_h, y2 + factor * h)
    return [int(x1_new), int(y1_new), int(x2_new), int(y2_new)]

def draw_premium_box(frame, box, label, color, is_weapon=False):
    x1, y1, x2, y2 = map(int, box)
    if is_weapon:
        # Glow
        cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 100), 4)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        L = 15
        for px, py, sx, sy in [(x1,y1,1,1), (x2,y1,-1,1), (x1,y2,1,-1), (x2,y2,-1,-1)]:
            cv2.line(frame, (px, py), (px+sx*L, py), color, 4)
            cv2.line(frame, (px, py), (px, py+sy*L), color, 4)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = 0.5
    (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
# DETECTION WORKER STATE
# ─────────────────────────────────────────────
# Thread-safe global detections state
latest_roi_detections = {
    "persons": [],
    "weapons": [],
    "is_alerting": False
}
roi_detections_lock = threading.Lock()


# ─────────────────────────────────────────────
# DETECTION WORKER
# ─────────────────────────────────────────────
def detection_worker(vstream):
    global latest_annotated_frame, latest_roi_detections
    
    print("📦 Loading AI Models...")
    person_model = YOLO(PERSON_MODEL_PATH)
    weapon_model = YOLO(WEAPON_MODEL_PATH)
    print("✅ Models Loaded.")
    
    person_tracker = TrackManager(iou_threshold=0.35, max_age=8)
    
    fps_timer = time.time()
    frames = 0
    last_snapshot_time = 0
    last_alert_latch_time = 0 
    alert_in_progress = False
    
    last_frame_id = -1
    target_fps = 30.0
    frame_interval = 1.0 / target_fps
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    RECORDING_DIR = os.path.join(base_dir, "cctv_recordings")
    if not os.path.exists(RECORDING_DIR):
        os.makedirs(RECORDING_DIR)

    video_writer = None
    recording_active = False
    last_threat_time = 0
    post_record_cooldown = 3.0  # Keep recording for 3 seconds after threat clears
    
    try:
        while True:
            # ── CHECK FOR MODEL RELOAD
            if model_reload_event.is_set():
                new_path = reload_model_path
                model_reload_event.clear()
                if new_path and os.path.exists(new_path):
                    print(f"🔄 Reloading weapon model: {new_path}")
                    with stats_lock:
                        stats["status"] = "Reloading..."
                    try:
                        weapon_model = YOLO(new_path)
                        with stats_lock:
                            stats["weapon_model"] = os.path.basename(new_path)
                            stats["status"] = "LIVE"
                        print(f"✅ Model successfully reloaded: {new_path}")
                    except Exception as e:
                        print(f"❌ Failed to reload model: {e}")
                        with stats_lock:
                            stats["status"] = "ERROR: Reload failed"

            loop_start = time.time()
            ret, frame, frame_id = vstream.read()
            
            if not ret or frame is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
                
            last_frame_id = frame_id
                
            with stats_lock: stats["cam_connected"] = True
            
            h, w = frame.shape[:2]
            
            # 1. Person Detection (Optimized imgsz)
            p_results = person_model(frame, classes=[0], conf=0.4, imgsz=256, verbose=False)
            
            any_weapon_now = False
            persons_list = []
            weapons_list = []
            
            raw_persons = []
            
            # Extract raw person detections
            for p_res in p_results:
                for p_box in p_res.boxes:
                    px1, py1, px2, py2 = p_box.xyxy[0].tolist()
                    raw_persons.append({"box": [px1, py1, px2, py2], "conf": float(p_box.conf[0])})
                    
            # 2. Update Person Tracker
            tracked_persons = person_tracker.update(raw_persons, frame_id)
            
            roi_crops = []
            roi_track_refs = []
            
            # 3. Extract ROIs for each tracked person
            for t_person in tracked_persons:
                # OPTIMIZATION: Only run weapon model on tracks that were actively detected in THIS frame.
                # If age > 0, it means it's a predicted/ghost track from a previous frame.
                if t_person.age > 0:
                    continue
                    
                px1, py1, px2, py2 = t_person.box
                p_area = (px2 - px1) * (py2 - py1)
                
                roi_box = expand_bbox([px1, py1, px2, py2], w, h, factor=EXPANSION_FACTOR)
                rx1, ry1, rx2, ry2 = roi_box
                roi_crop = frame[ry1:ry2, rx1:rx2]
                if roi_crop.size == 0: continue
                
                roi_crops.append(roi_crop)
                roi_track_refs.append((t_person, rx1, ry1, p_area, py1, py2, px1, py1))
            
            for t in tracked_persons:
                t.weapon_detected_this_frame = False
            
            is_alerting = False
            raw_weapons = []
            
            # 4. Weapon (Batched Inference on ROIs)
            if roi_crops:
                w_results = weapon_model(roi_crops, conf=CONF_THRESH, imgsz=640, verbose=False)
                for i, w_res in enumerate(w_results):
                    t_person, rx1, ry1, p_area, py1, py2, px1, py1_orig = roi_track_refs[i]
                    weapon_found = False
                    max_weapon_conf = 0.0
                    for w_box in w_res.boxes:
                        wx1, wy1, wx2, wy2 = w_box.xyxy[0].tolist()
                        fwx1, fwy1, fwx2, fwy2 = rx1+wx1, ry1+wy1, rx1+wx2, ry1+wy2
                        
                        # Filter: Size Constraint
                        w_w = wx2 - wx1
                        w_h = wy2 - wy1
                        size_ratio = (w_w * w_h / p_area)
                        if size_ratio < MIN_SIZE_RATIO:
                            print(f"DEBUG Filtered: Size Ratio {size_ratio:.4f} < {MIN_SIZE_RATIO}")
                            continue
                        
                        # Filter: Position/Anatomy Constraint
                        w_cy = (fwy1+fwy2)/2
                        anatomy_ratio = (w_cy-py1)/(py2-py1)
                        if anatomy_ratio < 0.1 or anatomy_ratio > 0.9:
                            print(f"DEBUG Filtered: Anatomy Ratio {anatomy_ratio:.2f} outside [0.1, 0.9] (Y position relative to body)")
                            continue
                        
                        # Filter: Aspect Ratio Constraint (phones/sticks vs handguns/rifles)
                        if w_h > 0:
                            aspect_ratio = w_w / w_h
                            if aspect_ratio < 0.20 or aspect_ratio > 3.0:
                                print(f"DEBUG Filtered: Aspect Ratio {aspect_ratio:.2f} outside [0.20, 3.0]")
                                continue
                        
                        weapon_found = True
                        conf_val = float(w_box.conf[0])
                        print(f"DEBUG Weapon Passed: Conf {conf_val:.2f}, Size {size_ratio:.3f}, Anatomy {anatomy_ratio:.2f}, Aspect {aspect_ratio:.2f}")
                        max_weapon_conf = max(max_weapon_conf, conf_val)
                        
                        # Relative box coordinates to the person box
                        rel_box = [fwx1 - px1, fwy1 - py1_orig, fwx2 - px1, fwy2 - py1_orig]
                        raw_weapons.append({
                            "rel_box": rel_box,
                            "conf": conf_val,
                            "person_id": t_person.id
                        })
                        
                    if weapon_found:
                        t_person.weapon_detected_this_frame = True
                        # High-Confidence Bypass: instantly trigger alert if confidence is >= 90%
                        if max_weapon_conf >= 0.90:
                            t_person.weapon_counter = max(t_person.weapon_counter, float(TEMPORAL_THRESHOLD))
                        else:
                            t_person.weapon_counter = min(5.0, t_person.weapon_counter + 1.0)
                    else:
                        # Lower decay penalty from 0.5 to 0.2 to handle frame drops
                        t_person.weapon_counter = max(0.0, t_person.weapon_counter - 0.2)
                        
                    if t_person.weapon_counter >= TEMPORAL_THRESHOLD:
                        t_person.is_armed = True
            
            # Check global alert status (only if the armed person is actively visible in this frame)
            for t in tracked_persons:
                if t.is_armed and t.age == 0:
                    is_alerting = True
                    break
            
            if is_alerting:
                # Save Snapshot with 3-second cooldown
                if time.time() - last_snapshot_time > 3.0:
                    last_snapshot_time = time.time()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = os.path.join(SNAPSHOT_DIR, f"detection_{ts}.jpg")
                    
                    # Annotate copy of the frame specifically for the snapshot
                    snap_frame = frame.copy()
                    for t in tracked_persons:
                        if t.is_armed:
                            draw_premium_box(snap_frame, t.box, "ARMED PERSON", (0, 0, 255), True)
                        else:
                            draw_premium_box(snap_frame, t.box, "Person", (150, 150, 150))
                    
                    # Draw actual weapon boxes relative to person box
                    for w_info in raw_weapons:
                        pid = w_info["person_id"]
                        p_box = None
                        for t in tracked_persons:
                            if t.id == pid:
                                p_box = t.box
                                break
                        if p_box is not None:
                            rel = w_info["rel_box"]
                            w_abs = [p_box[0] + rel[0], p_box[1] + rel[1], p_box[0] + rel[2], p_box[1] + rel[3]]
                            draw_premium_box(snap_frame, w_abs, f"WEAPON {w_info['conf']:.2f}", (0, 0, 255), False)
                    
                    cv2.rectangle(snap_frame, (0, 0), (w, 60), (0, 0, 200), -1)
                    cv2.putText(snap_frame, "!!! ARMED THREAT DETECTED !!!", (int(w/2)-250, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
                    cv2.rectangle(snap_frame, (0,0), (w,h), (0,0,255), 10)
                    
                    cv2.imwrite(snap_path, snap_frame)
                    print(f"📸 Saved snapshot: {snap_path}")

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
                for t in tracked_persons:
                    if t.is_armed:
                        draw_premium_box(record_frame, t.box, "ARMED PERSON", (0, 0, 255), True)
                    else:
                        draw_premium_box(record_frame, t.box, "Person", (150, 150, 150))
                
                for w_info in raw_weapons:
                    pid = w_info["person_id"]
                    p_box = None
                    for t in tracked_persons:
                        if t.id == pid:
                            p_box = t.box
                            break
                    if p_box is not None:
                        rel = w_info["rel_box"]
                        w_abs = [p_box[0] + rel[0], p_box[1] + rel[1], p_box[0] + rel[2], p_box[1] + rel[3]]
                        draw_premium_box(record_frame, w_abs, f"WEAPON {w_info['conf']:.2f}", (0, 0, 255), False)
                
                cv2.rectangle(record_frame, (0, 0), (w, 60), (0, 0, 200), -1)
                cv2.putText(record_frame, "!!! ARMED THREAT DETECTED !!!", (int(w/2)-250, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
                if int(time.time()*4)%2==0:
                    cv2.rectangle(record_frame, (0,0), (w,h), (0,0,255), 10)
                    
                video_writer.write(record_frame)
                
                # Check if alert cleared and cooldown expired
                if not is_alerting and (time.time() - last_threat_time > post_record_cooldown):
                    video_writer.release()
                    video_writer = None
                    recording_active = False
                    print(f"🎥 Threat cleared. Saved evidence video clip to: {RECORDING_DIR}")

            # Update thread-safe active detections state
            with roi_detections_lock:
                latest_roi_detections = {
                    "tracked_persons": [{"id": t.id, "box": t.box, "velocity": t.velocity, "is_armed": t.is_armed} for t in tracked_persons if t.age == 0],
                    "weapons": raw_weapons,
                    "is_alerting": is_alerting
                }

            # Cache the AI results for this frame_id in our history buffer
            with ai_history_lock:
                ai_history[frame_id] = {
                    "tracked_persons": [{"id": t.id, "box": t.box, "velocity": t.velocity, "is_armed": t.is_armed} for t in tracked_persons if t.age == 0],
                    "weapons": raw_weapons,
                    "is_alerting": is_alerting
                }
                # Keep history cache limited to 100 frames to prevent memory bloat
                if len(ai_history) > 100:
                    oldest_keys = sorted(ai_history.keys())[:-100]
                    for k in oldest_keys:
                        ai_history.pop(k, None)

            # Update Dashboard State Every Frame
            with stats_lock:
                # Latch: Keep alert active for 2s in UI for visibility
                if is_alerting:
                    last_alert_latch_time = time.time()
                    if not alert_in_progress:
                        stats["alerts"] += 1
                        stats["last_alert"] = datetime.now().strftime("%H:%M:%S")
                        alert_in_progress = True
                elif (time.time() - last_alert_latch_time) > 1.0:
                    alert_in_progress = False

                stats["alert_active"] = (time.time() - last_alert_latch_time) < 1.0

            # Stats (1-second tick)
            frames += 1
            if time.time() - fps_timer > 1.0:
                dfps = frames / (time.time() - fps_timer)
                fps_timer = time.time()
                frames = 0
                with stats_lock:
                    stats["detect_fps"] = round(dfps, 1)
                    stats["status"] = "LIVE"

            # Maintain Stable FPS
            elapsed = time.time() - loop_start
            wait_time = max(0.001, frame_interval - elapsed)
            time.sleep(wait_time)
    finally:
        if video_writer is not None:
            video_writer.release()
            print("🎥 Safely released video writer on thread exit.")


# ─────────────────────────────────────────────
# WEB SERVER
# ─────────────────────────────────────────────
app = Flask(__name__)

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SMS Vision AI | Command Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root { 
            --bg: #05080f; 
            --panel: #0b1220; 
            --border: #1a2f50; 
            --accent: #38bdf8; 
            --danger: #ef4444; 
            --safe: #22c55e;
            --text: #cbd5e1;
        }
        body { 
            background: var(--bg); 
            color: var(--text); 
            font-family: 'Outfit', sans-serif; 
            margin: 0; 
            display: flex; 
            flex-direction: column; 
            height: 100vh; 
            overflow: hidden; 
        }
        header { 
            padding: 12px 25px; 
            background: var(--panel); 
            border-bottom: 1px solid var(--border); 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }
        .logo-section {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .logo { 
            font-size: 1.3rem; 
            font-weight: 900; 
            color: var(--accent); 
            letter-spacing: 1px; 
        }
        .tabs { 
            display: flex; 
            gap: 8px; 
        }
        .tab-btn { 
            background: rgba(255,255,255,0.03); 
            border: 1px solid var(--border); 
            color: #94a3b8; 
            padding: 8px 18px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-weight: 700; 
            font-family: inherit; 
            font-size: 0.8rem; 
            transition: 0.2s; 
        }
        .tab-btn:hover { 
            background: rgba(255,255,255,0.08); 
            color: #fff; 
        }
        .tab-btn.active { 
            background: var(--accent); 
            border-color: var(--accent); 
            color: #05080f; 
        }
        .badge { 
            font-size: 0.7rem; 
            padding: 4px 12px; 
            border-radius: 12px; 
            background: #1e293b; 
            font-weight: 700;
        }
        
        .tab-content { 
            display: grid; 
            grid-template-columns: 1fr 320px; 
            gap: 15px; 
            padding: 15px; 
            flex: 1; 
            min-height: 0; 
            box-sizing: border-box;
        }
        
        /* Live Monitor Tab Styles */
        .video-box { 
            background: #000; 
            border: 1px solid var(--border); 
            border-radius: 12px; 
            position: relative; 
            overflow: hidden; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
        }
        #feed { 
            max-width: 100%; 
            max-height: 100%; 
            object-fit: contain; 
        }
        .sidebar { 
            display: flex; 
            flex-direction: column; 
            gap: 15px; 
            min-height: 0;
            overflow-y: auto;
        }
        .card { 
            background: var(--panel); 
            border: 1px solid var(--border); 
            border-radius: 12px; 
            padding: 15px; 
        }
        .alert-card { 
            text-align: center; 
            font-weight: 900; 
            padding: 25px 15px; 
            transition: 0.3s; 
        }
        .alert-card.safe { 
            border-color: var(--safe); 
            color: var(--safe); 
            background: rgba(34,197,94,0.05); 
        }
        .alert-card.danger { 
            border-color: var(--danger); 
            color: var(--danger); 
            background: rgba(239,68,68,0.1); 
            animation: pulse 1s infinite; 
        }
        @keyframes pulse { 
            0%, 100% { transform: scale(1); } 
            50% { transform: scale(1.02); } 
        }
        .stat-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 10px; 
        }
        .stat-item { 
            background: rgba(255,255,255,0.03); 
            padding: 10px; 
            border-radius: 8px; 
            text-align: center; 
        }
        .stat-val { 
            font-size: 1.2rem; 
            font-weight: 900; 
            color: var(--accent); 
        }
        .stat-label { 
            font-size: 0.6rem; 
            color: #64748b; 
            text-transform: uppercase; 
        }
        
        /* Evidence Vault Tab Styles */
        .vault-layout {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 15px;
            height: 100%;
        }
        .vault-sidebar {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: 100%;
        }
        .vault-header {
            padding: 15px;
            border-bottom: 1px solid var(--border);
            font-weight: 900;
            font-size: 0.85rem;
            color: #fff;
            letter-spacing: 0.5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .vault-items {
            overflow-y: auto;
            flex: 1;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .vault-item {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            transition: 0.2s;
        }
        .vault-item:hover {
            background: rgba(255,255,255,0.05);
            border-color: var(--border);
        }
        .vault-item.active {
            background: rgba(56,189,248,0.08);
            border-color: var(--accent);
        }
        .vault-item-title {
            font-weight: 700;
            font-size: 0.85rem;
            color: #fff;
            margin-bottom: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .vault-item-meta {
            font-size: 0.7rem;
            color: #64748b;
            display: flex;
            justify-content: space-between;
        }
        .vault-badge {
            font-size: 0.6rem;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 900;
            text-transform: uppercase;
        }
        .badge-video {
            background: rgba(239,68,68,0.1);
            color: var(--danger);
            border: 1px solid rgba(239,68,68,0.2);
        }
        .badge-image {
            background: rgba(56,189,248,0.1);
            color: var(--accent);
            border: 1px solid rgba(56,189,248,0.2);
        }
        
        .vault-preview-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden;
        }
        .vault-player-container {
            background: #000;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            border-bottom: 1px solid var(--border);
        }
        #vault-media-preview {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .vault-controls {
            padding: 15px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0,0,0,0.2);
        }
        .control-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .btn {
            padding: 8px 18px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.8rem;
            font-weight: 700;
            cursor: pointer;
            border: 1px solid transparent;
            transition: 0.2s;
        }
        .btn-primary {
            background: var(--accent);
            color: #05080f;
        }
        .btn-primary:hover {
            opacity: 0.9;
        }
        .btn-danger {
            background: transparent;
            border-color: var(--danger);
            color: var(--danger);
        }
        .btn-danger:hover {
            background: var(--danger);
            color: #fff;
        }
        .btn-secondary {
            background: rgba(255,255,255,0.05);
            border-color: var(--border);
            color: var(--text);
        }
        .btn-secondary:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #64748b;
            gap: 10px;
            text-align: center;
            height: 100%;
        }
        .empty-icon {
            font-size: 3rem;
            opacity: 0.4;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-section">
            <div class="logo">🛡️ SMS VISION AI <span style="font-weight: 400; color: #64748b; font-size: 0.8rem;">v2.0</span></div>
            
            <div style="margin-left: 20px; display: flex; align-items: center; gap: 10px;">
                <label style="font-size: 0.8rem; font-weight: 700; color: #94a3b8;">CAMERA:</label>
                <select id="camera-select" onchange="changeCamera(this.value)" style="background: var(--panel); border: 1px solid var(--border); color: #fff; padding: 6px 12px; border-radius: 6px; font-family: inherit; font-size: 0.8rem; cursor: pointer; outline: none;">
                    {% for cam in cameras.keys() %}
                        <option value="{{ cam }}" {% if cam == active_camera %}selected{% endif %}>{{ cam }}</option>
                    {% endfor %}
                    <option value="Custom Video" id="custom-video-option" {% if active_camera == "Custom Video" %}selected{% else %}style="display:none;"{% endif %}>Custom Video</option>
                </select>
            </div>

            <div style="margin-left: 20px; display: flex; align-items: center; gap: 10px; border-left: 1px solid var(--border); padding-left: 20px;">
                <label style="font-size: 0.8rem; font-weight: 700; color: #94a3b8;">MODEL:</label>
                <select id="model-select" onchange="changeModel(this.value)" style="background: var(--panel); border: 1px solid var(--border); color: #fff; padding: 6px 12px; border-radius: 6px; font-family: inherit; font-size: 0.8rem; cursor: pointer; outline: none;">
                    {% for model in models %}
                        <option value="{{ model }}" {% if model == current_model %}selected{% endif %}>{{ model.split('/')[-1] }}</option>
                    {% endfor %}
                </select>
            </div>

            <div style="margin-left: 20px; display: flex; align-items: center; gap: 10px; border-left: 1px solid var(--border); padding-left: 20px;">
                <label style="font-size: 0.8rem; font-weight: 700; color: #94a3b8;">📂 TEST VIDEO:</label>
                <button class="tab-btn" onclick="document.getElementById('video-upload-input').click()" style="padding: 6px 12px; font-size: 0.75rem;">📤 Upload File</button>
                <input type="file" id="video-upload-input" accept=".mp4,.avi,.mov,.mkv" style="display:none;" onchange="handleVideoUpload(this)">
                
                <input type="text" id="video-path-input" placeholder="Or enter local path..." style="background: rgba(0,0,0,0.3); border: 1px solid var(--border); color: #fff; padding: 6px 10px; border-radius: 6px; font-family: inherit; font-size: 0.75rem; width: 170px; outline: none; transition: 0.2s;" onfocus="this.style.borderColor='var(--accent)'" onblur="this.style.borderColor='var(--border)'">
                <button class="tab-btn" onclick="handleVideoPathLoad()" style="padding: 6px 12px; font-size: 0.75rem;">⚡ Run</button>
            </div>

            <div class="tabs" style="margin-left: 20px;">
                <button id="tab-live" class="tab-btn active" onclick="switchTab('live')">📺 LIVE MONITOR</button>
                <button id="tab-vault" class="tab-btn" onclick="switchTab('vault')">📂 EVIDENCE VAULT</button>
            </div>
        </div>
        <div class="badge" id="status-badge">SYSTEM LOADING...</div>
    </header>
    
    <!-- Tab 1: Live Monitor -->
    <main id="live-tab" class="tab-content">
        <div class="video-box">
            <img id="feed" src="/video_feed" onerror="this.src='https://via.placeholder.com/1280x720?text=Camera+Signal+Lost'">
        </div>
        <div class="sidebar">
            <div id="alert-card" class="card alert-card safe">✅ AREA SECURE</div>
            <div class="card">
                <div class="stat-label" style="margin-bottom:10px;">Performance Metrics</div>
                <div class="stat-grid">
                    <div class="stat-item"><div id="detect-fps" class="stat-val">0.0</div><div class="stat-label">AI FPS</div></div>
                    <div class="stat-item"><div id="total-alerts" class="stat-val">0</div><div class="stat-label">Total Alerts</div></div>
                </div>
            </div>
            <div class="card" style="flex:1;">
                <div class="stat-label" style="margin-bottom:10px;">System Parameters</div>
                <div style="font-size: 0.8rem; line-height: 1.8;">
                    <div>• Model: <span style="color:var(--accent)" id="active-model-name">{{ weapon_model.split('/')[-1] }}</span></div>
                    <div>• Threshold: <span style="color:var(--accent)">{{ conf_thresh }}</span></div>
                    <div>• Expansion: <span style="color:var(--accent)">20%</span></div>
                    <div>• Min Size: <span style="color:var(--accent)">1% ROI</span></div>
                    <div>• Temporal: <span style="color:var(--accent)">3 Frames</span></div>
                </div>
                <div id="last-alert-box" style="margin-top:20px; font-size:0.7rem; color:#64748b;">
                    Last Alert: <span id="last-alert-time" style="color:#94a3b8">None</span>
                </div>
            </div>
        </div>
    </main>
    
    <!-- Tab 2: Evidence Vault -->
    <main id="vault-tab" class="tab-content" style="display:none;">
        <div class="vault-layout">
            <div class="vault-sidebar">
                <div class="vault-header">
                    <span>Threat Incidents</span>
                    <button class="btn btn-secondary" style="padding: 4px 10px; font-size:0.7rem;" onclick="loadArchives()">🔄 Refresh</button>
                </div>
                <div id="vault-items" class="vault-items">
                    <!-- Dynamic List of Recordings -->
                </div>
            </div>
            <div class="vault-preview-card">
                <div class="vault-player-container">
                    <img id="vault-media-preview" style="display:none;">
                    <div id="vault-placeholder" class="empty-state">
                        <div class="empty-icon">🎥</div>
                        <div>Select a threat incident from the timeline to review evidence</div>
                    </div>
                </div>
                <div class="vault-controls">
                    <div class="control-group">
                        <div>
                            <div id="preview-title" style="font-weight: 700; font-size: 0.9rem; color: #fff;">Select an item</div>
                            <div id="preview-meta" style="font-size: 0.75rem; color: #64748b; margin-top:2px;">No file loaded</div>
                        </div>
                    </div>
                    <div class="control-group">
                        <button id="play-btn" class="btn btn-primary" style="display:none;" onclick="togglePlay()">▶ PLAY RECORDING</button>
                        <button id="download-btn" class="btn btn-secondary">📥 DOWNLOAD</button>
                        <button id="delete-btn" class="btn btn-danger">🗑️ DELETE</button>
                    </div>
                </div>
            </div>
        </div>
    </main>
    
    <script>
        // Camera Switching Handler
        async function changeCamera(camName) {
            const feedEl = document.getElementById('feed');
            document.getElementById('status-badge').innerText = "CONNECTING...";
            document.getElementById('status-badge').style.color = '#eab308';
            // Step 1: Blank the feed immediately to break old stream
            feedEl.src = '';
            
            // Hide custom video option if switching back to camera
            if (camName !== 'Custom Video') {
                const customOpt = document.getElementById('custom-video-option');
                if (customOpt) customOpt.style.display = 'none';
            }
            
            try {
                const res = await fetch('/api/change_camera', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ camera_name: camName })
                });
                const data = await res.json();
                if(data.status === 'success') {
                    // Step 2: Wait 2.5s for RTSP reconnect, then reload with fresh URL
                    setTimeout(() => {
                        feedEl.src = '/video_feed?t=' + Date.now();
                    }, 2500);
                } else {
                    alert(data.message);
                }
            } catch(e) {
                console.error(e);
                alert("Failed to switch camera.");
            }
        }

        // Video Upload Handler
        async function handleVideoUpload(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const feedEl = document.getElementById('feed');
            
            document.getElementById('status-badge').innerText = "UPLOADING...";
            document.getElementById('status-badge').style.color = '#eab308';
            feedEl.src = '';
            
            const formData = new FormData();
            formData.append('video', file);
            
            try {
                const res = await fetch('/api/upload_video', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.status === 'success') {
                    // Update Dropdown to Custom Video
                    const customOpt = document.getElementById('custom-video-option');
                    customOpt.style.display = 'block';
                    document.getElementById('camera-select').value = 'Custom Video';
                    
                    // Wait a moment, then load feed
                    setTimeout(() => {
                        feedEl.src = '/video_feed?t=' + Date.now();
                    }, 2000);
                } else {
                    alert(data.message);
                    document.getElementById('status-badge').innerText = "ERROR";
                }
            } catch(e) {
                console.error(e);
                alert("Upload failed.");
            }
        }

        // Local Video Path Handler
        async function handleVideoPathLoad() {
            const pathInput = document.getElementById('video-path-input');
            const path = pathInput.value.trim();
            if (!path) {
                alert("Please enter a valid local video file path.");
                return;
            }
            
            const feedEl = document.getElementById('feed');
            document.getElementById('status-badge').innerText = "LOADING PATH...";
            document.getElementById('status-badge').style.color = '#eab308';
            feedEl.src = '';
            
            try {
                const res = await fetch('/api/load_video_path', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ video_path: path })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    // Update Dropdown to Custom Video
                    const customOpt = document.getElementById('custom-video-option');
                    customOpt.style.display = 'block';
                    document.getElementById('camera-select').value = 'Custom Video';
                    
                    setTimeout(() => {
                        feedEl.src = '/video_feed?t=' + Date.now();
                    }, 2000);
                } else {
                    alert(data.message);
                    document.getElementById('status-badge').innerText = "ERROR";
                }
            } catch(e) {
                console.error(e);
                alert("Failed to load local video path.");
            }
        }

        // Live Statistics Handlers
        async function updateStats() {
            try {
                const res = await fetch('/stats');
                const d = await res.json();
                document.getElementById('detect-fps').innerText = d.detect_fps;
                document.getElementById('total-alerts').innerText = d.alerts;
                document.getElementById('status-badge').innerText = d.status;
                document.getElementById('status-badge').style.color = d.cam_connected ? '#22c55e' : '#ef4444';
                
                const ac = document.getElementById('alert-card');
                if (d.alert_active) {
                    ac.innerText = "🚨 WEAPON DETECTED";
                    ac.className = "card alert-card danger";
                } else {
                    ac.innerText = "✅ AREA SECURE";
                    ac.className = "card alert-card safe";
                }
                
                if (d.last_alert) {
                    document.getElementById('last-alert-time').innerText = d.last_alert;
                }
                
                if (d.weapon_model) {
                    const el = document.getElementById('active-model-name');
                    if (el) el.innerText = d.weapon_model;
                }
            } catch(e) {}
        }
        setInterval(updateStats, 500);

        // Model Switching Handler
        async function changeModel(modelName) {
            document.getElementById('status-badge').innerText = "RELOADING MODEL...";
            document.getElementById('status-badge').style.color = '#eab308';
            try {
                const res = await fetch('/api/change_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: modelName })
                });
                const data = await res.json();
                if(data.status !== 'success') {
                    alert(data.message);
                }
            } catch(e) {
                console.error(e);
                alert("Failed to switch model.");
            }
        }

        // Evidence Vault State & Handlers
        let selectedFile = null;
        let isPlaying = false;
        
        function switchTab(tab) {
            document.getElementById('tab-live').classList.remove('active');
            document.getElementById('tab-vault').classList.remove('active');
            
            if (tab === 'live') {
                document.getElementById('tab-live').classList.add('active');
                document.getElementById('live-tab').style.display = 'grid';
                document.getElementById('vault-tab').style.display = 'none';
            } else {
                document.getElementById('tab-vault').classList.add('active');
                document.getElementById('live-tab').style.display = 'none';
                document.getElementById('vault-tab').style.display = 'grid';
                loadArchives();
            }
        }
        
        async function loadArchives() {
            try {
                const res = await fetch('/api/archives');
                const data = await res.json();
                const itemsContainer = document.getElementById('vault-items');
                itemsContainer.innerHTML = '';
                
                if (data.length === 0) {
                    itemsContainer.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-icon">📂</div>
                            <div style="font-size:0.8rem;">No threat recordings or snapshots found.</div>
                        </div>
                    `;
                    showEmptyPreview();
                    return;
                }
                
                data.forEach(item => {
                    const div = document.createElement('div');
                    div.className = `vault-item ${selectedFile && selectedFile.name === item.name ? 'active' : ''}`;
                    div.onclick = () => selectArchive(item, div);
                    
                    const isVideo = item.type === 'video';
                    const badgeClass = isVideo ? 'badge-video' : 'badge-image';
                    
                    div.innerHTML = `
                        <div class="vault-item-title">
                            <span>${isVideo ? '🎥 Weapon Threat' : '📸 Alert Snapshot'}</span>
                            <span class="vault-badge ${badgeClass}">${item.type}</span>
                        </div>
                        <div class="vault-item-meta">
                            <span>${item.timestamp}</span>
                            <span>${item.size}</span>
                        </div>
                    `;
                    itemsContainer.appendChild(div);
                });
                
                if (!selectedFile && data.length > 0) {
                    selectArchive(data[0], itemsContainer.firstElementChild);
                }
            } catch (e) {
                console.error(e);
            }
        }
        
        function selectArchive(item, element) {
            document.querySelectorAll('.vault-item').forEach(el => el.classList.remove('active'));
            if (element) element.classList.add('active');
            
            selectedFile = item;
            isPlaying = false;
            
            const preview = document.getElementById('vault-media-preview');
            const placeholder = document.getElementById('vault-placeholder');
            const title = document.getElementById('preview-title');
            const meta = document.getElementById('preview-meta');
            const playBtn = document.getElementById('play-btn');
            
            preview.style.display = 'block';
            placeholder.style.display = 'none';
            
            title.innerText = item.name;
            meta.innerText = `${item.timestamp} • ${item.size}`;
            
            if (item.type === 'video') {
                playBtn.style.display = 'inline-block';
                playBtn.innerText = '▶ PLAY RECORDING';
                preview.src = 'https://via.placeholder.com/1280x720?text=Zero-Lag+MJPEG+Player+Ready';
            } else {
                playBtn.style.display = 'none';
                preview.src = `/archive_image/${item.name}?t=${Date.now()}`;
            }
            
            document.getElementById('download-btn').onclick = () => {
                window.location.href = `/download_archive/${item.type}/${item.name}`;
            };
            
            document.getElementById('delete-btn').onclick = () => deleteArchive(item);
        }
        
        function togglePlay() {
            if (!selectedFile || selectedFile.type !== 'video') return;
            
            const preview = document.getElementById('vault-media-preview');
            const playBtn = document.getElementById('play-btn');
            
            if (isPlaying) {
                preview.src = 'https://via.placeholder.com/1280x720?text=Stream+Paused';
                playBtn.innerText = '▶ PLAY RECORDING';
                isPlaying = false;
            } else {
                preview.src = `/archive_feed/${selectedFile.name}?t=${Date.now()}`;
                playBtn.innerText = '⏸ PAUSE';
                isPlaying = true;
            }
        }
        
        async function deleteArchive(item) {
            if (!confirm(`Are you sure you want to permanently delete this evidence file?\n\nFilename: ${item.name}`)) return;
            try {
                const res = await fetch(`/api/delete_archive/${item.type}/${item.name}`, { method: 'DELETE' });
                const d = await res.json();
                if (d.status === 'success') {
                    selectedFile = null;
                    loadArchives();
                } else {
                    alert('Error deleting file: ' + d.message);
                }
            } catch(e) {
                alert('Failed to connect to server.');
            }
        }
        
        function showEmptyPreview() {
            document.getElementById('vault-media-preview').style.display = 'none';
            document.getElementById('vault-placeholder').style.display = 'flex';
            document.getElementById('preview-title').innerText = 'Select an item';
            document.getElementById('preview-meta').innerText = 'No file loaded';
            document.getElementById('play-btn').style.display = 'none';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    active_cam = "Camera 13"
    curr_model = "models/bestcctv1.pt"
    with stats_lock:
        active_cam = stats.get("active_camera", "Camera 13")
        model_basename = stats.get("weapon_model", "bestcctv1.pt")
        for m in AVAILABLE_MODELS:
            if os.path.basename(m) == model_basename:
                curr_model = m
                break
    return render_template_string(DASHBOARD_HTML, models=AVAILABLE_MODELS, current_model=curr_model, weapon_model=WEAPON_MODEL_PATH, cameras=CAMERAS, active_camera=active_cam, conf_thresh=f"{CONF_THRESH:.2f}")

@app.route('/api/change_model', methods=['POST'])
def api_change_model():
    global reload_model_path
    data = request.json
    model_name = data.get('model_name')
    if model_name in AVAILABLE_MODELS:
        reload_model_path = model_name
        model_reload_event.set()
        return jsonify({"status": "success", "message": f"Initiated model reload to {model_name}"})
    return jsonify({"status": "error", "message": "Model not found"}), 404

@app.route('/api/change_camera', methods=['POST'])
def api_change_camera():
    global RTSP_SOURCE
    data = request.json
    cam_name = data.get('camera_name')
    if cam_name in CAMERAS:
        new_source = CAMERAS[cam_name]
        RTSP_SOURCE = new_source
        vstream.change_source(new_source)
        with stats_lock:
            stats["active_camera"] = cam_name
            stats["status"] = "Connecting..."
            stats["cam_connected"] = False
        return jsonify({"status": "success", "message": f"Switched to {cam_name}"})
    return jsonify({"status": "error", "message": "Camera not found"}), 404

@app.route('/api/upload_video', methods=['POST'])
def api_upload_video():
    global RTSP_SOURCE
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
        
    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)
    
    RTSP_SOURCE = filepath
    vstream.change_source(filepath)
    with stats_lock:
        stats["active_camera"] = "Custom Video"
        stats["status"] = "Loading Video..."
        stats["cam_connected"] = False
        
    return jsonify({"status": "success", "message": f"Switched to custom uploaded video: {filename}"})

@app.route('/api/load_video_path', methods=['POST'])
def api_load_video_path():
    global RTSP_SOURCE
    data = request.json
    video_path = data.get('video_path', '').strip()
    if not video_path:
        return jsonify({"status": "error", "message": "Video path cannot be empty"}), 400
        
    # Clean quotes
    if (video_path.startswith('"') and video_path.endswith('"')) or (video_path.startswith("'") and video_path.endswith("'")):
        video_path = video_path[1:-1]
        
    if not os.path.exists(video_path):
        return jsonify({"status": "error", "message": f"File not found at: {video_path}"}), 404
        
    # Check extension
    _, ext = os.path.splitext(video_path)
    if ext.lower() not in ['.mp4', '.avi', '.mkv', '.mov']:
        return jsonify({"status": "error", "message": f"Unsupported format '{ext}'. Use MP4, AVI, MKV, or MOV."}), 400
        
    RTSP_SOURCE = video_path
    vstream.change_source(video_path)
    with stats_lock:
        stats["active_camera"] = "Custom Video"
        stats["status"] = "Loading Video..."
        stats["cam_connected"] = False
        
    return jsonify({"status": "success", "message": f"Switched to custom local video path: {os.path.basename(video_path)}"})

@app.route('/video_feed')
def video_feed():
    def gen():
        boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        target_fps = 25.0
        frame_interval = 1.0 / target_fps
        
        last_played_id = -1
        
        while True:
            loop_start = time.time()
            
            if vstream is None:
                time.sleep(0.01)
                continue
                
            with vstream.buffer_lock:
                buf_len = len(vstream.buffer)
                if buf_len == 0:
                    time.sleep(0.01)
                    continue
                
                # Retrieve the absolute latest frame from the stream buffer
                frame_id, frame = vstream.buffer[-1]
                
                # If we already served this frame, wait briefly for the next frame
                if frame_id == last_played_id:
                    time.sleep(0.005)
                    continue
                
                frame = frame.copy()
                last_played_id = frame_id
                
            h, w = frame.shape[:2]
            
            # Retrieve cached AI detections for this frame_id
            active_dets = None
            nearest_id = None
            with ai_history_lock:
                if frame_id in ai_history:
                    active_dets = ai_history[frame_id]
                    nearest_id = frame_id
                else:
                    # Fallback: Find the nearest processed frame_id
                    available_ids = sorted(ai_history.keys())
                    for aid in available_ids:
                        if aid <= frame_id:
                            nearest_id = aid
                        else:
                            break
                    if nearest_id is not None:
                        active_dets = ai_history[nearest_id]
            
            # Draw synchronized bounding boxes with linear motion prediction
            if active_dets is not None:
                delta_frames = frame_id - nearest_id
                
                # Cap extrapolation at max 5 frames to prevent runaway boxes on sudden turns/stops
                extrap_delta = max(0, min(delta_frames, 5))
                
                # Decay velocity factor based on how old the detection is
                decay_factor = 0.9 ** extrap_delta
                
                extrapolated_persons = {}
                
                for tp in active_dets.get("tracked_persons", []):
                    box = tp["box"]
                    velocity = tp.get("velocity", [0.0, 0.0, 0.0, 0.0])
                    
                    # Extrapolate person box position for intermediate frames with decayed velocity
                    x1 = int(box[0] + velocity[0] * extrap_delta * decay_factor)
                    y1 = int(box[1] + velocity[1] * extrap_delta * decay_factor)
                    x2 = int(box[2] + velocity[2] * extrap_delta * decay_factor)
                    y2 = int(box[3] + velocity[3] * extrap_delta * decay_factor)
                    extrapolated_box = [x1, y1, x2, y2]
                    extrapolated_persons[tp["id"]] = extrapolated_box
                    
                    if tp["is_armed"]:
                        draw_premium_box(frame, extrapolated_box, "ARMED PERSON", (0, 0, 255), True)
                    else:
                        draw_premium_box(frame, extrapolated_box, "Person", (150, 150, 150))
                        
                for w_info in active_dets.get("weapons", []):
                    pid = w_info.get("person_id")
                    if pid in extrapolated_persons:
                        epx = extrapolated_persons[pid]
                        rel = w_info["rel_box"]
                        wx1 = epx[0] + rel[0]
                        wy1 = epx[1] + rel[1]
                        wx2 = epx[0] + rel[2]
                        wy2 = epx[1] + rel[3]
                        draw_premium_box(frame, [wx1, wy1, wx2, wy2], f"WEAPON {w_info['conf']:.2f}", (0, 0, 255), False)
                    
                if active_dets["is_alerting"]:
                    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
                    cv2.putText(frame, "!!! ARMED THREAT DETECTED !!!", (int(w/2)-250, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
                    if int(time.time()*4)%2==0:
                        cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 10)
                        
            # Encode frame to JPEG
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                yield (boundary + buf.tobytes() + b'\r\n')
                
            # Maintain a stable 25 FPS stream
            elapsed = time.time() - loop_start
            time.sleep(max(0.001, frame_interval - elapsed))
            
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    })

@app.route('/stats')
def stats_api():
    with stats_lock:
        return jsonify(stats)

@app.route('/api/archives')
def api_archives():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(base_dir, "cctv_recordings")
    images_dir = os.path.join(base_dir, "cctv1_images")
    files = []
    
    # 1. Videos (recordings)
    if os.path.exists(recordings_dir):
        for f in os.listdir(recordings_dir):
            if f.endswith('.avi'):
                path = os.path.join(recordings_dir, f)
                stat = os.stat(path)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                size_mb = round(stat.st_size / (1024 * 1024), 2)
                files.append({
                    "name": f,
                    "type": "video",
                    "timestamp": mtime,
                    "size": f"{size_mb} MB"
                })
                
    # 2. Images (snapshots)
    if os.path.exists(images_dir):
        for f in os.listdir(images_dir):
            if f.endswith('.jpg'):
                path = os.path.join(images_dir, f)
                stat = os.stat(path)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                size_kb = round(stat.st_size / 1024, 1)
                files.append({
                    "name": f,
                    "type": "image",
                    "timestamp": mtime,
                    "size": f"{size_kb} KB"
                })
                
    # Sort newest first
    files.sort(key=lambda x: x["timestamp"], reverse=True)
    return jsonify(files)

@app.route('/archive_feed/<path:filename>')
def archive_feed(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "cctv_recordings", filename)
    if not os.path.exists(file_path):
        return "File not found", 404
        
    def gen_archive():
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        if fps < 1.0 or np.isnan(fps):
            fps = 5.0
        delay = 1.0 / fps
        boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                yield (boundary + buf.tobytes() + b'\r\n')
            
            elapsed = time.time() - start_time
            sleep_time = max(0.001, delay - elapsed)
            time.sleep(sleep_time)
        cap.release()
        
    return Response(gen_archive(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/archive_image/<path:filename>')
def archive_image(filename):
    from flask import send_from_directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, "cctv1_images")
    return send_from_directory(directory, filename)

@app.route('/download_archive/<file_type>/<path:filename>')
def download_archive(file_type, filename):
    from flask import send_from_directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, "cctv_recordings" if file_type == "video" else "cctv1_images")
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/api/delete_archive/<file_type>/<path:filename>', methods=['DELETE'])
def delete_archive(file_type, filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, "cctv_recordings" if file_type == "video" else "cctv1_images")
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "not_found"}), 404

if __name__ == '__main__':
    threading.Thread(target=detection_worker, args=(vstream,), daemon=True).start()
    app.run(host='0.0.0.0', port=PORT, threaded=True)
