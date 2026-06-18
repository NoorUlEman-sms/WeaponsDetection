import cv2
import os
import time
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# ✅ CONFIGURATION
# ============================================================
PERSON_MODEL_PATH = "yolov8n.pt"  # Standard COCO model
WEAPON_MODEL_PATH = "models/best (50).pt"  # User's trained model

# RTSP Source (Using the one from app.py)
RTSP_SOURCE = "rtsp://admin:Sms786%40sms@192.168.51.238:554/cam/realmonitor?channel=1&subtype=1"

# Filtering Thresholds
WEAPON_CONF_THRESH = 0.65  # Be aggressive as requested
MIN_SIZE_RATIO = 0.01      # weapon_area / person_area
TEMPORAL_THRESHOLD = 5     # Consecutive frames

# ROI Expansion
EXPANSION_FACTOR = 0.2     # 20% expansion

# ============================================================

class VideoStream:
    """Threaded video stream for low-latency RTSP capture."""
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def expand_bbox(box, frame_w, frame_h, factor=0.2):
    """Expands the bounding box by a given factor and clips to image boundaries."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    x1_new = max(0, x1 - factor * w)
    y1_new = max(0, y1 - factor * h)
    x2_new = min(frame_w, x2 + factor * w)
    y2_new = min(frame_h, y2 + factor * h)
    
    return [int(x1_new), int(y1_new), int(x2_new), int(y2_new)]

def draw_premium_box(frame, box, label, color, thickness=2, is_weapon=False):
    """Draws a professional bounding box with specialized style for weapons."""
    x1, y1, x2, y2 = map(int, box)
    
    if is_weapon:
        # Outer glow for weapons
        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 150), 4)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        # Corner markers
        L = 20
        cv2.line(frame, (x1, y1), (x1+L, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1+L), color, 4)
        cv2.line(frame, (x2, y1), (x2-L, y1), color, 4)
        cv2.line(frame, (x2, y1), (x2, y1+L), color, 4)
    else:
        # Thin dashed-style box for persons
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Label
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = 0.6
    (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    print("🚀 Initializing Multi-Stage Detection Pipeline...")
    
    # 1. Load Models
    print(f"📦 Loading Person Model: {PERSON_MODEL_PATH}")
    person_model = YOLO(PERSON_MODEL_PATH)
    
    print(f"📦 Loading Weapon Model: {WEAPON_MODEL_PATH}")
    if not os.path.exists(WEAPON_MODEL_PATH):
        print(f"❌ Error: Weapon model not found at {WEAPON_MODEL_PATH}")
        return
    weapon_model = YOLO(WEAPON_MODEL_PATH)
    
    # 2. Connect to Camera
    print(f"📡 Connecting to: {RTSP_SOURCE}")
    vstream = VideoStream(RTSP_SOURCE).start()
    
    # Temporal tracking state
    weapon_counter = 0
    alert_active = False
    alert_frames_remaining = 0
    
    last_time = time.time()
    
    try:
        while True:
            ret, frame = vstream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            h, w = frame.shape[:2]
            display_frame = frame.copy()
            
            # --- STAGE 1: Person Detection ---
            # class 0 in COCO is 'person'
            person_results = person_model(frame, classes=[0], conf=0.4, verbose=False)
            
            any_weapon_in_this_frame = False
            detections_to_draw = []
            
            for p_res in person_results:
                for p_box in p_res.boxes:
                    px1, py1, px2, py2 = p_box.xyxy[0].tolist()
                    p_area = (px2 - px1) * (py2 - py1)
                    
                    # Draw person ROI (thin box)
                    detections_to_draw.append(([px1, py1, px2, py2], "Person", (150, 150, 150), False))
                    
                    # --- STAGE 2: ROI Extraction & Expansion ---
                    roi_box = expand_bbox([px1, py1, px2, py2], w, h, factor=EXPANSION_FACTOR)
                    rx1, ry1, rx2, ry2 = roi_box
                    
                    roi_crop = frame[ry1:ry2, rx1:rx2]
                    if roi_crop.size == 0: continue
                    
                    # --- STAGE 3: Weapon Detection in ROI ---
                    weapon_results = weapon_model(roi_crop, conf=WEAPON_CONF_THRESH, verbose=False)
                    
                    for w_res in weapon_results:
                        for w_box in w_res.boxes:
                            # Map weapon box back to full frame coordinates
                            wx1, wy1, wx2, wy2 = w_box.xyxy[0].tolist()
                            full_wx1 = rx1 + wx1
                            full_wy1 = ry1 + wy1
                            full_wx2 = rx1 + wx2
                            full_wy2 = ry1 + wy2
                            
                            w_conf = float(w_box.conf[0])
                            w_area = (wx2 - wx1) * (wy2 - wy1)
                            
                            # --- STAGE 4: Strict Filtering ---
                            
                            # Rule 3: Size Constraint
                            size_ratio = w_area / p_area
                            if size_ratio < MIN_SIZE_RATIO:
                                continue
                                
                            # Rule 4: Position Constraint (relative to person height)
                            # Weapon center Y relative to person box
                            w_cy = (full_wy1 + full_wy2) / 2
                            p_h = py2 - py1
                            rel_y = (w_cy - py1) / p_h
                            
                            # Reject if too low (feet) or too high (far above head)
                            if rel_y < 0.1 or rel_y > 0.9:
                                continue
                            
                            # Valid weapon detection!
                            any_weapon_in_this_frame = True
                            detections_to_draw.append(([full_wx1, full_wy1, full_wx2, full_wy2], f"WEAPON {w_conf:.2f}", (0, 0, 255), True))

            # --- STAGE 5: Temporal Filtering ---
            if any_weapon_in_this_frame:
                weapon_counter += 1
            else:
                weapon_counter = 0
                
            if weapon_counter >= TEMPORAL_THRESHOLD:
                alert_active = True
                alert_frames_remaining = 15 # Keep alert visible for 15 frames
            
            if alert_frames_remaining > 0:
                alert_frames_remaining -= 1
            else:
                alert_active = False

            # --- VISUALIZATION ---
            for box, label, color, is_w in detections_to_draw:
                draw_premium_box(display_frame, box, label, color, is_weapon=is_w)
            
            if alert_active:
                # Big warning banner
                cv2.rectangle(display_frame, (0, 0), (w, 80), (0, 0, 200), -1)
                cv2.putText(display_frame, "!!! ARMED THREAT DETECTED !!!", (int(w/2) - 350, 55), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                # Visual strobe effect
                if int(time.time() * 5) % 2 == 0:
                    cv2.rectangle(display_frame, (0,0), (w,h), (0,0,255), 10)

            # Performance & Info
            curr_fps = 1.0 / (time.time() - last_time)
            last_time = time.time()
            cv2.putText(display_frame, f"FPS: {curr_fps:.1f} | Person->ROI->Weapon Pipeline", (10, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("SMS Vision AI - ROI Pipeline", display_frame)
            cv2.imwrite("live_view.jpg", display_frame) # Update web view if running
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("🧹 Cleaning up...")
        vstream.stop()
        cv2.destroyAllWindows()
        print("✅ Pipeline Stopped.")

if __name__ == "__main__":
    main()
