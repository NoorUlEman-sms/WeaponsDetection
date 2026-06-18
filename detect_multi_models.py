import cv2
import threading
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# Camera 13 - All Three Models Comparison
# ============================================================
RTSP_SOURCE = "rtsp://admin:Sms786%40sms@192.168.51.238:554/cam/realmonitor?channel=1&subtype=1"
CONF_THRESHOLD = 0.25

MODELS_CONFIG = [
    {"path": "best.pt",    "label": "best.pt",    "color": (0, 200, 255), "out": "live_view_best.jpg"},
    {"path": "best (1).pt","label": "best(1).pt", "color": (0, 255, 0),   "out": "live_view_best1.jpg"},
    {"path": "best (6).pt","label": "best(6).pt", "color": (0, 0, 255),   "out": "live_view_best6.jpg"},
]

class VideoStream:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

def draw_beautiful_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    length = 20
    t = 5
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, t)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, t)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, t)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, t)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    label_size = cv2.getTextSize(label, font, font_scale, 2)[0]
    label_y = max(y1, label_size[1] + 15)
    cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), (x1 + label_size[0] + 10, label_y), color, -1)
    cv2.putText(frame, label, (x1 + 5, label_y - 6), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

def run_model_inference(frame, model, cfg):
    """Run one model on a frame copy and save output."""
    out = frame.copy()
    h, w = out.shape[:2]

    results = model(out, conf=CONF_THRESHOLD, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            name = model.names[int(box.cls[0])]
            if 'gun' in name.lower() or 'weapon' in name.lower():
                detections.append((box.xyxy[0], name, float(box.conf[0])))

    color = cfg["color"]

    # Top header bar
    header = out.copy()
    cv2.rectangle(header, (0, 0), (w, 55), (20, 20, 40), -1)
    cv2.addWeighted(header, 0.8, out, 0.2, 0, out)
    cv2.putText(out, f"Model: {cfg['label']}", (12, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)

    if detections:
        # Alert overlay
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 55), (w, 115), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
        cv2.putText(out, "!!! WEAPON DETECTED !!!", (12, 96),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        for box_coords, name, conf in detections:
            draw_beautiful_box(out, box_coords, f"{name.upper()} {conf:.2f}", color)
        
        # Log detection
        with open("detection_log.txt", "a") as log:
            log.write(f"{datetime.now().strftime('%H:%M:%S')} [{cfg['label']}] DETECTED ({detections[0][2]:.2f})\n")

    # Bottom status bar
    cv2.rectangle(out, (0, h - 30), (w, h), (20, 20, 40), -1)
    status = f"Conf: {CONF_THRESHOLD} | Detections: {len(detections)}"
    cv2.putText(out, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imwrite(cfg["out"], out)

def main():
    # Load all models
    models = []
    print("\n=== Loading All Models ===")
    for cfg in MODELS_CONFIG:
        if os.path.exists(cfg["path"]):
            try:
                m = YOLO(cfg["path"])
                models.append((m, cfg))
                print(f"[OK] {cfg['path']} -> Classes: {list(m.names.values())}")
            except Exception as e:
                print(f"[FAIL] {cfg['path']}: {e}")
        else:
            print(f"[MISS] {cfg['path']} not found, skipping.")

    if not models:
        print("No models loaded. Exiting.")
        return

    # Connect to camera
    print(f"\n=== Connecting to Camera 13 ===\n{RTSP_SOURCE}")
    vstream = VideoStream(RTSP_SOURCE).start()

    # Wait for first frame
    start = time.time()
    while not vstream.read()[0] and time.time() - start < 15:
        time.sleep(0.1)

    if not vstream.read()[0]:
        print("ERROR: Could not connect to camera.")
        vstream.stop()
        return

    print(f"\n=== {len(models)} Model(s) Running on Camera 13 ===")
    print("Open live_view.html in your browser to see results.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            ret, frame = vstream.read()
            if not ret or frame is None:
                continue

            # Run all models in parallel threads
            threads = []
            for model, cfg in models:
                t = threading.Thread(target=run_model_inference, args=(frame, model, cfg))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        vstream.stop()
        print("Done.")

if __name__ == "__main__":
    main()
