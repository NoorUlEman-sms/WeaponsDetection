import cv2
import argparse
import os
import time
import sys
import threading
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# ✅ DEFAULT CONFIGURATION
# ============================================================
MODEL_PATHS = [
    "best.pt",
    "best (1).pt",
    "best (6).pt",
    "best.onnx"
]
# Switch to Sub-Stream (subtype=1) for lower latency
DEFAULT_SOURCE = "rtsp://admin:Sms786%40sms@192.168.51.238:554/cam/realmonitor?channel=1&subtype=1"
RECORDINGS_DIR = "recordings"
CONF_THRESHOLD = 0.25
RECONNECT_ATTEMPTS = 5
# ============================================================

class VideoStream:
    """Threaded video stream for low-latency RTSP capture."""
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        # Optimization: Minimize buffer size
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

def get_args():
    parser = argparse.ArgumentParser(description="🔫 Ultra Gun Detection System Pro - Optimized")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO model")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="RTSP URL or Camera Index")
    parser.add_argument("--no-save", action="store_true", help="Disable recording")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    return parser.parse_args()

def draw_beautiful_box(frame, box, label, color):
    """Draws a premium bounding box with rounded corners and glowing effect."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    length = 25
    t = 6
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, t)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y2 if y1 == y2 else y1 + length), color, t)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, t)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, t)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    label_size = cv2.getTextSize(label, font, font_scale, 2)[0]
    label_y = max(y1, label_size[1] + 15)
    cv2.rectangle(frame, (x1, label_y - label_size[1] - 15), (x1 + label_size[0] + 15, label_y), color, -1)
    cv2.putText(frame, label, (x1 + 7, label_y - 10), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    args = get_args()
    save_video = not args.no_save
    if save_video and not os.path.exists(RECORDINGS_DIR):
        os.makedirs(RECORDINGS_DIR)

    # 1. Load Model
    model = None
    model_to_try = [args.model] if args.model else MODEL_PATHS
    print("\n--- Model Initialization ---")
    for path in model_to_try:
        if os.path.exists(path):
            print(f"🔃 Loading: {path}")
            try:
                model = YOLO(path)
                print(f"✅ Success! Using model: {os.path.basename(path)}")
                break
            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")

    if model is None:
        print("🛑 Error: No valid model found.")
        return

    # 2. Setup Threaded Camera
    print(f"\n📡 Connecting to (Optimized): {args.source}")
    vstream = VideoStream(args.source).start()
    
    # Wait for first frame
    start_wait = time.time()
    while not vstream.read()[0] and time.time() - start_wait < 10:
        time.sleep(0.1)

    if not vstream.read()[0]:
        print("🛑 Error: Could not connect to video source.")
        vstream.stop()
        return

    # Get properties
    ret, initial_frame = vstream.read()
    frame_height, frame_width = initial_frame.shape[:2]
    fps = 25
    
    # 3. Setup Video Writer
    writer = None
    if save_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDINGS_DIR, f"opt_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
        print(f"🎥 Recording active: {filename}")

    print("\n🚀 OPTIMIZED GUN DETECTION ONLINE")
    print("💡 Low-latency threading active. Press 'Q' to exit.\n")

    last_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = vstream.read()
            if not ret or frame is None:
                continue

            # Performance monitor
            frame_count += 1
            if time.time() - last_time >= 1.0:
                curr_fps = frame_count / (time.time() - last_time)
                last_time = time.time()
                frame_count = 0

            # 4. Inference
            results = model(frame, conf=args.conf, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    name = model.names[int(box.cls[0])]
                    # Only process detections identified as 'gun' or 'weapon'
                    if 'gun' in name.lower() or 'weapon' in name.lower():
                        detections.append((box.xyxy[0], name, float(box.conf[0])))

            # 5. UI Overlays
            if detections:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, 75), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(frame, "!!! WARNING: WEAPON DETECTED !!!", (int(frame_width/2) - 300, 48), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                for box_coords, name, conf in detections:
                    draw_beautiful_box(frame, box_coords, f"{name.upper()} {conf:.2f}", (0, 0, 255))

            # Save for Web UI (Every frame for smoothness)
            cv2.imwrite("live_view.jpg", frame)

            # Optional Window Display (can be disabled for pure headless)
            cv2.imshow("SMS Vision AI - Optimized", frame)

            if writer:
                writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("🧹 Cleaning up...")
        vstream.stop()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
