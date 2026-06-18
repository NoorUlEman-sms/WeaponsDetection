
import cv2
import os
import time
from ultralytics import YOLO

# Configuration
MODEL_PATH = r"C:\Users\CE\Downloads\best.pt"
live_source = "rtsp://admin:Sms786%40sms@192.168.51.238:554/cam/realmonitor?channel=1&subtype=0"
THRESHOLD = 0.5

def capture_result():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    model = YOLO(MODEL_PATH)
    
    print(f"📡 Attempting to connect to live stream: {live_source}")
    cap = cv2.VideoCapture(live_source)
    
    # Wait a bit for connection
    time.sleep(2)
    
    if not cap.isOpened():
        print("❌ Error: Could not open live camera stream. Please check if the camera is online and the URL is correct.")
        return

    # Try to grab a few frames to clear buffer
    for _ in range(5):
        cap.grab()
    
    ret, frame = cap.retrieve()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not retrieve frame from live stream.")
        return

    print("✅ Successfully captured frame from live stream. Running detection...")
    
    # Run detection
    results = model(frame, conf=THRESHOLD, verbose=False)
    
    detections_found = False
    from detect_live import draw_beautiful_box
    frame_width = frame.shape[1]
    
    for result in results:
        for box in result.boxes:
            name = model.names[int(box.cls[0])]
            if 'gun' in name.lower() or 'weapon' in name.lower():
                conf = float(box.conf[0])
                label = f"{name.upper()} {conf:.2f}"
                draw_beautiful_box(frame, box.xyxy[0], label, (0, 0, 255))
                detections_found = True
                print(f"🔥 GUN DETECTED: {conf:.2f}")

    # UI Overlay
    if detections_found:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, 75), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "!!! WARNING: WEAPON DETECTED !!!", (int(frame_width/2) - 280, 48), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # Save a "System Active" watermark
        cv2.putText(frame, "LIVE: NO WEAPONS DETECTED", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print("No guns detected in current frame.")

    cv2.imwrite("live_snapshot_result.jpg", frame)
    print("📁 Result saved to live_snapshot_result.jpg")

if __name__ == "__main__":
    capture_result()
