
import cv2
import os
import time
from ultralytics import YOLO
from datetime import datetime

# Configuration
MODEL_PATH = r"C:\Users\CE\Downloads\best.pt"
live_source = "rtsp://admin:hik%401245@192.168.51.237:554/Streaming/Channels/2201"
THRESHOLD = 0.25  # Using user's latest 0.25 threshold

def record_demo():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    model = YOLO(MODEL_PATH)
    
    print(f"📡 Connecting to live stream: {live_source}")
    cap = cv2.VideoCapture(live_source)
    
    if not cap.isOpened():
        print("❌ Error: Could not connect to camera.")
        return

    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10 # Force a lower FPS for smaller file size
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_file = "live_stream_demo.avi"
    writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    
    from detect_live import draw_beautiful_box
    
    print("🎥 Recording and processing 5 seconds of live feed...")
    start_time = time.time()
    
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret: break
        
        # Run detection
        results = model(frame, conf=THRESHOLD, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                name = model.names[int(box.cls[0])]
                if 'gun' in name.lower():
                    detections.append((box.xyxy[0], name, float(box.conf[0])))
        
        if detections:
            # Alert Background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 75), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, "!!! WARNING: WEAPON DETECTED !!!", (int(width/2) - 280, 48), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            
            for box, name, conf in detections:
                draw_beautiful_box(frame, box, f"{name.upper()} {conf:.2f}", (0, 0, 255))
        else:
             cv2.putText(frame, "LIVE: SECURE", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)
        
    cap.release()
    writer.release()
    print(f"✅ Demo recording saved to {out_file}")

if __name__ == "__main__":
    record_demo()
