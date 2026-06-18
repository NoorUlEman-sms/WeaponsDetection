
import cv2
import os
import time
import threading
from ultralytics import YOLO

# Configuration
MODEL_PATHS = ["best (6).pt", "best (1).pt", "best.onnx", "best.pt"]
THRESHOLD = 0.25

# Camera Stream Configuration
# Single Camera 14 as requested
SOURCE = "rtsp://:@192.168.51.239:554/cam/realmonitor?channel=1&subtype=0"

def start_single_stream():
    """Processes the single camera stream."""
    model_path = next((p for p in MODEL_PATHS if os.path.exists(p)), "best.pt")
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(SOURCE)
    
    if not cap.isOpened():
        print(f"Connection Failed: {SOURCE}")
    
    from detect_live import draw_beautiful_box
    
    while True:
        if not cap.isOpened():
            time.sleep(5)
            cap = cv2.VideoCapture(SOURCE)
            continue

        for _ in range(3): cap.grab()
        ret, frame = cap.read()
        
        if ret:
            results = model(frame, conf=THRESHOLD, verbose=False)
            frame_width = frame.shape[1]
            detections_found = False
            
            for result in results:
                for box in result.boxes:
                    name = model.names[int(box.cls[0])]
                    if 'gun' in name.lower():
                        conf = float(box.conf[0])
                        draw_beautiful_box(frame, box.xyxy[0], f"{name.upper()} {conf:.2f}", (0, 0, 255))
                        detections_found = True
            
            if detections_found:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, 70), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(frame, "!!! WEAPON DETECTED !!!", (int(frame_width/2)-200, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "MONITORING: SECURE", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save to live_view.jpg for the dashboard
            cv2.imwrite("live_view_tmp.jpg", frame)
            if os.path.exists("live_view_tmp.jpg"):
                try:
                    if os.path.exists("live_view.jpg"):
                        os.remove("live_view.jpg")
                    os.rename("live_view_tmp.jpg", "live_view.jpg")
                except:
                    pass
                    
        time.sleep(1)

if __name__ == "__main__":
    start_single_stream()
