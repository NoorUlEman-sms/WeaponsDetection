
import cv2
import os
import time
from ultralytics import YOLO

# Configuration
MODEL_PATH = r"C:\Users\CE\Downloads\best.pt"
live_source = "rtsp://admin:Sms786%40sms@192.168.51.238:554/cam/realmonitor?channel=1&subtype=0"
THRESHOLD = 0.25

def capture_multiple(count=3):
    if not os.path.exists(MODEL_PATH):
        return
    
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(live_source)
    
    if not cap.isOpened():
        print("❌ Connect Error")
        return

    from detect_live import draw_beautiful_box
    
    for i in range(count):
        print(f"📸 Capturing frame {i+1}...")
        # Clear buffer
        for _ in range(5): cap.grab()
        ret, frame = cap.retrieve()
        
        if ret:
            results = model(frame, conf=THRESHOLD, verbose=False)
            frame_width = frame.shape[1]
            detections_found = False
            
            for result in results:
                for box in result.boxes:
                    name = model.names[int(box.cls[0])]
                    if 'gun' in name.lower() or 'weapon' in name.lower():
                        conf = float(box.conf[0])
                        draw_beautiful_box(frame, box.xyxy[0], f"{name.upper()} {conf:.2f}", (0, 0, 255))
                        detections_found = True
            
            if detections_found:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, 75), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(frame, "!!! WARNING !!!", (30, 50), 1, 2, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"LIVE STREAM {i+1}: SECURE", (30, 50), 1, 2, (0, 255, 0), 2)
                
            cv2.imwrite(f"live_stream_{i+1}.jpg", frame)
            print(f"📁 Saved live_stream_{i+1}.jpg")
        
        time.sleep(1) # Gap between shots
        
    cap.release()

if __name__ == "__main__":
    capture_multiple()
