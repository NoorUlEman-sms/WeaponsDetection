
import cv2
import os
import time
from ultralytics import YOLO

# Paths based on user's environment
MODEL_PATH = r"C:\Users\CE\Downloads\best.pt"
VIDEO_PATH = r"c:\Users\CE\Downloads\guns_detection\recordings\detection_20260415_182052.avi"

def test_run():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
    
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found at {VIDEO_PATH}")
        return
        
    cap = cv2.VideoCapture(VIDEO_PATH)
    count = 0
    detected_frame_saved = False
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    while cap.isOpened() and not detected_frame_saved:
        ret, frame = cap.read()
        if not ret: break
        
        count += 1
        # Skip frames to speed up
        if count % 10 != 0: continue
        
        results = model(frame, conf=0.25, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                conf = float(box.conf[0])
                detections.append((box.xyxy[0], name, conf))
        
        if detections:
            print(f"Frame {count}: Detected {len(detections)} object(s)")
            
            # 1. Alert Background (Draw first)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, 75), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # 2. Alert Text
            cv2.putText(frame, "!!! WARNING: WEAPON DETECTED !!!", (int(frame_width/2) - 280, 48), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 3. Draw Bounding Boxes (Draw last to stay on top)
            from detect_live import draw_beautiful_box
            for box_coords, name, conf in detections:
                label = f"{name.upper()} {conf:.2f}"
                draw_beautiful_box(frame, box_coords, label, (0, 0, 255))
            
            cv2.imwrite("detection_verify_improved.jpg", frame)
            print("Saved detection_verify_improved.jpg")
            detected_frame_saved = True
            break
                
    cap.release()

if __name__ == "__main__":
    test_run()
