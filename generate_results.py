
import cv2
import os
from ultralytics import YOLO

# Configuration
MODEL_PATH = r"C:\Users\CE\Downloads\best.pt"
VIDEO_PATH = r"c:\Users\CE\Downloads\guns_detection\recordings\detection_20260415_182052.avi"
THRESHOLD = 0.5

def find_best_detection():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    model = YOLO(MODEL_PATH)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        # Fallback to webcam if possible or just exit?
        # User wants me to run everything for them. 
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    max_conf = 0
    best_frame = None
    best_detections = []
    
    print("🎬 Scanning video for high-confidence (0.5+) detections...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        # Sample every 5 frames for speed
        if frame_count % 5 != 0: continue
        
        results = model(frame, conf=THRESHOLD, verbose=False)
        
        current_detections = []
        for result in results:
            for box in result.boxes:
                name = model.names[int(box.cls[0])]
                if 'gun' in name.lower():
                    conf = float(box.conf[0])
                    current_detections.append((box.xyxy[0], name, conf))
                    if conf > max_conf:
                        max_conf = conf
                        best_frame = frame.copy()
                        best_detections = current_detections[:]

    cap.release()
    
    if best_frame is not None:
        print(f"✅ Found detection with confidence: {max_conf:.2f}")
        
        # Apply visualization
        from detect_live import draw_beautiful_box
        frame_width = best_frame.shape[1]
        
        # Alert Overlay
        overlay = best_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, 75), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, best_frame, 0.6, 0, best_frame)
        cv2.putText(best_frame, "!!! WARNING: WEAPON DETECTED !!!", (int(frame_width/2) - 280, 48), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Bboxes
        for box, name, conf in best_detections:
            draw_beautiful_box(best_frame, box, f"{name.upper()} {conf:.2f}", (0, 0, 255))
            
        cv2.imwrite("final_result.jpg", best_frame)
        print("📁 Saved result to final_result.jpg")
    else:
        print("❌ No detections found with 0.5+ confidence in this video.")
        # If no 0.5+ found, maybe save a "System Online" frame anyway?
        

if __name__ == "__main__":
    find_best_detection()
