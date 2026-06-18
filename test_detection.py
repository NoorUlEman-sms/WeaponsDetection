
import cv2
import os
from ultralytics import YOLO

MODEL_PATH = r"c:\Users\CE\Downloads\guns_detection\best (1).onnx" # Using ONNX as I saw it in list_dir
# Or try .pt if it exists in the downloads as per log
PT_PATH = r"C:\Users\CE\Downloads\best.pt"

def test():
    model_path = PT_PATH if os.path.exists(PT_PATH) else MODEL_PATH
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create a dummy frame (black image) with a white rectangle to simulate a "gun" 
    # Or better, try to read from an existing recording if any
    recording_path = r"c:\Users\CE\Downloads\guns_detection\recordings\detection_20260415_181630.mp4"
    if os.path.exists(recording_path):
        cap = cv2.VideoCapture(recording_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to read frame from recording")
            frame = None
    else:
        print("Recording not found, using dummy frame")
        frame = None
        
    if frame is None:
        # Create a dummy frame (1080p)
        import numpy as np
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    results = model(frame, conf=0.25)
    
    for result in results:
        # Drawing using the custom function from detect_live.py to see if it works
        from detect_live import draw_beautiful_box
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            label = f"{name.upper()} {conf:.2f}"
            draw_beautiful_box(frame, box.xyxy[0], label, (0, 0, 255))
            print(f"Detected {name} at {box.xyxy[0]}")

    cv2.imwrite("test_output.jpg", frame)
    print("Saved test_output.jpg")

if __name__ == "__main__":
    test()
