import cv2
import os
from ultralytics import YOLO

def expand_bbox(box, frame_w, frame_h, factor=0.2):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1_new = max(0, x1 - factor * w)
    y1_new = max(0, y1 - factor * h)
    x2_new = min(frame_w, x2 + factor * w)
    y2_new = min(frame_h, y2 + factor * h)
    return [int(x1_new), int(y1_new), int(x2_new), int(y2_new)]

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

def main():
    print("Loading models...")
    person_model = YOLO("yolov8n.pt")
    # Latest updated weapon model
    weapon_model = YOLO("models/bestcctv1.pt")
    print("Models loaded successfully.")
    
    images = ["cam13_live_test.jpg", "debug_cam13.jpg", "test_threat_frame.jpg", "test_threat_frame_2.jpg"]
    
    for img_name in images:
        if not os.path.exists(img_name):
            print(f"File {img_name} not found, skipping.")
            continue
            
        print(f"\n--- Processing {img_name} ---")
        # 1. Direct Single-Stage Detection
        img_direct = cv2.imread(img_name)
        h, w = img_direct.shape[:2]
        
        print("Running direct single-stage detection...")
        results_direct = weapon_model(img_direct, conf=0.15, imgsz=640, verbose=False)
        direct_count = 0
        for r in results_direct:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = weapon_model.names[cls]
                conf = float(box.conf[0])
                draw_beautiful_box(img_direct, box.xyxy[0], f"DIRECT {name.upper()} {conf:.2f}", (0, 165, 255))
                print(f"  [Direct] Detected {name} with conf {conf:.2f} at {box.xyxy[0].tolist()}")
                direct_count += 1
                
        # Draw status watermark
        if direct_count > 0:
            cv2.rectangle(img_direct, (0, 0), (w, 60), (0, 0, 200), -1)
            cv2.putText(img_direct, "!!! DIRECT WEAPON DETECTED !!!", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(img_direct, "DIRECT: SECURE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
        out_direct_name = f"result_direct_{img_name}"
        cv2.imwrite(out_direct_name, img_direct)
        print(f"Saved direct detection result to: {out_direct_name}")
        
        # 2. Two-Stage ROI Detection (using imgsz=640 for person detection)
        img_roi = cv2.imread(img_name)
        print("Running Two-Stage ROI pipeline...")
        p_results = person_model(img_roi, classes=[0], conf=0.25, imgsz=640, verbose=False)
        roi_count = 0
        
        for idx, p_box in enumerate(p_results[0].boxes):
            px1, py1, px2, py2 = p_box.xyxy[0].tolist()
            p_conf = float(p_box.conf[0])
            draw_beautiful_box(img_roi, [px1, py1, px2, py2], f"Person {p_conf:.2f}", (150, 150, 150))
            
            roi_box = expand_bbox([px1, py1, px2, py2], w, h, factor=0.2)
            rx1, ry1, rx2, ry2 = roi_box
            roi_crop = img_roi[ry1:ry2, rx1:rx2]
            
            if roi_crop.size > 0:
                w_results = weapon_model(roi_crop, conf=0.15, imgsz=640, verbose=False)
                for w_box in w_results[0].boxes:
                    wx1, wy1, wx2, wy2 = w_box.xyxy[0].tolist()
                    w_conf = float(w_box.conf[0])
                    # Absolute coords
                    fwx1, fwy1, fwx2, fwy2 = rx1+wx1, ry1+wy1, rx1+wx2, ry1+wy2
                    draw_beautiful_box(img_roi, [fwx1, fwy1, fwx2, fwy2], f"WEAPON {w_conf:.2f}", (0, 0, 255))
                    print(f"  [Two-Stage] Detected weapon on Person {idx} with conf {w_conf:.2f} at [{fwx1:.1f}, {fwy1:.1f}, {fwx2:.1f}, {fwy2:.1f}]")
                    roi_count += 1
                    
        if roi_count > 0:
            cv2.rectangle(img_roi, (0, 0), (w, 60), (0, 0, 200), -1)
            cv2.putText(img_roi, "!!! ARMED THREAT DETECTED (ROI) !!!", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(img_roi, "ROI PIPELINE: SECURE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
        out_roi_name = f"result_roi_{img_name}"
        cv2.imwrite(out_roi_name, img_roi)
        print(f"Saved ROI pipeline result to: {out_roi_name}")

if __name__ == "__main__":
    main()
