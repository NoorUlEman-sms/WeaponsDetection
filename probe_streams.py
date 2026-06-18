
import cv2
import time
import os

def probe_streams():
    base = "rtsp://admin:Sms786%40sms@192.168.51.238:554/cam/realmonitor?channel="
    # Try common Hikvision channels
    targets = ["1&subtype=0", "1&subtype=1"]
    
    for t in targets:
        url = base + t
        print(f"Probing {url}...")
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"SUCCESS on {t}! Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cv2.imwrite(f"probe_{t}.jpg", frame)
            else:
                print(f"FAILED to read frame from {t}.")
            cap.release()
        else:
            print(f"FAILED to open {t}.")
        time.sleep(0.5)

if __name__ == "__main__":
    probe_streams()
