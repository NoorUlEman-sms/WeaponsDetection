# SMS Vision AI: System Architecture & Project Overview

This document provides a detailed breakdown of the weapon detection project, how the AI models work together, and how the underlying logic optimizes for both performance and accuracy.

---

## 1. The Core Architecture: The Two-Stage Pipeline
Detecting small weapons on low-resolution CCTV feeds is incredibly difficult. If you run a high-resolution model on the entire video frame, the system becomes too slow (low FPS). If you lower the resolution, the AI misses the weapons entirely.

To solve this, the project uses a highly optimized **Two-Stage Detection Pipeline** (found in `app_roi.py`):

### Stage 1: Person Detection (Fast & Broad)
*   **Model:** `yolov8n.pt` (Nano model)
*   **How it works:** The system first scans the entire video frame at a low resolution (`imgsz=256`). The only goal here is to quickly find the coordinates of all humans in the scene. Because the nano model is extremely fast, this has minimal impact on FPS.

### Stage 2: Weapon Detection (Precise & Focused)
*   **Model:** `bestcctv1.pt` (Your custom trained CCTV weapon model)
*   **How it works:** Instead of scanning the whole frame for a gun, the system crops a "Region of Interest" (ROI) around each person detected in Stage 1. It expands this box by 20% to account for extended arms. It then feeds *only* these small cropped images into the weapon model at a high resolution (`imgsz=416`). 
*   **Why it's brilliant:** The weapon model gets to look at a high-definition, zoomed-in image of the person's hands/body without the computational cost of scanning the pavement, sky, or walls. 

---

## 2. Object Tracking (`tracker.py`)
Video feeds flicker. Sometimes a person turns around, or the camera drops a frame, causing the bounding box to disappear for a split second.

The project uses an **IoU (Intersection over Union) Tracker**:
*   It assigns a unique identity to every person in the frame.
*   It tracks their movement across frames by comparing where their bounding box is now vs. where it was a fraction of a second ago.
*   **Ghost Tracks:** If a person's detection drops for a few frames, the tracker remembers them for up to 8 frames (`max_age=8`), keeping the system stable and preventing alerts from resetting.

---

## 3. False-Positive Filtering (The Temporal Guard)
An AI might mistake a shiny watch or a phone for a gun for a single frame. To prevent the system from triggering the alarm constantly, the project uses several smart filters:

*   **Temporal Guard (3 Frames):** The system will *never* trigger an alarm on a single frame. The tracker demands that the weapon model detects a gun on the *same tracked person* for **3 consecutive frames**. If it's a glitch, it won't happen 3 times in a row.
*   **Size Filtering (`MIN_SIZE_RATIO = 1%`):** If the detected "weapon" is less than 1% of the person's total body size, it ignores it. (It's too small to be a rifle/handgun).
*   **Anatomy Filtering:** The system checks the vertical position of the weapon. If the weapon box is on the very top of their head (< 10%) or by their shoes (> 90%), it discards it, as people carry weapons in their midsection/hands.

---

## 4. Threat Handling & Evidence Recording
When a confirmed armed threat passes all the filters:
1.  **UI Alert:** The dashboard instantly flashes red, showing "🚨 WEAPON DETECTED" and highlighting the person in a glowing red bounding box.
2.  **Snapshot:** It takes a high-res screenshot of the frame, draws the boxes and a warning banner, and saves it to the `cctv1_images/` folder. It has a 3-second cooldown to avoid spamming your hard drive.
3.  **Video Recording:** It dynamically spins up a video writer (`cv2.VideoWriter`) and begins recording an `.avi` video clip to `cctv_recordings/`. 
4.  **Cooldown:** Even after the person leaves the frame, it keeps recording for 3 seconds (`post_record_cooldown`) to ensure it captures the full context of where they went.

---

## 5. The Web Dashboard (UI v2.0)
The entire system is served over a lightweight Flask web server running on `http://127.0.0.1:8082`.

*   **Live Monitor:** Displays the real-time MJPEG stream (meaning you can view it on any device on the network without special plugins). It features a dropdown allowing you to hot-swap between multiple RTSP streams (Camera 13, 14, 15, Gate, etc.) without restarting the server.
*   **Evidence Vault:** A dynamic tab that scans your hard drive for saved threat snapshots and video clips. It allows security personnel to playback video evidence or download/delete snapshots directly from the web browser. 

---

## Summary of Files
*   `app_roi.py`: The main flagship application featuring the two-stage pipeline, tracker, and v2.0 UI.
*   `app.py`: The legacy/older version of the project that runs a single model on the full frame (useful for older hardware, but less accurate).
*   `tracker.py`: The mathematics for tracking people frame-by-frame.
*   `config_detection.json`: A simple file to persist your default settings.
*   `cctv1_images/` & `cctv_recordings/`: Where the system stores its evidence.
