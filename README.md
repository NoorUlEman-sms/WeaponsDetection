# SMS Vision AI: High-Performance Weapon Detection System

SMS Vision AI is a high-performance, real-time weapon detection system designed for CCTV camera streams. It uses a highly optimized **Two-Stage ROI Detection Pipeline** combined with an **IoU Tracker** and smart **False-Positive Filtering** to detect weapons (rifles, handguns, knives) with high frame rates (FPS) and low alert latency.

---

## Key Features
* 🚀 **Two-Stage Pipeline**: 
  * **Stage 1 (Person Scan)**: Standard YOLOv8n scans the entire frame at a low resolution (`imgsz=256`) to locate people rapidly.
  * **Stage 2 (Weapon Scan)**: Crops a Region of Interest (ROI) around each person, expands it by 20%, and runs a custom CCTV weapon model (`bestcctv1.pt`) at high resolution (`imgsz=640`).
* 📊 **Hungarian Algorithm Tracker**: Keeps track of individuals across frames, handles temporary occlusions (ghost tracks up to 8 frames), and estimates box velocities.
* 🛡️ **False-Positive Guards**: 
  * *Temporal Guard*: Requires consecutive frames of detection before alerting (or instant bypass for $\ge 90\%$ confidence).
  * *Size Guard*: Ignores weapon boxes that are less than 1% of the person's total size.
  * *Anatomy Guard*: Rejects detections on the very top of the head or at the feet.
* 📺 **Interactive Web Dashboard**:
  * Real-time MJPEG live video feed.
  * Hot-swap cameras and AI models dynamically.
  * Timeline of threat alerts and interactive Evidence Vault for review, download, or deletion of snapshots and recordings.

---

## 🛠️ Installation & Setup

### 1. Install Dependencies
Open your command prompt or terminal and run:
```bash
pip install -r requirements.txt
```

### 2. Configure Your Cameras (Crucial)
To prevent private camera passwords and IP addresses from leaking to GitHub, settings are kept in a local `config_detection.json` file which is automatically ignored by Git.

1. Copy the example configuration template:
   ```bash
   copy config_detection.json.example config_detection.json
   ```
2. Open `config_detection.json` in a text editor and update:
   * `cameras`: Add your RTSP camera names and URLs (incorporating usernames/passwords).
   * `default_camera`: The name of the camera you want to launch on startup (e.g. `"Camera 13"`).
   * Adjust other thresholds (`conf_thresh`, `temporal_threshold`) as needed.

---

## 🚀 How to Run the Project

### Running the Live CCTV Server
Run the Flask server with the unbuffered Python flag:
```bash
python -u app_roi.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:8082](http://127.0.0.1:8082)**

### Running the Offline Video Demo
If you want to test the ROI detection pipeline on a local video file (without connecting to active camera streams):
```bash
python app_roi_video.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:8084](http://127.0.0.1:8084)**

---

## 📖 Deep Technical Documentation
For in-depth mathematical formulas, flow diagrams, tracker explanations, and filter definitions, refer to the full documentation:
📄 **[SMS_Vision_AI_Project_Documentation.md](./SMS_Vision_AI_Project_Documentation.md)**
