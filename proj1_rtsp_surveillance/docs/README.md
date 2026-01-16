# Modular RTSP AI Surveillance System

A robust and modular Python application for real-time video surveillance using RTSP streams.  
The system leverages **YOLOv8** for fast and accurate person detection, **OpenCV** for image processing, and provides an interactive interface for restricted zone monitoring and privacy protection.

---

## ✨ Features

- **Person Detection**  
  High-performance human detection using YOLOv8 (Nano) for efficient inference.

- **GPU Acceleration**  
  Automatically uses CUDA-enabled NVIDIA GPUs when available and falls back to CPU without failure.

- **Restricted Zone Intrusion Detection**  
  Detects and highlights individuals entering a defined polygonal restricted area in real time.

- **Interactive Zone Editor**
  - **Drag & Drop Editing** – Adjust zone points directly on the video feed.
  - **Persistent Storage** – Zone configurations are saved and restored automatically.

- **Privacy Protection**  
  Optional real-time face blurring using Haar Cascade classifiers.

- **Performance Monitoring**  
  Live FPS counter for monitoring system performance.

- **Auto-Reconnect**  
  Automatically reconnects to the RTSP stream if the connection drops.

- **Modular Code Architecture**  
  Clean separation of functionality into reusable modules for easy maintenance and scalability.

---

## 📁 Project Structure

```
rtsp_surveillance/
├── main.py
├── config.py
├── zone_config.json
├── modules/
│   ├── streamer.py
│   ├── detector.py
│   └── privacy.py
└── utils/
    └── fps_counter.py
```

---

## 🚀 Installation

```bash
pip install opencv-python numpy torch ultralytics
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
USER = "admin"
PASS = ""
IP = "192.168.0.27"
PORT = 554
```

---

## 📖 Usage

```bash
python main.py
```

---

## 🎹 Keyboard Controls

| Key | Action |
|---|---|
| q | Quit |
| f | Toggle face blur |
| z | Toggle zone overlay |
| s | Save zone |

---

## 📺 Demo Video

[(PROJ1 RTSP CAM SURVEILANCE)](https://youtu.be/DjbIjHJHGqA)
