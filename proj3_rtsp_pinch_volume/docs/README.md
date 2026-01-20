# RTSP Hand Gesture Control System

A modular, high-performance Python application that leverages Computer Vision (MediaPipe) to control Windows system settings via hand gestures.  
The application supports streaming from RTSP IP cameras or a local webcam with automatic fallback.

---

## 🌟 Features

- **Modular Architecture**: Clean separation between `core`, `modules`, and `utils`.
- **Threaded Video Capture**: Non-blocking RTSP capture for smooth UI even with network latency.
- **Multi-Zone Control**:
  - 📢 **Left Lane**: Master Volume Control
  - ☀️ **Right Lane**: Screen Brightness Control
  - 🖱️ **Middle Zone**: Virtual Mouse Cursor (Move & Click)
- **Auto-Fallback**: Automatically switches to local webcam if RTSP stream fails.
- **Environment Configuration**: Uses `.env` file for secure credential handling.
- **Smart Interaction**: Hysteresis-based pinch detection to prevent accidental toggles.

---

## 🛠️ Prerequisites

- **Operating System**: Windows 10 / Windows 11
- **Python**: Version 3.8 or higher
- **Hardware**:
  - RTSP IP Camera **OR**
  - USB Webcam

---

## 📦 Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy pycaw screen-brightness-control pyautogui python-dotenv
```

> **Note:** `pycaw` requires Microsoft Visual C++ Redistributable.

Create a `.env` file:

```env
RTSP_USER=admin
RTSP_PASS=
RTSP_IP=192.168.0.27
RTSP_PORT=554
RTSP_PATH=Streaming/Channels/101
```

---

## 🚀 Usage

```bash
python core/main.py
```

---

## 🖐️ Gesture Guide

| Zone | Gesture | Action |
|-----|--------|--------|
| Left Lane | Pinch + Move | Volume |
| Right Lane | Pinch + Move | Brightness |
| Middle | Move Hand | Mouse Move |
| Middle | Pinch | Click / Drag |
| Keyboard | m | Mute |
| Keyboard | q | Quit |

---

## 📂 Project Structure

```text
project/
├── .env
├── README.md
├── core/
│   ├── config.py
│   └── main.py
├── modules/
│   ├── audio_controller.py
│   ├── brightness_controller.py
│   ├── mouse_controller.py
│   ├── hand_processor.py
│   └── ui_manager.py
└── utils/
    ├── drawing.py
    └── video_thread.py
```

---

## 📝 License

Open-source for educational use.
