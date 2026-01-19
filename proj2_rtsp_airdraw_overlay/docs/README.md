# RTSP AirDraw Overlay

A Python application that connects to an RTSP camera stream and allows you to draw in the air using hand gestures (via MediaPipe).

---

## Features

- **RTSP Streaming**: Connects to IP cameras.
- **Air Drawing**: Draw on the video feed using your index finger.
- **Gesture Control**:
  - Index Finger: Draw.
  - Open Palm (Hold): Toggle drawing On/Off.
- **FPS Counter**: Displays real-time frames per second.
- **Robust Reconnection**: Automatically attempts to reconnect if the RTSP stream drops.

---

## Installation

Ensure you have **Python 3.8+** installed.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit `core/config.py` to change:
- RTSP defaults
- Brush size
- Gesture sensitivity

Or supply RTSP settings via environment variables:

```bash
export RTSP_USER="admin"
export RTSP_PASS="password"
export RTSP_IP="192.168.0.27"
export RTSP_PORT="554"
export RTSP_PATH="/stream1"
```

You can also set `RTSP_URL` directly to override all fields.

---

## Usage

Run the main script:

```bash
python core/main.py
```

Or run as a module from the root directory:

```bash
python -m core.main
```

---

## Controls

- **Index Finger**: Draw lines
- **Palm Hold (0.6s)**: Toggle drawing mode
- **c**: Clear canvas
- **q**: Quit application

---

## Documentation

Detailed system design is available in:

- `docs/architecture.md`
