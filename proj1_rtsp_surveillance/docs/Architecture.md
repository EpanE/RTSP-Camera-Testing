# System Architecture — RTSP AI Surveillance System

This document outlines the **software architecture** of the RTSP AI Surveillance System.  
The project is built using a **modular, separation-of-concerns design**, ensuring that video streaming, AI inference, UI interaction, and utilities remain decoupled for maintainability and scalability.

---

## 🔁 High-Level Overview

The system follows a continuous cyclic workflow:

1. **Acquisition**  
   Raw video frames are fetched from the RTSP source.

2. **Processing**  
   Frames are analyzed for human presence and facial regions.

3. **Interaction**  
   User inputs (mouse and keyboard) dynamically modify system state.

4. **Visualization**  
   Processed frames, overlays, and performance statistics are rendered to the display.

---

## 📁 Project Layout

```
rtsp_surveillance/
├── main.py                 # Entry point & main application loop
├── config.py               # Global configuration & state persistence
├── zone_config.json        # Saved restricted zone coordinates
├── modules/
│   ├── __init__.py         # Module interface
│   ├── streamer.py         # RTSP input handler
│   ├── detector.py         # AI inference engine
│   └── privacy.py          # Privacy / face blurring handler
└── utils/
    ├── __init__.py         # Utility interface
    └── fps_counter.py      # Performance monitoring
```

---

## 🧠 File-by-File Architecture

### 1. Root Level

#### `main.py`
**Role:** Orchestrator / Controller  

The core runtime of the application. This file coordinates all system components without embedding heavy logic.

**Responsibilities**
- Initializes `RTSPStreamer`, `PersonDetector`, `PrivacyFilter`, and `FPSCounter`
- Runs the infinite frame-by-frame event loop
- Handles keyboard input:
  - Toggle face blur
  - Save zone configuration
  - Quit application
- Handles mouse input:
  - Drag-and-drop polygon vertices
- Renders:
  - Bounding boxes
  - Restricted zone overlays
  - FPS and alert text
- Binds `config.py` state to mouse callbacks

---

#### `config.py`
**Role:** Source of Truth  

Centralizes all static configuration and persistence logic to avoid hardcoded values.

**Responsibilities**
- Network configuration (RTSP credentials, IP, port)
- AI configuration:
  - Model path (`yolov8n.pt`)
  - Confidence thresholds
  - Input image size
- Persistence:
  - `load_zone()` and `save_zone()` methods
  - Reads and writes `zone_config.json`
- Ensures restricted zone settings persist across restarts

---

### 2. `modules/` Package

#### `modules/__init__.py`
**Role:** Package Interface  

Exposes core classes (`RTSPStreamer`, `PersonDetector`, `PrivacyFilter`) for clean imports in `main.py`.

---

#### `modules/streamer.py`
**Role:** Input Abstraction  

Encapsulates all RTSP stream handling complexity.

**Responsibilities**
- Manages `cv2.VideoCapture`
- Sets buffer size to `1` for low-latency streaming
- Implements heartbeat monitoring
- Automatically reconnects on stream failure
- Exposes a simple `.read_frame()` interface

---

#### `modules/detector.py`
**Role:** Intelligence Layer  

Handles computationally intensive AI and geometric logic.

**Responsibilities**
- Loads YOLOv8 model
- Automatically assigns GPU or CPU
- Implements frame skipping (`SKIP_EVERY_N`) for performance
- Caches detection results to maintain visual continuity
- Executes inference using `model.predict()`
- Converts bounding boxes to center points
- Uses `cv2.pointPolygonTest()` to detect restricted zone intrusion

---

#### `modules/privacy.py`
**Role:** Data Sanitization  

Provides privacy protection independent of person detection.

**Responsibilities**
- Detects faces using OpenCV Haar Cascades
- Applies Gaussian blur to face regions
- Includes boundary safety checks
- Modifies frames in-place for efficiency

---

### 3. `utils/` Package

#### `utils/__init__.py`
**Role:** Utility Interface  

Exposes `FPSCounter` for clean imports.

---

#### `utils/fps_counter.py`
**Role:** Telemetry  

Dedicated performance measurement utility.

**Responsibilities**
- Calculates FPS using time delta
- Renders FPS value onto video frames
- Keeps performance logic out of `main.py`

---

## 🔄 Data Flow

1. **Configuration Load**
   - `main.py` starts
   - Loads settings from `config.py`
   - Restores zone data from `zone_config.json`

2. **Stream Acquisition**
   - `main.py` → `streamer.read_frame()`
   - Returns frame or `None`

3. **Inference Pipeline**
   - `main.py` → `detector.detect(frame)`
   - Detector checks frame skip logic
   - Runs YOLO inference or returns cached results
   - Checks restricted zone intersection

4. **Privacy Processing**
   - If enabled: `privacy_filter.apply_face_blur(frame)`
   - Frame modified in-place

5. **UI & Overlay**
   - Draws bounding boxes and zone overlays
   - Updates FPS via `fps_counter.update()`

6. **Display**
   - Final frame rendered using `cv2.imshow()`

7. **Interaction & Persistence**
   - Mouse drag updates `config.RESTRICTED_ZONE`
   - Pressing `s` triggers `config.save_zone()`

---

## 📌 Architectural Benefits

- Clear separation of responsibilities
- Easy extensibility (alerts, logging, cloud integration)
- Fault-tolerant streaming
- Performance-aware AI inference
- Clean and maintainable codebase
