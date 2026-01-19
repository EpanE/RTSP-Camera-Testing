# System Architecture — RTSP AI Surveillance System

This document describes the architecture of the `proj1_rtsp_surveillance` prototype. The system uses a modular, separation-of-concerns design with dedicated threads for capture and AI inference, keeping the UI and rendering responsive.

---

## 🔁 High-Level Overview

The system follows a continuous cyclic workflow:

1. **Acquisition**  
   Raw video frames are fetched from the RTSP source.

2. **Processing**  
   Frames are analyzed for people, zone occupancy, and optional face blur.

3. **Interaction**  
   User inputs (mouse and keyboard) dynamically modify system state.

4. **Visualization**  
   Processed frames, overlays, and performance statistics are rendered to the display (OpenCV window or CustomTkinter GUI).

---

## 📁 Project Layout

```
proj1_rtsp_surveillance/
├── core/
│   ├── main.py             # OpenCV UI entry point
│   ├── gui_main.py         # CustomTkinter GUI entry point
│   └── config.py           # Global configuration & zone persistence
├── modules/
│   ├── __init__.py         # Module interface
│   ├── producer.py         # Camera capture thread
│   ├── consumer.py         # AI inference thread
│   ├── detector.py         # YOLO inference logic
│   ├── privacy.py          # Face blur helper
│   └── logger.py           # Alert logging + snapshots
├── utils/
│   ├── __init__.py         # Utility interface
│   └── fps_counter.py      # Performance monitoring
├── zone_config.json        # Saved restricted zone coordinates
└── yolov8n.pt              # Person detection model
```

---

## 🧠 File-by-File Architecture

### 1. Core Entry Points

#### `core/main.py`
**Role:** Orchestrator / Controller  

The OpenCV-based runtime. This file coordinates system components and draws overlays without blocking on inference.

**Responsibilities**
- Initializes `CameraProducer`, `AIConsumer`, `PrivacyFilter`, `FPSCounter`, and `AlertLogger`
- Runs the frame-by-frame render loop
- Handles keyboard input:
  - Toggle face blur
  - Toggle zone visibility
  - Save zone configuration
  - Quit application
- Handles mouse input:
  - Drag-and-drop polygon vertices
- Renders:
  - Bounding boxes and IDs
  - Restricted zone overlays
  - Occupancy count, FPS, and alert text

---

#### `core/gui_main.py`
**Role:** GUI Controller  

CustomTkinter-based UI that mirrors `main.py` behavior while providing a control panel, occupancy count, and log preview.

---

#### `core/config.py`
**Role:** Source of Truth  

Centralizes all static configuration and persistence logic to avoid hardcoded values.

**Responsibilities**
- Network configuration (RTSP credentials, IP, port)
- AI configuration:
  - Model path (`yolov8n.pt`)
  - Confidence thresholds
  - Input image size
- Performance settings (frame skipping, half precision)
- Persistence:
  - `load_zone()` and `save_zone()` methods
  - Reads and writes `zone_config.json`
- Ensures restricted zone settings persist across restarts

---

### 2. `modules/` Package

#### `modules/__init__.py`
**Role:** Package Interface  

Exposes core classes (`CameraProducer`, `AIConsumer`, `PrivacyFilter`, `AlertLogger`) for clean imports.

---

#### `modules/producer.py`
**Role:** Input Thread  

Owns camera capture and reconnection logic.

**Responsibilities**
- Manages `cv2.VideoCapture`
- Falls back to the local webcam on RTSP failure
- Continuously captures frames in a background thread
- Provides the latest frame to the UI thread

---

#### `modules/consumer.py`
**Role:** Inference Thread  

Consumes frames from the producer and runs detection on a background thread.

**Responsibilities**
- Hosts the `PersonDetector` instance
- Stores the latest detections for rendering
- Keeps inference work off the UI thread

---

#### `modules/detector.py`
**Role:** Intelligence Layer

Handles computationally intensive AI and geometric logic.

**Responsibilities**
- Loads YOLOv8 model
- Automatically assigns GPU or CPU
- Implements frame skipping (`SKIP_EVERY_N`) for performance
- Caches detection results to maintain visual continuity
- Executes inference using `model.track()` for consistent IDs
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

#### `modules/logger.py`
**Role:** Alerting & Storage  

Logs intrusion events and saves snapshot images for audits.

**Responsibilities**
- Manages `logs/` and `snapshots/` directories
- Appends event rows to a CSV file
- Saves frame captures when alerts trigger

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
   - `core/main.py` or `core/gui_main.py` starts
   - Loads settings from `core/config.py`
   - Restores zone data from `zone_config.json`

2. **Stream Acquisition**
   - `CameraProducer` captures frames
   - UI thread pulls the latest frame from the producer

3. **Inference Pipeline**
   - `AIConsumer` pulls frames from the producer
   - `PersonDetector` checks frame skip logic
   - Runs YOLO inference or returns cached results
   - Checks restricted zone intersection

4. **Privacy Processing**
   - If enabled: `privacy_filter.apply_face_blur(frame)`
   - Frame modified in-place

5. **UI & Overlay**
   - Draws bounding boxes, zone overlays, and occupancy count
   - Updates FPS via `fps_counter.update()`

6. **Display**
   - Final frame rendered using `cv2.imshow()` or CustomTkinter

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
