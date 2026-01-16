# config.py

# ===================== RTSP CONFIG =====================
USER = "admin"
PASS = ""
IP = "192.168.0.27"
PORT = 554
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/Streaming/Channels/101"

# ===================== AI CONFIG =====================
PERSON_MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35
DEVICE = 0  # 0 for GPU, "cpu" for CPU

# ===================== PERFORMANCE SETTINGS =====================
SKIP_EVERY_N = 2
INFER_IMGSZ = 512
USE_HALF = True  # Set False if using CPU

# ===================== ALERT ZONE =====================
# Format: [[x1,y1], [x2,y2], ...]
RESTRICTED_ZONE = [
    [120, 120],
    [520, 120],
    [640, 420],
    [140, 450],
]

# ===================== UI =====================
WINDOW_NAME = "RTSP-AI"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 650