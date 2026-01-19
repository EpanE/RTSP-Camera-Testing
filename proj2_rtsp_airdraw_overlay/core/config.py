# ===================== RTSP IPC CONFIG =====================
#USER = "admin"
#PASS = ""
#IP = "192.168.0.27"
#PORT = 554
#RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/Streaming/Channels/101"

# ===================== RTSP C520WS CONFIG =====================

USER = "Robomy"
PASS = "Patent15051%25"
IP = "192.168.0.9"
PORT = 554
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/stream1"

# ===================== DRAWING CONFIG =====================
BRUSH_THICKNESS = 6
ERASER_THICKNESS = 40
DRAW_COLOR = (0, 0, 255) # Red in BGR

# Smooth the fingertip to reduce jitter (higher = smoother, more lag)
SMOOTH_ALPHA = 0.35  # 0..1 (try 0.25 to be smoother)

# Palm toggle safety
TOGGLE_HOLD_SEC = 0.6
TOGGLE_COOLDOWN_SEC = 1.0

# ===================== MEDIAPIPE HANDS =====================
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 1
MODEL_COMPLEXITY = 1