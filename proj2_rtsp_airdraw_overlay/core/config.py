import os
from urllib.parse import quote

# ===================== RTSP CONFIG =====================
# Provide values via environment variables or edit defaults below.
# If RTSP_URL is set explicitly, it overrides the individual parts.

USER = os.getenv("RTSP_USER", "")
PASS = os.getenv("RTSP_PASS", "")
IP = os.getenv("RTSP_IP", "127.0.0.1")
PORT = int(os.getenv("RTSP_PORT", "554"))
PATH = os.getenv("RTSP_PATH", "/stream1")

def build_rtsp_url(user, password, ip, port, path):
    if not path.startswith("/"):
        path = f"/{path}"
    auth = ""
    if user or password:
        auth_user = quote(user, safe="")
        auth_pass = quote(password, safe="")
        auth = f"{auth_user}:{auth_pass}@"
    return f"rtsp://{auth}{ip}:{port}{path}"

RTSP_URL = os.getenv("RTSP_URL") or build_rtsp_url(USER, PASS, IP, PORT, PATH)

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
