import os
from urllib.parse import quote
from dotenv import load_dotenv

# ===================== LOAD ENVIRONMENT =====================
# This loads variables from the .env file in the project root
load_dotenv()

# ===================== RTSP CONFIG =====================
# These values look for environment variables first. 
# If not found, they fall back to the defaults defined below.

USER = os.getenv("RTSP_USER", "")
PASS = os.getenv("RTSP_PASS", "")
IP = os.getenv("RTSP_IP", "")
PORT = int(os.getenv("RTSP_PORT", ""))
PATH = os.getenv("RTSP_PATH", "")

def build_rtsp_url(user, password, ip, port, path):
    """
    Constructs an RTSP URL with proper URL encoding for authentication.
    This ensures special characters in passwords (like '@', ':', etc.) don't break the URL.
    """
    if not path.startswith("/"):
        path = f"/{path}"
    
    auth = ""
    if user or password:
        # Encode credentials to handle special characters
        auth_user = quote(user, safe="")
        auth_pass = quote(password, safe="")
        auth = f"{auth_user}:{auth_pass}@"
        
    return f"rtsp://{auth}{ip}:{port}/{path}"

# If RTSP_URL is defined in .env, use it directly. Otherwise build it from parts.
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