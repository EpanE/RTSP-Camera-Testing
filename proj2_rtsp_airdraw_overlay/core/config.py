import os
from dotenv import load_dotenv

# ===================== LOAD ENVIRONMENT =====================
load_dotenv()

# ===================== RTSP CONFIG =====================
USER = os.getenv("RTSP_USER", "")
PASS = os.getenv("RTSP_PASS", "")
IP = os.getenv("RTSP_IP", "127.0.0.1")
PORT = int(os.getenv("RTSP_PORT", "554"))
PATH = os.getenv("RTSP_PATH", "/stream1")

def build_rtsp_url(user, password, ip, port, path):
    """
    Constructs an RTSP URL.
    REMOVED URL ENCODING: We send the password exactly as defined in .env
    to prevent double-encoding issues that cause 401 errors.
    """
    if not path.startswith("/"):
        path = f"/{path}"
    
    auth = ""
    if user or password:
        # Send raw string - do not encode
        auth = f"{user}:{password}@"
        
    return f"rtsp://{auth}{ip}:{port}{path}"

# Construct the final URL
RTSP_URL = os.getenv("RTSP_URL") or build_rtsp_url(USER, PASS, IP, PORT, PATH)

# ===================== DRAWING CONFIG =====================
BRUSH_THICKNESS = 6
ERASER_THICKNESS = 40
DRAW_COLOR = (0, 0, 255) # Default Red in BGR

# Define a palette of colors (BGR format)
PALETTE_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (255, 255, 255) # White
]

# Palette UI settings
PALETTE_RECT_SIZE = 40  # Size of the color squares
PALETTE_START_Y = 20    # Y position to start drawing squares
PALETTE_START_X = 20    # X position

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