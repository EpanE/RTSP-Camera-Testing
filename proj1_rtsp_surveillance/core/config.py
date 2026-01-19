# config.py
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONE_FILE = os.path.join(BASE_DIR, "zone_config.json")

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

# ===================== FALLBACK CONFIG =====================
# If RTSP fails, switch to this camera index (0 is usually the default webcam)
FALLBACK_CAM_INDEX = 0

# ===================== AI CONFIG =====================
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
CONFIDENCE_THRESHOLD = 0.35
DEVICE = 0

# ===================== PERFORMANCE SETTINGS =====================
SKIP_EVERY_N = 2
INFER_IMGSZ = 512
USE_HALF = True

# ===================== ALERT ZONE =====================
DEFAULT_ZONE = [
    [120, 120],
    [520, 120],
    [640, 420],
    [140, 450],
]

def load_zone():
    if os.path.exists(ZONE_FILE):
        try:
            with open(ZONE_FILE, "r") as f:
                data = json.load(f)
                print(f"✅ Loaded zone from {ZONE_FILE}")
                return data
        except Exception as e:
            print(f"⚠️ Error loading zone file: {e}. Using default.")
    return DEFAULT_ZONE

def save_zone(zone_points):
    try:
        with open(ZONE_FILE, "w") as f:
            json.dump(zone_points, f)
        print(f"💾 Zone saved to {ZONE_FILE}")
    except Exception as e:
        print(f"⚠️ Error saving zone file: {e}")

RESTRICTED_ZONE = load_zone()

# ===================== UI =====================
WINDOW_NAME = "RTSP-AI"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 650
