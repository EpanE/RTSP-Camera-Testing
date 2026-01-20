import time
import cv2
import numpy as np

import mediapipe as mp

# ===== Windows Volume Control (pycaw) =====
from pycaw.pycaw import AudioUtilities


"""
===================== PROJECT: RTSP PINCH SLIDER → WINDOWS MASTER VOLUME =====================

How it works:
- Pinch (thumb tip + index tip close) = "grab" volume control
- Volume changes ONLY when your index fingertip is inside the slider lane
- While pinching + inside lane, move your hand UP/DOWN to change volume
- Release pinch to stop changing volume

Controls:
- Pinch and hold to control volume (only in slider area)
- m = mute/unmute
- q = quit

Tuning knobs:
- PINCH_ON / PINCH_OFF : pinch sensitivity (raise values if pinch rarely turns ON)
- SLIDER_PAD_X         : widen/narrow the lane acceptance region
- VOL_SMOOTH_ALPHA     : lower = smoother (less jitter), higher = faster response
- MEDIUM_STEP          : 0.05 = 5% steps, None = fully smooth

===============================================================================
"""

# ===================== RTSP CONFIG =====================
USER = "admin"
PASS = ""               # set if needed
IP   = "192.168.0.27"
PORT = 554
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/Streaming/Channels/101"

# ===================== CAMERA FLIP =====================
FLIP_HORIZONTAL = True  # mirror like a webcam

# ===================== PINCH SETTINGS =====================
PINCH_ON  = 0.045   # pinch starts when thumb-index distance < this
PINCH_OFF = 0.070   # pinch releases when distance > this (hysteresis)

# Medium sensitivity and smoothness
VOL_SMOOTH_ALPHA = 0.25   # 0..1 (higher=faster response, lower=smoother)
MEDIUM_STEP = 0.05        # 5% steps (set to None for fully smooth)

# ===================== SLIDER UI REGION =====================
SLIDER_X = 40
SLIDER_Y1 = 120
SLIDER_Y2 = 520
SLIDER_W = 16
SLIDER_PAD_X = 40  # forgiveness zone left/right of slider

# ===================== MEDIAPIPE =====================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ===================== WINDOWS AUDIO SETUP (YOUR PYCAW VERSION) =====================
device = AudioUtilities.GetSpeakers()
endpoint = device.EndpointVolume  # correct for pycaw.utils.AudioDevice

def get_master_volume_scalar():
    return float(endpoint.GetMasterVolumeLevelScalar())  # 0..1

def set_master_volume_scalar(v):
    v = float(max(0.0, min(1.0, v)))
    endpoint.SetMasterVolumeLevelScalar(v, None)

def toggle_mute():
    endpoint.SetMute(not endpoint.GetMute(), None)

# ===================== UTILS =====================
def norm_dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return float(np.sqrt(dx*dx + dy*dy))

def draw_text(img, text, x, y, scale=0.8, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)

def draw_slider(img, vol_scalar, is_grabbed):
    # track
    cv2.rectangle(img, (SLIDER_X, SLIDER_Y1), (SLIDER_X + SLIDER_W, SLIDER_Y2), (200, 200, 200), 2)

    # fill (0 bottom -> 1 top)
    y_fill = int(SLIDER_Y2 - vol_scalar * (SLIDER_Y2 - SLIDER_Y1))
    cv2.rectangle(
        img,
        (SLIDER_X + 2, y_fill),
        (SLIDER_X + SLIDER_W - 2, SLIDER_Y2 - 2),
        (0, 255, 0) if is_grabbed else (0, 180, 255),
        -1
    )
    # knob
    cv2.circle(img, (SLIDER_X + SLIDER_W // 2, y_fill), 10, (0, 255, 0) if is_grabbed else (0, 180, 255), -1)

def y_to_volume(y):
    y = max(SLIDER_Y1, min(SLIDER_Y2, y))
    t = (SLIDER_Y2 - y) / (SLIDER_Y2 - SLIDER_Y1)
    return float(max(0.0, min(1.0, t)))

def quantize_medium(v):
    if MEDIUM_STEP is None:
        return v
    step = float(MEDIUM_STEP)
    return round(v / step) * step

def in_slider_area(x, y):
    # Finger must be inside this lane for volume changes to happen
    return (SLIDER_X - SLIDER_PAD_X <= x <= SLIDER_X + SLIDER_W + SLIDER_PAD_X) and (SLIDER_Y1 <= y <= SLIDER_Y2)

# ===================== VIDEO CAPTURE =====================
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream. Check URL/creds/path.")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("RTSP Pinch Volume", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP Pinch Volume", 1100, 650)

# ===================== STATE =====================
pinching = False
smoothed_vol = get_master_volume_scalar()

# reconnect tracking
last_frame_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        if time.time() - last_frame_time > 2.0:
            print("Frame read failed. Reconnecting RTSP...")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            last_frame_time = time.time()
        continue
    last_frame_time = time.time()

    if FLIP_HORIZONTAL:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "NO HAND"
    lane_ok = False

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_lms = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        lm = hand_lms.landmark

        # Thumb tip (4), Index tip (8)
        tx, ty = lm[4].x, lm[4].y
        ix, iy = lm[8].x, lm[8].y

        # normalized pinch distance (in landmark space)
        d = norm_dist(tx, ty, ix, iy)

        # index tip pixels
        ix_px = int(ix * w)
        iy_px = int(iy * h)

        # Pinch hysteresis
        if not pinching and d < PINCH_ON:
            pinching = True
        elif pinching and d > PINCH_OFF:
            pinching = False

        lane_ok = in_slider_area(ix_px, iy_px)

        # UI markers
        cv2.circle(frame, (int(tx*w), int(ty*h)), 8, (255, 255, 255), -1)  # thumb tip
        cv2.circle(frame, (ix_px, iy_px), 8, (0, 255, 0), -1)              # index tip

        gesture = f"PINCH={'ON' if pinching else 'OFF'}  dist={d:.3f}  lane={'OK' if lane_ok else 'OUT'}"

        # Control volume ONLY when pinching AND finger is inside slider lane
        if pinching and lane_ok:
            target_vol = y_to_volume(iy_px)
            target_vol = quantize_medium(target_vol)  # 5% steps, or smooth if MEDIUM_STEP=None
            smoothed_vol = (1 - VOL_SMOOTH_ALPHA) * smoothed_vol + VOL_SMOOTH_ALPHA * target_vol
            set_master_volume_scalar(smoothed_vol)
        else:
            # reset smoothing reference to current volume so it doesn't jump later
            smoothed_vol = get_master_volume_scalar()

    # Draw slider
    draw_slider(frame, get_master_volume_scalar(), pinching and lane_ok)

    # If pinching, show valid lane box
    if pinching:
        cv2.rectangle(
            frame,
            (SLIDER_X - SLIDER_PAD_X, SLIDER_Y1),
            (SLIDER_X + SLIDER_W + SLIDER_PAD_X, SLIDER_Y2),
            (0, 255, 0) if lane_ok else (0, 0, 255),
            2
        )

    # Status text
    vol_percent = int(get_master_volume_scalar() * 100)
    mute_state = "MUTED" if endpoint.GetMute() else "UNMUTED"

    draw_text(frame, f"Volume: {vol_percent}%  |  {mute_state}", 20, 40, 0.9, 2)
    draw_text(frame, gesture, 20, 80, 0.75, 2)
    draw_text(frame, "Pinch to grab. Volume changes only inside slider lane. Keys: [m]=mute  [q]=quit",
              20, h - 20, 0.65, 2)

    cv2.imshow("RTSP Pinch Volume", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        toggle_mute()

cap.release()
cv2.destroyAllWindows()
hands.close()
