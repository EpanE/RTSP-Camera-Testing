import time
import cv2
import numpy as np

import mediapipe as mp

# ===================== RTSP CONFIG =====================
USER = "admin"
PASS = ""               # set if needed
IP   = "192.168.0.27"
PORT = 554
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/ch0"

# ===================== DRAWING CONFIG =====================
BRUSH_THICKNESS = 6
ERASER_THICKNESS = 40

# Smooth the fingertip to reduce jitter (higher = smoother, more lag)
SMOOTH_ALPHA = 0.35  # 0..1 (try 0.25 to be smoother)

# Palm toggle safety
TOGGLE_HOLD_SEC = 0.6
TOGGLE_COOLDOWN_SEC = 1.0

# ===================== MEDIAPIPE HANDS =====================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def finger_states(hand_lms, handedness_label):
    """Return dict of finger up/down."""
    lm = hand_lms.landmark
    fingers = {}

    fingers["index"]  = lm[8].y  < lm[6].y
    fingers["middle"] = lm[12].y < lm[10].y
    fingers["ring"]   = lm[16].y < lm[14].y
    fingers["pinky"]  = lm[20].y < lm[18].y

    # Thumb: depends left/right
    if handedness_label == "Right":
        fingers["thumb"] = lm[4].x > lm[3].x
    else:
        fingers["thumb"] = lm[4].x < lm[3].x

    return fingers

def is_palm(fingers):
    return sum(fingers.values()) == 5

def is_fist(fingers):
    return sum(fingers.values()) == 0

def put_hud(img, line1, line2=None, y=40):
    cv2.putText(img, line1, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    if line2:
        cv2.putText(img, line2, (20, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

# ===================== VIDEO CAPTURE =====================
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream. Check URL/creds/path.")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("RTSP AirDraw (Overlay)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP AirDraw (Overlay)", 1100, 650)

# ===================== STATE =====================
draw_enabled = False
last_toggle_time = 0.0
palm_start_time = None
prev_draw_pt = None
smoothed_pt = None

canvas = None  # created once we know frame size

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

    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "NONE"
    now = time.time()

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_lms = result.multi_hand_landmarks[0]
        handedness_label = result.multi_handedness[0].classification[0].label  # Left/Right

        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        fingers = finger_states(hand_lms, handedness_label)
        up_count = sum(fingers.values())

        # ---- Palm toggle logic (hold) ----
        palm = is_palm(fingers)
        in_toggle_cooldown = (now - last_toggle_time) < TOGGLE_COOLDOWN_SEC

        if palm and not in_toggle_cooldown:
            if palm_start_time is None:
                palm_start_time = now
            held = now - palm_start_time
            gesture = f"PALM (hold {held:.2f}s)"
            if held >= TOGGLE_HOLD_SEC:
                draw_enabled = not draw_enabled
                last_toggle_time = now
                palm_start_time = None
                prev_draw_pt = None
                smoothed_pt = None
        else:
            palm_start_time = None

        # ---- Drawing logic (index finger tip) ----
        # Only draw when enabled AND hand is not palm (to avoid scribbling while toggling)
        if draw_enabled and not palm:
            # Use index finger tip landmark 8
            tip = hand_lms.landmark[8]
            x = int(tip.x * w)
            y = int(tip.y * h)

            # Smooth fingertip to reduce jitter
            if smoothed_pt is None:
                smoothed_pt = (x, y)
            else:
                sx, sy = smoothed_pt
                sx = int((1 - SMOOTH_ALPHA) * sx + SMOOTH_ALPHA * x)
                sy = int((1 - SMOOTH_ALPHA) * sy + SMOOTH_ALPHA * y)
                smoothed_pt = (sx, sy)

            # Decide draw vs hover:
            # We draw only if index finger is up (to avoid accidental drawing when hand relaxed)
            if fingers["index"]:
                if prev_draw_pt is None:
                    prev_draw_pt = smoothed_pt
                cv2.line(canvas, prev_draw_pt, smoothed_pt, (0, 0, 255), BRUSH_THICKNESS)
                prev_draw_pt = smoothed_pt
                gesture = "DRAWING"
            else:
                prev_draw_pt = None
                gesture = "HOVER"

        # Show quick info
        cv2.putText(frame, f"Hand: {handedness_label} | Fingers up: {up_count}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        # No hand detected: stop connecting lines
        prev_draw_pt = None
        smoothed_pt = None
        palm_start_time = None

    # ---- Overlay canvas on frame ----
    # Make canvas drawn pixels visible without hiding the camera
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    out = cv2.add(bg, fg)

    # ---- HUD ----
    put_hud(
        out,
        f"DRAW: {'ON' if draw_enabled else 'OFF'} | Gesture: {gesture}",
        "Keys: [c]=clear  [q]=quit"
    )

    cv2.imshow("RTSP AirDraw (Overlay)", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0
        prev_draw_pt = None
        smoothed_pt = None

cap.release()
cv2.destroyAllWindows()
hands.close()
