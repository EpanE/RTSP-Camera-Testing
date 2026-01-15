import time
import cv2
import numpy as np

import mediapipe as mp

"""
===================== PROJECT: RTSP AIR-DRAW + VIRTUAL HUD BUTTONS (LEFT) =====================

What you can do:
1) Toggle drawing ON/OFF using OPEN PALM (hold ~0.6s)
2) Draw using INDEX FINGER TIP (index finger up)
3) Press virtual buttons (fixed on LEFT side) using your index fingertip:
   - CLEAR  : clears drawing canvas
   - SAVE   : saves a screenshot (camera + drawing overlay) to ./captures/
   - DRAW   : toggles drawing ON/OFF
   - QUIT   : exits program

How to "press" a button:
- Put your index fingertip inside the button box and HOLD briefly (~0.35s).
- The button will fill as a progress indicator.
- Cooldown prevents double-triggering.

Keys:
- q : quit
- c : clear canvas (same as CLEAR button)

===============================================================================
"""

# ===================== RTSP CONFIG =====================
USER = "admin"
PASS = ""               # set if needed
IP   = "192.168.0.27"
PORT = 554
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/Streaming/Channels/101"

# ===================== DRAWING CONFIG =====================
BRUSH_THICKNESS = 6

# Smooth fingertip to reduce jitter (higher = smoother, more lag)
SMOOTH_ALPHA = 0.35  # 0..1

# Palm toggle safety
PALM_TOGGLE_HOLD_SEC = 0.6
PALM_TOGGLE_COOLDOWN_SEC = 1.0

# ===================== BUTTON CONFIG =====================
BUTTON_DWELL_SEC = 0.35     # time fingertip must stay inside to click
BUTTON_COOLDOWN_SEC = 0.60  # cooldown after any button press

# Layout (left side)
BTN_W = 220
BTN_H = 60
BTN_GAP = 14
BTN_X = 20
BTN_Y0 = 170

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
    if handedness_label == "Right":
        fingers["thumb"] = lm[4].x > lm[3].x
    else:
        fingers["thumb"] = lm[4].x < lm[3].x
    return fingers

def is_palm(fingers):
    return sum(fingers.values()) == 5

def put_text(img, text, x, y, scale=0.8, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)

def clamp_pt(x, y, w, h):
    return max(0, min(x, w - 1)), max(0, min(y, h - 1))

def overlay_canvas(frame, canvas):
    """Overlay drawn canvas (white strokes) onto frame without hiding video."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(bg, fg)

def ensure_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

# ===================== BUTTONS =====================
def build_buttons():
    # Each button: name, rect (x1,y1,x2,y2)
    names = ["CLEAR", "SAVE", "DRAW", "QUIT"]
    btns = []
    y = BTN_Y0
    for name in names:
        x1, y1 = BTN_X, y
        x2, y2 = BTN_X + BTN_W, y + BTN_H
        btns.append({"name": name, "rect": (x1, y1, x2, y2)})
        y += BTN_H + BTN_GAP
    return btns

def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return (x1 <= px <= x2) and (y1 <= py <= y2)

def draw_button(panel, btn, progress=0.0, active=False):
    x1, y1, x2, y2 = btn["rect"]
    # background box
    border_col = (0, 255, 0) if active else (180, 180, 180)
    cv2.rectangle(panel, (x1, y1), (x2, y2), border_col, 2)

    # progress fill
    if progress > 0:
        fill_w = int((x2 - x1) * max(0.0, min(progress, 1.0)))
        cv2.rectangle(panel, (x1, y1), (x1 + fill_w, y2), (0, 255, 0), -1)

    # label (draw on top)
    label = btn["name"]
    tx = x1 + 18
    ty = y1 + 40
    cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0) if progress > 0 else (255, 255, 255), 2)

# ===================== VIDEO CAPTURE =====================
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream. Check URL/creds/path.")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("RTSP AirDraw + Buttons (Left)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP AirDraw + Buttons (Left)", 1100, 650)

# ===================== STATE =====================
draw_enabled = False

# Palm toggle timers
last_palm_toggle_time = 0.0
palm_start_time = None

# Drawing points
prev_draw_pt = None
smoothed_pt = None

# Button click state
buttons = build_buttons()
hover_btn = None
hover_start = None
last_button_time = 0.0

# Canvas (created once we know frame size)
canvas = None

# Save path
CAPTURE_DIR = "captures"
ensure_dir(CAPTURE_DIR)

# reconnect tracking
last_frame_time = time.time()

while True:
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)   # horizontal mirror (left-right)
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

    now = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_text = "NONE"
    tip_px = None  # index fingertip pixel coordinate

    # Reset drawing linkage when no hand
    if not (result.multi_hand_landmarks and result.multi_handedness):
        prev_draw_pt = None
        smoothed_pt = None
        palm_start_time = None
        hover_btn = None
        hover_start = None

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_lms = result.multi_hand_landmarks[0]
        handedness_label = result.multi_handedness[0].classification[0].label

        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        fingers = finger_states(hand_lms, handedness_label)
        palm = is_palm(fingers)

        # Index fingertip pixel
        tip = hand_lms.landmark[8]
        x = int(tip.x * w)
        y = int(tip.y * h)
        x, y = clamp_pt(x, y, w, h)
        tip_px = (x, y)

        # ----------------- PALM TOGGLE (hold) -----------------
        in_palm_cd = (now - last_palm_toggle_time) < PALM_TOGGLE_COOLDOWN_SEC
        if palm and not in_palm_cd:
            if palm_start_time is None:
                palm_start_time = now
            held = now - palm_start_time
            gesture_text = f"PALM hold {held:.2f}s"
            if held >= PALM_TOGGLE_HOLD_SEC:
                draw_enabled = not draw_enabled
                last_palm_toggle_time = now
                palm_start_time = None
                prev_draw_pt = None
                smoothed_pt = None
        else:
            palm_start_time = None

        # ----------------- BUTTON HOVER/CLICK -----------------
        # Only allow button clicks when:
        # - index finger is up (intent)
        # - not palm (avoid toggle gesture scribble/click)
        in_btn_cd = (now - last_button_time) < BUTTON_COOLDOWN_SEC

        hovering = None
        if tip_px and fingers["index"] and not palm and not in_btn_cd:
            for btn in buttons:
                if point_in_rect(tip_px[0], tip_px[1], btn["rect"]):
                    hovering = btn["name"]
                    break

        if hovering != hover_btn:
            hover_btn = hovering
            hover_start = now if hovering else None

        # Trigger click if dwell long enough
        clicked = None
        if hover_btn and hover_start and not in_btn_cd:
            dwell = now - hover_start
            if dwell >= BUTTON_DWELL_SEC:
                clicked = hover_btn
                last_button_time = now
                hover_btn = None
                hover_start = None

        # Execute button action
        if clicked == "CLEAR":
            canvas[:] = 0
            prev_draw_pt = None
            smoothed_pt = None
        elif clicked == "SAVE":
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_img = overlay_canvas(frame.copy(), canvas)
            path = f"{CAPTURE_DIR}/airdraw_{ts}.png"
            cv2.imwrite(path, out_img)
            print("Saved:", path)
        elif clicked == "DRAW":
            draw_enabled = not draw_enabled
            prev_draw_pt = None
            smoothed_pt = None
        elif clicked == "QUIT":
            break

        # ----------------- DRAWING -----------------
        # Only draw when enabled, NOT palm, and index finger is up.
        if draw_enabled and not palm and fingers["index"]:
            # Smooth fingertip
            if smoothed_pt is None:
                smoothed_pt = tip_px
            else:
                sx, sy = smoothed_pt
                sx = int((1 - SMOOTH_ALPHA) * sx + SMOOTH_ALPHA * tip_px[0])
                sy = int((1 - SMOOTH_ALPHA) * sy + SMOOTH_ALPHA * tip_px[1])
                smoothed_pt = (sx, sy)

            if prev_draw_pt is None:
                prev_draw_pt = smoothed_pt
            cv2.line(canvas, prev_draw_pt, smoothed_pt, (255, 255, 255), BRUSH_THICKNESS)
            prev_draw_pt = smoothed_pt
            gesture_text = "DRAWING"
        else:
            prev_draw_pt = None
            smoothed_pt = None if palm else smoothed_pt
            if gesture_text == "NONE":
                gesture_text = "HOVER"

    # ----------------- COMPOSE OUTPUT -----------------
    out = overlay_canvas(frame, canvas)

    # HUD header
    put_text(out, f"DRAW: {'ON' if draw_enabled else 'OFF'} | {gesture_text}", 20, 40, scale=0.85)

    # Draw fingertip marker
    if tip_px:
        cv2.circle(out, tip_px, 8, (0, 255, 0), -1)

    # ----------------- BUTTON PANEL -----------------
    # We draw buttons directly on the output frame.
    # Compute progress fill if hovering
    for btn in buttons:
        prog = 0.0
        active = False
        if hover_btn == btn["name"] and hover_start:
            prog = min(1.0, (now - hover_start) / BUTTON_DWELL_SEC)
            active = True
        draw_button(out, btn, progress=prog, active=active)

    # Footer hints
    put_text(out, "Palm hold: toggle draw | Index into button: click | Keys: [c]=clear [q]=quit", 20, h - 20, scale=0.65, thickness=2)

    cv2.imshow("RTSP AirDraw + Buttons (Left)", out)

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
