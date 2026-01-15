import time
import cv2
import numpy as np
from collections import deque

import mediapipe as mp
from pynput.keyboard import Controller, Key

"""
===================== HOW TO USE: GESTURE CONTROL FOR POWERPOINT =====================

1) Start the script
   - Run this Python file.
   - A window will open showing your RTSP camera feed and hand landmarks.

2) Open your PowerPoint presentation
   - Launch PowerPoint.
   - Open your .pptx file.

3) Arm the gesture system
   - Click on the Python camera window.
   - Press the key:   g
   - You will see:  "ARMED: True" on the video overlay.

4) Give focus to PowerPoint
   - Click on the PowerPoint window once.
   - REQUIRED so keyboard commands go to PowerPoint.

5) Use gestures (Swipe + Palm + Fist)

   Gesture → Action in PowerPoint
   ---------------------------------------
   Open Palm (5 fingers, hold ~0.6s) → Start Reading View (Windowed)  (Alt -> W -> R)
   Closed Fist (0 fingers, hold ~0.6s) → Black / Unblack screen  (B)
   Swipe Right (quick wave to the right) → Next slide            (Right Arrow)
   Swipe Left  (quick wave to the left)  → Previous slide        (Left Arrow)

   Notes:
   - Swipe works best with an open hand.
   - "Reading View" opens the slide in a simple windowed mode.
   - After a command is triggered, there is a cooldown (~1s).

6) Pause or resume gesture control
   - Press:  g  (toggles ARM / DISARM)

7) Quit
   - Press:  q

===============================================================================
"""

# ===================== RTSP CONFIG =====================
USER = "admin"
PASS = ""               # set if needed
IP   = "192.168.0.27"
PORT = 554
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/ch01"

# ===================== POWERPOINT KEY MAPPINGS =====================
kbd = Controller()

def press_key(key):
    """Press a single key (or Key.*) with pynput."""
    kbd.press(key)
    kbd.release(key)

def enter_reading_view():
    """
    Activates Reading View via Ribbon navigation: Alt -> W -> R.
    Short sleeps ensure PowerPoint registers the key sequences.
    """
    kbd.press(Key.alt)
    time.sleep(0.05)
    kbd.press('w')
    time.sleep(0.05)
    kbd.release('w')
    kbd.press('r')
    kbd.release('r')
    kbd.release(Key.alt)

ACTIONS = {
    "PALM": lambda: enter_reading_view(),   # Start Reading View (Windowed)
    "FIST": lambda: press_key("b"),        # Black screen toggle
    "SWIPE_RIGHT": lambda: press_key(Key.right),  # Next slide
    "SWIPE_LEFT": lambda: press_key(Key.left),    # Previous slide
}

# ===================== STABILITY / COOLDOWN =====================
HOLD_SECONDS = 0.6
COOLDOWN_SECONDS = 1.0

# ===================== SWIPE SETTINGS =====================
SWIPE_WINDOW_SEC = 0.35   # compare movement over last X seconds
SWIPE_MIN_DX_NORM = 0.18  # distance threshold (normalized 0..1)
SWIPE_MAX_DY_NORM = 0.10  # reject swipes with too much vertical movement

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

# ===================== HELPERS =====================
def finger_states(hand_lms, handedness_label):
    """
    Returns dict for finger up/down using simple landmark geometry.
    """
    lm = hand_lms.landmark
    fingers = {}

    # Index / Middle / Ring / Pinky: tip.y < pip.y => up
    fingers["index"]  = lm[8].y  < lm[6].y
    fingers["middle"] = lm[12].y < lm[10].y
    fingers["ring"]   = lm[16].y < lm[14].y
    fingers["pinky"]  = lm[20].y < lm[18].y

    # Thumb x-direction depends on left/right hand
    if handedness_label == "Right":
        fingers["thumb"] = lm[4].x > lm[3].x
    else:
        fingers["thumb"] = lm[4].x < lm[3].x

    return fingers

def detect_swipe(wrist_hist, window_sec, min_dx, max_dy):
    """
    Returns: "SWIPE_RIGHT", "SWIPE_LEFT", or None
    Uses wrist movement over last window_sec.
    """
    if len(wrist_hist) < 5:
        return None

    t_now, x_now, y_now = wrist_hist[-1]
    t_target = t_now - window_sec

    older = None
    for i in range(len(wrist_hist) - 1, -1, -1):
        if wrist_hist[i][0] <= t_target:
            older = wrist_hist[i]
            break
    if older is None:
        older = wrist_hist[0]

    _, x_old, y_old = older
    dx = x_now - x_old
    dy = y_now - y_old

    if abs(dy) > max_dy:
        return None

    if dx >= min_dx:
        return "SWIPE_RIGHT"
    if dx <= -min_dx:
        return "SWIPE_LEFT"
    return None

def draw_status(frame, text, y, ok=True):
    color = (0, 255, 0) if ok else (0, 0, 255)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# ===================== VIDEO CAPTURE =====================
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream. Check URL/creds/path.")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("RTSP Gesture Control (PPT)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP Gesture Control (PPT)", 1100, 650)

# ===================== CONTROL STATE =====================
armed = False
last_trigger_time = 0.0

stable_gesture = None
gesture_start_time = None

# Wrist history buffer: (timestamp, x_norm, y_norm)
wrist_hist = deque(maxlen=60)

# Reconnect tracking
last_frame_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        if time.time() - last_frame_time > 2.0:
            print("Frame read failed. Reconnecting RTSP...")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(RTSP_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            last_frame_time = time.time()
        continue

    last_frame_time = time.time()

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    current_gesture = "NONE"
    handedness_label = "Unknown"
    held_time = 0.0

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_lms = result.multi_hand_landmarks[0]
        handedness_label = result.multi_handedness[0].classification[0].label

        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # Track wrist for swipe detection
        lm = hand_lms.landmark
        wrist_x = lm[0].x
        wrist_y = lm[0].y
        wrist_hist.append((time.time(), wrist_x, wrist_y))

        fingers = finger_states(hand_lms, handedness_label)
        up_count = sum(fingers.values())

        # 1. Check for Swipe FIRST (Dynamic movement overrides static pose)
        swipe = detect_swipe(
            wrist_hist,
            window_sec=SWIPE_WINDOW_SEC,
            min_dx=SWIPE_MIN_DX_NORM,
            max_dy=SWIPE_MAX_DY_NORM
        )

        if swipe:
            current_gesture = swipe
        else:
            # 2. If no swipe, check for Static Poses
            if up_count == 5:
                current_gesture = "PALM"
            elif up_count == 0:
                current_gesture = "FIST"
            else:
                current_gesture = "NONE"

        cv2.putText(frame, f"Hand: {handedness_label} | Fingers up: {up_count}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    now = time.time()

    # ===================== STABILITY LOGIC =====================
    if current_gesture != stable_gesture:
        stable_gesture = current_gesture
        gesture_start_time = now

    held_time = (now - gesture_start_time) if gesture_start_time else 0.0
    ready_to_fire = held_time >= HOLD_SECONDS

    # ===================== COOLDOWN LOGIC =====================
    in_cooldown = (now - last_trigger_time) < COOLDOWN_SECONDS

    # ===================== TRIGGER LOGIC =====================
    if armed and not in_cooldown:
        # Case A: Swipes trigger IMMEDIATELY (Action-based)
        if stable_gesture.startswith("SWIPE") and stable_gesture in ACTIONS:
            ACTIONS[stable_gesture]()
            last_trigger_time = now
            
            # Reset tracking to prevent multiple triggers from same motion
            wrist_hist.clear()
            stable_gesture = "NONE" 
            print(f"Triggered: {stable_gesture} (Swipe)")

        # Case B: Poses require HOLD duration (State-based)
        elif ready_to_fire and stable_gesture in ACTIONS:
            ACTIONS[stable_gesture]()
            last_trigger_time = now
            
            # Reset timer to require re-hold for next trigger
            gesture_start_time = now
            print(f"Triggered: {stable_gesture}")

    # ===================== OVERLAY UI =====================
    draw_status(frame, f"ARMED: {armed} (press 'g' to toggle)", 40, ok=armed)
    cv2.putText(frame, f"Gesture: {stable_gesture} | Hold: {held_time:.2f}s",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if in_cooldown:
        cv2.putText(frame, f"Cooldown: {COOLDOWN_SECONDS - (now - last_trigger_time):.2f}s",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.putText(frame,
                "PPT controls: PALM=Reading View | SWIPE=Next/Prev | FIST=Black",
                (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Keys: [g]=arm/disarm  [q]=quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("RTSP Gesture Control (PPT)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("g"):
        armed = not armed
        print("ARMED =", armed)
        last_trigger_time = time.time()
        gesture_start_time = time.time()
        wrist_hist.clear()

cap.release()
cv2.destroyAllWindows()
hands.close()