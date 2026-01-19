import sys
import os
import time
import cv2
import numpy as np

# ===================== PATH SETUP =====================
# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the project root)
project_root = os.path.dirname(current_dir)
# Add the project root to Python's system path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import *
from modules.hand_tracker import HandTracker
from modules.canvas_manager import CanvasManager
from utils.fps import FPSCounter
from utils.capture_thread import VideoCaptureThreaded  # Import the threaded class

def put_hud(img, lines, start_y=40):
    """Helper to put multiple lines of text on screen."""
    y = start_y
    for i, line in enumerate(lines):
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        y += 35

def main():
    # ===================== INITIALIZATION =====================
    
    # Force TCP transport globally (redundant safety, thread also sets it)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    
    # Debug: Print the URL to console
    print(f"--- [DEBUG] Connecting to: {RTSP_URL} ---")

    # Initialize the threaded video capture
    video_thread = VideoCaptureThreaded(RTSP_URL)
    video_thread.start()
    
    # Give the thread a moment to connect before we start processing
    print("Waiting for stream to initialize...")
    time.sleep(2.0)

    cv2.namedWindow("RTSP AirDraw (Overlay)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RTSP AirDraw (Overlay)", 1100, 650)

    tracker = HandTracker(MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE)
    canvas_mgr = CanvasManager()
    fps_counter = FPSCounter()
    fps_counter.start()

    # ===================== STATE =====================
    draw_enabled = False
    last_toggle_time = 0.0
    palm_start_time = None
    prev_draw_pt = None
    smoothed_pt = None

    try:
        while True:
            # ===================== VIDEO CAPTURE (THREADED) =====================
            # This call is now non-blocking. It returns the latest frame instantly
            # or None if no frame is ready (e.g., during reconnection).
            frame = video_thread.read()

            # If frame is None, the thread is likely connecting/reconnecting.
            # We just skip this loop iteration to keep the UI alive.
            if frame is None:
                cv2.waitKey(1)
                continue

            # Update FPS counter
            fps_counter.update()

            h, w = frame.shape[:2]
            canvas_mgr.ensure_size(h, w)

            # ===================== HAND TRACKING =====================
            result = tracker.process(frame)
            gesture = "NONE"
            now = time.time()

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_lms = result.multi_hand_landmarks[0]
                handedness_label = result.multi_handedness[0].classification[0].label

                tracker.draw_landmarks(frame, hand_lms)

                fingers = tracker.get_finger_states(hand_lms, handedness_label)
                up_count = sum(fingers.values())

                # ---- Palm toggle logic (hold) ----
                palm = tracker.is_palm(fingers)
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
                if draw_enabled and not palm:
                    # Use index finger tip landmark 8
                    tip = hand_lms.landmark[8]
                    x = int(tip.x * w)
                    y = int(tip.y * h)

                    # Smooth fingertip
                    if smoothed_pt is None:
                        smoothed_pt = (x, y)
                    else:
                        sx, sy = smoothed_pt
                        sx = int((1 - SMOOTH_ALPHA) * sx + SMOOTH_ALPHA * x)
                        sy = int((1 - SMOOTH_ALPHA) * sy + SMOOTH_ALPHA * y)
                        smoothed_pt = (sx, sy)

                    if fingers["index"]:
                        if prev_draw_pt is None:
                            prev_draw_pt = smoothed_pt
                        canvas_mgr.draw_line(prev_draw_pt, smoothed_pt, DRAW_COLOR, BRUSH_THICKNESS)
                        prev_draw_pt = smoothed_pt
                        gesture = "DRAWING"
                    else:
                        prev_draw_pt = None
                        gesture = "HOVER"

                # Add quick info text to frame
                cv2.putText(frame, f"Hand: {handedness_label} | Fingers up: {up_count}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                prev_draw_pt = None
                smoothed_pt = None
                palm_start_time = None

            # ===================== COMPOSITING & UI =====================
            # Overlay canvas on frame
            out = canvas_mgr.get_overlay(frame)

            # HUD
            hud_lines = [
                f"DRAW: {'ON' if draw_enabled else 'OFF'} | Gesture: {gesture}",
                f"FPS: {fps_counter.get_fps():.2f}",
                "Keys: [c]=clear  [q]=quit"
            ]
            put_hud(out, hud_lines)

            cv2.imshow("RTSP AirDraw (Overlay)", out)

            # ===================== INPUT HANDLING =====================
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                canvas_mgr.clear()
                prev_draw_pt = None
                smoothed_pt = None

    finally:
        print("Stopping capture thread and closing window...")
        video_thread.stop()  # Safely stop the background thread
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()