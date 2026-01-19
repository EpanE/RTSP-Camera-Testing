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
from utils.capture_thread import VideoCaptureThreaded

# ===================== HELPER FUNCTIONS =====================

def put_hud(img, lines, start_y=40):
    """Helper to put multiple lines of text on screen."""
    y = start_y
    for i, line in enumerate(lines):
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        y += 35

def draw_palette(frame, current_color):
    """Draws the color palette on the screen."""
    for i, color in enumerate(PALETTE_COLORS):
        x = PALETTE_START_X + (i * (PALETTE_RECT_SIZE + 10))
        y = PALETTE_START_Y
        
        # Draw the rectangle
        cv2.rectangle(frame, (x, y), (x + PALETTE_RECT_SIZE, y + PALETTE_RECT_SIZE), color, -1)
        
        # Highlight the currently selected color
        if color == current_color:
            cv2.rectangle(frame, (x-2, y-2), (x + PALETTE_RECT_SIZE+2, y + PALETTE_RECT_SIZE+2), (255, 255, 255), 2)

def check_palette_selection(x, y):
    """Checks if coordinates (x, y) are inside a palette box."""
    for i, color in enumerate(PALETTE_COLORS):
        px = PALETTE_START_X + (i * (PALETTE_RECT_SIZE + 10))
        py = PALETTE_START_Y
        
        if px <= x <= px + PALETTE_RECT_SIZE and py <= y <= py + PALETTE_RECT_SIZE:
            return color
    return None

# ===================== MAIN FUNCTION =====================

def main():
    # ===================== INITIALIZATION =====================
    
    # Force TCP transport globally
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
            frame = video_thread.read()

            if frame is None:
                cv2.waitKey(1)
                continue

            fps_counter.update()
            h, w = frame.shape[:2]
            canvas_mgr.ensure_size(h, w)

            # ===================== HAND TRACKING =====================
            result = tracker.process(frame)
            gesture = "NONE"
            now = time.time()

            # Default active tool settings
            active_color = DRAW_COLOR
            active_thickness = BRUSH_THICKNESS
            is_drawing = False

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_lms = result.multi_hand_landmarks[0]
                handedness_label = result.multi_handedness[0].classification[0].label

                tracker.draw_landmarks(frame, hand_lms)

                fingers = tracker.get_finger_states(hand_lms, handedness_label)

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

                # ---- Drawing & Interaction Logic ----
                # Only proceed if Draw is Enabled and NOT a Palm (to avoid scribbling)
                if draw_enabled and not palm:
                    # Get Index Finger Tip (Landmark 8)
                    tip_x = int(hand_lms.landmark[8].x * w)
                    tip_y = int(hand_lms.landmark[8].y * h)

                    # 1. Check Palette Selection
                    selected_color = check_palette_selection(tip_x, tip_y)
                    if selected_color:
                        active_color = selected_color
                        gesture = "SELECT COLOR"
                        # Stop drawing when selecting colors
                        prev_draw_pt = None
                        smoothed_pt = None
                    else:
                        # 2. Check Gestures (Eraser vs Brush)
                        
                        # Gesture: Index + Middle = ERASER
                        if fingers["index"] and fingers["middle"]:
                            active_color = (0, 0, 0) # Black
                            active_thickness = ERASER_THICKNESS
                            gesture = "ERASER"
                            is_drawing = True
                        
                        # Gesture: Only Index = BRUSH
                        elif fingers["index"] and not fingers["middle"]:
                            gesture = "DRAWING"
                            is_drawing = True
                            
                        else:
                            # Other gestures (Fist, etc.)
                            gesture = "HOVER"

                    # 3. Execute Drawing
                    if is_drawing:
                        # Smoothing logic
                        if smoothed_pt is None:
                            smoothed_pt = (tip_x, tip_y)
                        else:
                            sx, sy = smoothed_pt
                            sx = int((1 - SMOOTH_ALPHA) * sx + SMOOTH_ALPHA * tip_x)
                            sy = int((1 - SMOOTH_ALPHA) * sy + SMOOTH_ALPHA * tip_y)
                            smoothed_pt = (sx, sy)

                        if prev_draw_pt is None:
                            prev_draw_pt = smoothed_pt
                        
                        canvas_mgr.draw_line(prev_draw_pt, smoothed_pt, active_color, active_thickness)
                        prev_draw_pt = smoothed_pt
                    else:
                        prev_draw_pt = None
                        smoothed_pt = None
                
                # Add quick info text to frame
                cv2.putText(frame, f"Hand: {handedness_label}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            else:
                prev_draw_pt = None
                smoothed_pt = None
                palm_start_time = None

            # ===================== COMPOSITING & UI =====================
            
            # Draw the palette on the frame (before overlay)
            draw_palette(frame, active_color)

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
        video_thread.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()