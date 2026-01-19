# core/main.py

import sys
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (project root)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to Python's path so it can find 'utils' and 'modules'
sys.path.append(parent_dir)

import cv2
import numpy as np
import time
import config
from utils import FPSCounter
from modules import RTSPStreamer, PersonDetector, PrivacyFilter, AlertLogger

# Global variables for mouse interaction
current_points = np.array(config.RESTRICTED_ZONE, dtype=np.int32).tolist()
drag_idx = -1
radius = 10

def mouse_callback(event, x, y, flags, param):
    global drag_idx, current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, point in enumerate(current_points):
            px, py = point
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist <= radius:
                drag_idx = i
                break
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_idx != -1:
            current_points[drag_idx] = [x, y]
            config.RESTRICTED_ZONE = current_points
    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = -1

def main():
    # ===================== INIT MODULES =====================
    streamer = RTSPStreamer()
    detector = PersonDetector()
    privacy_filter = PrivacyFilter()
    fps_counter = FPSCounter()
    logger = AlertLogger()

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback)

    show_zone = True
    blur_faces = False

    SNAPSHOT_COOLDOWN = 5.0
    last_snapshot_time = 0

    # ===================== OCCUPANCY STATE =====================
    # A set to store IDs of people currently inside the zone
    active_intruders = set()

    print("System started. Occupancy Counting enabled.")
    print("Controls: [s]=Save Zone  [f]=Blur  [z]=Zone  [q]=Quit")

    while True:
        ok, frame = streamer.read_frame()
        if not ok:
            continue

        fps_counter.update()
        out = frame.copy()
        detections = detector.detect(out)
        
        # 1. Identify all Track IDs currently visible in the frame
        visible_ids = set()
        
        # Temporary set for this frame to update the main 'active_intruders' set
        current_frame_intruders = set()
        
        alert_triggered = False
        intruders_log = [] # List of (id, conf) for logging

        for (x1, y1, x2, y2, cf, inside, cx, cy, t_id) in detections:
            # Ignore -1 ID (uninitialized tracks) for occupancy counting to avoid noise
            if t_id != -1:
                visible_ids.add(t_id)
            
            if inside:
                alert_triggered = True
                if t_id != -1:
                    current_frame_intruders.add(t_id)
                intruders_log.append((t_id, cf))
            
            color = (0, 0, 255) if inside else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out, (cx, cy), 4, color, -1)
            
            label_text = f"ID:{t_id}" if t_id != -1 else "Unknown"
            cv2.putText(out, label_text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ===================== UPDATE OCCUPANCY STATE =====================
        # Logic: 
        # 1. Keep people who are active_intruders AND still visible in this frame.
        # 2. Add people who are 'current_frame_intruders'.
        # 3. Remove people who are NOT in 'visible_ids' (they left the scene).
        
        # Intersection: Keep only those who were inside AND are still seen anywhere on screen
        active_intruders = active_intruders.intersection(visible_ids)
        # Union: Add the new ones who just entered the zone
        active_intruders = active_intruders.union(current_frame_intruders)

        occupancy_count = len(active_intruders)

        # ===================== ALERT LOGIC =====================
        if alert_triggered or occupancy_count > 0:
            # Visual Alert
            color_alert = (0, 0, 255)
            cv2.putText(out, "⚠️ ALERT: RESTRICTED ZONE ⚠️", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_alert, 3)

            # Log specific tracked IDs
            for (tid, conf) in intruders_log:
                event_id = f"ID:{tid}" if tid != -1 else "Unknown"
                logger.log_event(event_id, conf, "Inside Zone")

            # Save Snapshot (Cooldown)
            current_time = time.time()
            if current_time - last_snapshot_time > SNAPSHOT_COOLDOWN:
                logger.save_snapshot(out)
                last_snapshot_time = current_time
                cv2.putText(out, "💾 SAVED SNAPSHOT", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ===================== DRAWING =====================
        if show_zone:
            pts = np.array(current_points, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 255), 2)
            for pt in pts:
                cv2.circle(out, tuple(pt), radius, (0, 255, 0), -1)

        # ===================== DRAW OCCUPANCY COUNTER =====================
        # Display the count prominently
        count_color = (0, 165, 255) # Orange
        cv2.rectangle(out, (out.shape[1] - 200, 20), (out.shape[1] - 20, 90), count_color, -1)
        cv2.putText(out, "INSIDE ZONE:", (out.shape[1] - 190, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, f"{occupancy_count}", (out.shape[1] - 190, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        if blur_faces:
            privacy_filter.apply_face_blur(out)
            cv2.putText(out, "FACE BLUR: ON", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fps_counter.draw(out)
        
        help_text = "Keys: [s]=Save Zone  [f]=Blur  [z]=Zone  [q]=Quit"
        cv2.putText(out, help_text, (20, out.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(out, f"Device: {config.DEVICE}", (20, out.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.imshow(config.WINDOW_NAME, out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            blur_faces = not blur_faces
        elif key == ord("z"):
            show_zone = not show_zone
        elif key == ord("s"):
            config.save_zone(current_points)

    streamer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()