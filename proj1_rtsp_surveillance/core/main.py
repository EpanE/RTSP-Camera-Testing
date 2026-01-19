# core/main.py
import sys
import os
# Fix for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import cv2
import numpy as np
import time
import config
import threading

from utils import FPSCounter
from modules import CameraProducer, AIConsumer, PrivacyFilter, AlertLogger

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
    # ===================== INIT THREADS =====================
    print("Initializing Threads...")
    producer = CameraProducer()
    consumer = AIConsumer(producer)
    
    # Start the threads
    producer.start()
    consumer.start()

    # ===================== INIT UTILS =====================
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
    active_intruders = set()

    print("System Running (Multi-threaded). Press 'q' to quit.")

    while True:
        # 1. GET DATA (Non-blocking)
        # Get the latest raw frame from the camera thread
        frame = producer.get_frame()
        
        # Get the latest AI results from the processing thread
        detections = consumer.get_detections()

        if frame is None:
            # If camera not ready, wait a bit
            cv2.waitKey(100)
            continue

        # 2. PROCESSING (Main Thread is now only for Drawing!)
        fps_counter.update()
        out = frame.copy() # Work on a copy for drawing

        # Identify visible IDs
        visible_ids = set()
        current_frame_intruders = set()
        alert_triggered = False
        intruders_log = []

        for (x1, y1, x2, y2, cf, inside, cx, cy, t_id) in detections:
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

        # Update Occupancy
        active_intruders = active_intruders.intersection(visible_ids)
        active_intruders = active_intruders.union(current_frame_intruders)
        occupancy_count = len(active_intruders)

        # ===================== ALERT LOGIC =====================
        if alert_triggered or occupancy_count > 0:
            cv2.putText(out, "⚠️ ALERT: RESTRICTED ZONE ⚠️", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            for (tid, conf) in intruders_log:
                event_id = f"ID:{tid}" if tid != -1 else "Unknown"
                logger.log_event(event_id, conf, "Inside Zone")

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

        # Draw Occupancy Counter
        count_color = (0, 165, 255)
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

    # Cleanup
    producer.stop()
    consumer.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()