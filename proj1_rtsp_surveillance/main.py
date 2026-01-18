# main.py
import cv2
import numpy as np
import config
import time
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

    print("System started. Object Tracking enabled.")
    print("Controls: [s]=Save Zone  [f]=Blur  [z]=Zone  [q]=Quit")

    while True:
        ok, frame = streamer.read_frame()
        if not ok:
            continue

        fps_counter.update()
        out = frame.copy()
        
        # Detect now returns (x1, y1, x2, y2, cf, inside, cx, cy, track_id)
        detections = detector.detect(out)
        
        alert_triggered = False
        intruders = []

        for (x1, y1, x2, y2, cf, inside, cx, cy, t_id) in detections:
            if inside:
                alert_triggered = True
                intruders.append((t_id, cf))
            
            color = (0, 0, 255) if inside else (0, 255, 0)
            
            # Draw Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out, (cx, cy), 4, color, -1)
            
            # ===================== CHANGE: Display ID on screen =====================
            label_text = f"ID:{t_id} {cf:.2f}"
            cv2.putText(out, label_text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ===================== ALERT LOGIC =====================
        if alert_triggered:
            cv2.putText(out, "ALERT: person in restricted zone", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # 1. Log specific tracked IDs
            for (tid, conf) in intruders:
                # Use the unique ID (or "Unknown" if tracking hasn't locked on yet)
                event_id = f"ID:{tid}" if tid != -1 else "Unknown"
                logger.log_event(event_id, conf, "Inside Zone")

            # 2. Save Snapshot (Cooldown)
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

        if blur_faces:
            privacy_filter.apply_face_blur(out)
            cv2.putText(out, "FACE BLUR: ON", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fps_counter.draw(out)
        
        if alert_triggered:
            help_text = "⚠️ ALERT ACTIVE ⚠️ | Tracking & Logging..."
            text_color = (0, 0, 255)
        else:
            help_text = "Keys: [s]=Save Zone  [f]=Blur  [z]=Zone  [q]=Quit"
            text_color = (255, 255, 255)
            
        cv2.putText(out, help_text, (20, out.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
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