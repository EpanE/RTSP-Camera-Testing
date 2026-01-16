# main.py
import cv2
import numpy as np
import config
import time
from utils import FPSCounter
from modules import RTSPStreamer, PersonDetector, PrivacyFilter, AlertLogger  # <--- ADD AlertLogger

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
    logger = AlertLogger() # <--- INIT LOGGER

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback)

    show_zone = True
    blur_faces = False

    # Snapshot Cooldown Logic
    SNAPSHOT_COOLDOWN = 5.0  # Seconds to wait before saving next snapshot
    last_snapshot_time = 0

    print("System started. Logging enabled.")
    print("Controls: [s]=Save Zone  [f]=Blur  [z]=Zone  [q]=Quit")

    while True:
        ok, frame = streamer.read_frame()
        if not ok:
            continue

        fps_counter.update()
        out = frame.copy()
        detections = detector.detect(out)
        alert_triggered = False
        
        # List to hold people currently inside zone for logging
        intruders = []

        for (x1, y1, x2, y2, cf, inside, cx, cy) in detections:
            if inside:
                alert_triggered = True
                intruders.append(cf) # Add confidence to list of intruders
            
            color = (0, 0, 255) if inside else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out, (cx, cy), 4, color, -1)
            cv2.putText(out, f"person {cf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ===================== ALERT LOGIC & LOGGING =====================
        if alert_triggered:
            cv2.putText(out, "ALERT: person in restricted zone", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # 1. Log each intruder
            for idx, conf in enumerate(intruders):
                # We use index+1 as an ID since we don't have deep tracking yet
                logger.log_event(f"Person_{idx+1}", conf, "Inside Zone")

            # 2. Save Snapshot (With Cooldown)
            current_time = time.time()
            if current_time - last_snapshot_time > SNAPSHOT_COOLDOWN:
                logger.save_snapshot(out)
                last_snapshot_time = current_time
                
                # Visual feedback on screen
                cv2.putText(out, "💾 SAVED SNAPSHOT", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ===================== DRAWING & UI =====================
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
        
        # Dynamic Help Text (Change based on state)
        if alert_triggered:
            help_text = "⚠️ ALERT ACTIVE ⚠️ | Saving Logs/Snapshots..."
            text_color = (0, 0, 255)
        else:
            help_text = "Keys: [s]=Save Zone  [f]=Blur  [z]=Zone  [q]=Quit"
            text_color = (255, 255, 255)
            
        cv2.putText(out, help_text, (20, out.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Draw Device Info at bottom left
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