# main.py
import cv2
import numpy as np
import config

from utils import FPSCounter
from modules import RTSPStreamer, PersonDetector, PrivacyFilter

def main():
    # ===================== INIT MODULES =====================
    streamer = RTSPStreamer()
    detector = PersonDetector()
    privacy_filter = PrivacyFilter()
    fps_counter = FPSCounter()

    # Setup Window
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    # State flags
    show_zone = True
    blur_faces = False

    zone_poly = np.array(config.RESTRICTED_ZONE, dtype=np.int32)

    print("System started. Press 'q' to quit.")

    while True:
        # 1. Read Frame
        ok, frame = streamer.read_frame()
        if not ok:
            continue # Wait for reconnect

        # 2. Update FPS
        fps_counter.update()
        
        out = frame.copy()

        # 3. Run AI Detection
        detections = detector.detect(out)
        alert_triggered = False

        # 4. Draw Detection Results & Check Zone
        for (x1, y1, x2, y2, cf, inside, cx, cy) in detections:
            if inside:
                alert_triggered = True
            
            color = (0, 0, 255) if inside else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out, (cx, cy), 4, color, -1)
            cv2.putText(out, f"person {cf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 5. Draw Zone & Alert
        if show_zone:
            cv2.polylines(out, [zone_poly], True, (0, 255, 255), 2)

        if alert_triggered:
            cv2.putText(out, "ALERT: person in restricted zone", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 6. Face Blur
        if blur_faces:
            privacy_filter.apply_face_blur(out)
            cv2.putText(out, "FACE BLUR: ON (press 'f')", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 7. Draw FPS & Info
        fps_counter.draw(out) # The requested FPS function
        cv2.putText(out, "Keys: [f]=face blur  [z]=zone overlay  [q]=quit",
                    (20, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(out, f"Device: {config.DEVICE} | imgsz={config.INFER_IMGSZ}",
                    (20, out.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 8. Display
        cv2.imshow(config.WINDOW_NAME, out)

        # Handle Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            blur_faces = not blur_faces
        elif key == ord("z"):
            show_zone = not show_zone

    # Cleanup
    streamer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()