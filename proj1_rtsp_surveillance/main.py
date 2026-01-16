# main.py
import cv2
import numpy as np
import config
from utils import FPSCounter
from modules import RTSPStreamer, PersonDetector, PrivacyFilter

# Global variables for mouse interaction
current_points = np.array(config.RESTRICTED_ZONE, dtype=np.int32).tolist()
drag_idx = -1  # Index of the point being dragged (-1 means none)
radius = 10    # Radius of the handle circles for clicking

def mouse_callback(event, x, y, flags, param):
    """Handles mouse events to drag polygon points."""
    global drag_idx, current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if mouse is near any point
        for i, point in enumerate(current_points):
            px, py = point
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist <= radius:
                drag_idx = i
                break
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_idx != -1:
            # Update the position of the dragged point
            current_points[drag_idx] = [x, y]
            # Update global config so detector uses the new zone immediately
            config.RESTRICTED_ZONE = current_points

    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = -1

def main():
    # ===================== INIT MODULES =====================
    streamer = RTSPStreamer()
    detector = PersonDetector()
    privacy_filter = PrivacyFilter()
    fps_counter = FPSCounter()

    # Setup Window
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
    
    # Register Mouse Callback
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback)

    # State flags
    show_zone = True
    blur_faces = False

    print("System started. Controls:")
    print(" [Mouse Drag] : Edit Zone")
    print(" [s]          : Save Zone to file")
    print(" [f]          : Toggle Face Blur")
    print(" [z]          : Toggle Zone Overlay")
    print(" [q]          : Quit")

    while True:
        # 1. Read Frame
        ok, frame = streamer.read_frame()
        if not ok:
            continue

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

        # 5. Draw Zone (Interactive) & Alert
        if show_zone:
            # Convert list to numpy for drawing
            pts = np.array(current_points, dtype=np.int32)
            
            # Draw filled polygon (slightly transparent look via lines or just fill)
            # OpenCV doesn't support alpha fill easily on main image, so we stick to lines
            cv2.polylines(out, [pts], True, (0, 255, 255), 2)

            # Draw handles (circles) at corners to show they are draggable
            for pt in pts:
                cv2.circle(out, tuple(pt), radius, (0, 255, 0), -1)

        if alert_triggered:
            cv2.putText(out, "ALERT: person in restricted zone", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 6. Face Blur
        if blur_faces:
            privacy_filter.apply_face_blur(out)
            cv2.putText(out, "FACE BLUR: ON (press 'f')", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 7. Draw FPS & Info
        fps_counter.draw(out)
        
        help_text = "Keys: [s]=Save Zone  [f]=Blur  [z]=Zone  [q]=Quit"
        cv2.putText(out, help_text, (20, out.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Zone editing hint
        if show_zone:
            cv2.putText(out, "MODE: Drag green dots to edit zone", (20, out.shape[0] - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        else:
            cv2.putText(out, f"Device: {config.DEVICE}", (20, out.shape[0] - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

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
        elif key == ord("s"):
            # Save the current points to config file
            config.save_zone(current_points)

    # Cleanup
    streamer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()