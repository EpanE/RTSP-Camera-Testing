import cv2
import numpy as np
import time
from ultralytics import YOLO

# ===================== RTSP CONFIG =====================
USER = "admin"
PASS = ""              # put password if you have one
IP   = "192.168.0.27"
PORT = 554

# Hikvision-style path (your current working pattern)
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/Streaming/Channels/101"

# ===================== AI CONFIG =====================
# Person detection: COCO class 0
person_model = YOLO("yolov8n.pt")

# Face detection (for blur): use OpenCV Haar Cascade (offline, lightweight)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===================== ALERT ZONE =====================
# Edit these points to match your scene (x,y pixel coords)
# Tip: start rough, run, adjust, repeat.
restricted_zone = np.array([
    [120, 120],
    [520, 120],
    [640, 420],
    [140, 450],
], dtype=np.int32)

# ===================== STREAM OPEN =====================
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream. Check URL, creds, and camera path.")

# Reduce latency (may or may not be respected depending on backend/camera)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("RTSP-AI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP-AI", 1100, 650)

show_zone = True
blur_faces = False

# Simple reconnect logic state
last_frame_time = time.time()

def blur_roi(img, x1, y1, x2, y2, ksize=35):
    """Blur a region-of-interest safely."""
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    # Ensure odd kernel size
    k = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1:y2, x1:x2] = blurred

while True:
    ok, frame = cap.read()

    if not ok or frame is None:
        # Reconnect if stream stalls
        if time.time() - last_frame_time > 2.0:
            print("Frame read failed. Reconnecting RTSP...")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            last_frame_time = time.time()
        continue

    last_frame_time = time.time()

    out = frame.copy()

    # -------- (A) Person detection --------
    alert = False

    results = person_model.predict(out, conf=0.35, imgsz=640, verbose=False)
    r = results[0]

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        clss  = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), c, cf in zip(boxes, clss, confs):
            if c != 0:  # only "person"
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Center point of person bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            inside = cv2.pointPolygonTest(restricted_zone, (cx, cy), False) >= 0
            if inside:
                alert = True

            color = (0, 0, 255) if inside else (0, 255, 0)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out, (cx, cy), 4, color, -1)
            cv2.putText(
                out, f"person {cf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    # -------- (B) Zone overlay + alert text --------
    if show_zone:
        cv2.polylines(out, [restricted_zone], True, (0, 255, 255), 2)

    if alert:
        cv2.putText(
            out, "ALERT: person in restricted zone",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
        )

    # -------- (C) Face blur (privacy) --------
    # Run after drawing person boxes so blur sits on top nicely.
    if blur_faces:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (fx, fy, fw, fh) in faces:
            # Slightly expand box for better coverage
            pad = int(0.15 * fw)
            x1 = fx - pad
            y1 = fy - pad
            x2 = fx + fw + pad
            y2 = fy + fh + pad
            blur_roi(out, x1, y1, x2, y2, ksize=45)

        cv2.putText(
            out, "FACE BLUR: ON (press 'f' to toggle)",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    # -------- UI hints --------
    cv2.putText(
        out, "Keys: [f]=face blur  [z]=zone overlay  [q]=quit",
        (20, out.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    cv2.imshow("RTSP-AI", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        blur_faces = not blur_faces
    elif key == ord("z"):
        show_zone = not show_zone

cap.release()
cv2.destroyAllWindows()
