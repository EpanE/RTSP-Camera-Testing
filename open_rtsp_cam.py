import cv2

USER = "admin"
PASS = ""
IP   = "192.168.0.27"   # from Fing
PORT = 554

# Try one URL pattern at a time
rtsp_url = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/Streaming/Channels/101"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream. Check URL / username / password / path.")

# Create resizable window
cv2.namedWindow("RTSP", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP", 800, 600)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame read failed (network/codec/URL issue).")
        break

    cv2.imshow("RTSP", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
