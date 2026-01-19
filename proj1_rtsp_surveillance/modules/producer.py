# modules/producer.py
import cv2
import time
import threading
from core import config

class CameraProducer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True  # Thread will die when main program exits
        self.cap = None
        self.last_frame = None
        self.is_running = True
        
        # Connect immediately
        self.connect()

    def connect(self):
        """Attempts RTSP connection first, falls back to Local Webcam."""
        # 1. Attempt RTSP Connection
        print(f"Producer: Connecting to RTSP: {config.RTSP_URL}")
        self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
        
        if self.cap.isOpened():
            print("Producer: ✅ Connected to RTSP Stream.")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return

        # 2. Fallback to Local Webcam
        print(f"Producer: ⚠️ RTSP failed. Switching to Webcam ({config.FALLBACK_CAM_INDEX})...")
        self.cap = cv2.VideoCapture(config.FALLBACK_CAM_INDEX)
        
        if self.cap.isOpened():
            print("Producer: ✅ Connected to Webcam.")
        else:
            print("Producer: ❌ Failed to connect to any camera.")

    def run(self):
        """The loop that runs in the background thread."""
        while self.is_running:
            if self.cap is None:
                time.sleep(1)
                continue

            ok, frame = self.cap.read()

            if not ok or frame is None:
                print("Producer: Frame read failed. Reconnecting...")
                self.cap.release()
                time.sleep(0.5)
                self.connect()
                continue
            
            # Store the frame. 
            # We make a copy to ensure the Consumer/Main thread doesn't read 
            # this exact memory address while we overwrite it next loop.
            self.last_frame = frame.copy()

    def get_frame(self):
        """Returns the latest available frame."""
        if self.last_frame is not None:
            return self.last_frame.copy()
        return None

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()