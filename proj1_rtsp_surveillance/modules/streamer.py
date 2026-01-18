# modules/streamer.py
import cv2
import time
import config

class RTSPStreamer:
    def __init__(self):
        self.cap = None
        self.connect() # Initial connection attempt
        self.last_frame_time = time.time()

    def connect(self):
        """Attempts RTSP connection first, falls back to Local Webcam."""
        
        # 1. Attempt RTSP Connection
        print(f"Attempting connection to RTSP: {config.RTSP_URL}")
        self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
        
        if self.cap.isOpened():
            print("✅ Connected to RTSP Stream.")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return

        # 2. Fallback to Local Webcam if RTSP failed
        print(f"⚠️ RTSP connection failed.")
        print(f"🔄 Switching to Local Webcam (Index {config.FALLBACK_CAM_INDEX})...")
        
        self.cap = cv2.VideoCapture(config.FALLBACK_CAM_INDEX)
        
        if self.cap.isOpened():
            print("✅ Connected to Local Webcam.")
        else:
            # If both fail, stop the program
            raise RuntimeError("Failed to connect to both RTSP and Local Webcam.")

    def read_frame(self):
        """
        Reads a frame. 
        Returns (True, frame) if successful.
        Returns (False, None) if stream needs to reconnect.
        """
        ok, frame = self.cap.read()

        if not ok or frame is None:
            # Simple reconnect logic
            if time.time() - self.last_frame_time > 2.0:
                print("Stream stalled. Attempting to reconnect...")
                self.cap.release()
                time.sleep(0.5)
                self.connect() # This will now try RTSP then Webcam again
                self.last_frame_time = time.time()
            return False, None

        self.last_frame_time = time.time()
        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()