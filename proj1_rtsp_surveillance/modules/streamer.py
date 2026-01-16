# modules/streamer.py
import cv2
import time
import config

class RTSPStreamer:
    def __init__(self):
        self.cap = None
        self.connect()
        self.last_frame_time = time.time()

    def connect(self):
        """Initialize or re-initialize the video capture."""
        print(f"Connecting to {config.RTSP_URL}...")
        self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open RTSP stream. Check URL and network.")
        
        # Reduce buffering for lower latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Connected.")

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
                print("Stream stalled. Reconnecting...")
                self.cap.release()
                time.sleep(0.5)
                self.connect()
                self.last_frame_time = time.time()
            return False, None

        self.last_frame_time = time.time()
        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()