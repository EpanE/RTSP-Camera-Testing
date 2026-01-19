import threading
import queue
import cv2
import time
import os

class VideoCaptureThreaded:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.q = queue.Queue(maxsize=1) # Queue size 1 ensures we always get the latest frame
        self.stopped = False
        self.cap = None
        self.thread = None

    def start(self):
        """Start the background thread."""
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Thread dies when main program exits
        self.thread.start()
        return self

    def update(self):
        """Read frames from the RTSP stream in a loop."""
        # Initialize capture inside the thread
        # Force TCP for reliability
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            print("[Thread] Initial connection failed. Retrying...")

        while not self.stopped:
            # --- Connection Check ---
            if not self.cap.isOpened():
                # Try to reconnect if stream drops
                print("[Thread] Connection lost. Reconnecting...")
                time.sleep(1.0) # Wait a second before retrying
                self.cap.open(self.rtsp_url, cv2.CAP_FFMPEG)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print("[Thread] Reconnected successfully.")
                continue

            # --- Read Frame ---
            ret, frame = self.cap.read()
            
            if ret:
                # If queue is full, remove old frame (prevents lag buildup)
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)
            else:
                # Frame read failed (might be temporary glitch or disconnect)
                print("[Thread] Frame read failed. Releasing cap...")
                self.cap.release()

    def read(self):
        """Returns the latest frame. Returns None if no frame is available yet."""
        try:
            # Wait up to 10ms for a frame. If none, return None
            return self.q.get(timeout=0.01)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the thread and release resources."""
        self.stopped = True
        if self.thread is not None:
            self.thread.join() # Wait for thread to finish
        if self.cap is not None:
            self.cap.release()