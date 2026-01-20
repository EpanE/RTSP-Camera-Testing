import threading
import cv2
import time

class VideoCaptureThread:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.last_read_time = time.time()
        
        if not self.cap.isOpened():
            print(f"Error: Cannot open stream {self.src}")

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.last_read_time = time.time()
                
            # Small sleep to prevent CPU hogging if stream is very fast
            time.sleep(0.01)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.grabbed else None
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

    def reconnect(self):
        print("Attempting to reconnect...")
        self.stop()
        time.sleep(0.5)
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.start()