# modules/consumer.py
import threading
import time
from modules.detector import PersonDetector

class AIConsumer(threading.Thread):
    def __init__(self, producer):
        super().__init__()
        self.daemon = True
        self.producer = producer
        self.detector = PersonDetector() # Load model here in separate thread
        self.last_detections = []
        self.is_running = True

    def run(self):
        """The loop that runs AI in the background."""
        while self.is_running:
            # 1. Get latest frame from producer
            frame = self.producer.get_frame()
            
            if frame is not None:
                # 2. Run Detection
                # Note: This blocks the thread but NOT the UI/Main thread
                results = self.detector.detect(frame)
                
                # 3. Store results for Main thread to draw
                self.last_detections = results
            else:
                time.sleep(0.01) # Wait if no frame available

    def get_detections(self):
        """Returns the latest AI results."""
        return self.last_detections

    def stop(self):
        self.is_running = False