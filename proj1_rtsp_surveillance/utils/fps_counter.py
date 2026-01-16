# utils/fps_counter.py
import cv2
import time

class FPSCounter:
    def __init__(self):
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0

    def update(self):
        """
        Updates the internal time and calculates FPS.
        Call this once per frame.
        """
        self.curr_frame_time = time.time()
        
        # Avoid division by zero on the very first frame
        if self.prev_frame_time != 0:
            diff = self.curr_frame_time - self.prev_frame_time
            self.fps = 1 / diff
        
        self.prev_frame_time = self.curr_frame_time

    def draw(self, frame, position=(20, 110), color=(0, 255, 0)):
        """
        Draws the FPS value onto the frame.
        """
        cv2.putText(
            frame, 
            f"FPS: {int(self.fps)}", 
            position, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            color, 
            2
        )