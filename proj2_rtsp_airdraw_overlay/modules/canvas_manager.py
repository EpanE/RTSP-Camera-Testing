import cv2
import numpy as np

class CanvasManager:
    def __init__(self):
        self.canvas = None

    def ensure_size(self, h, w):
        if self.canvas is None or self.canvas.shape[0] != h or self.canvas.shape[1] != w:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def clear(self):
        if self.canvas is not None:
            self.canvas[:] = 0

    def draw_line(self, pt1, pt2, color, thickness):
        if self.canvas is not None:
            cv2.line(self.canvas, pt1, pt2, color, thickness)

    def get_overlay(self, frame):
        """Blends the canvas onto the frame."""
        if self.canvas is None:
            return frame
            
        # Create mask
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Mask out parts of frame
        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        # Mask out parts of canvas
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        
        return cv2.add(bg, fg)