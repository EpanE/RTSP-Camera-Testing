import pyautogui
import time

# Safety: Failsafe prevents dragging mouse to corner to abort
pyautogui.FAILSAFE = True 

class MouseController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.last_click_time = 0
        self.is_pressed = False

    def move(self, norm_x: float, norm_y: float):
        """
        Moves the mouse to the normalized position.
        norm_x, norm_y are 0.0-1.0 (from MediaPipe).
        """
        # Clamp values to stay on screen
        x = max(0, min(1, norm_x))
        y = max(0, min(1, norm_y))
        
        # Convert to screen pixels
        pixel_x = int(x * self.screen_width)
        pixel_y = int(y * self.screen_height)
        
        # Move
        pyautogui.moveTo(pixel_x, pixel_y)

    def handle_pinch(self, is_pinching):
        """
        Handles Mouse Down (Press) and Mouse Up (Release).
        """
        if is_pinching:
            if not self.is_pressed:
                pyautogui.mouseDown()
                self.is_pressed = True
        else:
            if self.is_pressed:
                pyautogui.mouseUp()
                self.is_pressed = False

    def reset(self):
        """Release mouse button on exit/reset"""
        if self.is_pressed:
            pyautogui.mouseUp()
            self.is_pressed = False