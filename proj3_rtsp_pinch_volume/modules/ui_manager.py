import cv2
from typing import Tuple  # <--- THIS LINE WAS MISSING
from core.config import Config

class VolumeSliderUI:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _to_pixel(self, norm_x: float, norm_y: float, w: int, h: int) -> Tuple[int, int]:
        return int(norm_x * w), int(norm_y * h)

    def is_in_lane(self, x_px: int, y_px: int) -> bool:
        """Check if finger is horizontally within the slider lane."""
        x_ok = (self.cfg.slider_x - self.cfg.slider_pad_x) <= x_px <= (self.cfg.slider_x + self.cfg.slider_w + self.cfg.slider_pad_x)
        y_ok = self.cfg.slider_y1 <= y_px <= self.cfg.slider_y2
        return x_ok and y_ok

    def y_to_volume(self, y_px: int) -> float:
        """Convert pixel Y to 0.0-1.0 volume scalar."""
        y_clamped = max(self.cfg.slider_y1, min(self.cfg.slider_y2, y_px))
        t = (self.cfg.slider_y2 - y_clamped) / (self.cfg.slider_y2 - self.cfg.slider_y1)
        return float(max(0.0, min(1.0, t)))

    def quantize(self, vol: float) -> float:
        """Apply stepping if configured."""
        if self.cfg.medium_step is None:
            return vol
        step = float(self.cfg.medium_step)
        return round(vol / step) * step

    def draw_slider(self, img, vol: float, is_active: bool):
        # Track
        cv2.rectangle(img, (self.cfg.slider_x, self.cfg.slider_y1), 
                      (self.cfg.slider_x + self.cfg.slider_w, self.cfg.slider_y2), (200, 200, 200), 2)

        # Fill
        y_fill = int(self.cfg.slider_y2 - vol * (self.cfg.slider_y2 - self.cfg.slider_y1))
        color = (0, 255, 0) if is_active else (0, 180, 255)
        
        cv2.rectangle(img, (self.cfg.slider_x + 2, y_fill), 
                      (self.cfg.slider_x + self.cfg.slider_w - 2, self.cfg.slider_y2 - 2), color, -1)
        
        # Knob
        cv2.circle(img, (self.cfg.slider_x + self.cfg.slider_w // 2, y_fill), 10, color, -1)

    def draw_overlay(self, img, vol_percent, gesture_text, mute_state):
        from utils.drawing import draw_text
        
        draw_text(img, f"Volume: {vol_percent}%  |  {mute_state}", 20, 40, 0.9, 2)
        draw_text(img, gesture_text, 20, 80, 0.75, 2)
        h = img.shape[0]
        draw_text(img, "Pinch to grab. Volume changes only inside slider lane. Keys: [m]=mute  [q]=quit",
                  20, h - 20, 0.65, 2)

    def draw_finger_markers(self, img, tx, ty, ix, iy):
        cv2.circle(img, (tx, ty), 8, (255, 255, 255), -1)  # thumb
        cv2.circle(img, (ix, iy), 8, (0, 255, 0), -1)      # index