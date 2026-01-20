import cv2
from typing import Tuple, Optional
from core.config import Config

class VolumeSliderUI:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _to_pixel(self, norm_x: float, norm_y: float, w: int, h: int) -> Tuple[int, int]:
        return int(norm_x * w), int(norm_y * h)

    def get_active_lane(self, x_px: int, y_px: int) -> Optional[str]:
        """
        Returns 'VOLUME', 'BRIGHTNESS', or None based on finger position.
        """
        # Check Volume Lane (Left)
        in_vol_x = (self.cfg.vol_x - self.cfg.vol_pad_x) <= x_px <= (self.cfg.vol_x + self.cfg.vol_w + self.cfg.vol_pad_x)
        
        # Check Brightness Lane (Right)
        in_bright_x = (self.cfg.bright_x - self.cfg.bright_pad_x) <= x_px <= (self.cfg.bright_x + self.cfg.bright_w + self.cfg.bright_pad_x)
        
        in_y = self.cfg.slider_y1 <= y_px <= self.cfg.slider_y2

        if in_vol_x and in_y:
            return 'VOLUME'
        elif in_bright_x and in_y:
            return 'BRIGHTNESS'
        return None

    def y_to_norm(self, y_px: int) -> float:
        """Convert pixel Y to 0.0-1.0 scalar."""
        y_clamped = max(self.cfg.slider_y1, min(self.cfg.slider_y2, y_px))
        t = (self.cfg.slider_y2 - y_clamped) / (self.cfg.slider_y2 - self.cfg.slider_y1)
        return float(max(0.0, min(1.0, t)))

    def quantize(self, val: float) -> float:
        if self.cfg.medium_step is None:
            return val
        step = float(self.cfg.medium_step)
        return round(val / step) * step

    def draw_generic_slider(self, img, x_pos, width, value: float, label: str, active: bool, color=(0, 180, 255)):
        # Track
        cv2.rectangle(img, (x_pos, self.cfg.slider_y1), 
                      (x_pos + width, self.cfg.slider_y2), (200, 200, 200), 2)

        # Fill
        y_fill = int(self.cfg.slider_y2 - value * (self.cfg.slider_y2 - self.cfg.slider_y1))
        
        # Green if active, else default color
        fill_color = (0, 255, 0) if active else color
        
        cv2.rectangle(img, (x_pos + 2, y_fill), 
                      (x_pos + width - 2, self.cfg.slider_y2 - 2), fill_color, -1)
        
        # Knob
        cv2.circle(img, (x_pos + width // 2, y_fill), 10, fill_color, -1)
        
        # Label
        text_y = self.cfg.slider_y1 - 15
        cv2.putText(img, label, (x_pos, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fill_color, 2)

    def draw_overlay(self, img, vol_percent, bright_percent, gesture_text, mute_state):
        from utils.drawing import draw_text
        
        # Top Left Info
        draw_text(img, f"Vol: {vol_percent}%  |  Brt: {bright_percent}%", 20, 40, 0.9, 2)
        draw_text(img, gesture_text, 20, 80, 0.75, 2)
        
        h = img.shape[0]
        draw_text(img, "Pinch in LEFT Lane for Vol | RIGHT Lane for Brightness. Keys: [m]=mute  [q]=quit",
                  20, h - 20, 0.65, 2)

    def draw_finger_markers(self, img, tx, ty, ix, iy):
        cv2.circle(img, (tx, ty), 8, (255, 255, 255), -1)  # thumb
        cv2.circle(img, (ix, iy), 8, (0, 255, 0), -1)      # index