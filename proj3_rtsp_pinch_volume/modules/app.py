import time
import cv2
import numpy as np

from core.config import Config
from modules.audio_controller import AudioController
from modules.hand_processor import HandProcessor
from modules.ui_manager import VolumeSliderUI

class RTSPVolumeApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.audio = AudioController()
        self.hand_proc = HandProcessor()
        self.ui = VolumeSliderUI(cfg)
        
        # State
        self.pinching = False
        self.smoothed_vol = self.audio.get_master_volume_scalar()
        self.last_frame_time = time.time()
        
        # Video Capture
        self.cap = cv2.VideoCapture(cfg.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open RTSP stream.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cv2.namedWindow("RTSP Pinch Volume", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RTSP Pinch Volume", 1100, 650)

    def _get_frame(self):
        """Reads frame with auto-reconnect logic."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if time.time() - self.last_frame_time > 2.0:
                print("Frame read failed. Reconnecting RTSP...")
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.cfg.rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.last_frame_time = time.time()
            return None
        self.last_frame_time = time.time()

        if self.cfg.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    def run(self):
        try:
            while True:
                frame = self._get_frame()
                if frame is None:
                    continue

                # 1. Process Vision
                hand_data, frame = self.hand_proc.process(frame)
                
                gesture = "NO HAND"
                lane_ok = False
                control_active = False

                # 2. Logic Update
                if hand_data:
                    # Extract coordinates
                    tx, ty = hand_data['thumb']
                    ix, iy = hand_data['index']
                    w, h = hand_data['w_px'], hand_data['h_px']
                    
                    ix_px, iy_px = self.ui._to_pixel(ix, iy, w, h)
                    tx_px, ty_px = self.ui._to_pixel(tx, ty, w, h)

                    # Calculate Distance
                    dist = np.sqrt((tx - ix)**2 + (ty - iy)**2)

                    # Pinch State Machine (Hysteresis)
                    if not self.pinching and dist < self.cfg.pinch_on_threshold:
                        self.pinching = True
                    elif self.pinching and dist > self.cfg.pinch_off_threshold:
                        self.pinching = False

                    lane_ok = self.ui.is_in_lane(ix_px, iy_px)
                    control_active = self.pinching and lane_ok

                    # Volume Control
                    if control_active:
                        target_vol = self.ui.y_to_volume(iy_px)
                        target_vol = self.ui.quantize(target_vol)
                        
                        # Smoothing
                        self.smoothed_vol = (1 - self.cfg.vol_smooth_alpha) * self.smoothed_vol + \
                                            self.cfg.vol_smooth_alpha * target_vol
                        self.audio.set_master_volume_scalar(self.smoothed_vol)
                    else:
                        # Reset smoothing reference to actual hardware volume
                        self.smoothed_vol = self.audio.get_master_volume_scalar()

                    # Update UI Strings
                    gesture = f"PINCH={'ON' if self.pinching else 'OFF'}  dist={dist:.3f}  lane={'OK' if lane_ok else 'OUT'}"
                    
                    # Visuals
                    self.ui.draw_finger_markers(frame, tx_px, ty_px, ix_px, iy_px)
                    
                    # Draw Lane Guide if pinching
                    if self.pinching:
                        color = (0, 255, 0) if lane_ok else (0, 0, 255)
                        cv2.rectangle(frame, 
                                      (self.cfg.slider_x - self.cfg.slider_pad_x, self.cfg.slider_y1),
                                      (self.cfg.slider_x + self.cfg.slider_w + self.cfg.slider_pad_x, self.cfg.slider_y2),
                                      color, 2)

                # 3. Draw UI
                current_vol = self.audio.get_master_volume_scalar()
                self.ui.draw_slider(frame, current_vol, control_active)
                self.ui.draw_overlay(frame, int(current_vol * 100), gesture, 
                                     "MUTED" if self.audio.is_muted() else "UNMUTED")

                cv2.imshow("RTSP Pinch Volume", frame)

                # Input Handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("m"):
                    self.audio.toggle_mute()

        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_proc.close()