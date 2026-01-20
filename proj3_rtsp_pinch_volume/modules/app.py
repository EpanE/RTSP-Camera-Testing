import time
import cv2
import numpy as np

from core.config import Config
from modules.audio_controller import AudioController
from modules.brightness_controller import BrightnessController
from modules.hand_processor import HandProcessor
from modules.ui_manager import VolumeSliderUI
from utils.video_thread import VideoCaptureThread

class RTSPVolumeApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Controllers
        self.audio = AudioController()
        self.brightness = BrightnessController()
        self.hand_proc = HandProcessor()
        self.ui = VolumeSliderUI(cfg)
        
        # State
        self.pinching = False
        self.last_frame_time = time.time()
        
        # Smoothed values (0.0 - 1.0)
        self.smoothed_vol = self.audio.get_master_volume_scalar()
        self.smoothed_bright = self.brightness.get_brightness() / 100.0
        
        # ==========================================
        # SOURCE DETECTION (RTSP vs WEBCAM)
        # ==========================================
        print(f"Attempting to connect to RTSP: {cfg.rtsp_url}")
        
        # Try to open RTSP stream
        temp_cap = cv2.VideoCapture(cfg.rtsp_url, cv2.CAP_FFMPEG)
        temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        connected = False
        if temp_cap.isOpened():
            # Try to read one frame to confirm actual stream data
            ret, _ = temp_cap.read()
            if ret:
                connected = True
        
        temp_cap.release()

        if connected:
            self.active_source = cfg.rtsp_url
            print("[SUCCESS] RTSP Stream connected.")
        else:
            self.active_source = 0  # 0 is usually the default laptop webcam
            print("[WARNING] RTSP Connection failed. Falling back to Local Webcam (Source 0).")
        # ==========================================

        # Start Threaded Video Capture with the resolved source
        self.video_thread = VideoCaptureThread(self.active_source).start()

        cv2.namedWindow("RTSP Pinch Volume", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RTSP Pinch Volume", 1100, 650)

    def _get_frame(self):
        ok, frame = self.video_thread.read()
        if not ok or frame is None:
            # Auto reconnect logic is now handled via time comparison
            # Note: If we started on Webcam, this will try to reconnect to Webcam. 
            # If we started on RTSP, it tries RTSP.
            if time.time() - self.last_frame_time > 3.0:
                print("Stream timeout. Reconnecting...")
                self.video_thread.reconnect()
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
                active_lane = None

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

                    active_lane = self.ui.get_active_lane(ix_px, iy_px)
                    control_active = self.pinching and active_lane is not None

                    # Update UI Strings
                    lane_str = active_lane if active_lane else "NONE"
                    gesture = f"PINCH={'ON' if self.pinching else 'OFF'}  dist={dist:.3f}  lane={lane_str}"
                    
                    # Control Logic
                    if control_active:
                        target_norm = self.ui.y_to_norm(iy_px)
                        target_norm = self.ui.quantize(target_norm)
                        
                        if active_lane == 'VOLUME':
                            self.smoothed_vol = (1 - self.cfg.smooth_alpha) * self.smoothed_vol + \
                                                self.cfg.smooth_alpha * target_norm
                            self.audio.set_master_volume_scalar(self.smoothed_vol)
                        
                        elif active_lane == 'BRIGHTNESS':
                            self.smoothed_bright = (1 - self.cfg.smooth_alpha) * self.smoothed_bright + \
                                                    self.cfg.smooth_alpha * target_norm
                            # Convert back to 0-100 for the controller
                            self.brightness.set_brightness(int(self.smoothed_bright * 100))
                    
                    else:
                        # Reset smoothing references to reality to prevent jumps when re-entering lane
                        self.smoothed_vol = self.audio.get_master_volume_scalar()
                        self.smoothed_bright = self.brightness.get_brightness() / 100.0

                    # Visuals
                    self.ui.draw_finger_markers(frame, tx_px, ty_px, ix_px, iy_px)
                    
                    # Draw Lane Guides if pinching
                    if self.pinching:
                        # Highlight active lane
                        if active_lane == 'VOLUME':
                            color = (0, 255, 0)
                            x = self.cfg.vol_x
                            w = self.cfg.vol_w
                            pad = self.cfg.vol_pad_x
                        elif active_lane == 'BRIGHTNESS':
                            color = (0, 255, 255) # Cyan for Brightness
                            x = self.cfg.bright_x
                            w = self.cfg.bright_w
                            pad = self.cfg.bright_pad_x
                        else:
                            color = (0, 0, 255)
                            x, w, pad = 0,0,0 

                        if x != 0:
                            cv2.rectangle(frame, 
                                          (x - pad, self.cfg.slider_y1),
                                          (x + w + pad, self.cfg.slider_y2),
                                          color, 2)

                # 3. Draw UI
                # Volume Slider (Green base)
                self.ui.draw_generic_slider(frame, self.cfg.vol_x, self.cfg.vol_w, 
                                            self.smoothed_vol, "VOLUME", 
                                            active_lane == 'VOLUME' and self.pinching, 
                                            color=(0, 180, 255))
                
                # Brightness Slider (Orange base)
                self.ui.draw_generic_slider(frame, self.cfg.bright_x, self.cfg.bright_w, 
                                            self.smoothed_bright, "BRIGHTNESS", 
                                            active_lane == 'BRIGHTNESS' and self.pinching, 
                                            color=(0, 165, 255))

                self.ui.draw_overlay(frame, int(self.smoothed_vol * 100), 
                                     int(self.smoothed_bright * 100), gesture, 
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
        self.video_thread.stop()
        cv2.destroyAllWindows()
        self.hand_proc.close()