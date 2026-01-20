import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    # RTSP Settings (Loaded from .env)
    rtsp_user: str = os.getenv("RTSP_USER", "admin")
    rtsp_pass: str = os.getenv("RTSP_PASS", "")
    rtsp_ip: str = os.getenv("RTSP_IP", "192.168.0.27")
    rtsp_port: int = int(os.getenv("RTSP_PORT", "554"))
    rtsp_path: str = os.getenv("RTSP_PATH", "Streaming/Channels/101")
    
    flip_horizontal: bool = True

    # Pinch Settings
    pinch_on_threshold: float = 0.045
    pinch_off_threshold: float = 0.070
    
    # Volume Smoothing
    vol_smooth_alpha: float = 0.25
    medium_step: float = 0.05  # None for fully smooth

    # UI Dimensions
    slider_x: int = 40
    slider_y1: int = 120
    slider_y2: int = 520
    slider_w: int = 16
    slider_pad_x: int = 40

    @property
    def rtsp_url(self) -> str:
        return f"rtsp://{self.rtsp_user}:{self.rtsp_pass}@{self.rtsp_ip}:{self.rtsp_port}/{self.rtsp_path}"