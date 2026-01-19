# modules/__init__.py
from .streamer import RTSPStreamer  # Kept for legacy, but not used in threaded mode
from .detector import PersonDetector
from .privacy import PrivacyFilter
from .logger import AlertLogger
from .producer import CameraProducer  # NEW
from .consumer import AIConsumer      # NEW