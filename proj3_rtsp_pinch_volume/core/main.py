import sys
import os

# Add the project root to path so modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from modules.app import RTSPVolumeApp

if __name__ == "__main__":
    # You can adjust config values here before launching
    config = Config()
    
    app = RTSPVolumeApp(config)
    app.run()