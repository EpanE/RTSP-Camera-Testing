import screen_brightness_control as sbc

class BrightnessController:
    def __init__(self):
        # Ensure valid range on init
        self.set_brightness(self.get_brightness())

    def get_brightness(self) -> int:
        """Returns brightness 0-100"""
        try:
            val = sbc.get_brightness()
            return int(val[0]) if isinstance(val, list) else int(val)
        except:
            return 100

    def set_brightness(self, v: int):
        """Sets brightness 0-100"""
        v = int(max(0, min(100, v)))
        try:
            sbc.set_brightness(v)
        except Exception as e:
            print(f"Error setting brightness: {e}")