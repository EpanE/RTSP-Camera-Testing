from pycaw.pycaw import AudioUtilities

class AudioController:
    def __init__(self):
        self.device = AudioUtilities.GetSpeakers()
        self.endpoint = self.device.EndpointVolume

    def get_master_volume_scalar(self) -> float:
        return float(self.endpoint.GetMasterVolumeLevelScalar())

    def set_master_volume_scalar(self, v: float):
        v = float(max(0.0, min(1.0, v)))
        self.endpoint.SetMasterVolumeLevelScalar(v, None)

    def toggle_mute(self):
        self.endpoint.SetMute(not self.endpoint.GetMute(), None)

    def is_muted(self) -> bool:
        return self.endpoint.GetMute()