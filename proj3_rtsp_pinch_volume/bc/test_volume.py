from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def get_audio_endpoint():
    device = AudioUtilities.GetSpeakers()
    imm = getattr(device, "_device", None) or device
    interface = imm.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))

endpoint = get_audio_endpoint()
print("Current volume:", endpoint.GetMasterVolumeLevelScalar())
print("Mute:", endpoint.GetMute())
