import time

class FPSCounter:
    def __init__(self):
        self._start_time = None
        self._end_time = None
        self._num_frames = 0
        self._fps = 0.0

    def start(self):
        self._start_time = time.time()
        self._end_time = self._start_time
        self._num_frames = 0
        self._fps = 0.0

    def update(self):
        self._end_time = time.time()
        self._num_frames += 1
        
        # Update FPS every 0.5 seconds to avoid flickering numbers
        if self._end_time - self._start_time >= 0.5:
            self._fps = self._num_frames / (self._end_time - self._start_time)
            self._start_time = self._end_time
            self._num_frames = 0

    def get_fps(self):
        return self._fps