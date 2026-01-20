import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class HandProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def process(self, frame: np.ndarray) -> Tuple[Optional[dict], Optional[np.ndarray]]:
        """
        Processes frame to find hand landmarks.
        Returns (hand_data, annotated_image).
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hand_data = None
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_lms = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
            
            lm = hand_lms.landmark
            # Thumb (4) and Index (8) tips
            hand_data = {
                'thumb': (lm[4].x, lm[4].y),
                'index': (lm[8].x, lm[8].y),
                'w_px': w,
                'h_px': h
            }
            
        return hand_data, frame

    def close(self):
        self.hands.close()