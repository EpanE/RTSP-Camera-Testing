import mediapipe as mp
import cv2

class HandTracker:
    def __init__(self, min_detection_conf=0.6, min_tracking_conf=0.6, max_num_hands=1, model_complexity=1):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        """Returns (multi_hand_landmarks, multi_handedness) or (None, None)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw_landmarks(self, frame, hand_lms):
        self.mp_draw.draw_landmarks(frame, hand_lms, self.hands.HAND_CONNECTIONS)

    def close(self):
        self.hands.close()

    @staticmethod
    def get_finger_states(hand_lms, handedness_label):
        """Return dict of finger up/down."""
        lm = hand_lms.landmark
        fingers = {}

        fingers["index"]  = lm[8].y  < lm[6].y
        fingers["middle"] = lm[12].y < lm[10].y
        fingers["ring"]   = lm[16].y < lm[14].y
        fingers["pinky"]  = lm[20].y < lm[18].y

        # Thumb: depends left/right
        if handedness_label == "Right":
            fingers["thumb"] = lm[4].x > lm[3].x
        else:
            fingers["thumb"] = lm[4].x < lm[3].x

        return fingers

    @staticmethod
    def is_palm(fingers):
        return sum(fingers.values()) == 5

    @staticmethod
    def is_fist(fingers):
        return sum(fingers.values()) == 0
