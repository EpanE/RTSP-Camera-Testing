# modules/privacy.py
import cv2

class PrivacyFilter:
    def __init__(self):
        # Load Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def blur_roi(self, img, x1, y1, x2, y2, ksize=35):
        """Blur a region-of-interest safely."""
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img.shape[1]-1, x2)
        y2 = min(img.shape[0]-1, y2)
        
        if x2 <= x1 or y2 <= y1:
            return
            
        roi = img[y1:y2, x1:x2]
        k = ksize if ksize % 2 == 1 else ksize + 1
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        img[y1:y2, x1:x2] = blurred

    def apply_face_blur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (fx, fy, fw, fh) in faces:
            pad = int(0.15 * fw)
            x1, y1 = fx - pad, fy - pad
            x2, y2 = fx + fw + pad, fy + fh + pad
            self.blur_roi(frame, x1, y1, x2, y2, ksize=45)
        
        return len(faces) > 0