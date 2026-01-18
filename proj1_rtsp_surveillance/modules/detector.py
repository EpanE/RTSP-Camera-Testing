# modules/detector.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import config

class PersonDetector:
    def __init__(self):
        print(f"Loading model: {config.PERSON_MODEL_PATH}...")
        self.model = YOLO(config.PERSON_MODEL_PATH)
        
        # Device configuration
        if torch.cuda.is_available():
            self.model.to("cuda")
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            config.DEVICE = "cpu"
            config.USE_HALF = False
            print("⚠️ CUDA not available, using CPU")

        self.last_person_dets = [] 
        self.frame_count = 0

    def detect(self, frame):
        """
        Runs tracking/detection.
        Returns list: (x1, y1, x2, y2, conf, inside_zone, cx, cy, track_id)
        """
        self.frame_count += 1
        do_detect = (self.frame_count % config.SKIP_EVERY_N == 0)
        
        if do_detect:
            self._run_inference(frame)
        
        return self.last_person_dets

    def _run_inference(self, frame):
        """Internal method to run YOLO Tracking."""
        zone_poly = np.array(config.RESTRICTED_ZONE, dtype=np.int32)
        self.last_person_dets = []

        # ===================== CHANGE: Using .track() instead of .predict() =====================
        # persist=True keeps the tracking history across frames
        results = self.model.track(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.INFER_IMGSZ,
            device=config.DEVICE,
            half=config.USE_HALF,
            verbose=False,
            persist=True 
        )
        
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            
            # Extract track IDs. If tracking hasn't initialized yet, this might be None.
            if r.boxes.id is not None:
                track_ids = r.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(boxes)

            for (x1, y1, x2, y2), c, cf, t_id in zip(boxes, clss, confs, track_ids):
                if c != 0: # 0 is 'person'
                    continue

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                inside = cv2.pointPolygonTest(zone_poly, (cx, cy), False) >= 0
                
                # Pass the track_id (-1 if unknown) to the main loop
                self.last_person_dets.append((x1, y1, x2, y2, float(cf), inside, cx, cy, t_id))