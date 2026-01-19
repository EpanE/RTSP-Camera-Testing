# modules/logger.py
import os
import cv2
import csv
from datetime import datetime

class AlertLogger:
    def __init__(self):
        # Define directories
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(base_dir, "logs")
        self.snapshots_dir = os.path.join(base_dir, "snapshots")
        
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # CSV File Path
        self.csv_path = os.path.join(self.log_dir, "intrusion_log.csv")
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Event_ID", "Confidence", "Status"])
            print(f"Created log file at {self.csv_path}")

    def log_event(self, event_id, confidence, status):
        """
        Appends a single row to the CSV log.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, event_id, f"{confidence:.2f}", status])
        except Exception as e:
            print(f"Error writing to log: {e}")

    def save_snapshot(self, frame):
        """
        Saves the current frame as a JPG image with a timestamp filename.
        Returns the path to the saved image.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{timestamp}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            print(f"📸 Snapshot saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return None
