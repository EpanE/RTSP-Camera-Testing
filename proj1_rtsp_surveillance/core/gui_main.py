# core/gui_custom.py
import sys
import os
# Fix for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import cv2
import numpy as np
import time
import config
import threading
import customtkinter as ctk

# PIL is required by CustomTkinter to display images
from PIL import Image, ImageTk 

from utils import FPSCounter
from modules import CameraProducer, AIConsumer, PrivacyFilter, AlertLogger

# CustomTkinter Setup
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Global variables for zone interaction
current_points = np.array(config.RESTRICTED_ZONE, dtype=np.int32).tolist()
drag_idx = -1
radius = 10

class SurveillanceApp(ctk.CTk):
    def __init__(self, producer, consumer):
        super().__init__()

        # Configure Window
        self.title("RTSP AI Surveillance System")
        self.geometry("1100x700")

        # Init Modules
        self.producer = producer
        self.consumer = consumer
        self.privacy_filter = PrivacyFilter()
        self.fps_counter = FPSCounter()
        self.logger = AlertLogger()

        # State Variables
        self.show_zone = True
        self.blur_faces = False
        self.active_intruders = set()
        self.last_snapshot_time = 0
        self.drag_idx = -1

        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        # === GRID LAYOUT ===
        self.grid_columnconfigure(0, weight=3) # Video area
        self.grid_columnconfigure(1, weight=1) # Controls area
        self.grid_rowconfigure(0, weight=1)

        # === LEFT: VIDEO FEED ===
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # We use a Label to display the image
        self.video_label = ctk.CTkLabel(self.video_frame, text="Connecting...")
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Bind Mouse Events for Zone Editing
        # Note: We bind to the Frame, but coordinates need adjustment based on where the Label is
        self.video_label.bind("<Button-1>", self.on_click)
        self.video_label.bind("<B1-Motion>", self.on_drag)
        self.video_label.bind("<ButtonRelease-1>", self.on_release)

        # === RIGHT: CONTROLS ===
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Title
        self.title_lbl = ctk.CTkLabel(self.control_frame, text="Control Panel", font=("Roboto", 20, "bold"))
        self.title_lbl.pack(pady=20)

        # Status
        self.status_lbl = ctk.CTkLabel(self.control_frame, text="System Active", text_color="green", font=("Roboto", 14))
        self.status_lbl.pack(pady=5)

        # Toggle Switches
        self.zone_switch = ctk.CTkSwitch(self.control_frame, text="Show Zone", command=self.toggle_zone)
        self.zone_switch.pack(pady=10)
        self.zone_switch.select() # Default on

        self.blur_switch = ctk.CTkSwitch(self.control_frame, text="Face Blur", command=self.toggle_blur)
        self.blur_switch.pack(pady=10)

        # Save Button
        self.save_btn = ctk.CTkButton(self.control_frame, text="Save Zone Config", command=self.save_zone)
        self.save_btn.pack(pady=20, padx=20)

        # Occupancy Count
        self.count_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.count_frame.pack(pady=10)
        ctk.CTkLabel(self.count_frame, text="Occupancy:", font=("Roboto", 14)).pack()
        self.count_lbl = ctk.CTkLabel(self.count_frame, text="0", font=("Roboto", 40, "bold"), text_color="orange")
        self.count_lbl.pack()

        # Log Preview
        self.log_box = ctk.CTkTextbox(self.control_frame, width=250, height=150)
        self.log_box.pack(pady=20, padx=20)
        
        # FPS
        self.fps_lbl = ctk.CTkLabel(self.control_frame, text="FPS: 0", font=("Roboto", 12))
        self.fps_lbl.pack(side="bottom", pady=10)

    # --- INTERACTION HANDLERS ---
    def toggle_zone(self):
        self.show_zone = self.zone_switch.get()

    def toggle_blur(self):
        self.blur_faces = self.blur_switch.get()

    def save_zone(self):
        config.save_zone(current_points)
        self.status_lbl.configure(text="Zone Saved!", text_color="yellow")

    def on_click(self, event):
        x, y = event.x, event.y
        # Check if clicked near a point (Scale logic might be needed if video is resized, assuming 1:1 for now)
        # Assuming 640x480 canvas roughly, but dynamic is better.
        # We will map mouse coordinates relative to the displayed image.
        
        # Get actual label size
        w = self.video_label.winfo_width()
        h = self.video_label.winfo_height()
        
        # Normalize mouse position to 640x480 (Internal drawing resolution)
        norm_x = int((x / w) * 640)
        norm_y = int((y / h) * 480)

        for i, point in enumerate(current_points):
            px, py = point
            dist = np.sqrt((norm_x - px)**2 + (norm_y - py)**2)
            if dist <= 15:
                self.drag_idx = i
                break

    def on_drag(self, event):
        if self.drag_idx != -1:
            x, y = event.x, event.y
            w = self.video_label.winfo_width()
            h = self.video_label.winfo_height()
            
            # Normalize to 640x480
            norm_x = int((x / w) * 640)
            norm_y = int((y / h) * 480)
            
            # Clamp
            norm_x = max(0, min(640, norm_x))
            norm_y = max(0, min(480, norm_y))

            current_points[self.drag_idx] = [norm_x, norm_y]
            config.RESTRICTED_ZONE = current_points

    def on_release(self, event):
        self.drag_idx = -1

    # --- MAIN UPDATE LOOP ---
    def update_loop(self):
        # 1. Get Data
        frame = self.producer.get_frame()
        detections = self.consumer.get_detections()

        if frame is not None:
            # Resize for display (Performance & Fit in UI)
            # We assume the internal drawing coordinates are 640x480
            resized_frame = cv2.resize(frame, (640, 480))
            
            # 2. Logic
            self.fps_counter.update()
            out = resized_frame.copy()

            visible_ids = set()
            current_frame_intruders = set()
            alert_triggered = False
            intruders_log = []

            for (x1, y1, x2, y2, cf, inside, cx, cy, t_id) in detections:
                # Scale coords to 640x480 if model output is different
                # Assuming model imgsz is 512 or 640, let's scale to fit our 640x480 view
                scale_x = 640 / config.INFER_IMGSZ
                scale_y = 480 / config.INFER_IMGSZ # Aspect ratio might be slightly off, but fine for demo
                
                sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
                sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
                
                if t_id != -1: visible_ids.add(t_id)
                
                if inside:
                    alert_triggered = True
                    if t_id != -1: current_frame_intruders.add(t_id)
                    intruders_log.append((t_id, cf))
                
                color = (0, 0, 255) if inside else (0, 255, 0)
                cv2.rectangle(out, (sx1, sy1), (sx2, sy2), color, 2)
                cv2.putText(out, f"ID:{t_id}", (sx1, max(20, sy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update Occupancy
            self.active_intruders = self.active_intruders.intersection(visible_ids)
            self.active_intruders = self.active_intruders.union(current_frame_intruders)
            count = len(self.active_intruders)
            self.count_lbl.configure(text=str(count))

            # Alerts
            if alert_triggered:
                self.status_lbl.configure(text="⚠️ ALERT ACTIVE ⚠️", text_color="red")
                for (tid, conf) in intruders_log:
                    self.logger.log_event(f"ID:{tid}", conf, "Inside Zone")
                if time.time() - self.last_snapshot_time > 5.0:
                    self.logger.save_snapshot(out)
                    self.last_snapshot_time = time.time()
            else:
                self.status_lbl.configure(text="Monitoring...", text_color="green")

            # Draw Zone
            if self.show_zone:
                pts = np.array(current_points, dtype=np.int32)
                cv2.polylines(out, [pts], True, (0, 255, 255), 2)
                for pt in pts:
                    cv2.circle(out, tuple(pt), 6, (0, 255, 0), -1)

            if self.blur_faces:
                self.privacy_filter.apply_face_blur(out)

            # Update Log Preview
            try:
                if os.path.exists(self.logger.csv_path):
                    with open(self.logger.csv_path, 'r') as f:
                        lines = f.readlines()[-3:]
                        self.log_box.delete("0.0", "end")
                        self.log_box.insert("end", ''.join(lines))
            except: pass

            # Update FPS
            self.fps_lbl.configure(text=f"FPS: {int(self.fps_counter.fps)}")

            # 3. Convert to Image for CustomTkinter
            img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk # Keep reference

        # Schedule next update (30ms approx = 33 FPS)
        self.after(30, self.update_loop)

def main():
    # ===================== INIT THREADS =====================
    producer = CameraProducer()
    consumer = AIConsumer(producer)
    producer.start()
    consumer.start()

    # ===================== INIT GUI =====================
    app = SurveillanceApp(producer, consumer)
    app.mainloop()

    # Cleanup
    producer.stop()
    consumer.stop()

if __name__ == "__main__":
    main()