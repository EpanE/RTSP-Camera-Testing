from ultralytics import YOLO
import torch
m = YOLO("yolov8n.pt")
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    m.to("cuda")
    print("YOLO device:", m.device)