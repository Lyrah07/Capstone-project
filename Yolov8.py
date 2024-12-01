from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

result = model(source="speed.mp4", show=True, conf=0.3, save=True)