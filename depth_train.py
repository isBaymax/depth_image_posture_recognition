from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')

results = model.train(data='/proj/pan_ws/src/posture_recognition/scripts/depth-pose.v4i.yolov8/data.yaml', epochs=500, imgsz=640)