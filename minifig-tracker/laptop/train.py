"""Train a YOLO detector on the Roboflow-exported green minifig dataset.

Unzip the Roboflow export (format: YOLOv8 or YOLO11) into ./dataset so that
./dataset/data.yaml exists, then run:  python train.py
"""
import argparse

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dataset/data.yaml")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--model", default="yolov8n.pt", help="pretrained starting weights")
args = parser.parse_args()

model = YOLO(args.model)
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, name="minifig")
print("Best weights: runs/detect/minifig/weights/best.pt")
