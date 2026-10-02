# Fine-tunes a YOLOv8 detector to find the green LEGO minifigure.
#
# Dataset: dataset/ (Roboflow YOLOv8 export, class "green_lego_minifigure", 31 train / 9 valid images).
# Training progress and the final weights land under
# runs/detect/green_minifig/weights/best.pt

from pathlib import Path

import torch
from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "dataset" / "data.yaml"
EPOCHS = 100  # small dataset (31 images) benefits from more epochs than the usual 50
IMAGE_SIZE = 640

# Use the fastest device available: NVIDIA GPU, Apple Silicon GPU, or CPU.
if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Training on device={DEVICE!r}")

# Start from COCO-pretrained weights and fine-tune -- much faster than
# training from scratch, and works well with a few dozen images.
model = YOLO("yolov8n.pt")
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    name="green_minifig",
    exist_ok=True,  # reuse runs/detect/green_minifig instead of making green_minifig2, 3, ...
    device=DEVICE,
)

print("Done. Weights saved to runs/detect/green_minifig/weights/best.pt")
