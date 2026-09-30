from ultralytics import YOLO

DATA_YAML = "dataset/data.yaml"
EPOCHS = 100
IMG_SIZE = 640


def main():
    model = YOLO("yolov8n.pt")
    model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE)


if __name__ == "__main__":
    main()
