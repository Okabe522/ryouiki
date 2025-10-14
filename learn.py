from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    results = model.train(data="dataset.yaml", epochs=300)