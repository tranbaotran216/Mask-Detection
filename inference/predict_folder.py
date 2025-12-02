from ultralytics import YOLO
import sys

def predict_folder(path):
    model = YOLO("runs/detect/train/weights/best.pt")
    model.predict(source=path, imgsz=640, conf=0.4, save=True)

if __name__ == "__main__":
    predict_folder(sys.argv[1])
