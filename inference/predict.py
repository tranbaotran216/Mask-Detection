from ultralytics import YOLO
import sys

def predict(image_path):
    model = YOLO('runs/detect/yolov11-mask-detection/weights/best.pt')  # Load the trained YOLOv11 model

    model.predict(source=image_path, imgsz=640, conf=0.4, save=True)  # Perform prediction with a confidence threshold of 0.25

if __name__ == "__main__":
    predict(sys.argv[1])  