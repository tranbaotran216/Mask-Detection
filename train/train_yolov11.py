from ultralytics import YOLO

def train_yolov11():
    print(">>> Starting Training Script...")   # debug here
    model = YOLO('yolo11n.pt')
    print(">>> Model Loaded")

    model.train(
        data='./config/mask.yaml',  # path to dataset config file
        epochs=100,                   
        imgsz=640,                   
        batch=16,                    
        lr0=1e-3,                        
        name='yolov11-mask-detection'
    )

if __name__ == "__main__":
    train_yolov11()