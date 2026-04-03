from ultralytics import YOLO

def train_model():
    # 加载模型
    model = YOLO("yolov8n.pt")
    
    # 配置
    results = model.train(
        data='fire-detect-data/data.yaml',
        epochs=100,
        batch=-1,           
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        
        
    )

if __name__ == "__main__":
    train_model()