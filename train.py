from ultralytics import YOLO
import os

def train_model():
    # 加载模型
    model = YOLO("yolo11n.pt")
    
    # 训练配置
    train_results = model.train(
        data='fire-detect-data/data.yaml',
        epochs=100,
        batch=-1,
        optimizer='auto',
        patience=10
    )

if __name__ == "__main__":
    train_model()