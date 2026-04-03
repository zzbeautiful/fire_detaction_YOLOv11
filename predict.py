import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

def inference_test(model_path, test_images_dir):
    """
    使用YOLOv8模型推理，为fire和smoke分配不同颜色
    
    Args:
        model_path (str): 模型路径（如 'runs/detect/train/weights/best.pt'）
        test_images_dir (str): 测试图片路径（支持单张图片或文件夹）
    """
    # 检查路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"图片路径不存在: {test_images_dir}")

    # 加载模型
    model = YOLO(model_path)

    # 执行推理
    results = model(test_images_dir, verbose=False)

    def plot_results(res, n_ims=1, rows=1, cols=1):
        """
        可视化推理结果（fire和smoke用不同颜色）
        
        Args:
            res (list): YOLO推理结果列表
            n_ims (int): 最多显示的图片数量
            rows (int): 显示的行数
            cols (int): 显示的列数
        """
        plt.figure(figsize=(10, 10))
        for idx, r in enumerate(res):
            if idx == n_ims: 
                break
            
            im = np.array(Image.open(r.path).convert("RGB"))
            
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                
                # 根据类别选择颜色（BGR格式）
                if cls_name == "fire":    # 火灾用红色
                    color = (0, 0, 255)   # 红色
                elif cls_name == "smoke":  # 烟雾用蓝色
                    color = (255, 0, 0)   # 蓝色
                else:                     # 其他类别用绿色
                    color = (0, 255, 0)   # 绿色
                
                # 绘制检测框（边框加粗）
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 3)  # 线宽=3
                
                # 添加标签（背景与边框同色）
                label = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(im, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)  # 填充背景
                cv2.putText(
                    im, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (255, 255, 255), 2, cv2.LINE_AA  # 白色文字，加粗
                )
            
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(im)
            plt.axis("off")
            plt.title(f"Detection Result {idx + 1}")
        
        plt.tight_layout()
        plt.show()

    # 显示结果
    plot_results(results, n_ims=1)

if __name__ == "__main__":
    # 示例调用（替换为你的实际路径）
    inference_test(
        model_path="runs/detect/train/weights/best.pt",
        test_images_dir="屏幕截图 2025-06-26 193943.png"
    )