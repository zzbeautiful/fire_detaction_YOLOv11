import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

video_path = 'D:\\FireDetect\\vi.mp4'
best_model = YOLO("runs/detect/train/weights/best.pt") 

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
text_position = (40, 80)
font_color = (255, 255, 255)    
background_color = (0, 0, 255) 

damage_deque = deque(maxlen=10)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("无法打开视频文件")

# 获取视频的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('vi.avi', fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 模型进行预测
    results = best_model.predict(source=frame, imgsz=640, conf=0.25)
    processed_frame = results[0].plot(boxes=False)
    
    percentage_damage = 0 
    
    if results[0].masks is not None:
        total_area = 0
        masks = results[0].masks.data.cpu().numpy()
        image_area = frame.shape[0] * frame.shape[1]  
        for mask in masks:
            binary_mask = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:  # 检查是否有轮廓
                total_area += cv2.contourArea(contours[0])
        
        percentage_damage = (total_area / image_area) * 100

    damage_deque.append(percentage_damage)
    smoothed_percentage_damage = sum(damage_deque) / len(damage_deque)
        
    # 绘制背景条
    cv2.line(processed_frame, (text_position[0], text_position[1] - 10),
             (text_position[0] + 350, text_position[1] - 10), background_color, 40)
    
    # 显示文字（修复：添加文本内容）
    cv2.putText(processed_frame, 
                f'Road Damage: {smoothed_percentage_damage:.2f}%',  # 添加文本
                text_position, font, font_scale, font_color, 2, cv2.LINE_AA)         
    
    out.write(processed_frame)

cap.release()
out.release()
print("视频处理完成！")