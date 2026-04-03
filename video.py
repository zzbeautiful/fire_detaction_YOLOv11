import os
from ultralytics import YOLO
import cv2
import argparse

# 禁用自动下载预训练模型
os.environ['YOLO_DOWNLOAD'] = 'False'

def process_video(input_path, output_path, model_path, conf_threshold=0.5):
    """
    处理视频进行火灾烟雾检测
    
    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径
        model_path: 模型权重路径(best.pt)
        conf_threshold: 置信度阈值(默认0.5)
    """
    try:
        # 加载自定义训练模型
        model = YOLO(model_path)
        
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {input_path}")
            
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"开始处理视频: {input_path}")
        print(f"视频信息: {width}x{height}, {fps}FPS, 总帧数: {total_frames}")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 使用模型进行预测
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                imgsz=(640, 640),  # 可根据需要调整
                verbose=False  # 禁用详细输出
            )
            
            # 在帧上绘制检测结果
            annotated_frame = results[0].plot()
            
            # 写入输出视频
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames:.1%})")
                
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"处理完成! 输出视频已保存至: {output_path}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        # 确保资源被释放
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='火灾烟雾视频检测')
    parser.add_argument('--input', type=str, required=True, help='fire.mp4')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    parser.add_argument('--model', type=str, default='best.pt', help='runs/detect/train/weights/best.pt')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值(0-1)')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件 {args.model} 不存在!")
    
    # 运行视频处理
    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        conf_threshold=args.conf
    )