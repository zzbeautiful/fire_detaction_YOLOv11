import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas, StringVar, OptionMenu,messagebox,Entry
from PIL import Image, ImageTk
import numpy as np
import os
import pygame
import matplotlib.pyplot as plt
from ultralytics import YOLO

class FireSmokeDetectionApp:
    def __init__(self, model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.model = YOLO(model_path)

        self.root = tk.Tk()
        self.root.title("火御智眼系统")

        # 设置窗口大小并允许调整
        self.root.geometry("1200x700")

        # 加载背景图片
        self.set_background(r"88.jpg")

        # 创建主框架，使用网格布局更灵活
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # --- 左侧视频显示区域 ---
        self.video_frame = tk.LabelFrame(self.main_frame, text="实时监控画面", font=('Arial', 12, 'bold'))
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 创建画布用于显示视频，添加边框
        self.canvas = Canvas(self.video_frame, width=100, height=550, bg="white")
        self.info_frame = tk.Frame(self.video_frame)
        self.info_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = Label(self.info_frame, text="状态: 就绪", font=('Arial', 10), anchor=tk.W)
        self.status_label.pack(side=tk.LEFT)

        self.detection_info_label = Label(self.info_frame, text="", font=('Arial', 10), fg='red', anchor=tk.E)
        self.detection_info_label.pack(side=tk.RIGHT)

        # --- 右侧控制面板 ---
        self.control_panel = tk.LabelFrame(self.main_frame, text="控制面板", font=('Arial', 12, 'bold'), width=200,
                                           height=600)
        self.control_panel.pack_propagate(False)
        self.control_panel.configure(bg="white")
        self.canvas.pack(fill=tk.X, expand=False)

        # 状态信息显示区（在视频下方）
        self.control_panel.pack(side=tk.RIGHT, fill=tk.X)

        # 加载视频按钮
        self.load_video_button = Button(self.control_panel, text="加载视频", command=self.load_video, padx=10, pady=5)
        self.load_video_button.pack(fill=tk.X, pady=(0, 10))

        # 加载图片按钮
        self.load_photo_button = Button(self.control_panel, text="加载图片", command=self.load_image, padx=10, pady=5)
        self.load_photo_button.pack(fill=tk.X, pady=(0, 10))

        # 加载文件按钮
        self.load_photo_button = Button(self.control_panel, text="批量图片", command=self.load_images_batch, padx=10,
                                        pady=5)
        self.load_photo_button.pack(fill=tk.X, pady=(0, 10))
        self.next_image_button = Button(self.control_panel, text="下一张图片", command=self.next_image,
                                        state=tk.DISABLED, padx=10, pady=5)
        self.next_image_button.pack(fill=tk.X, pady=(0, 10))
        self.previous_image_button = Button(self.control_panel, text="上一张图片", command=self.prev_image,
                                            state=tk.DISABLED, padx=10, pady=5)
        self.previous_image_button.pack(fill=tk.X, pady=(0, 10))

        # 开始检测按钮
        self.start_detection_button = Button(self.control_panel, text="开始检测", command=self.start_detection,
                                             state=tk.DISABLED, padx=10, pady=5)
        self.start_detection_button.pack(fill=tk.X, pady=(0, 10))

        # 停止检测按钮
        self.stop_detection_button = Button(self.control_panel, text="停止检测", command=self.stop_detection,
                                            state=tk.DISABLED, padx=10, pady=5)
        self.stop_detection_button.pack(fill=tk.X, pady=(0, 10))

        # 设置输入置信度阈值文本框
        self.entry = Entry(self.control_panel)
        self.entry.insert(0, "请输入置信度阈值")
        # 设置默认文本为灰色
        self.entry.config(fg="grey")
        self.entry.pack(fill=tk.X)

        # 绑定焦点事件
        self.entry.bind("<FocusIn>", self.clear_default_text)
        self.entry.bind("<FocusOut>", self.restore_default_text)

        self.input_button = Button(self.control_panel, text="确认", command=self.get_confidence_threshold)
        self.input_button.pack(fill=tk.X)

        # 检测算法选择
        self.algorithm_var = StringVar(self.root)
        self.algorithm_var.set("火焰-烟雾检测")
        self.algorithm_menu = OptionMenu(self.control_panel, self.algorithm_var, "火焰-烟雾检测")
        self.algorithm_menu.pack(fill=tk.X, pady=5)

        # 状态标签
        self.status_label = Label(self.control_panel, text="状态: 就绪", anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=5)

        self.control_panel = tk.Frame(self.root)
        self.control_panel.pack(pady=10)

        # 初始化Pygame
        pygame.init()
        pygame.mixer.init()

        # 尝试加载音频文件
        self.alarmsound = 'fire_alarm.wav'
        if os.path.exists(self.alarmsound):
            pygame.mixer.music.load(self.alarmsound)
        else:
            print(f"警告: 音频文件 {self.alarmsound} 未找到")

        # 初始化变量
        self.video_path = None
        self.image_path = None
        self.folder_path = None
        self.cap = None
        self.image = None
        self.imagefiles = []
        self.is_detecting = False
        self.bg_image = None
        self.conf_threshold = 0.5

    def clear_default_text(self, event):
        if self.entry.get() == "请输入置信度阈值":
            self.entry.delete(0, tk.END)
            self.entry.config(fg="black")  # 设置正常文本颜色

    def restore_default_text(self, event):
        if not self.entry.get():
            self.entry.insert(0, "请输入置信度阈值")
            self.entry.config(fg="grey")  # 设置提示文本颜色为灰色
    
    def get_confidence_threshold(self):
        # 获取用户输入的置信度阈值
        threshold = self.entry.get()
        try:
            # 将输入转换为浮点数
            self.conf_threshold = float(threshold)
        except ValueError:
            alarm_window = tk.Toplevel(self.root)
            alarm_window.title("输入非法")
            alarm_window.geometry("300x150")
            alarm_label = tk.Label(
                alarm_window,
                text="输入非法！",
                font=("Arial", 20),
                bg='white',
                fg='black'
            )
            alarm_label.pack(pady=20)

    def set_background(self, image_path):
        # 加载图片
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)

        # 创建一个标签，并设置图片为背景
        label = tk.Label(self.root, image=photo)
        label.image = photo  # 保持对图片的引用，防止被垃圾回收
        label.place(x=0, y=0, relwidth=1, relheight=1)

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("视频文件", ".mp4;.avi;.mov")])
        if self.video_path:
            try:
                self.imagefiles = []
                self.image = None
                self.stop_detection()  # 先停止任何可能的检测
                self.cap = cv2.VideoCapture(self.video_path)
                if self.cap.isOpened():
                    self.status_label.config(text=f"状态: 视频已加载 - {os.path.basename(self.video_path)}")
                    self.start_detection_button.config(state=tk.NORMAL)
                else:
                    self.status_label.config(text="状态: 无法打开视频文件")
                    self.cap = None
            except Exception as e:
                self.status_label.config(text=f"状态: 加载视频出错 - {e}")
                self.cap = None
        else:
            self.status_label.config(text="状态: 未选择视频文件")

    def display_frame(self):
        if self.is_detecting and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 处理帧（例如检测）
                # 转换帧格式并显示
                frame = self.detect_fire_smoke(frame,self.conf_threshold)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                

                # 调整帧大小
                frame_width = self.video_frame.winfo_width()
                frame_height = self.video_frame.winfo_height()
                resized_image = image.resize((frame_width, frame_height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image=resized_image)

                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.image = photo  # 保持引用
                # 继续显示下一帧
                self.root.after(30, self.display_frame)
            else:
                # 视频结束
                self.status_label.config(text="状态: 视频播放结束")
                self.stop_detection()  # 自动停止检测

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.image_path:
            try:
                self.cap = None
                self.imagefiles = []
                self.stop_detection()
                self.image = cv2.imread(self.image_path)
                if self.image is not None:
                    self.start_detection_button.config(state=tk.NORMAL)
                    self.status_label.config(text=f"状态: 已加载图片 - {os.path.basename(self.image_path)}")
                    self.display_image()
                else:
                    self.status_label.config(text="状态: 图片无法打开")
                    self.image = None
                    self.stop_detection()  # 确保检测停止
            except Exception as e:
                self.status_label.config(text=f"状态: 加载图片失败 - {str(e)}")
                self.image = None
                self.stop_detection()  # 确保检测停止

    def display_image(self):
        if self.image is not None:
            try:
                if self.is_detecting:
                    annotated_image = self.detect_fire_smoke(self.image,self.conf_threshold)
                else:
                    annotated_image = self.image.copy()
                
                # 调整图片大小
                frame_width = self.video_frame.winfo_width()
                frame_height = self.video_frame.winfo_height()
                resized_image = cv2.resize(annotated_image, (frame_width, frame_height))
                
                # 将OpenCV的BGR格式转换为RGB格式
                image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                
                # 将NumPy数组转换为Tkinter可显示的图像
                img = Image.fromarray(image_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.canvas.delete("all")  # 清除之前的图像
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk  # 保持引用
                
            except Exception as e:
                self.status_label.config(text=f"状态: 显示图片失败 - {str(e)}")
                self.image = None
                self.stop_detection()  # 确保检测停止
        else:
            self.status_label.config(text="状态: 无图片源")
            self.stop_detection()  # 确保检测停止


    def load_images_batch(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.cap = None
            self.image = None
            self.stop_detection()
            self.imagefiles = [f for f in os.listdir(self.folder_path) if
                               f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if not self.imagefiles:
                messagebox.showinfo("提示", "所选文件夹中没有图片文件")
                return
            self.status_label.config(text=f"状态: 已加载文件夹 - {os.path.basename(self.folder_path)}")
            self.start_detection_button.config(state=tk.NORMAL)
            self.currentimageindex = 0
            self.load_and_display_image(self.currentimageindex)

    def load_and_display_image(self, index):
        if 0 <= index < len(self.imagefiles):
            imagepath = os.path.join(self.folder_path, self.imagefiles[index])
            try:
                # 读取图像
                image = cv2.imread(imagepath)
                if image is not None:
                    self.image = image
                    
                    # 执行检测并获取带标注框的图像
                    if self.is_detecting:
                        annotated_image = self.detect_fire_smoke(image,self.conf_threshold)
                    else:
                        annotated_image = image.copy()
                    
                    # 调整图片大小
                    frame_width = self.video_frame.winfo_width()
                    frame_height = self.video_frame.winfo_height()
                    resized_image = cv2.resize(annotated_image, (frame_width, frame_height))
                    
                    # 将OpenCV的BGR格式转换为RGB格式
                    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                    
                    # 将NumPy数组转换为Tkinter可显示的图像
                    img = Image.fromarray(image_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.canvas.delete("all")  # 清除之前的图像
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                    self.canvas.imgtk = imgtk  # 保持引用
                    
                    self.status_label.config(text=f"状态: 显示图片 {index + 1}/{len(self.imagefiles)}")
                else:
                    self.status_label.config(text=f"状态: 无法加载图片 {self.imagefiles[index]}")
            except Exception as e:
                self.status_label.config(text=f"状态: 加载图片失败 - {str(e)}")
        else:
            self.status_label.config(text="状态: 所有图片已处理")

    def next_image(self):
        if self.imagefiles and self.is_detecting:
            self.currentimageindex = (self.currentimageindex + 1) % len(self.imagefiles)
            self.load_and_display_image(self.currentimageindex)

    def prev_image(self):
        if self.imagefiles and self.is_detecting:
            self.currentimageindex = (self.currentimageindex - 1) % len(self.imagefiles)
            self.load_and_display_image(self.currentimageindex)

    def start_detection(self):
        if (self.cap and self.cap.isOpened()) or (self.image is not None) or (self.imagefiles):
            self.is_detecting = True
            self.start_detection_button.config(state=tk.DISABLED)
            self.stop_detection_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"状态: 正在检测 - {self.algorithm_var.get()}")
            if self.image is not None:
                self.display_image()
            if self.cap and self.cap.isOpened():
                self.display_frame()
            if self.imagefiles:
                self.next_image_button.config(state=tk.NORMAL)
                self.previous_image_button.config(state=tk.NORMAL)
                self.load_and_display_image(self.currentimageindex)

    def stop_detection(self):
        self.is_detecting = False
        self.start_detection_button.config(state=tk.NORMAL)
        self.stop_detection_button.config(state=tk.DISABLED)
        self.next_image_button.config(state=tk.DISABLED)
        self.previous_image_button.config(state=tk.DISABLED)
        self.status_label.config(text="状态: 检测已停止")
        if self.image is not None:
            self.display_image()
        if self.imagefiles:
            self.load_and_display_image(self.currentimageindex)
        

    def detect_fire_smoke(self, frame, conf_threshold):
        try:
            # 检查模型是否已加载
            if not hasattr(self, 'model') or self.model is None:
                raise RuntimeError("模型未加载，请先初始化模型")

            # 转换图像格式
            if isinstance(frame, str):  # 如果是文件路径
                frame = cv2.imread(frame)
            elif isinstance(frame, Image.Image):  # 如果是PIL图像
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 执行推理，设置置信度阈值
            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False
            )
            
            # 初始化检测结果
            fire_detected = False
            smoke_detected = False
            
            # 处理检测结果并绘制框
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    # 记录检测结果
                    if cls_name == "fire":
                        fire_detected = True
                        color = (0, 0, 255)  # 红色框
                    elif cls_name == "smoke":
                        smoke_detected = True
                        color = (255, 0, 0)  # 蓝色框
                    else:
                        color = (0, 255, 0)  # 绿色框
                    
                    # 绘制检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加标签
                    label = f"{cls_name} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                               (255, 255, 255), 1, cv2.LINE_AA)

            # 触发警报
            if self.is_detecting:
                if fire_detected:
                    self.show_fire_alarm()
                elif smoke_detected:
                    self.show_smoke_alarm()
                
            # 返回带标注框的图像
            return frame

        except Exception as e:
            print(f"检测过程中发生错误: {str(e)}")
            return frame

    def show_fire_alarm(self):
        # 创建报警窗口
        alarm_window = tk.Toplevel(self.root)
        alarm_window.title("火灾报警")

        window_width = 300
        window_height = 150
        screen_width = alarm_window.winfo_screenwidth()
        screen_height = alarm_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        alarm_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # 设置窗口背景为红色
        alarm_window.configure(bg='red')

        # 添加报警文本
        alarm_label = tk.Label(
            alarm_window,
            text="火焰报警！",
            font=("Arial", 16),
            bg='red',
            fg='white'
        )
        alarm_label.pack(pady=20)

        # 添加“我知道了”按钮
        ok_button = tk.Button(
            alarm_window,
            text="我知道了",
            command=lambda: self.stop_alarm(alarm_window)
        )
        ok_button.pack(pady=10)

        # 设置窗口行为
        alarm_window.transient(self.root)
        alarm_window.grab_set()
        alarm_window.focus_set()

        # 开始循环播放音频
        if os.path.exists(self.alarmsound):
            pygame.mixer.music.play(-1)  # -1 表示无限循环

    def show_smoke_alarm(self):
        # 创建报警窗口
        alarm_window = tk.Toplevel(self.root)
        alarm_window.title("烟雾报警")

        window_width = 300
        window_height = 150
        screen_width = alarm_window.winfo_screenwidth()
        screen_height = alarm_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        alarm_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # 设置窗口背景为黄色
        alarm_window.configure(bg='yellow')

        # 添加报警文本
        alarm_label = tk.Label(
            alarm_window,
            text="烟雾报警！",
            font=("Arial", 16),
            bg='yellow',
            fg='black'
        )
        alarm_label.pack(pady=20)

        # 添加“我知道了”按钮
        ok_button = tk.Button(
            alarm_window,
            text="我知道了",
            command=lambda: alarm_window.destroy()
        )
        ok_button.pack(pady=10)

        # 设置窗口行为
        alarm_window.transient(self.root)
        alarm_window.grab_set()
        alarm_window.focus_set()

    def stop_alarm(self, window):
        # 停止音频播放
        pygame.mixer.music.stop()
        # 关闭窗口
        window.destroy()

if __name__ == "__main__":
    app = FireSmokeDetectionApp("runs/detect/train/weights/best.pt")
    app.root.mainloop()
    pygame.quit()