import os
import cv2
import yaml
import random
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

class Visualization:
    """YOLO数据集可视化与分析工具类"""
    
    def __init__(self, root, data_types, n_ims, rows, cmap=None):
        """
        初始化可视化工具
        :param root: 数据集根目录
        :param data_types: 数据类型列表 ['train','val','test']
        :param n_ims: 要显示的图片数量
        :param rows: 显示图片的行数
        :param cmap: 颜色映射方式
        """
        self.root = root
        self.n_ims, self.rows = n_ims, rows
        self.cmap, self.data_types = cmap, data_types
        self.colors = ["firebrick", "darkorange", "blueviolet"]  # 不同数据集的柱状图颜色
        self.get_cls_names()  # 获取类别名称
        self.get_bboxes()     # 获取边界框数据

    def get_cls_names(self):
        """从data.yaml文件中读取类别名称"""
        with open(f"{self.root}/data.yaml", 'r', encoding='utf-8') as file: 
            data = yaml.safe_load(file)
        # 创建类别索引到名称的映射字典
        self.class_dict = {index: name for index, name in enumerate(data['names'])}

    def get_bboxes(self):
        """获取所有边界框数据和统计分析数据"""
        self.vis_datas, self.analysis_datas, self.im_paths = {}, {}, {}
        
        for data_type in self.data_types:
            all_bboxes, all_analysis_datas = [], {}
            # 获取所有图片路径
            im_paths = glob(f"{self.root}/{data_type}/images/*")

            for idx, im_path in enumerate(im_paths):
                bboxes = []
                # 构建对应的标签文件路径
                im_ext = os.path.splitext(im_path)[-1]
                lbl_path = im_path.replace(im_ext, ".txt").replace("images", "labels")
                
                if not os.path.isfile(lbl_path): continue  # 跳过没有标签的图片
                
                # 读取标签文件内容
                meta_data = open(lbl_path).readlines()
                for data in meta_data:
                    parts = data.strip().split()[:5]  # 提取前5个值[class, x_center, y_center, w, h]
                    cls_name = self.class_dict[int(parts[0])]  # 获取类别名称
                    # 转换为[class_name, x_center, y_center, w, h]格式
                    bboxes.append([cls_name] + [float(x) for x in parts[1:]])
                    # 统计各类别数量
                    all_analysis_datas[cls_name] = all_analysis_datas.get(cls_name, 0) + 1
                
                all_bboxes.append(bboxes)
            
            # 存储各数据集的结果
            self.vis_datas[data_type] = all_bboxes
            self.analysis_datas[data_type] = all_analysis_datas
            self.im_paths[data_type] = im_paths

    def plot(self, rows, cols, count, im_path, bboxes):
        """
        绘制单张图片及其边界框
        :param rows: 总行数
        :param cols: 总列数
        :param count: 当前子图位置
        :param im_path: 图片路径
        :param bboxes: 边界框列表
        :return: 下一个子图位置
        """
        plt.subplot(rows, cols, count)
        or_im = np.array(Image.open(im_path).convert("RGB"))
        height, width, _ = or_im.shape

        for bbox in bboxes:
            class_id, x_center, y_center, w, h = bbox
            # 将YOLO格式(归一化坐标)转换为像素坐标
            x_min = int((x_center - w/2) * width)   # 左上角x
            y_min = int((y_center - h/2) * height)  # 左上角y
            x_max = int((x_center + w/2) * width)   # 右下角x
            y_max = int((y_center + h/2) * height)  # 右下角y
            
            # 随机颜色绘制边界框
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(or_im, (x_min, y_min), (x_max, y_max), color, thickness=3)
        
        plt.imshow(or_im)
        plt.axis("off")
        plt.title(f"Objects: {len(bboxes)}")
        return count + 1

    def vis(self, save_name):
        """可视化指定数据集类型的图片"""
        print(f"{save_name.upper()} Data Visualization is in process...\n")
        assert self.cmap in ["rgb", "gray"], "Please choose rgb or gray cmap"
        
        cols = self.n_ims // self.rows
        count = 1
        plt.figure(figsize=(25, 20))
        
        # 随机选择要显示的图片索引
        indices = [random.randint(0, len(self.vis_datas[save_name])-1) for _ in range(self.n_ims)]
        
        for index in indices:
            if count > self.n_ims: break
            im_path = self.im_paths[save_name][index]
            bboxes = self.vis_datas[save_name][index]
            count = self.plot(self.rows, cols, count, im_path, bboxes)
        
        plt.show()

    def data_analysis(self, save_name, color):
        """绘制类别分布柱状图"""
        print("Data analysis is in process...\n")
        
        width, text_width, text_height = 0.5, 0.25, 5  # 调整柱宽和文字位置
        cls_names = list(self.analysis_datas[save_name].keys())
        counts = list(self.analysis_datas[save_name].values())

        _, ax = plt.subplots(figsize=(10, 6))  # 调整图形大小
        indices = np.arange(len(counts))
        
        # 绘制柱状图
        bars = ax.bar(indices, counts, width, color=color)
        ax.set_xlabel("Class Names", fontsize=14)
        ax.set_ylabel("Data Counts", fontsize=14)
        ax.set_title(f"{save_name.upper()} Class Distribution", fontsize=16)
        
        # 设置x轴刻度
        ax.set_xticks(indices)
        ax.set_xticklabels(cls_names, rotation=45, ha='right', fontsize=12)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+text_height,
                   str(count), ha='center', va='bottom',
                   color='royalblue', fontsize=12, fontweight='bold')
        
        plt.tight_layout()  # 自动调整布局

    def visualization(self):
        """可视化所有数据集类型"""
        [self.vis(save_name) for save_name in self.data_types]

    def analysis(self):
        """分析所有数据集类型"""
        [self.data_analysis(save_name, color) for save_name, color in zip(self.data_types, self.colors)]

def main():
    root = "fire-detect-data"
    vis = Visualization(
        root=root,
        data_types=["train", "val", "test"],
        n_ims=20,
        rows=5,
        cmap="rgb"
    )
    vis.analysis()
    vis.visualization()

if __name__ == "__main__":
    main()