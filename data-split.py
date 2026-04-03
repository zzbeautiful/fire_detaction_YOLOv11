import os
import random
import shutil
from sklearn.model_selection import train_test_split


def split_yolo_dataset(
    dataset_path, 
    output_path, 
    train_ratio=0.7, 
    val_ratio=0.2, 
    test_ratio=0.1,
    max_samples=None  # 新增：限制总样本数
):
    """
    划分YOLO数据集（可选择限制总样本数）

    参数:
        dataset_path: 原始数据集路径 (包含images和labels文件夹)
        output_path: 输出路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        max_samples: 最大样本数（若为None，则使用全部数据）
    """
    # 确保比例总和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "比例总和必须为1"

    # 创建输出目录结构
    os.makedirs(os.path.join(output_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test', 'labels'), exist_ok=True)

    # 获取所有图片文件名（不带扩展名）
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    base_names = [os.path.splitext(f)[0] for f in image_files]

    # 随机打乱
    random.shuffle(base_names)

    # 如果指定了最大样本数，则截取前 max_samples 个
    if max_samples is not None and max_samples < len(base_names):
        base_names = base_names[:max_samples]

    # 划分数据集
    train_val, test = train_test_split(base_names, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

    # 复制文件到对应目录
    def copy_files(names, subset):
        for name in names:
            # 查找图片文件（考虑不同扩展名）
            for ext in ['.jpg', '.jpeg', '.png']:
                src_img = os.path.join(images_dir, name + ext)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(output_path, subset, 'images', name + ext))
                    break

            # 复制标签文件
            src_label = os.path.join(labels_dir, name + '.txt')
            if os.path.exists(src_label):
                shutil.copy(src_label, os.path.join(output_path, subset, 'labels', name + '.txt'))

    copy_files(train, 'train')
    copy_files(val, 'val')
    copy_files(test, 'test')

    print(f"数据集划分完成: 训练集 {len(train)} 个, 验证集 {len(val)} 个, 测试集 {len(test)} 个")


# 使用示例
if __name__ == "__main__":
    # 原始数据集路径 (包含images和labels文件夹)
    dataset_path = "data/fire-data"
    # 输出路径
    output_path = "fire-detect-data"

    # 划分比例 (训练集70%, 验证集20%, 测试集10%)，并限制总样本数为1500
    split_yolo_dataset(
        dataset_path, 
        output_path, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        test_ratio=0.1,
        max_samples=1500  # 只取1500张图片
    )