#!/usr/bin/env python3
"""
食物数据集准备脚本
Food Dataset Preparation Script

帮助准备和组织食物分类数据集
"""

import os
import shutil
import argparse
from pathlib import Path
import random
from PIL import Image
import json

def create_dataset_structure(dataset_path):
    """创建标准的数据集目录结构"""
    dataset_path = Path(dataset_path)
    
    # 创建目录结构
    dirs_to_create = [
        dataset_path / 'images' / 'train',
        dataset_path / 'images' / 'val',
        dataset_path / 'images' / 'test',
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")

def organize_images_by_class(source_dir, target_dir, train_ratio=0.7, val_ratio=0.2):
    """
    将按类别组织的图像分割为训练集、验证集和测试集
    
    预期的源目录结构:
    source_dir/
    ├── 苹果/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── 香蕉/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── ...
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录结构
    create_dataset_structure(target_path)
    
    # 类别映射
    class_mapping = {}
    class_id = 0
    
    # 统计信息
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    print(f"📁 处理源目录: {source_path}")
    
    # 遍历每个类别目录
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        class_mapping[class_id] = class_name
        
        # 获取该类别的所有图像
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(class_dir.glob(ext))
            image_files.extend(class_dir.glob(ext.upper()))
        
        if not image_files:
            print(f"⚠️ 类别 '{class_name}' 中没有找到图像文件")
            continue
        
        print(f"📊 类别 '{class_name}': {len(image_files)} 张图像")
        
        # 随机打乱图像列表
        random.shuffle(image_files)
        
        # 计算分割点
        total_images = len(image_files)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        # 分割数据集
        train_images = image_files[:train_count]
        val_images = image_files[train_count:train_count + val_count]
        test_images = image_files[train_count + val_count:]
        
        # 复制图像到相应目录
        for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            if not images:
                continue
                
            for img_file in images:
                # 生成新的文件名（包含类别ID）
                new_filename = f"{class_id:02d}_{class_name}_{img_file.stem}{img_file.suffix}"
                target_file = target_path / 'images' / split / new_filename
                
                try:
                    # 验证图像文件
                    with Image.open(img_file) as img:
                        img.verify()
                    
                    # 复制文件
                    shutil.copy2(img_file, target_file)
                    stats[split] += 1
                    
                except Exception as e:
                    print(f"❌ 跳过损坏的图像: {img_file} ({e})")
        
        class_id += 1
    
    # 保存类别映射
    mapping_file = target_path / 'class_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"\n📈 数据集统计:")
    print(f"训练集: {stats['train']} 张图像")
    print(f"验证集: {stats['val']} 张图像") 
    print(f"测试集: {stats['test']} 张图像")
    print(f"总计: {sum(stats.values())} 张图像")
    print(f"类别数: {class_id}")
    print(f"💾 类别映射已保存到: {mapping_file}")

def download_food101_sample():
    """下载 Food-101 数据集的示例"""
    print("📥 Food-101 数据集下载指南:")
    print("1. 访问: https://www.vision.ee.ethz.ch/datasets_extra/food-101/")
    print("2. 下载 food-101.tar.gz 文件")
    print("3. 解压到 ../datasets/ 目录")
    print("4. 运行本脚本整理数据集结构")
    
    print("\n💡 或者，您可以:")
    print("1. 使用自己收集的食物图像")
    print("2. 从网络爬取食物图像")
    print("3. 使用 Google Images, Unsplash 等免费图像资源")

def create_sample_dataset(target_dir, num_classes=5, images_per_class=50):
    """创建示例数据集结构（用于测试）"""
    target_path = Path(target_dir)
    create_dataset_structure(target_path)
    
    sample_classes = ['苹果', '香蕉', '披萨', '汉堡', '蛋糕'][:num_classes]
    
    print(f"🧪 创建示例数据集结构")
    print(f"📍 目标目录: {target_path}")
    print(f"📊 类别数: {num_classes}")
    print(f"🖼️ 每类图像数: {images_per_class}")
    
    for class_id, class_name in enumerate(sample_classes):
        for split in ['train', 'val', 'test']:
            split_dir = target_path / 'images' / split
            
            # 根据分割比例分配图像数量
            if split == 'train':
                count = int(images_per_class * 0.7)
            elif split == 'val':
                count = int(images_per_class * 0.2)
            else:
                count = int(images_per_class * 0.1)
            
            for i in range(count):
                # 创建占位符文件
                placeholder_file = split_dir / f"{class_id:02d}_{class_name}_{i:03d}.txt"
                with open(placeholder_file, 'w', encoding='utf-8') as f:
                    f.write(f"这是 {class_name} 类别的第 {i+1} 张图像的占位符\n")
                    f.write(f"请替换为实际的 {class_name} 图像文件\n")
    
    print("✅ 示例数据集结构创建完成")
    print("📝 请将占位符文件替换为实际的图像文件")

def main():
    parser = argparse.ArgumentParser(description='食物数据集准备工具')
    parser.add_argument('--mode', choices=['organize', 'download', 'sample'], 
                       required=True, help='运行模式')
    parser.add_argument('--source', type=str, 
                       help='源数据目录（organize 模式）')
    parser.add_argument('--target', type=str, 
                       help='目标数据目录')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='示例类别数（sample 模式）')
    parser.add_argument('--images-per-class', type=int, default=50,
                       help='每类图像数（sample 模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'organize':
        if not args.source or not args.target:
            print("❌ organize 模式需要指定 --source 和 --target 参数")
            return
        organize_images_by_class(args.source, args.target, 
                               args.train_ratio, args.val_ratio)
    
    elif args.mode == 'download':
        download_food101_sample()
    
    elif args.mode == 'sample':
        if not args.target:
            args.target = '../datasets/food-sample'
        create_sample_dataset(args.target, args.num_classes, args.images_per_class)

if __name__ == '__main__':
    main()
