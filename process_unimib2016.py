#!/usr/bin/env python3
"""
UNIMIB2016 数据集处理脚本
UNIMIB2016 Dataset Processing Script

专门用于处理从 Kaggle 下载的 UNIMIB2016 食物数据集
"""

import os
import shutil
import argparse
from pathlib import Path
import random
from PIL import Image
import json
import zipfile

# UNIMIB2016 类别映射
UNIMIB2016_CLASSES = {
    'bread': 0, 'pasta_with_tomato_sauce': 1, 'pasta_with_meat_sauce': 2,
    'pasta_with_clam_sauce': 3, 'pasta_with_pesto_sauce': 4, 'pasta_with_oil_and_garlic': 5,
    'gnocchi_with_tomato_sauce': 6, 'gnocchi_with_pesto_sauce': 7, 'risotto': 8,
    'polenta': 9, 'pizza_margherita': 10, 'pizza_four_cheese': 11,
    'pizza_with_vegetables': 12, 'pizza_with_ham': 13, 'focaccia': 14,
    'fagottini_peas_ham': 15, 'tagliatelle_with_sauce': 16, 'meatballs_with_tomato_sauce': 17,
    'baked_pasta': 18, 'pasta_salad': 19, 'minestrone': 20,
    'fish_soup': 21, 'vegetable_soup': 22, 'tripe': 23,
    'pasta_e_fagioli': 24, 'ribollita': 25, 'grilled_fish': 26,
    'mixed_fried_fish': 27, 'battered_fish': 28, 'roasted_chicken': 29,
    'chicken_breast': 30, 'chicken_wings': 31, 'fried_chicken': 32,
    'veal_cutlet': 33, 'grilled_beef': 34, 'beef_stew': 35,
    'roasted_beef': 36, 'hamburger': 37, 'pork_cutlet': 38,
    'pork_loin': 39, 'roasted_pork': 40, 'raw_ham': 41,
    'cooked_ham': 42, 'fried_egg': 43, 'scrambled_egg': 44,
    'boiled_egg': 45, 'omelette': 46, 'cheese': 47,
    'mozzarella': 48, 'cottage_cheese': 49, 'yogurt': 50,
    'apple': 51, 'banana': 52, 'orange': 53,
    'strawberry': 54, 'grapes': 55, 'pear': 56,
    'peach': 57, 'lemon': 58, 'kiwi': 59,
    'pineapple': 60, 'mixed_salad': 61, 'carrots': 62,
    'green_beans': 63, 'spinach': 64, 'tomatoes': 65,
    'potatoes': 66, 'french_fries': 67, 'roasted_potatoes': 68,
    'boiled_potatoes': 69, 'potato_gnocchi': 70, 'wine': 71, 'water': 72
}

def extract_kaggle_dataset(zip_path, extract_dir):
    """解压从 Kaggle 下载的数据集"""
    print(f"📦 解压数据集: {zip_path}")
    
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print(f"✅ 解压完成到: {extract_path}")
    return extract_path

def process_unimib2016(source_dir, target_dir, train_ratio=0.7, val_ratio=0.2):
    """
    处理 UNIMIB2016 数据集
    
    预期的源目录结构可能是：
    source_dir/
    ├── pre8/
    │   ├── bread/
    │   ├── pasta_with_tomato_sauce/
    │   └── ...
    或者直接：
    source_dir/
    ├── bread/
    ├── pasta_with_tomato_sauce/
    └── ...
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录结构
    target_dirs = [
        target_path / 'images' / 'train',
        target_path / 'images' / 'val',
        target_path / 'images' / 'test'
    ]
    
    for dir_path in target_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 处理 UNIMIB2016 数据集")
    print(f"🔍 源目录: {source_path}")
    print(f"📦 目标目录: {target_path}")
    
    # 查找实际的数据目录
    data_dirs = []
    
    # 检查是否有 pre8 子目录
    if (source_path / 'pre8').exists():
        data_root = source_path / 'pre8'
        print(f"📂 找到 pre8 目录，使用: {data_root}")
    else:
        data_root = source_path
        print(f"📂 直接使用源目录: {data_root}")
    
    # 查找所有类别目录
    for item in data_root.iterdir():
        if item.is_dir() and item.name in UNIMIB2016_CLASSES:
            data_dirs.append(item)
    
    if not data_dirs:
        print("❌ 未找到有效的食物类别目录")
        print("🔍 请检查数据集结构是否正确")
        return
    
    print(f"✅ 找到 {len(data_dirs)} 个食物类别")
    
    # 统计信息
    stats = {'train': 0, 'val': 0, 'test': 0}
    class_stats = {}
    
    # 处理每个类别
    for class_dir in data_dirs:
        class_name = class_dir.name
        class_id = UNIMIB2016_CLASSES[class_name]
        
        # 获取该类别的所有图像
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(class_dir.glob(ext))
        
        if not image_files:
            print(f"⚠️ 类别 '{class_name}' 中没有找到图像文件")
            continue
        
        print(f"📊 处理类别 '{class_name}': {len(image_files)} 张图像")
        
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
        
        class_stats[class_name] = {
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images),
            'total': len(image_files)
        }
        
        # 复制图像到相应目录
        for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            if not images:
                continue
                
            for i, img_file in enumerate(images):
                # 生成新的文件名
                new_filename = f"{class_id:02d}_{class_name}_{i:04d}{img_file.suffix}"
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
    
    # 保存详细统计信息
    stats_file = target_path / 'dataset_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_stats': stats,
            'class_stats': class_stats,
            'dataset_info': {
                'name': 'UNIMIB2016',
                'total_classes': len(class_stats),
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': 1 - train_ratio - val_ratio
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📈 UNIMIB2016 数据集处理完成:")
    print(f"训练集: {stats['train']} 张图像")
    print(f"验证集: {stats['val']} 张图像")
    print(f"测试集: {stats['test']} 张图像")
    print(f"总计: {sum(stats.values())} 张图像")
    print(f"类别数: {len(class_stats)}")
    print(f"💾 详细统计已保存到: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description='UNIMIB2016 数据集处理工具')
    parser.add_argument('--source', type=str, required=True,
                       help='数据集源目录或zip文件路径')
    parser.add_argument('--target', type=str, default='../datasets/unimib2016',
                       help='目标数据目录')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--extract', action='store_true',
                       help='如果源是zip文件，先解压')
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    
    # 如果是zip文件，先解压
    if args.extract and source_path.suffix.lower() == '.zip':
        extract_dir = source_path.parent / 'extracted'
        extracted_path = extract_kaggle_dataset(source_path, extract_dir)
        source_path = extracted_path
    
    # 处理数据集
    process_unimib2016(source_path, args.target, args.train_ratio, args.val_ratio)
    
    print(f"\n🎉 数据集处理完成！")
    print(f"📁 数据集位置: {args.target}")
    print(f"⚡ 现在可以开始训练了:")
    print(f"   python train_food_classification.py --data data/unimib2016.yaml --model yolov5s-cls.pt --epochs 50")

if __name__ == '__main__':
    main()
