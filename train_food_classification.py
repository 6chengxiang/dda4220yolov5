#!/usr/bin/env python3
"""
食物分类训练脚本
Food Classification Training Script

使用 YOLOv5 进行食物图像分类训练
"""

import argparse
import os
import sys
from pathlib import Path

# 添加 YOLOv5 路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from classify.train import run as classify_train

def main():
    parser = argparse.ArgumentParser(description='食物分类训练')
    
    # 基础参数
    parser.add_argument('--data', type=str, default='data/unimib2016.yaml', 
                       help='数据配置文件路径')
    parser.add_argument('--model', type=str, default='yolov5s-cls.pt', 
                       help='预训练模型路径')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='批处理大小')
    parser.add_argument('--imgsz', '--img', type=int, default=224, 
                       help='训练图像尺寸')
    
    # 训练参数
    parser.add_argument('--lr0', type=float, default=0.001, 
                       help='初始学习率')
    parser.add_argument('--device', default='', 
                       help='训练设备 (cpu, 0, 1, 2, 3, ...)')
    parser.add_argument('--workers', type=int, default=8, 
                       help='数据加载器工作线程数')
    
    # 保存参数
    parser.add_argument('--project', default='runs/train-cls', 
                       help='保存结果的项目目录')
    parser.add_argument('--name', default='food-classification', 
                       help='保存结果的实验名称')
    parser.add_argument('--save-period', type=int, default=10, 
                       help='每隔多少轮保存一次模型')
    
    # 数据增强
    parser.add_argument('--cache', action='store_true', 
                       help='缓存图像以加快训练速度')
    parser.add_argument('--mixup', type=float, default=0.2, 
                       help='Mixup 数据增强概率')
    parser.add_argument('--cutmix', type=float, default=0.2, 
                       help='CutMix 数据增强概率')
    
    args = parser.parse_args()
    
    print("🍕 开始食物分类训练...")
    print(f"📊 数据配置: {args.data}")
    print(f"🏗️  模型: {args.model}")
    print(f"🔄 训练轮数: {args.epochs}")
    print(f"📦 批处理大小: {args.batch_size}")
    print(f"📏 图像尺寸: {args.imgsz}")
    
    # 调用 YOLOv5 分类训练
    classify_train(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        lr0=args.lr0,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        cache=args.cache,
        mixup=args.mixup,
        cutmix=args.cutmix
    )
    
    print("✅ 训练完成！")

if __name__ == '__main__':
    main()
