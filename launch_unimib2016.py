#!/usr/bin/env python3
"""
UNIMIB2016 一键启动脚本
UNIMIB2016 One-Click Launch Script

自动完成数据集下载、处理、训练和测试的完整流程
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n🔄 {description}")
    print(f"💻 执行命令: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} - 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败: {e}")
        return False

def check_requirements():
    """检查环境要求"""
    print("🔍 检查环境要求...")
    
    # 检查 Python 版本
    if sys.version_info < (3, 8):
        print("❌ Python 版本需要 >= 3.8")
        return False
    
    # 检查关键依赖
    try:
        import torch
        import PIL
        print(f"✅ PyTorch 版本: {torch.__version__}")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return False
    
    return True

def setup_environment():
    """设置环境"""
    print("📦 安装依赖包...")
    
    commands = [
        "pip install -r requirements.txt",
        "pip install opencv-python Pillow pandas"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"安装依赖: {cmd}"):
            return False
    
    return True

def process_dataset(dataset_path, target_dir):
    """处理数据集"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ 数据集文件不存在: {dataset_path}")
        return False
    
    if dataset_path.suffix.lower() == '.zip':
        cmd = f"python process_unimib2016.py --source \"{dataset_path}\" --target \"{target_dir}\" --extract"
    else:
        cmd = f"python process_unimib2016.py --source \"{dataset_path}\" --target \"{target_dir}\""
    
    return run_command(cmd, "处理 UNIMIB2016 数据集")

def train_model(model_size='s', epochs=50, batch_size=32):
    """训练模型"""
    model_name = f"yolov5{model_size}-cls.pt"
    
    cmd = (f"python train_food_classification.py "
           f"--model {model_name} "
           f"--epochs {epochs} "
           f"--batch-size {batch_size} "
           f"--name unimib2016-{model_size}")
    
    return run_command(cmd, f"训练 YOLOv5{model_size} 模型")

def test_model(model_path, test_image=None):
    """测试模型"""
    if test_image and Path(test_image).exists():
        cmd = f"python predict_food_classification.py --weights \"{model_path}\" --source \"{test_image}\""
        return run_command(cmd, "测试模型预测")
    else:
        # 使用验证集进行测试
        cmd = f"python classify/val.py --weights \"{model_path}\" --data data/unimib2016.yaml"
        return run_command(cmd, "验证模型性能")

def main():
    parser = argparse.ArgumentParser(description='UNIMIB2016 一键启动脚本')
    parser.add_argument('--dataset', type=str, required=True,
                       help='UNIMIB2016 数据集路径（zip文件或解压后的目录）')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], 
                       default='s', help='模型大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('--test-image', type=str,
                       help='测试图像路径（可选）')
    parser.add_argument('--skip-setup', action='store_true',
                       help='跳过环境设置')
    parser.add_argument('--skip-processing', action='store_true',
                       help='跳过数据处理（如果已经处理过）')
    parser.add_argument('--skip-training', action='store_true',
                       help='跳过训练（仅测试）')
    
    args = parser.parse_args()
    
    print("🍕 UNIMIB2016 食物分类项目一键启动")
    print("=" * 50)
    
    # 检查环境
    if not check_requirements():
        print("❌ 环境检查失败，请先解决依赖问题")
        return
    
    # 设置环境
    if not args.skip_setup:
        if not setup_environment():
            print("❌ 环境设置失败")
            return
    
    # 处理数据集
    target_dataset_dir = "../datasets/unimib2016"
    if not args.skip_processing:
        if not process_dataset(args.dataset, target_dataset_dir):
            print("❌ 数据集处理失败")
            return
    
    # 训练模型
    if not args.skip_training:
        if not train_model(args.model_size, args.epochs, args.batch_size):
            print("❌ 模型训练失败")
            return
    
    # 测试模型
    model_path = f"runs/train-cls/unimib2016-{args.model_size}/weights/best.pt"
    if Path(model_path).exists():
        test_model(model_path, args.test_image)
    else:
        print(f"⚠️ 模型文件不存在: {model_path}")
    
    print("\n🎉 UNIMIB2016 食物分类项目启动完成！")
    print("📊 您可以查看以下结果:")
    print(f"   - 训练日志: runs/train-cls/unimib2016-{args.model_size}/")
    print(f"   - 最佳模型: {model_path}")
    print("📈 下一步操作:")
    print(f"   - 预测新图像: python predict_food_classification.py --weights \"{model_path}\" --source \"your_image.jpg\"")
    print("   - 查看训练曲线: tensorboard --logdir runs/train-cls")

if __name__ == '__main__':
    main()
