#!/usr/bin/env python3
"""
快速测试脚本
Quick Test Script

简单易用的模型测试工具
"""

import argparse
import torch
import time
from pathlib import Path
import sys
from PIL import Image
import numpy as np

# 添加 YOLOv5 路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 简化的类别名称（前20个常见食物）
SIMPLE_CLASSES = {
    0: '面包', 1: '番茄酱意面', 2: '肉酱意面', 10: '玛格丽特披萨', 
    11: '四种奶酪披萨', 29: '烤鸡', 37: '汉堡', 43: '煎蛋',
    51: '苹果', 52: '香蕉', 53: '橙子', 54: '草莓', 55: '葡萄',
    61: '混合沙拉', 65: '番茄', 66: '土豆', 67: '薯条'
}

def quick_test(weights_path, image_path, device='cpu'):
    """快速测试单张图像"""
    print(f"🍕 快速食物分类测试")
    print(f"📂 模型: {weights_path}")
    print(f"🖼️ 图像: {image_path}")
    print(f"💻 设备: {device}")
    print("-" * 40)
    
    try:
        # 加载模型
        print("🏗️ 加载模型...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                              path=weights_path, device=device, force_reload=True)
        
        # 加载图像
        print("🖼️ 加载图像...")
        img = Image.open(image_path).convert('RGB')
        
        # 预测
        print("🔮 开始预测...")
        start_time = time.time()
        results = model(img)
        inference_time = (time.time() - start_time) * 1000
        
        # 获取预测结果
        if hasattr(results, 'pandas'):
            # YOLOv5 detection format
            df = results.pandas().xyxy[0]
            if len(df) > 0:
                print("🎯 检测结果:")
                for idx, row in df.iterrows():
                    class_name = row['name']
                    confidence = row['confidence']
                    print(f"  ✅ {class_name}: {confidence:.3f}")
            else:
                print("❌ 未检测到任何物体")
        else:
            # Classification format
            print("📊 分类结果:")
            print(f"⏱️ 推理时间: {inference_time:.2f}ms")
            
            # 这里需要手动处理分类结果
            # 因为torch.hub加载的模型可能格式不同
            print("✅ 预测完成（详细结果需要使用完整测试脚本）")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def batch_test(weights_path, image_dir, device='cpu', max_images=10):
    """批量测试图像文件夹"""
    print(f"🍕 批量食物分类测试")
    print(f"📂 模型: {weights_path}")
    print(f"📁 图像目录: {image_dir}")
    print(f"🔢 最大图像数: {max_images}")
    print("-" * 40)
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"❌ 目录不存在: {image_dir}")
        return False
    
    # 查找图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ 未找到图像文件")
        return False
    
    # 限制数量
    image_files = image_files[:max_images]
    print(f"🔍 找到 {len(image_files)} 张图像")
    
    try:
        # 加载模型
        print("🏗️ 加载模型...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                              path=weights_path, device=device, force_reload=True)
        
        # 测试每张图像
        success_count = 0
        total_time = 0
        
        for i, img_path in enumerate(image_files):
            print(f"\n📸 测试 {i+1}/{len(image_files)}: {img_path.name}")
            
            try:
                img = Image.open(img_path).convert('RGB')
                
                start_time = time.time()
                results = model(img)
                inference_time = (time.time() - start_time) * 1000
                total_time += inference_time
                
                print(f"  ⏱️ 推理时间: {inference_time:.2f}ms")
                print(f"  ✅ 预测完成")
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
        
        # 统计结果
        print(f"\n📊 批量测试结果:")
        print(f"✅ 成功: {success_count}/{len(image_files)}")
        print(f"⏱️ 平均推理时间: {total_time/len(image_files):.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量测试失败: {e}")
        return False

def performance_test(weights_path, device='cpu', test_runs=100):
    """性能测试"""
    print(f"🚀 性能测试")
    print(f"📂 模型: {weights_path}")
    print(f"🔢 测试次数: {test_runs}")
    print("-" * 40)
    
    try:
        # 加载模型
        print("🏗️ 加载模型...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                              path=weights_path, device=device, force_reload=True)
        
        # 创建测试图像
        print("🖼️ 创建测试图像...")
        test_img = Image.new('RGB', (224, 224), color='red')
        
        # 预热
        print("🔥 模型预热...")
        for _ in range(10):
            _ = model(test_img)
        
        # 性能测试
        print(f"⏱️ 开始 {test_runs} 次推理测试...")
        times = []
        
        for i in range(test_runs):
            start_time = time.time()
            _ = model(test_img)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000
            times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"  进度: {i + 1}/{test_runs}")
        
        # 统计结果
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        print(f"\n📊 性能测试结果:")
        print(f"⏱️ 平均推理时间: {avg_time:.2f}ms")
        print(f"🏃 最快推理时间: {min_time:.2f}ms")
        print(f"🐌 最慢推理时间: {max_time:.2f}ms")
        print(f"📈 标准差: {std_time:.2f}ms")
        print(f"🔥 推理速度: {1000/avg_time:.1f} FPS")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='快速测试脚本')
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重文件路径')
    parser.add_argument('--source', type=str,
                       help='测试图像路径或目录')
    parser.add_argument('--device', default='cpu',
                       help='推理设备 (cpu, 0, 1, 2, 3, ...)')
    parser.add_argument('--mode', choices=['single', 'batch', 'performance'], 
                       default='single', help='测试模式')
    parser.add_argument('--max-images', type=int, default=10,
                       help='批量测试最大图像数')
    parser.add_argument('--test-runs', type=int, default=100,
                       help='性能测试运行次数')
    
    args = parser.parse_args()
    
    # 检查权重文件
    if not Path(args.weights).exists():
        print(f"❌ 权重文件不存在: {args.weights}")
        return
    
    if args.mode == 'single':
        if not args.source:
            print("❌ 单图像测试需要指定 --source 参数")
            return
        quick_test(args.weights, args.source, args.device)
        
    elif args.mode == 'batch':
        if not args.source:
            print("❌ 批量测试需要指定 --source 参数")
            return
        batch_test(args.weights, args.source, args.device, args.max_images)
        
    elif args.mode == 'performance':
        performance_test(args.weights, args.device, args.test_runs)

if __name__ == '__main__':
    main()
