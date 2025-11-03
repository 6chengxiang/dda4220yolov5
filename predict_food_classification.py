#!/usr/bin/env python3
"""
食物分类预测脚本
Food Classification Prediction Script

使用训练好的模型对食物图像进行分类预测
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# 添加 YOLOv5 路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages
from utils.plots import Annotator, colors

# UNIMIB2016 食物类别映射（与 data/unimib2016.yaml 保持一致）
FOOD_CLASSES = {
    0: '面包', 1: '番茄酱意面', 2: '肉酱意面', 3: '蛤蜊酱意面', 4: '青酱意面',
    5: '蒜蓉橄榄油意面', 6: '番茄酱土豆团子', 7: '青酱土豆团子', 8: '意式烩饭', 9: '玉米粥',
    10: '玛格丽特披萨', 11: '四种奶酪披萨', 12: '蔬菜披萨', 13: '火腿披萨', 14: '佛卡夏面包',
    15: '豌豆火腿包', 16: '宽面条配酱', 17: '番茄酱肉丸', 18: '烤意面', 19: '意面沙拉',
    20: '蔬菜汤', 21: '鱼汤', 22: '蔬菜汤', 23: '牛肚', 24: '意面豆汤',
    25: '托斯卡纳蔬菜汤', 26: '烤鱼', 27: '什锦炸鱼', 28: '裹粉炸鱼', 29: '烤鸡',
    30: '鸡胸肉', 31: '鸡翅', 32: '炸鸡', 33: '小牛排', 34: '烤牛肉',
    35: '炖牛肉', 36: '烤牛肉', 37: '汉堡', 38: '猪排', 39: '猪里脊',
    40: '烤猪肉', 41: '生火腿', 42: '熟火腿', 43: '煎蛋', 44: '炒蛋',
    45: '水煮蛋', 46: '煎蛋卷', 47: '奶酪', 48: '马苏里拉奶酪', 49: '茅屋奶酪',
    50: '酸奶', 51: '苹果', 52: '香蕉', 53: '橙子', 54: '草莓',
    55: '葡萄', 56: '梨', 57: '桃子', 58: '柠檬', 59: '猕猴桃',
    60: '菠萝', 61: '混合沙拉', 62: '胡萝卜', 63: '青豆', 64: '菠菜',
    65: '番茄', 66: '土豆', 67: '薯条', 68: '烤土豆', 69: '水煮土豆',
    70: '土豆团子', 71: '葡萄酒', 72: '水'
}

def predict_food(model, image_path, device, conf_thres=0.25):
    """
    预测单张图像的食物类别
    """
    # 加载图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法加载图像: {image_path}")
        return None
    
    # 预处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).to(device)
    img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    # 预测
    with torch.no_grad():
        pred = model(img_tensor)
        
    # 如果是分类模型，直接返回类别概率
    if hasattr(pred, 'softmax'):
        probs = pred.softmax(1)
        top5_indices = probs.argsort(1, descending=True)[0][:5]
        
        results = []
        for i, idx in enumerate(top5_indices):
            class_id = idx.item()
            confidence = probs[0][class_id].item()
            class_name = FOOD_CLASSES.get(class_id, f'Unknown_{class_id}')
            results.append({
                'rank': i + 1,
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence
            })
        return results
    
    return None

def main():
    parser = argparse.ArgumentParser(description='食物分类预测')
    parser.add_argument('--weights', type=str, required=True,
                       help='训练好的模型权重文件路径')
    parser.add_argument('--source', type=str, required=True,
                       help='输入图像路径或目录')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--device', default='',
                       help='推理设备 (cpu, 0, 1, 2, 3, ...)')
    parser.add_argument('--save-results', action='store_true',
                       help='保存预测结果')
    parser.add_argument('--view-img', action='store_true',
                       help='显示预测结果')
    
    args = parser.parse_args()
    
    # 选择设备
    device = select_device(args.device)
    
    # 加载模型
    print(f"🏗️ 加载模型: {args.weights}")
    model = DetectMultiBackend(args.weights, device=device)
    
    # 处理输入源
    source = Path(args.source)
    
    if source.is_file():
        # 单张图像
        print(f"🖼️ 预测图像: {source}")
        results = predict_food(model, source, device, args.conf_thres)
        
        if results:
            print(f"\n📊 预测结果 - {source.name}:")
            print("-" * 50)
            for result in results:
                print(f"第{result['rank']}名: {result['class_name']} "
                      f"(置信度: {result['confidence']:.3f})")
        else:
            print("❌ 预测失败")
            
    elif source.is_dir():
        # 图像目录
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in source.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"📁 处理目录: {source}")
        print(f"🔍 找到 {len(image_files)} 张图像")
        
        all_results = {}
        for img_file in image_files:
            print(f"\n🖼️ 预测: {img_file.name}")
            results = predict_food(model, img_file, device, args.conf_thres)
            
            if results:
                all_results[img_file.name] = results
                top_result = results[0]
                print(f"✅ 预测结果: {top_result['class_name']} "
                      f"(置信度: {top_result['confidence']:.3f})")
            else:
                print("❌ 预测失败")
        
        # 保存结果
        if args.save_results and all_results:
            save_path = source / 'prediction_results.txt'
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("食物分类预测结果\n")
                f.write("=" * 50 + "\n\n")
                
                for img_name, results in all_results.items():
                    f.write(f"图像: {img_name}\n")
                    f.write("-" * 30 + "\n")
                    for result in results:
                        f.write(f"第{result['rank']}名: {result['class_name']} "
                               f"(置信度: {result['confidence']:.3f})\n")
                    f.write("\n")
            
            print(f"💾 结果已保存到: {save_path}")
    
    else:
        print(f"❌ 无效的输入源: {source}")

if __name__ == '__main__':
    main()
