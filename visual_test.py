#!/usr/bin/env python3
"""
可视化测试脚本
Visual Test Script

生成带有预测结果的可视化图像
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns

# 添加 YOLOv5 路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# UNIMIB2016 类别名称
FOOD_CLASSES = {
    0: '面包', 1: '番茄酱意面', 2: '肉酱意面', 3: '蛤蜊酱意面', 4: '青酱意面',
    5: '蒜蓉橄榄油意面', 6: '番茄酱土豆团子', 7: '青酱土豆团子', 8: '意式烩饭', 9: '玉米粥',
    10: '玛格丽特披萨', 11: '四种奶酪披萨', 12: '蔬菜披萨', 13: '火腿披萨', 14: '佛卡夏面包',
    15: '豌豆火腿包', 16: '宽面条配酱', 17: '番茄酱肉丸', 18: '烤意面', 19: '意面沙拉',
    20: '蔬菜汤', 21: '鱼汤', 22: '蔬菜汤2', 23: '牛肚', 24: '意面豆汤',
    25: '托斯卡纳蔬菜汤', 26: '烤鱼', 27: '什锦炸鱼', 28: '裹粉炸鱼', 29: '烤鸡',
    30: '鸡胸肉', 31: '鸡翅', 32: '炸鸡', 33: '小牛排', 34: '烤牛肉',
    35: '炖牛肉', 36: '烤牛肉2', 37: '汉堡', 38: '猪排', 39: '猪里脊',
    40: '烤猪肉', 41: '生火腿', 42: '熟火腿', 43: '煎蛋', 44: '炒蛋',
    45: '水煮蛋', 46: '煎蛋卷', 47: '奶酪', 48: '马苏里拉奶酪', 49: '茅屋奶酪',
    50: '酸奶', 51: '苹果', 52: '香蕉', 53: '橙子', 54: '草莓',
    55: '葡萄', 56: '梨', 57: '桃子', 58: '柠檬', 59: '猕猴桃',
    60: '菠萝', 61: '混合沙拉', 62: '胡萝卜', 63: '青豆', 64: '菠菜',
    65: '番茄', 66: '土豆', 67: '薯条', 68: '烤土豆', 69: '水煮土豆',
    70: '土豆团子', 71: '葡萄酒', 72: '水'
}

def predict_image(model, image_path, device='cpu'):
    """预测单张图像"""
    try:
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # 转换为tensor
        img_tensor = torch.from_numpy(img_array).to(device)
        img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            pred = model(img_tensor)
        
        # 处理预测结果
        if hasattr(pred, 'softmax'):
            probs = pred.softmax(1)
        else:
            probs = torch.softmax(pred, dim=1)
        
        # 获取top5预测
        top5_probs, top5_indices = torch.topk(probs[0], 5)
        
        results = []
        for prob, idx in zip(top5_probs, top5_indices):
            class_id = idx.item()
            confidence = prob.item()
            class_name = FOOD_CLASSES.get(class_id, f'未知_{class_id}')
            results.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence
            })
        
        return img, results
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return None, None

def create_result_image(image, predictions, save_path):
    """创建带有预测结果的图像"""
    # 转换为PIL图像
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image.copy()
    
    # 创建绘图对象
    draw = ImageDraw.Draw(img)
    
    # 尝试加载字体
    try:
        # Windows系统的中文字体
        font_large = ImageFont.truetype("msyh.ttc", 32)  # 微软雅黑
        font_small = ImageFont.truetype("msyh.ttc", 24)
    except:
        try:
            # 备用字体
            font_large = ImageFont.truetype("arial.ttf", 32)
            font_small = ImageFont.truetype("arial.ttf", 24)
        except:
            # 默认字体
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # 获取图像尺寸
    width, height = img.size
    
    # 创建半透明背景
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 绘制结果背景
    result_height = 200
    overlay_draw.rectangle(
        [(0, height - result_height), (width, height)],
        fill=(0, 0, 0, 180)
    )
    
    # 合成图像
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # 绘制预测结果
    y_start = height - result_height + 10
    
    # 标题
    draw.text((10, y_start), "🍕 食物分类结果", fill='white', font=font_large)
    
    # 预测结果
    for i, pred in enumerate(predictions[:3]):  # 只显示前3个
        y_pos = y_start + 40 + i * 35
        
        # 置信度条
        conf_width = int(300 * pred['confidence'])
        draw.rectangle(
            [(10, y_pos + 20), (310, y_pos + 30)],
            outline='white', width=1
        )
        draw.rectangle(
            [(10, y_pos + 20), (10 + conf_width, y_pos + 30)],
            fill='green' if i == 0 else 'orange'
        )
        
        # 文本
        text = f"{i+1}. {pred['class_name']} ({pred['confidence']:.1%})"
        draw.text((10, y_pos), text, fill='white', font=font_small)
    
    # 保存图像
    img.save(save_path)
    print(f"💾 结果图像已保存: {save_path}")
    
    return img

def create_comparison_grid(image_results, save_path, grid_size=(3, 3)):
    """创建对比网格图像"""
    rows, cols = grid_size
    max_images = min(len(image_results), rows * cols)
    
    if max_images == 0:
        print("❌ 没有图像结果可显示")
        return
    
    # 计算单个图像大小
    img_width, img_height = 300, 300
    
    # 创建网格图像
    grid_width = cols * img_width
    grid_height = rows * (img_height + 60)  # 额外空间用于文本
    
    grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # 字体
    try:
        font = ImageFont.truetype("msyh.ttc", 16)
    except:
        font = ImageFont.load_default()
    
    for i in range(max_images):
        row = i // cols
        col = i % cols
        
        img_path, predictions = image_results[i]
        
        # 加载并调整图像大小
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_width, img_height))
            
            # 粘贴到网格
            x = col * img_width
            y = row * (img_height + 60)
            grid_img.paste(img, (x, y))
            
            # 添加预测文本
            draw = ImageDraw.Draw(grid_img)
            text_y = y + img_height + 5
            
            # 文件名
            filename = Path(img_path).name
            draw.text((x + 5, text_y), f"📁 {filename}", fill='black', font=font)
            
            # 最佳预测
            if predictions:
                best_pred = predictions[0]
                pred_text = f"🍕 {best_pred['class_name']} ({best_pred['confidence']:.1%})"
                draw.text((x + 5, text_y + 20), pred_text, fill='green', font=font)
            
        except Exception as e:
            print(f"⚠️ 处理图像失败 {img_path}: {e}")
    
    # 保存网格图像
    grid_img.save(save_path)
    print(f"📊 对比网格已保存: {save_path}")
    
    return grid_img

def main():
    parser = argparse.ArgumentParser(description='可视化测试脚本')
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重文件路径')
    parser.add_argument('--source', type=str, required=True,
                       help='图像文件或目录路径')
    parser.add_argument('--device', default='cpu',
                       help='推理设备 (cpu, 0, 1, 2, 3, ...)')
    parser.add_argument('--output', type=str, default='runs/visual_test',
                       help='输出目录')
    parser.add_argument('--grid', action='store_true',
                       help='创建对比网格')
    parser.add_argument('--max-images', type=int, default=9,
                       help='最大处理图像数')
    
    args = parser.parse_args()
    
    print("🎨 可视化测试脚本")
    print(f"📂 模型: {args.weights}")
    print(f"🖼️ 源: {args.source}")
    print("-" * 40)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("🏗️ 加载模型...")
    try:
        # 使用YOLOv5分类模型
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                              path=args.weights, device=args.device)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    source_path = Path(args.source)
    image_results = []
    
    if source_path.is_file():
        # 单张图像
        print(f"📸 处理单张图像: {source_path.name}")
        
        img, predictions = predict_image(model, source_path, args.device)
        if img and predictions:
            # 创建结果图像
            result_path = output_dir / f"result_{source_path.stem}.jpg"
            create_result_image(img, predictions, result_path)
            
            # 显示结果
            print(f"\n📊 预测结果:")
            for i, pred in enumerate(predictions):
                print(f"  {i+1}. {pred['class_name']}: {pred['confidence']:.3f}")
    
    elif source_path.is_dir():
        # 图像目录
        print(f"📁 处理图像目录: {source_path}")
        
        # 查找图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print("❌ 未找到图像文件")
            return
        
        # 限制数量
        image_files = image_files[:args.max_images]
        print(f"🔍 找到 {len(image_files)} 张图像")
        
        # 处理每张图像
        for i, img_path in enumerate(image_files):
            print(f"\n📸 处理 {i+1}/{len(image_files)}: {img_path.name}")
            
            img, predictions = predict_image(model, img_path, args.device)
            if img and predictions:
                # 创建结果图像
                result_path = output_dir / f"result_{img_path.stem}.jpg"
                create_result_image(img, predictions, result_path)
                
                # 保存结果用于网格
                image_results.append((img_path, predictions))
                
                # 显示最佳预测
                best_pred = predictions[0]
                print(f"  🏆 最佳预测: {best_pred['class_name']} ({best_pred['confidence']:.3f})")
        
        # 创建对比网格
        if args.grid and image_results:
            print(f"\n📊 创建对比网格...")
            grid_path = output_dir / "comparison_grid.jpg"
            create_comparison_grid(image_results, grid_path)
    
    else:
        print(f"❌ 无效的源路径: {source_path}")
        return
    
    print(f"\n🎉 可视化测试完成！")
    print(f"📁 结果保存在: {output_dir}")

if __name__ == '__main__':
    main()
