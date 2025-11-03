#!/usr/bin/env python3
"""
UNIMIB2016 食物分类测试脚本
UNIMIB2016 Food Classification Test Script

全面测试训练好的食物分类模型，包括：
- 模型验证和评估
- 性能指标计算
- 混淆矩阵生成
- 分类报告
- 错误案例分析
- 可视化结果
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import time
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# 添加 YOLOv5 路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import check_img_size, increment_path
from utils.torch_utils import select_device
from utils.dataloaders import create_classification_dataloader

# UNIMIB2016 类别名称
UNIMIB2016_CLASSES = {
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

class FoodClassificationTester:
    def __init__(self, weights_path, data_yaml, device=''):
        """初始化测试器"""
        self.weights_path = Path(weights_path)
        self.data_yaml = data_yaml
        self.device = select_device(device)
        self.model = None
        self.results = {}
        
        print(f"🏗️ 初始化测试器")
        print(f"📂 模型权重: {self.weights_path}")
        print(f"📊 数据配置: {self.data_yaml}")
        print(f"💻 设备: {self.device}")
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        try:
            self.model = DetectMultiBackend(self.weights_path, device=self.device)
            print(f"✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def test_single_image(self, image_path, show_result=True):
        """测试单张图像"""
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"❌ 图像不存在: {image_path}")
            return None
        
        print(f"\n🖼️ 测试图像: {image_path.name}")
        
        # 加载和预处理图像
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # 转换为tensor
            img_tensor = torch.from_numpy(img_array).to(self.device)
            img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            # 预测
            start_time = time.time()
            with torch.no_grad():
                pred = self.model(img_tensor)
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 处理预测结果
            if hasattr(pred, 'softmax'):
                probs = pred.softmax(1)
            else:
                probs = torch.softmax(pred, dim=1)
            
            # 获取top5预测
            top5_probs, top5_indices = torch.topk(probs[0], 5)
            
            results = []
            for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
                class_id = idx.item()
                confidence = prob.item()
                class_name = UNIMIB2016_CLASSES.get(class_id, f'未知类别_{class_id}')
                results.append({
                    'rank': i + 1,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
            
            # 显示结果
            if show_result:
                print(f"⏱️ 推理时间: {inference_time:.2f}ms")
                print("📊 Top5 预测结果:")
                print("-" * 50)
                for result in results:
                    print(f"第{result['rank']}名: {result['class_name']} "
                          f"(置信度: {result['confidence']:.3f})")
            
            return {
                'image_path': str(image_path),
                'inference_time_ms': inference_time,
                'predictions': results
            }
            
        except Exception as e:
            print(f"❌ 图像处理失败: {e}")
            return None
    
    def test_dataset(self, data_yaml, batch_size=32, save_results=True):
        """测试整个数据集"""
        print(f"\n📊 开始数据集评估")
        print(f"📂 数据配置: {data_yaml}")
        print(f"📦 批处理大小: {batch_size}")
        
        try:
            # 创建数据加载器
            dataloader = create_classification_dataloader(
                path=data_yaml,
                imgsz=224,
                batch_size=batch_size,
                augment=False,
                cache=False,
                rank=-1,
                workers=4,
                shuffle=False
            )[0]
            
            print(f"📈 测试集大小: {len(dataloader.dataset)} 张图像")
            
            # 评估模型
            self.model.eval()
            all_predictions = []
            all_targets = []
            inference_times = []
            
            print("🔄 开始评估...")
            for batch_i, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 推理
                start_time = time.time()
                with torch.no_grad():
                    pred = self.model(images)
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
                
                # 获取预测类别
                if hasattr(pred, 'softmax'):
                    probs = pred.softmax(1)
                else:
                    probs = torch.softmax(pred, dim=1)
                
                predicted_classes = torch.argmax(probs, dim=1)
                
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                if (batch_i + 1) % 10 == 0:
                    print(f"  处理进度: {batch_i + 1}/{len(dataloader)} 批次")
            
            # 计算指标
            accuracy = accuracy_score(all_targets, all_predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted'
            )
            
            avg_inference_time = np.mean(inference_times)
            
            # 保存结果
            self.results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_inference_time_ms': avg_inference_time,
                'total_samples': len(all_targets),
                'predictions': all_predictions,
                'targets': all_targets
            }
            
            # 打印结果
            print(f"\n📈 评估结果:")
            print(f"✅ 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"🎯 精确率: {precision:.4f}")
            print(f"🔍 召回率: {recall:.4f}")
            print(f"🏆 F1分数: {f1:.4f}")
            print(f"⏱️ 平均推理时间: {avg_inference_time:.2f}ms/批次")
            
            if save_results:
                self.save_detailed_results()
                self.generate_confusion_matrix()
                self.generate_classification_report()
                self.analyze_errors()
            
            return self.results
            
        except Exception as e:
            print(f"❌ 数据集评估失败: {e}")
            return None
    
    def save_detailed_results(self):
        """保存详细结果"""
        save_dir = Path('runs/test') / 'unimib2016_results'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存数值结果
        results_file = save_dir / 'test_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            results_copy = self.results.copy()
            results_copy['predictions'] = [int(x) for x in self.results['predictions']]
            results_copy['targets'] = [int(x) for x in self.results['targets']]
            json.dump(results_copy, f, ensure_ascii=False, indent=2)
        
        print(f"💾 详细结果已保存到: {results_file}")
    
    def generate_confusion_matrix(self):
        """生成混淆矩阵"""
        save_dir = Path('runs/test') / 'unimib2016_results'
        
        # 计算混淆矩阵
        cm = confusion_matrix(self.results['targets'], self.results['predictions'])
        
        # 创建可视化
        plt.figure(figsize=(20, 16))
        
        # 由于类别太多，只显示类别ID
        class_ids = list(range(len(UNIMIB2016_CLASSES)))
        
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                   xticklabels=class_ids, yticklabels=class_ids)
        plt.title('UNIMIB2016 食物分类混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        # 保存图像
        cm_file = save_dir / 'confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 混淆矩阵已保存到: {cm_file}")
        
        # 保存混淆矩阵数据
        cm_data_file = save_dir / 'confusion_matrix.csv'
        cm_df = pd.DataFrame(cm, index=class_ids, columns=class_ids)
        cm_df.to_csv(cm_data_file)
        
        return cm
    
    def generate_classification_report(self):
        """生成分类报告"""
        save_dir = Path('runs/test') / 'unimib2016_results'
        
        # 生成分类报告
        class_names = [UNIMIB2016_CLASSES[i] for i in range(len(UNIMIB2016_CLASSES))]
        report = classification_report(
            self.results['targets'], 
            self.results['predictions'],
            target_names=class_names,
            output_dict=True
        )
        
        # 保存为CSV
        report_df = pd.DataFrame(report).transpose()
        report_file = save_dir / 'classification_report.csv'
        report_df.to_csv(report_file, encoding='utf-8-sig')
        
        # 保存为文本
        text_report = classification_report(
            self.results['targets'], 
            self.results['predictions'],
            target_names=class_names
        )
        
        report_text_file = save_dir / 'classification_report.txt'
        with open(report_text_file, 'w', encoding='utf-8') as f:
            f.write("UNIMIB2016 食物分类详细报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(text_report)
        
        print(f"📋 分类报告已保存到: {report_file}")
        print(f"📄 文本报告已保存到: {report_text_file}")
    
    def analyze_errors(self, top_errors=10):
        """分析错误案例"""
        save_dir = Path('runs/test') / 'unimib2016_results'
        
        predictions = np.array(self.results['predictions'])
        targets = np.array(self.results['targets'])
        
        # 找出错误预测
        errors = predictions != targets
        error_indices = np.where(errors)[0]
        
        print(f"\n🔍 错误分析:")
        print(f"❌ 错误预测数量: {len(error_indices)}")
        print(f"✅ 正确预测数量: {len(targets) - len(error_indices)}")
        print(f"📊 错误率: {len(error_indices)/len(targets)*100:.2f}%")
        
        # 统计每个类别的错误
        error_stats = {}
        for idx in error_indices:
            true_class = targets[idx]
            pred_class = predictions[idx]
            
            if true_class not in error_stats:
                error_stats[true_class] = {'total_errors': 0, 'confused_with': {}}
            
            error_stats[true_class]['total_errors'] += 1
            
            if pred_class not in error_stats[true_class]['confused_with']:
                error_stats[true_class]['confused_with'][pred_class] = 0
            error_stats[true_class]['confused_with'][pred_class] += 1
        
        # 保存错误分析
        error_analysis = {}
        for class_id, stats in error_stats.items():
            class_name = UNIMIB2016_CLASSES[class_id]
            confused_with = []
            for confused_class_id, count in stats['confused_with'].items():
                confused_class_name = UNIMIB2016_CLASSES[confused_class_id]
                confused_with.append({
                    'class_id': confused_class_id,
                    'class_name': confused_class_name,
                    'count': count
                })
            
            error_analysis[str(class_id)] = {
                'class_name': class_name,
                'total_errors': stats['total_errors'],
                'confused_with': sorted(confused_with, key=lambda x: x['count'], reverse=True)
            }
        
        error_file = save_dir / 'error_analysis.json'
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"🔍 错误分析已保存到: {error_file}")
        
        # 显示最容易混淆的类别
        print(f"\n🤔 最容易出错的前 {top_errors} 个类别:")
        print("-" * 60)
        sorted_errors = sorted(error_stats.items(), 
                              key=lambda x: x[1]['total_errors'], reverse=True)
        
        for i, (class_id, stats) in enumerate(sorted_errors[:top_errors]):
            class_name = UNIMIB2016_CLASSES[class_id]
            print(f"{i+1:2d}. {class_name} (ID: {class_id}) - {stats['total_errors']} 个错误")
            
            # 显示最常混淆的类别
            sorted_confused = sorted(stats['confused_with'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for confused_id, count in sorted_confused[:3]:
                confused_name = UNIMIB2016_CLASSES[confused_id]
                print(f"     ↳ 常被误认为: {confused_name} ({count} 次)")

def main():
    parser = argparse.ArgumentParser(description='UNIMIB2016 食物分类测试脚本')
    parser.add_argument('--weights', type=str, required=True,
                       help='训练好的模型权重文件路径')
    parser.add_argument('--data', type=str, default='data/unimib2016.yaml',
                       help='数据集配置文件')
    parser.add_argument('--device', default='',
                       help='推理设备 (cpu, 0, 1, 2, 3, ...)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('--single-image', type=str,
                       help='测试单张图像路径')
    parser.add_argument('--test-dataset', action='store_true',
                       help='测试整个数据集')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='保存测试结果')
    
    args = parser.parse_args()
    
    print("🍕 UNIMIB2016 食物分类模型测试")
    print("=" * 50)
    
    # 初始化测试器
    tester = FoodClassificationTester(args.weights, args.data, args.device)
    
    # 测试单张图像
    if args.single_image:
        result = tester.test_single_image(args.single_image)
        if result:
            print(f"\n✅ 单图像测试完成")
    
    # 测试数据集
    if args.test_dataset:
        results = tester.test_dataset(args.data, args.batch_size, args.save_results)
        if results:
            print(f"\n✅ 数据集测试完成")
            print(f"📊 最终准确率: {results['accuracy']*100:.2f}%")
    
    print(f"\n🎉 测试完成！")
    if args.save_results:
        print(f"📁 结果保存在: runs/test/unimib2016_results/")

if __name__ == '__main__':
    main()
