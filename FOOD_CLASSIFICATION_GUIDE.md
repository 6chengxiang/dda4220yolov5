# 🍕 YOLOv5 食物分类项目快速启动指南

## 🎯 专为 UNIMIB2016 数据集定制

本指南专门针对 [UNIMIB2016 Kaggle 数据集](https://www.kaggle.com/datasets/dangvanthuc0209/unimib2016) 进行了优化，包含73个意大利食物类别。

## 📋 第一步：环境准备

### 1. 安装依赖
```bash
cd c:\Users\dell\Desktop\yolov5
pip install -r requirements.txt
pip install opencv-python Pillow
```

### 2. 下载预训练分类模型
```bash
# 下载 YOLOv5 分类模型
python -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s-cls', pretrained=True)"
```

## 📁 第二步：准备 UNIMIB2016 数据集

### 1. 从 Kaggle 下载数据集

1. **访问数据集页面**
   - 打开 https://www.kaggle.com/datasets/dangvanthuc0209/unimib2016
   - 登录您的 Kaggle 账户

2. **下载数据集**
   - 点击页面上的 "Download" 按钮
   - 下载 `unimib2016.zip` 文件（约2.73GB）

### 2. 处理数据集

```bash
# 解压并处理 UNIMIB2016 数据集
python process_unimib2016.py --source "path/to/unimib2016.zip" --target "../datasets/unimib2016" --extract

# 或者如果您已经解压了
python process_unimib2016.py --source "path/to/extracted/folder" --target "../datasets/unimib2016"
```

### 3. 验证数据集结构

处理完成后，您的数据集结构应该是：
```
../datasets/unimib2016/
├── images/
│   ├── train/          # 训练集图像 (70%)
│   ├── val/            # 验证集图像 (20%)
│   └── test/           # 测试集图像 (10%)
├── dataset_stats.json  # 数据集统计信息
└── class_mapping.json  # 类别映射文件
```

### 🍝 UNIMIB2016 数据集特点

- **73个食物类别**：涵盖意大利经典食物
- **总计约2700张图像**：高质量食物图片
- **类别示例**：
  - 🍝 意面类：番茄酱意面、肉酱意面、青酱意面等
  - 🍕 披萨类：玛格丽特披萨、四种奶酪披萨等
  - 🥩 肉类：烤鸡、牛排、火腿等
  - 🍎 水果：苹果、香蕉、橙子等
  - 🥗 蔬菜：沙拉、胡萝卜、菠菜等

## 🚀 第三步：开始训练 UNIMIB2016

### 基础训练命令（推荐开始）
```bash
# 使用小型模型快速训练（约30分钟，适合测试）
python train_food_classification.py --model yolov5s-cls.pt --epochs 30 --batch-size 32

# 使用中型模型获得更好效果（约1小时）
python train_food_classification.py --model yolov5m-cls.pt --epochs 50 --batch-size 16

# 使用大型模型获得最佳效果（约2-3小时，需要更多GPU内存）
python train_food_classification.py --model yolov5l-cls.pt --epochs 100 --batch-size 8
```

### 针对 UNIMIB2016 优化的训练参数
```bash
# 推荐配置：平衡速度和精度
python train_food_classification.py \
    --model yolov5s-cls.pt \
    --epochs 80 \
    --batch-size 32 \
    --lr0 0.001 \
    --imgsz 224 \
    --mixup 0.15 \
    --cutmix 0.15 \
    --cache \
    --device 0 \
    --name unimib2016-v1

# 高精度配置（如果有足够的时间和GPU）
python train_food_classification.py \
    --model yolov5m-cls.pt \
    --epochs 150 \
    --batch-size 16 \
    --lr0 0.0008 \
    --imgsz 256 \
    --mixup 0.2 \
    --cutmix 0.2 \
    --cache \
    --device 0 \
    --name unimib2016-high-acc
```

## 🔍 第四步：测试和预测

### 单张图像预测
```bash
# 预测单张意大利食物图像
python predict_food_classification.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --source "test_pizza.jpg"

# 示例输出：
# 📊 预测结果 - test_pizza.jpg:
# --------------------------------------------------
# 第1名: 玛格丽特披萨 (置信度: 0.892)
# 第2名: 四种奶酪披萨 (置信度: 0.098)
# 第3名: 蔬菜披萨 (置信度: 0.007)
```

### 批量预测意大利食物
```bash
# 预测整个文件夹的意大利食物图像
python predict_food_classification.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --source "italian_food_images/" \
    --save-results

# 会生成 prediction_results.txt 文件，包含所有预测结果
```

### 使用 YOLOv5 内置分类脚本
```bash
# 验证 UNIMIB2016 模型
python classify/val.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --data data/unimib2016.yaml \
    --batch-size 32

# 预测（原版接口）
python classify/predict.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --source italian_food_images/
```

## 🧪 第五步：全面测试模型

### 🔍 完整模型评估
```bash
# 全面测试：包括混淆矩阵、分类报告、错误分析
python test_unimib2016.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --data data/unimib2016.yaml \
    --test-dataset \
    --save-results

# 测试单张图像
python test_unimib2016.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --single-image "test_pizza.jpg"
```

### ⚡ 快速测试
```bash
# 快速测试单张图像
python quick_test.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --source "test_image.jpg" \
    --mode single

# 批量快速测试
python quick_test.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --source "test_images_folder/" \
    --mode batch \
    --max-images 20

# 性能基准测试
python quick_test.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --mode performance \
    --test-runs 100
```

### 🎨 可视化测试
```bash
# 创建带预测结果的可视化图像
python visual_test.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --source "test_food.jpg" \
    --output "runs/visual_test"

# 批量可视化并创建对比网格
python visual_test.py \
    --weights runs/train-cls/unimib2016-v1/weights/best.pt \
    --source "test_images_folder/" \
    --output "runs/visual_test" \
    --grid \
    --max-images 9
```

### 📊 测试结果解读

#### 完整评估结果位置：
- `runs/test/unimib2016_results/test_results.json` - 数值结果
- `runs/test/unimib2016_results/confusion_matrix.png` - 混淆矩阵图
- `runs/test/unimib2016_results/classification_report.csv` - 详细分类报告
- `runs/test/unimib2016_results/error_analysis.json` - 错误案例分析

#### 关键指标说明：
- **准确率 (Accuracy)**: 整体分类正确率
- **精确率 (Precision)**: 预测为某类别的样本中真正属于该类别的比例
- **召回率 (Recall)**: 某类别的样本中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均数

## 📊 第六步：评估和优化

### 查看训练结果
```bash
# 查看训练日志
tensorboard --logdir runs/train-cls

# 或者查看 wandb 日志（如果配置了）
```

### 模型评估
```bash
python classify/val.py --weights runs/train-cls/food-classification/weights/best.pt --data data/food-classification.yaml --batch-size 32
```

## 🎯 使用技巧

### 1. 数据集质量优化
- ✅ 确保每个类别至少有100张图像
- ✅ 图像质量要好，避免模糊图像
- ✅ 类别之间要有明显区别
- ✅ 数据分布要均衡

### 2. 训练优化
- 🔧 从小模型开始（yolov5s-cls）
- 🔧 使用数据增强（mixup, cutmix）
- 🔧 适当调整学习率
- 🔧 使用早停机制

### 3. 常见问题解决

**内存不足？**
```bash
# 减小批处理大小
--batch-size 8

# 减小图像尺寸
--imgsz 128
```

**训练太慢？**
```bash
# 使用缓存
--cache

# 减少工作线程
--workers 4

# 使用混合精度训练
--amp
```

**精度不够高？**
```bash
# 增加训练轮数
--epochs 200

# 使用更大的模型
--model yolov5l-cls.pt

# 调整学习率
--lr0 0.0005
```

## 🚀 UNIMIB2016 快速开始命令

```bash
# 一键启动完整流程
# 1. 下载并处理数据集
python process_unimib2016.py --source "path/to/unimib2016.zip" --target "../datasets/unimib2016" --extract

# 2. 训练模型（快速版本）
python train_food_classification.py --model yolov5s-cls.pt --epochs 30 --name unimib2016-quick

# 3. 测试预测
python predict_food_classification.py \
    --weights runs/train-cls/unimib2016-quick/weights/best.pt \
    --source "test_image.jpg"
```

## 🎯 预期结果

使用 UNIMIB2016 数据集，您可以期待：

- **训练精度**: 85-95% (取决于模型大小和训练轮数)
- **验证精度**: 80-90% 
- **训练时间**: 
  - YOLOv5s: 30-60分钟 (30-50 epochs)
  - YOLOv5m: 1-2小时 (50-100 epochs)
  - YOLOv5l: 2-4小时 (100-150 epochs)
- **推理速度**: 10-50ms/图像 (取决于硬件)

## 🇮🇹 意大利食物识别示例

模型可以识别的意大利食物包括：
- 🍝 **意面类**: 各种酱料的意大利面
- 🍕 **披萨类**: 不同口味的披萨
- 🥩 **肉类**: 各种烹饪方式的肉食
- 🥚 **蛋类**: 煎蛋、炒蛋、水煮蛋等
- 🧀 **奶制品**: 各种意大利奶酪
- 🍎 **水果**: 新鲜水果
- 🥗 **蔬菜**: 沙拉和各种蔬菜制品

需要帮助？查看详细文档或提交 Issue！
