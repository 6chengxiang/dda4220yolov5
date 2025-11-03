# 我的 YOLOv5 食物分类项目 🍕

这是基于 [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) 的**食物分类**项目，专门用于识别和分类各种食物图像。

## 📋 项目简介

本项目使用 YOLOv5 的分类模块来实现食物图像分类，具有以下特点：
- 🍎 多类别食物识别：支持数十种常见食物分类
- 🚀 高性能：快速准确的食物分类
- 📱 易使用：简单的 Python 接口
- 🔧 可定制：支持自定义食物数据集训练
- 📊 实用性：可用于营养分析、食物推荐等应用

## 🥗 支持的食物类别

目前支持以下食物类别（可根据需要扩展）：
- 🍎 水果类：苹果、香蕉、橙子、草莓、葡萄等
- 🥬 蔬菜类：胡萝卜、西兰花、番茄、黄瓜、洋葱等  
- 🍞 主食类：面包、米饭、面条、披萨、汉堡等
- 🥩 蛋白质：鸡肉、牛肉、鱼类、鸡蛋、豆腐等
- 🥛 乳制品：牛奶、酸奶、奶酪等
- 🍰 甜品类：蛋糕、饼干、冰淇淋等

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
pip install opencv-python Pillow  # 食物分类额外依赖
```

### 🍕 食物分类模式

#### 1. 准备数据集
```bash
# 整理您的食物图像数据
python prepare_food_dataset.py --mode organize --source "your_food_images" --target "../datasets/food-101"

# 或创建示例数据集进行测试
python prepare_food_dataset.py --mode sample --target "../datasets/food-sample"
```

#### 2. 训练食物分类模型
```bash
# 快速训练
python train_food_classification.py --model yolov5s-cls.pt --epochs 50

# 高质量训练
python train_food_classification.py --model yolov5m-cls.pt --epochs 100 --batch-size 16
```

#### 3. 预测食物类别
```bash
# 单张图像预测
python predict_food_classification.py --weights runs/train-cls/food-classification/weights/best.pt --source "food_image.jpg"

# 批量预测
python predict_food_classification.py --weights runs/train-cls/food-classification/weights/best.pt --source "food_images_folder/"
```

### 🎯 传统目标检测模式

#### 运行检测
```bash
# 检测图像
python detect.py --weights yolov5s.pt --source data/images/

# 使用摄像头实时检测
python detect.py --weights yolov5s.pt --source 0
```

#### 训练自定义模型
```bash
python train.py --data your_dataset.yaml --weights yolov5s.pt --epochs 100
```

## 📁 项目结构

```
yolov5/
├── data/              # 数据集配置和示例图像
├── models/            # 模型架构定义
├── utils/             # 工具函数
├── weights/           # 预训练权重
├── detect.py          # 检测脚本
├── train.py           # 训练脚本
├── val.py             # 验证脚本
└── export.py          # 模型导出脚本
```

## 🎯 使用示例

### Python API
```python
import torch

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 检测
results = model('path/to/image.jpg')
results.show()
```

### 命令行
```bash
# 基础检测
python detect.py --weights yolov5s.pt --source image.jpg

# 批量检测
python detect.py --weights yolov5s.pt --source images/

# 视频检测
python detect.py --weights yolov5s.pt --source video.mp4
```

## 📊 性能对比

| 模型 | 尺寸 | mAP@0.5:0.95 | 速度 | 参数量 |
|------|------|--------------|------|--------|
| YOLOv5n | 640 | 28.0 | 6.3ms | 1.9M |
| YOLOv5s | 640 | 37.4 | 6.4ms | 7.2M |
| YOLOv5m | 640 | 45.4 | 8.2ms | 21.2M |
| YOLOv5l | 640 | 49.0 | 10.1ms | 46.5M |
| YOLOv5x | 640 | 50.7 | 12.1ms | 86.7M |

## 🛠️ 环境要求

- Python >= 3.8
- PyTorch >= 1.8
- CUDA（推荐，用于 GPU 加速）

## 📚 参考资料

- [原始 YOLOv5 仓库](https://github.com/ultralytics/yolov5)
- [YOLOv5 官方文档](https://docs.ultralytics.com/yolov5/)
- [自定义数据集训练教程](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)

## 📄 许可证

本项目基于 [GPL-3.0 许可证](LICENSE)。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

⭐ 如果这个项目对您有帮助，请给个星标！
