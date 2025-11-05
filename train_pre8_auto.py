"""
YOLOv5 Pre8数据集训练脚本 - 直接启动版本（无需确认）
用于训练Hot Dog实例分割模型
"""

import os
import subprocess
import sys
from pathlib import Path


def detect_device():
    """自动检测可用的训练设备"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ 检测到GPU: {device_name}")
            print(f"   CUDA版本: {torch.version.cuda}")
            return '0'
        else:
            print("⚠️  未检测到可用的GPU，将使用CPU训练")
            print("   提示: CPU训练速度会很慢，建议使用GPU")
            return 'cpu'
    except Exception as e:
        print(f"⚠️  设备检测失败: {e}")
        print("   默认使用CPU训练")
        return 'cpu'


def get_optimal_batch_size(device):
    """根据设备获取推荐的batch size"""
    if device == 'cpu':
        return 4  # CPU推荐较小的batch size
    else:
        # GPU默认batch size
        try:
            import torch
            # 粗略估算GPU显存
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_mem < 4:
                return 4
            elif gpu_mem < 8:
                return 8
            else:
                return 16
        except:
            return 16


# 自动检测设备
AUTO_DEVICE = detect_device()
AUTO_BATCH_SIZE = get_optimal_batch_size(AUTO_DEVICE)

print(f"   推荐批量大小: {AUTO_BATCH_SIZE}")
print()

# 训练参数配置
CONFIG = {
    # 数据集配置
    'data': 'data/pre8.yaml',
    
    # 模型配置
    'weights': 'yolov5s-seg.pt',
    
    # 训练参数
    'epochs': 100,
    'batch_size': AUTO_BATCH_SIZE,  # 自动根据设备调整
    'imgsz': 640,
    'device': AUTO_DEVICE,  # 自动检测 (GPU 或 CPU)
    
    # 优化参数
    'optimizer': 'SGD',
    
    # 其他设置
    'project': 'runs/train-seg',
    'name': 'pre8_hotdog',
    'exist_ok': False,
    'workers': 8 if AUTO_DEVICE != 'cpu' else 4,  # CPU使用较少workers
    'patience': 100,
}


def check_python_packages():
    """检查Python包是否安装"""
    import importlib.util
    
    required_packages = {
        'ultralytics': 'ultralytics',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'yaml': 'PyYAML',
        'numpy': 'numpy',
    }
    
    missing_packages = []
    installed_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                missing_packages.append(package_name)
            else:
                installed_packages.append(import_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            missing_packages.append(package_name)
    
    print(f"当前Python: {sys.executable}")
    print()
    
    if missing_packages:
        print(f"❌ 缺少必需的包: {', '.join(missing_packages)}")
        print()
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    print(f"✅ 已安装的包: {', '.join(installed_packages)}")
    return True


def check_requirements():
    """检查必要的文件和依赖"""
    print("检查环境...")
    print()
    
    # 检查Python包
    print("[1/3] 检查Python包...")
    if not check_python_packages():
        return False
    print("✅ 所有必需的包已安装")
    print()
    
    # 检查数据集配置文件
    print("[2/3] 检查数据集配置...")
    if not Path(CONFIG['data']).exists():
        print(f"❌ 错误: 数据集配置文件不存在: {CONFIG['data']}")
        return False
    print(f"✅ 数据集配置文件: {CONFIG['data']}")
    print()
    
    # 检查训练脚本
    print("[3/3] 检查训练脚本...")
    if not Path('segment/train.py').exists():
        print("❌ 错误: 找不到 segment/train.py")
        return False
    print("✅ 训练脚本: segment/train.py")
    print()
    
    print("✅ 环境检查通过")
    return True


def build_command():
    """构建训练命令"""
    cmd = [
        sys.executable,
        'segment/train.py',
        '--data', CONFIG['data'],
        '--weights', CONFIG['weights'],
        '--epochs', str(CONFIG['epochs']),
        '--batch-size', str(CONFIG['batch_size']),
        '--imgsz', str(CONFIG['imgsz']),
        '--device', CONFIG['device'],
        '--project', CONFIG['project'],
        '--name', CONFIG['name'],
        '--optimizer', CONFIG['optimizer'],
        '--workers', str(CONFIG['workers']),
        '--patience', str(CONFIG['patience']),
    ]
    
    if CONFIG['exist_ok']:
        cmd.append('--exist-ok')
    
    return cmd


def main():
    print("=" * 70)
    print("YOLOv5 实例分割训练 - Pre8 Hot Dog 数据集")
    print("=" * 70)
    print()
    
    # 显示配置
    print("📋 训练配置:")
    print(f"  数据集: {CONFIG['data']}")
    print(f"  预训练权重: {CONFIG['weights']}")
    print(f"  训练轮数: {CONFIG['epochs']}")
    print(f"  批量大小: {CONFIG['batch_size']}")
    print(f"  图片大小: {CONFIG['imgsz']}")
    print(f"  设备: {CONFIG['device']}")
    print(f"  优化器: {CONFIG['optimizer']}")
    print(f"  输出目录: {CONFIG['project']}/{CONFIG['name']}")
    print()
    
    # 检查环境
    if not check_requirements():
        print()
        print("❌ 环境检查失败，训练已终止")
        sys.exit(1)
    
    # 构建并显示命令
    cmd = build_command()
    print()
    print("🚀 训练命令:")
    print(" ".join(cmd))
    print()
    print("=" * 70)
    print("开始训练...")
    print("=" * 70)
    print()
    
    # 执行训练
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
        sys.exit(0)
    
    print()
    print("=" * 70)
    print("✅ 训练完成!")
    print("=" * 70)
    print(f"结果保存在: {CONFIG['project']}/{CONFIG['name']}")


if __name__ == '__main__':
    main()
