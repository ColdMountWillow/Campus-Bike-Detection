"""
使用训练好的模型在验证集上评估
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='验证 YOLOv8 bicycle 检测模型')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径（例如: runs/bicycle_exp/weights/best.pt）')
    parser.add_argument('--data', type=str, default='data/coco_bicycle.yaml',
                        help='数据集配置文件路径（默认: data/coco_bicycle.yaml）')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图片尺寸（默认: 640）')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值（默认: 0.25）')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU 阈值（默认: 0.45）')
    
    args = parser.parse_args()
    
    # 检查权重文件
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")
    
    # 检查数据集配置文件
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_path}")
    
    print("=" * 60)
    print("开始验证模型")
    print("=" * 60)
    print(f"模型权重: {args.weights}")
    print(f"数据集配置: {args.data}")
    print(f"图片尺寸: {args.imgsz}")
    print(f"置信度阈值: {args.conf}")
    print(f"IoU 阈值: {args.iou}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(args.weights)
    
    # 验证
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=True
    )
    
    # 输出关键指标
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("=" * 60)
    
    # 输出保存路径（ultralytics 会自动保存到 runs/detect/val）
    # 尝试获取保存路径，如果无法获取则使用默认路径
    try:
        if hasattr(model, 'trainer') and model.trainer is not None and hasattr(model.trainer, 'save_dir'):
            results_dir = Path(model.trainer.save_dir)
            if results_dir.exists():
                print(f"\n验证结果已保存到: {results_dir.absolute()}")
        else:
            # 默认保存路径（ultralytics 的标准路径）
            default_results_dir = Path('runs/detect/val')
            if default_results_dir.exists():
                print(f"\n验证结果已保存到: {default_results_dir.absolute()}")
    except Exception:
        # 如果无法获取路径，跳过（ultralytics 已经在终端输出了保存路径）
        pass


if __name__ == '__main__':
    main()

