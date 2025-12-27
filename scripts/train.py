"""
使用 ultralytics YOLOv8 训练 bicycle 检测模型
"""
import argparse
from pathlib import Path
from ultralytics import YOLO 


def main():
    parser = argparse.ArgumentParser(description='训练 YOLOv8 bicycle 检测模型')
    parser.add_argument('--data', type=str, default='data/coco_bicycle.yaml',
                        help='数据集配置文件路径（默认: data/coco_bicycle.yaml）')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='模型文件（默认: yolov8n.pt，可选: yolov8n.pt/yolov8s.pt/yolov8m.pt/yolov8l.pt/yolov8x.pt）')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图片尺寸（默认: 640）')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数（默认: 20）')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小（默认: 16）')
    parser.add_argument('--device', type=str, default='',
                        help='设备（默认: 自动选择，可选: 0, 1, cpu）')
    parser.add_argument('--project', type=str, default='runs',
                        help='项目目录（默认: runs）')
    parser.add_argument('--name', type=str, default='bicycle_exp',
                        help='实验名称（默认: bicycle_exp）')
    
    args = parser.parse_args()
    
    # 检查数据集配置文件
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_path}")
    
    print("=" * 60)
    print("开始训练 YOLOv8 bicycle 检测模型")
    print("=" * 60)
    print(f"数据集配置: {args.data}")
    print(f"模型: {args.model}")
    print(f"图片尺寸: {args.imgsz}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"设备: {args.device if args.device else '自动选择'}")
    print(f"项目目录: {args.project}/{args.name}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(args.model)
    
    # 训练
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        verbose=True
    )
    
    # 输出权重路径
    project_dir = Path(args.project) / args.name
    best_weights = project_dir / 'weights' / 'best.pt'
    last_weights = project_dir / 'weights' / 'last.pt'
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳模型权重: {best_weights}")
    print(f"最新模型权重: {last_weights}")
    print(f"训练结果目录: {project_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

