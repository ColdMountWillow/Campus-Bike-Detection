"""
使用训练好的模型进行推理，只保留 bicycle 检测结果
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import sys
from pathlib import Path

# 添加 src 目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils import filter_bicycle_results, draw_boxes, save_results_json


def process_image(model, img_path, conf_threshold, iou_threshold, save_dir, save_txt):
    """
    处理单张图片
    
    Args:
        model: YOLO 模型
        img_path: 图片路径
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值
        save_dir: 保存目录
        save_txt: 是否保存 txt/json 结果
    
    Returns:
        dict: 包含检测结果的字典
    """
    # 推理
    results = model.predict(
        source=str(img_path),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    # 过滤只保留 bicycle（单类数据集 class_id=0，如果是 COCO80 则 class_id=1）
    # 这里假设是单类数据集，class_id=0 就是 bicycle
    filtered_results = filter_bicycle_results(results[0], is_single_class=True)
    
    # 读取原图
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"警告: 无法读取图片 {img_path}")
        return None
    
    # 绘制检测框
    vis_img = draw_boxes(img.copy(), filtered_results)
    
    # 保存可视化结果
    save_path = save_dir / img_path.name
    cv2.imwrite(str(save_path), vis_img)
    
    # 保存 txt/json 结果（可选）
    if save_txt:
        # 保存 txt（YOLO 格式）
        txt_path = save_dir / (img_path.stem + '.txt')
        with open(txt_path, 'w') as f:
            for box in filtered_results['boxes']:
                # YOLO 格式: class x_center y_center w h
                f.write(f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} "
                       f"{box['width']:.6f} {box['height']:.6f}\n")
        
        # 保存 json
        json_path = save_dir / (img_path.stem + '.json')
        save_results_json(json_path, img_path.name, filtered_results)
    
    # 统计信息
    num_detections = len(filtered_results['boxes'])
    max_score = max([box['confidence'] for box in filtered_results['boxes']]) if filtered_results['boxes'] else 0.0
    
    return {
        'image': img_path.name,
        'num_detections': num_detections,
        'max_score': max_score,
        'boxes': filtered_results['boxes']
    }


def main():
    parser = argparse.ArgumentParser(description='使用 YOLOv8 进行 bicycle 检测推理')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径（例如: runs/bicycle_exp/weights/best.pt）')
    parser.add_argument('--source', type=str, required=True,
                        help='输入图片路径或目录')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值（默认: 0.25）')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU 阈值（默认: 0.45）')
    parser.add_argument('--save_dir', type=str, default='outputs/vis',
                        help='可视化结果保存目录（默认: outputs/vis）')
    parser.add_argument('--save_txt', action='store_true',
                        help='是否保存检测结果到 txt/json 文件')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图片尺寸（默认: 640）')
    
    args = parser.parse_args()
    
    # 检查权重文件
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")
    
    # 检查输入源
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {source_path}")
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始推理")
    print("=" * 60)
    print(f"模型权重: {args.weights}")
    print(f"输入源: {args.source}")
    print(f"置信度阈值: {args.conf}")
    print(f"IoU 阈值: {args.iou}")
    print(f"保存目录: {args.save_dir}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(args.weights)
    
    # 收集所有图片路径
    if source_path.is_file():
        image_paths = [source_path]
    else:
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = [p for p in source_path.rglob('*') 
                      if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        raise ValueError(f"在 {source_path} 中未找到图片文件")
    
    print(f"\n找到 {len(image_paths)} 张图片")
    
    # 处理每张图片
    all_results = []
    for img_path in tqdm(image_paths, desc="处理图片"):
        result = process_image(model, img_path, args.conf, args.iou, save_dir, args.save_txt)
        if result:
            all_results.append(result)
            print(f"\n{result['image']}: 检测到 {result['num_detections']} 个 bicycle, "
                  f"最高置信度: {result['max_score']:.4f}")
    
    # 汇总统计
    print("\n" + "=" * 60)
    print("推理完成")
    print("=" * 60)
    print(f"处理图片数: {len(all_results)}")
    total_detections = sum(r['num_detections'] for r in all_results)
    print(f"总检测数: {total_detections}")
    avg_detections = total_detections / len(all_results) if all_results else 0
    print(f"平均每张图片检测数: {avg_detections:.2f}")
    print(f"可视化结果已保存到: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

