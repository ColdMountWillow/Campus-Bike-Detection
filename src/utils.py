"""
工具函数：可视化、过滤、结果保存等
"""
import cv2
import json
import numpy as np
from pathlib import Path


def filter_bicycle_results(result, is_single_class=True):
    """
    从 YOLO 检测结果中过滤出 bicycle
    
    Args:
        result: ultralytics YOLO 的检测结果对象
        is_single_class: 是否为单类数据集
    
    Returns:
        dict: {
            'boxes': [
                {
                    'class_id': int,
                    'x_center': float (归一化),
                    'y_center': float (归一化),
                    'width': float (归一化),
                    'height': float (归一化),
                    'confidence': float,
                    'x1': int (像素),
                    'y1': int (像素),
                    'x2': int (像素),
                    'y2': int (像素)
                },
                ...
            ]
        }
    """
    boxes = []
    
    # 检查是否有检测框
    if result.boxes is None or len(result.boxes) == 0:
        return {'boxes': boxes}
    
    # 获取检测框数据
    box_data = result.boxes.data.cpu().numpy()  # [N, 6]: x1, y1, x2, y2, conf, cls
    
    # 获取图片尺寸，用于归一化
    img_height, img_width = result.orig_shape[:2]
    
    for box in box_data:
        x1, y1, x2, y2, conf, cls_id = box
        cls_id = int(cls_id)
        
        # 过滤 bicycle
        target_class = 0 if is_single_class else 1
        if cls_id != target_class:
            continue
        
        # 计算归一化坐标（YOLO 格式）
        x_center = ((x1 + x2) / 2.0) / img_width
        y_center = ((y1 + y2) / 2.0) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        boxes.append({
            'class_id': cls_id,
            'x_center': float(x_center),
            'y_center': float(y_center),
            'width': float(width),
            'height': float(height),
            'confidence': float(conf),
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2)
        })
    
    return {'boxes': boxes}


def draw_boxes(img, results, color=(0, 255, 0), thickness=2):
    """
    在图片上绘制检测框
    
    Args:
        img: numpy array，BGR 格式
        results: filter_bicycle_results 返回的字典
        color: 框的颜色 (B, G, R)
        thickness: 线条粗细
    
    Returns:
        numpy array: 绘制了检测框的图片
    """
    img_copy = img.copy()
    
    for box in results['boxes']:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        conf = box['confidence']
        
        # 绘制矩形框
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签文本
        label = f"bicycle {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # 文本背景
        cv2.rectangle(
            img_copy,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # 文本
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return img_copy


def save_results_json(json_path, image_name, results):
    """
    保存检测结果到 JSON 文件
    
    Args:
        json_path: JSON 文件保存路径
        image_name: 图片文件名
        results: filter_bicycle_results 返回的字典
    """
    output = {
        'image': image_name,
        'detections': []
    }
    
    for box in results['boxes']:
        output['detections'].append({
            'class': 'bicycle',
            'confidence': box['confidence'],
            'bbox': {
                'x1': box['x1'],
                'y1': box['y1'],
                'x2': box['x2'],
                'y2': box['y2']
            },
            'bbox_normalized': {
                'x_center': box['x_center'],
                'y_center': box['y_center'],
                'width': box['width'],
                'height': box['height']
            }
        })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def get_image_paths(source_path):
    """
    从文件或目录中收集所有图片路径
    
    Args:
        source_path: 文件路径或目录路径
    
    Returns:
        list: 图片路径列表
    """
    source_path = Path(source_path)
    
    if source_path.is_file():
        return [source_path]
    
    if source_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return [p for p in source_path.rglob('*') 
                if p.suffix.lower() in image_extensions]
    
    return []

