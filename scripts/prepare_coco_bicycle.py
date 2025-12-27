"""
从 COCO 2017 数据集中提取 bicycle 类别，生成 YOLO 格式的单类数据集
"""
import os
import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_coco_to_yolo(coco_bbox, img_width, img_height):
    """
    将 COCO 格式的 bbox [x, y, w, h] 转换为 YOLO 格式 [x_center, y_center, w, h] (归一化)
    
    Args:
        coco_bbox: [x, y, width, height] 像素坐标
        img_width: 图片宽度
        img_height: 图片高度
    
    Returns:
        [x_center, y_center, w, h] 归一化坐标，如果无效返回 None
    """
    x, y, w, h = coco_bbox
    
    # 检查有效性
    if w <= 0 or h <= 0:
        return None
    
    # 计算中心点和归一化
    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    # 检查是否越界
    if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
        return None
    if w_norm <= 0 or h_norm <= 0 or w_norm > 1 or h_norm > 1:
        return None
    
    return [x_center, y_center, w_norm, h_norm]


def prepare_coco_bicycle(coco_root, out_dir, split='train'):
    """
    从 COCO 数据集中提取 bicycle 类别，生成 YOLO 格式标签
    
    Args:
        coco_root: COCO 数据集根目录（包含 annotations/ 和 train2017/ 或 val2017/）
        out_dir: 输出目录
        split: 'train' 或 'val'
    """
    coco_root = Path(coco_root)
    out_dir = Path(out_dir)
    
    # 路径设置
    ann_file = coco_root / 'annotations' / f'instances_{split}2017.json'
    img_dir = coco_root / f'{split}2017'
    out_img_dir = out_dir / 'images' / split
    out_label_dir = out_dir / 'labels' / split
    
    # 创建输出目录
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件是否存在
    if not ann_file.exists():
        raise FileNotFoundError(f"标注文件不存在: {ann_file}")
    if not img_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")
    
    # 加载 COCO 标注
    print(f"加载标注文件: {ann_file}")
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 构建映射
    images_dict = {img['id']: img for img in coco_data['images']}
    categories_dict = {cat['id']: cat for cat in coco_data['categories']}
    
    # bicycle 的 COCO category id 是 2（注意：COCO 的 id 从 1 开始，但 bicycle 是 2）
    bicycle_cat_id = None
    for cat_id, cat in categories_dict.items():
        if cat['name'] == 'bicycle':
            bicycle_cat_id = cat_id
            break
    
    if bicycle_cat_id is None:
        raise ValueError("在 COCO 数据集中未找到 bicycle 类别")
    
    print(f"Bicycle 类别 ID: {bicycle_cat_id}")
    
    # 按图片组织标注
    img_annotations = {}
    for ann in coco_data['annotations']:
        # 跳过 crowd 标注（iscrowd=1）
        if ann.get('iscrowd', 0) == 1:
            continue
        
        # 只保留 bicycle
        if ann['category_id'] != bicycle_cat_id:
            continue
        
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    print(f"找到 {len(img_annotations)} 张包含 bicycle 的图片")
    
    # 处理每张图片
    valid_count = 0
    total_boxes = 0
    
    for img_id, anns in tqdm(img_annotations.items(), desc=f"处理 {split} 集"):
        img_info = images_dict[img_id]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 源图片路径
        src_img_path = img_dir / img_filename
        
        if not src_img_path.exists():
            print(f"警告: 图片不存在 {src_img_path}，跳过")
            continue
        
        # 目标路径
        dst_img_path = out_img_dir / img_filename
        label_path = out_label_dir / (Path(img_filename).stem + '.txt')
        
        # 复制或创建软链接
        try:
            if not dst_img_path.exists():
                # Windows 上直接复制
                if os.name == 'nt':  # Windows
                    shutil.copy2(src_img_path, dst_img_path)
                else:
                    # Linux/Mac 尝试创建软链接，失败则复制
                    try:
                        os.symlink(src_img_path, dst_img_path)
                    except (OSError, NotImplementedError):
                        shutil.copy2(src_img_path, dst_img_path)
        except Exception as e:
            print(f"警告: 无法复制图片 {src_img_path}: {e}")
            continue
        
        # 生成 YOLO 标签
        yolo_labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            yolo_bbox = convert_coco_to_yolo(bbox, img_width, img_height)
            
            if yolo_bbox is not None:
                # YOLO 格式: class_id x_center y_center w h (归一化)
                # 单类数据集，class_id 固定为 0
                yolo_labels.append(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                total_boxes += 1
        
        # 保存标签文件，即使没有有效 bbox 也创建空文件，但跳过这张图片
        if yolo_labels:
            with open(label_path, 'w') as f:
                f.writelines(yolo_labels)
            valid_count += 1
    
    print(f"\n{split} 集处理完成:")
    print(f"  - 有效图片数: {valid_count}")
    print(f"  - 总 bounding box 数: {total_boxes}")
    print(f"  - 输出目录: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='从 COCO 2017 提取 bicycle 类别生成 YOLO 数据集')
    parser.add_argument('--coco_root', type=str, required=True,
                        help='COCO 数据集根目录（包含 annotations/ train2017/ val2017/）')
    parser.add_argument('--out_dir', type=str, default='data/coco_bicycle',
                        help='输出 YOLO 数据集目录（默认: data/coco_bicycle）')
    
    args = parser.parse_args()
    
    # 处理训练集和验证集
    print("=" * 60)
    print("处理训练集...")
    print("=" * 60)
    prepare_coco_bicycle(args.coco_root, args.out_dir, split='train')
    
    print("\n" + "=" * 60)
    print("处理验证集...")
    print("=" * 60)
    prepare_coco_bicycle(args.coco_root, args.out_dir, split='val')
    
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print(f"输出目录: {args.out_dir}")
    print(f"目录结构:")
    print(f"  {args.out_dir}/")
    print(f"    images/")
    print(f"      train/")
    print(f"      val/")
    print(f"    labels/")
    print(f"      train/")
    print(f"      val/")


if __name__ == '__main__':
    main()

