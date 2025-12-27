# 课程实验四：校园共享单车检测

基于 **ultralytics YOLOv8** 的共享单车目标检测项目，使用 COCO 2017 数据集中的 bicycle 类别进行训练和推理。

## 项目简介

本项目实现了一个轻量级的共享单车检测系统，使用 YOLOv8 模型在校园场景图片中检测共享单车的位置。项目结构简单清晰，代码可直接运行，便于理解和复现。

- **输入**：校园场景图片（道路/停车区等）
- **输出**：共享单车位置（bounding boxes）
- **数据集**：COCO 2017，只关注 bicycle 类别（COCO class id = 2，类名 "bicycle"）
- **模型**：YOLOv8n（轻量级，适合快速训练和部署）

## 环境安装

### 1. 创建 Conda 环境（推荐）

```bash
conda create -n cj python=3.10
conda activate cj
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "from ultralytics import YOLO; print('YOLOv8 安装成功')"
```

### 环境要求

- Python >= 3.8
- PyTorch >= 1.8.0（会自动安装，支持 CPU/GPU）
- CUDA（可选，用于 GPU 加速）

## 数据集准备

### 下载 COCO 2017 数据集

请从 [COCO 官网](https://cocodataset.org/#download) 下载以下文件：

1. **标注文件**（annotations）：

   - `instances_train2017.json`（训练集标注，约 241MB）
   - `instances_val2017.json`（验证集标注，约 1GB）

2. **图片文件**：
   - `train2017.zip`（训练集图片，约 18GB，解压后约 118287 张）
   - `val2017.zip`（验证集图片，约 1GB，解压后约 5000 张）

### COCO 数据集存放路径

**COCO 数据集可以放在任意位置**，建议两种方式：

#### 方式 1：放在项目外部（推荐）

如果数据集很大，建议放在项目外部，例如：

```
D:/datasets/coco/          # 或 C:/datasets/coco/，任意路径都可以
  ├── annotations/
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  ├── train2017/
  │   ├── 000000000009.jpg
  │   ├── 000000000025.jpg
  │   └── ... (118287 张图片)
  └── val2017/
      ├── 000000000139.jpg
      ├── 000000000285.jpg
      └── ... (5000 张图片)
```

使用时指定绝对路径：

```bash
python scripts/prepare_coco_bicycle.py --coco_root D:/datasets/coco --out_dir data/coco_bicycle
```

#### 方式 2：放在项目内部

如果空间足够，也可以放在项目目录下：

```
Campus-Bike-Detection/
  ├── coco/                 # COCO 数据集目录
  │   ├── annotations/
  │   ├── train2017/
  │   └── val2017/
  ├── data/
  ├── scripts/
  └── ...
```

使用时指定相对路径：

```bash
python scripts/prepare_coco_bicycle.py --coco_root coco --out_dir data/coco_bicycle
```

### 目录结构要求

**重要：** COCO 数据集必须按照以下结构组织：

```
<你的COCO路径>/
  ├── annotations/
  │   ├── instances_train2017.json    # 必须存在
  │   └── instances_val2017.json       # 必须存在
  ├── train2017/                       # 必须存在
  │   ├── 000000000009.jpg
  │   ├── 000000000025.jpg
  │   └── ... (所有训练图片)
  └── val2017/                         # 必须存在
      ├── 000000000139.jpg
      ├── 000000000285.jpg
      └── ... (所有验证图片)
```

### 验证数据集路径

运行数据准备脚本前，可以手动检查路径是否正确：

**Windows:**

```powershell
# 检查标注文件
dir D:\datasets\coco\annotations\instances_train2017.json
dir D:\datasets\coco\annotations\instances_val2017.json

# 检查图片目录
dir D:\datasets\coco\train2017 | measure-object  # 应该看到约 118287 个文件
dir D:\datasets\coco\val2017 | measure-object     # 应该看到约 5000 个文件
```

**Linux/Mac:**

```bash
# 检查标注文件
ls /path/to/coco/annotations/instances_train2017.json
ls /path/to/coco/annotations/instances_val2017.json

# 检查图片数量
ls /path/to/coco/train2017 | wc -l  # 应该约 118287
ls /path/to/coco/val2017 | wc -l    # 应该约 5000
```

## 快速开始

### 步骤 1：数据准备

从 COCO 数据集中提取 bicycle 类别，生成 YOLO 格式的单类数据集：

```bash
python scripts/prepare_coco_bicycle.py --coco_root ~/autodl-tmp/COCO2017 --out_dir data/coco_bicycle
```

**参数说明：**

- `--coco_root`: COCO 数据集根目录（包含 `annotations/`、`train2017/`、`val2017/`）
- `--out_dir`: 输出 YOLO 数据集目录（默认: `data/coco_bicycle`）

**输出结构：**

```
data/coco_bicycle/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
```

**注意：**

- Windows 系统不支持软链接，脚本会自动使用文件复制
- 处理时间取决于数据集大小（约 10-30 分钟）

### 步骤 2：训练模型

#### 模型选择

YOLOv8 提供 5 种不同规模的模型，可根据需求选择：

| 模型         | 参数量 | 速度 | 精度 | 推荐场景                 |
| ------------ | ------ | ---- | ---- | ------------------------ |
| `yolov8n.pt` | 3.2M   | 最快 | 较低 | 快速测试、CPU 推理       |
| `yolov8s.pt` | 11.2M  | 快   | 中等 | **推荐：平衡速度和精度** |
| `yolov8m.pt` | 25.9M  | 中等 | 较高 | 追求更高精度             |
| `yolov8l.pt` | 43.7M  | 较慢 | 高   | 高精度需求               |
| `yolov8x.pt` | 68.2M  | 最慢 | 最高 | 最高精度要求             |

**推荐配置：**

- **快速训练/测试**：使用 `yolov8n.pt`（默认）
- **平衡性能**：使用 `yolov8s.pt` 或 `yolov8m.pt`（推荐）
- **追求精度**：使用 `yolov8l.pt` 或 `yolov8x.pt`

#### 训练命令

**使用默认模型（yolov8n，轻量快速）：**

```bash
python scripts/train.py --data data/coco_bicycle.yaml --model yolov8n.pt --epochs 20 --batch 16 --device 0
```

**使用更强大的模型（推荐 yolov8s 或 yolov8m）：**

```bash
# 使用 yolov8s（推荐：平衡速度和精度）
python scripts/train.py --data data/coco_bicycle.yaml --model yolov8s.pt --epochs 20 --batch 16 --device 0

# 使用 yolov8m（更高精度）
python scripts/train.py --data data/coco_bicycle.yaml --model yolov8m.pt --epochs 20 --batch 12 --device 0

# 使用 yolov8l（高精度）
python scripts/train.py --data data/coco_bicycle.yaml --model yolov8l.pt --epochs 20 --batch 8 --device 0

# 使用 yolov8x（最高精度）
python scripts/train.py --data data/coco_bicycle.yaml --model yolov8x.pt --epochs 20 --batch 4 --device 0
```

**参数说明：**

- `--data`: 数据集配置文件（默认: `data/coco_bicycle.yaml`）
- `--model`: 模型文件（默认: `yolov8n.pt`，可选: `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`）
- `--epochs`: 训练轮数（默认: 20，大模型可适当增加到 30-50）
- `--batch`: 批次大小（默认: 16，大模型需要减小，建议：n/s=16, m=12, l=8, x=4）
- `--device`: 设备（默认: 自动选择，可选: `0`, `1`, `cpu`）
- `--project`: 项目目录（默认: `runs`）
- `--name`: 实验名称（默认: `bicycle_exp`）

**注意：**

- 模型越大，训练时间越长，显存占用越多
- 如果显存不足，减小 `--batch` 参数
- 大模型通常需要更多训练轮数才能收敛（建议 30-50 epochs）

**训练输出：**

- 模型权重保存在 `runs/bicycle_exp/weights/`
  - `best.pt`: 验证集上表现最好的模型
  - `last.pt`: 最后一轮的模型
- 训练日志和可视化保存在 `runs/bicycle_exp/`

### 步骤 3：验证模型

在验证集上评估模型性能：

```bash
python scripts/val.py --weights runs/bicycle_exp/weights/best.pt --data data/coco_bicycle.yaml
```

**输出指标：**

- `mAP50`: 在 IoU=0.5 时的平均精度
- `mAP50-95`: 在 IoU=0.5:0.95 时的平均精度
- `Precision`: 精确率
- `Recall`: 召回率

### 步骤 4：推理检测

对单张图片或图片目录进行推理：

```bash
python scripts/infer.py --weights runs/bicycle_exp/weights/best.pt --source D:\Project\Campus-Bike-Detection\bike.jpg --conf 0.25 --save_dir outputs/vis --save_txt
```

**参数说明：**

- `--weights`: 模型权重路径
- `--source`: 输入图片路径或目录
- `--conf`: 置信度阈值（默认: 0.25）
- `--iou`: IoU 阈值（默认: 0.45）
- `--save_dir`: 可视化结果保存目录（默认: `outputs/vis`）
- `--save_txt`: 是否保存检测结果到 txt/json 文件

**输出：**

- 可视化图片保存在 `outputs/vis/`
- 如果使用 `--save_txt`，还会生成 `.txt`（YOLO 格式）和 `.json` 文件

## 项目结构

```
Campus-Bike-Detection/
  ├── README.md                    # 项目说明文档
  ├── requirements.txt             # Python 依赖
  ├── data/
  │   └── coco_bicycle.yaml        # 数据集配置文件
  ├── scripts/
  │   ├── prepare_coco_bicycle.py  # 数据准备脚本
  │   ├── train.py                 # 训练脚本
  │   ├── val.py                   # 验证脚本
  │   └── infer.py                 # 推理脚本
  ├── src/
  │   └── utils.py                 # 工具函数（可视化、过滤等）
  ├── runs/                        # 训练输出（自动生成）
  └── outputs/                     # 推理输出（自动生成）
      └── vis/
```

## 输出目录说明

### `runs/`

YOLOv8 训练过程中自动生成的目录，包含：

- `weights/`: 模型权重文件
- `train/`: 训练过程可视化
- `val/`: 验证结果可视化

### `outputs/vis/`

推理结果保存目录，包含：

- 可视化图片（带检测框）
- `.txt` 文件（YOLO 格式标签，如果使用 `--save_txt`）
- `.json` 文件（检测结果详情，如果使用 `--save_txt`）

## 算法分析

### YOLOv8 检测流程

1. **特征提取**：使用 CSPDarknet 骨干网络提取多尺度特征
2. **检测头**：通过 PAN-FPN 结构融合不同尺度的特征，生成检测预测
3. **后处理**：
   - **NMS (Non-Maximum Suppression)**：去除重复检测框
   - **置信度过滤**：根据阈值过滤低置信度检测
4. **输出**：返回 bounding boxes（坐标、类别、置信度）

### 为什么选择 YOLOv8

- **模型系列丰富**：提供 n/s/m/l/x 五种规模，可根据需求选择
- **高精度**：在 COCO 数据集上表现优异（YOLOv8n mAP 37.3，YOLOv8x mAP 53.9）
- **易部署**：支持 ONNX、TensorRT 等格式导出
- **简单易用**：ultralytics 提供了简洁的 API，便于快速开发

### 模型选择建议

- **YOLOv8n（nano）**：适合快速测试、CPU 推理、资源受限场景
- **YOLOv8s（small）**：**推荐**，平衡速度和精度，适合大多数应用
- **YOLOv8m（medium）**：追求更高精度时的好选择
- **YOLOv8l/x（large/xlarge）**：最高精度需求，需要更多计算资源

### 单类检测策略

本项目采用**方式 A（推荐）**：

- 从 COCO 标注中过滤出 bicycle 类别
- 生成单类 YOLO 格式数据集（class_id = 0 表示 bicycle）
- 训练时只学习 bicycle 特征，避免其他类别干扰
- 推理时直接输出 bicycle 检测结果

**优势：**

- 模型专注于 bicycle 检测，精度更高
- 推理速度更快（单类输出）
- 代码逻辑更清晰

## 环境配置说明

### GPU 配置（推荐）

如果使用 GPU 训练，需要安装 CUDA 版本的 PyTorch：

```bash
# 查看 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch（示例：CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CPU 配置

如果只有 CPU，也可以运行（速度较慢）：

```bash
python scripts/train.py \
    --data data/coco_bicycle.yaml \
    --model yolov8n.pt \
    --epochs 5 \
    --batch 4 \
    --device cpu
```

**注意：** CPU 训练建议使用较小的 `batch` 和 `epochs` 进行测试。

## 常见问题

### 1. 数据准备脚本运行失败

**问题**：`FileNotFoundError: 标注文件不存在`

**解决**：检查 `--coco_root` 路径是否正确，确保包含 `annotations/` 目录。

### 2. 训练时显存不足

**解决**：减小 `--batch` 参数（例如改为 8 或 4），或使用更小的模型（`yolov8n.pt`）。

### 3. Windows 上软链接问题

**说明**：Windows 系统不支持软链接，脚本会自动使用文件复制。如果图片很多，可能需要较长时间。

### 4. 推理结果中没有检测到 bicycle

**可能原因**：

- 置信度阈值过高，尝试降低 `--conf`（例如 0.1）
- 模型训练不充分，增加训练轮数
- 输入图片与训练数据分布差异较大

## 提交内容

- ✅ 完整源码（所有 Python 脚本）
- ✅ 训练出的模型权重（`runs/bicycle_exp/weights/best.pt`）
- ✅ 推理可视化结果（`outputs/vis/`）
- ✅ 算法分析与环境说明（本文档）

## 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [COCO 数据集官网](https://cocodataset.org/)
- [YOLO 论文](https://arxiv.org/abs/1506.02640)

## 许可证

本项目仅用于课程实验，请遵守 COCO 数据集的使用许可。
