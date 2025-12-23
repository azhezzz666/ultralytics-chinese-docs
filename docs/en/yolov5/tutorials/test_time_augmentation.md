---
comments: true
description: 使用测试时增强（TTA）提升您的 YOLOv5 性能。学习设置、测试和推理技术以提高 mAP 和召回率。
keywords: YOLOv5, 测试时增强, TTA, 机器学习, 深度学习, 目标检测, mAP, 召回率, PyTorch
---

# 测试时增强（TTA）

📚 本指南介绍如何在测试和推理期间使用测试时增强（TTA）来提高 YOLOv5 🚀 的 mAP 和[召回率](https://www.ultralytics.com/glossary/recall)。

## 开始之前

克隆仓库并在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)会从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)自动下载。

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆
cd yolov5
pip install -r requirements.txt # 安装
```

## 正常测试

在尝试 TTA 之前，我们想建立一个基线性能进行比较。此命令在图像大小 640 像素下测试 COCO val2017 上的 YOLOv5x。

```bash
python val.py --weights yolov5x.pt --data coco.yaml --img 640 --half
```

## 使用 TTA 测试

在任何现有的 `val.py` 命令后添加 `--augment` 以启用 TTA，并将图像大小增加约 30% 以获得更好的结果。

```bash
python val.py --weights yolov5x.pt --data coco.yaml --img 832 --augment --half
```


启用 TTA 的推理通常需要正常推理的 2-3 倍时间，因为图像会被左右翻转并以 3 种不同分辨率处理，输出在 [NMS](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) 之前合并。

## 使用 TTA 推理

`detect.py` TTA 推理与 `val.py` TTA 操作相同：只需在任何现有的 `detect.py` 命令后添加 `--augment`：

```bash
python detect.py --weights yolov5s.pt --img 832 --source data/images --augment
```

<img src="https://github.com/ultralytics/docs/releases/download/0/yolov5-test-time-augmentations.avif" width="500" alt="YOLOv5 测试时增强">

### PyTorch Hub TTA

TTA 自动集成到所有 [YOLOv5 PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/) 模型中，可以通过在推理时传递 `augment=True` 来访问。

```python
import torch

# 模型
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # 或 yolov5m、yolov5x、custom

# 图像
img = "https://ultralytics.com/images/zidane.jpg"  # 或文件、PIL、OpenCV、numpy、多个

# 推理
results = model(img, augment=True)  # <--- TTA 推理

# 结果
results.print()  # 或 .show()、.save()、.crop()、.pandas() 等
```

### 自定义

您可以在 [YOLOv5 `forward_augment()` 方法](https://github.com/ultralytics/yolov5/blob/8c6f9e15bfc0000d18b976a95b9d7c17d407ec91/models/yolo.py#L125-L137)中自定义应用的 TTA 操作。

## 测试时增强的优势

测试时增强为[目标检测](https://www.ultralytics.com/glossary/object-detection)任务提供了几个关键优势：

- **提高准确率**：如上面的结果所示，TTA 将 mAP 从 0.504 提高到 0.516，mAR 从 0.681 提高到 0.696。
- **更好的小目标检测**：TTA 特别增强了小目标的检测，小面积 AP 从 0.351 提高到 0.361。
- **增强鲁棒性**：通过测试每张图像的多个变体，TTA 减少了视角、光照和其他环境因素的影响。
- **简单实现**：只需在现有命令中添加 `--augment` 标志。

权衡是推理时间增加，使 TTA 更适合准确率优先于速度的应用。

## 支持的环境

Ultralytics 提供一系列开箱即用的环境，每个环境都预装了 [CUDA](https://developer.nvidia.com/cuda)、[CUDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/) 等基本依赖项，以便快速启动您的项目。

- **免费 GPU Notebook**：<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="在 Gradient 上运行"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>
- **Google Cloud**：[GCP 快速入门指南](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**：[AWS 快速入门指南](../environments/aws_quickstart_tutorial.md)
- **Azure**：[AzureML 快速入门指南](../environments/azureml_quickstart_tutorial.md)
- **Docker**：[Docker 快速入门指南](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## 项目状态

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

此徽章表示所有 [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) 持续集成（CI）测试均成功通过。
