---
comments: true
description: 学习如何从 PyTorch Hub 加载 YOLOv5 以实现无缝模型推理和自定义。按照 Ultralytics 文档的分步指南操作。
keywords: YOLOv5, PyTorch Hub, 模型加载, Ultralytics, 目标检测, 机器学习, AI, 教程, 推理
---

# 从 PyTorch Hub 加载 YOLOv5

📚 本指南介绍如何从 [PyTorch](https://www.ultralytics.com/glossary/pytorch) Hub 加载 YOLOv5 🚀，网址为 [https://pytorch.org/hub/ultralytics_yolov5](https://pytorch.org/hub/ultralytics_yolov5)。

## 开始之前

在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)会从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)自动下载。

```bash
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
```

💡 专业提示：**不需要**克隆 [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) 😃

## 使用 PyTorch Hub 加载 YOLOv5

### 简单示例

此示例从 PyTorch Hub 加载预训练的 YOLOv5s 模型作为 `model` 并传递图像进行推理。`'yolov5s'` 是最轻量和最快的 YOLOv5 模型。有关所有可用模型的详细信息，请参阅 [README](https://github.com/ultralytics/yolov5#pretrained-checkpoints)。

```python
import torch

# 模型
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# 图像
im = "https://ultralytics.com/images/zidane.jpg"

# 推理
results = model(im)

results.pandas().xyxy[0]
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
```

### 详细示例

此示例展示使用 **PIL** 和 **[OpenCV](https://www.ultralytics.com/glossary/opencv)** 图像源的**批量推理**。`results` 可以**打印**到控制台、**保存**到 `runs/hub`、在支持的环境中**显示**到屏幕，并作为**张量**或 **pandas** 数据帧返回。

```python
import cv2
import torch
from PIL import Image

# 模型
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# 图像
for f in "zidane.jpg", "bus.jpg":
    torch.hub.download_url_to_file("https://ultralytics.com/images/" + f, f)  # 下载 2 张图像
im1 = Image.open("zidane.jpg")  # PIL 图像
im2 = cv2.imread("bus.jpg")[..., ::-1]  # OpenCV 图像（BGR 转 RGB）

# 推理
results = model([im1, im2], size=640)  # 图像批次

# 结果
results.print()
results.save()  # 或 .show()

results.xyxy[0]  # im1 预测（张量）
results.pandas().xyxy[0]  # im1 预测（pandas）
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
```

<img src="https://github.com/ultralytics/docs/releases/download/0/yolo-inference-results-zidane.avif" width="500" alt="YOLO 在 zidane.jpg 上的推理结果">
<img src="https://github.com/ultralytics/docs/releases/download/0/yolo-inference-results-on-bus.avif" width="300" alt="YOLO 在 bus.jpg 上的推理结果">

有关所有推理选项，请参阅 YOLOv5 `AutoShape()` forward [方法](https://github.com/ultralytics/yolov5/blob/30e4c4f09297b67afedf8b2bcd851833ddc9dead/models/common.py#L243-L252)。

### 推理设置

YOLOv5 模型包含各种推理属性，如**置信度阈值**、**IoU 阈值**等，可以通过以下方式设置：

```python
model.conf = 0.25  # NMS 置信度阈值
model.iou = 0.45  # NMS IoU 阈值
model.agnostic = False  # NMS 类别无关
model.multi_label = False  # NMS 每个框多个标签
model.classes = None  # （可选列表）按类别过滤，例如 = [0, 15, 16] 表示 COCO 的人、猫和狗
model.max_det = 1000  # 每张图像的最大检测数
model.amp = False  # 自动混合精度（AMP）推理

results = model(im, size=320)  # 自定义推理大小
```

### 设备

模型创建后可以转移到任何设备：

```python
model.cpu()  # CPU
model.cuda()  # GPU
model.to(device)  # 例如 device=torch.device(0)
```

模型也可以直接在任何 `device` 上创建：

```python
model = torch.hub.load("ultralytics/yolov5", "yolov5s", device="cpu")  # 在 CPU 上加载
```

💡 专业提示：输入图像在推理前会自动转移到正确的模型设备。


### 静默输出

可以使用 `_verbose=False` 静默加载模型：

```python
model = torch.hub.load("ultralytics/yolov5", "yolov5s", _verbose=False)  # 静默加载
```

### 输入通道

要加载具有 4 个输入通道而不是默认 3 个的预训练 YOLOv5s 模型：

```python
model = torch.hub.load("ultralytics/yolov5", "yolov5s", channels=4)
```

在这种情况下，模型将由预训练权重组成，**除了**第一个输入层，它不再与预训练输入层具有相同的形状。输入层将保持由随机权重初始化。

### 类别数量

要加载具有 10 个输出类别而不是默认 80 个的预训练 YOLOv5s 模型：

```python
model = torch.hub.load("ultralytics/yolov5", "yolov5s", classes=10)
```

在这种情况下，模型将由预训练权重组成，**除了**输出层，它们不再与预训练输出层具有相同的形状。输出层将保持由随机权重初始化。

### 强制重新加载

如果您在上述步骤中遇到问题，设置 `force_reload=True` 可能会有所帮助，它会丢弃现有缓存并强制从 PyTorch Hub 重新下载最新的 YOLOv5 版本。缓存副本位于 `~/.cache/torch/hub`；删除该文件夹可达到相同效果。

```python
model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # 强制重新加载
```

### 截图推理

要在桌面屏幕上运行推理：

```python
import torch
from PIL import ImageGrab

# 模型
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# 图像
im = ImageGrab.grab()  # 截取屏幕截图

# 推理
results = model(im)
```

### 多 GPU 推理

YOLOv5 模型可以使用线程推理并行加载到多个 GPU：

```python
import threading

import torch


def run(model, im):
    """使用给定模型对图像执行推理并保存输出；模型必须支持 `.save()` 方法。"""
    results = model(im)
    results.save()


# 模型
model0 = torch.hub.load("ultralytics/yolov5", "yolov5s", device=0)
model1 = torch.hub.load("ultralytics/yolov5", "yolov5s", device=1)

# 推理
threading.Thread(target=run, args=[model0, "https://ultralytics.com/images/zidane.jpg"], daemon=True).start()
threading.Thread(target=run, args=[model1, "https://ultralytics.com/images/bus.jpg"], daemon=True).start()
```

### 训练

要加载用于训练而非推理的 YOLOv5 模型，请设置 `autoshape=False`。要加载具有随机初始化权重的模型（从头开始训练），请使用 `pretrained=False`。在这种情况下，您必须提供自己的训练脚本。或者参阅我们的 YOLOv5 [训练自定义数据教程](./train_custom_data.md)进行模型训练。

```python
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s", autoshape=False)  # 加载预训练
model = torch.hub.load("ultralytics/yolov5", "yolov5s", autoshape=False, pretrained=False)  # 从头加载
```

### Base64 结果

用于 API 服务。详情参见 [Flask REST API](https://github.com/ultralytics/yolov5/tree/master/utils/flask_rest_api) 示例。

```python
import base64
from io import BytesIO

from PIL import Image

results = model(im)  # 推理

results.ims  # 传递给模型进行推理的原始图像数组（作为 np 数组）
results.render()  # 使用边界框和标签更新 results.ims
for im in results.ims:
    buffered = BytesIO()
    im_base64 = Image.fromarray(im)
    im_base64.save(buffered, format="JPEG")
    print(base64.b64encode(buffered.getvalue()).decode("utf-8"))  # 带结果的 base64 编码图像
```

### 裁剪结果

结果可以作为检测裁剪返回和保存：

```python
results = model(im)  # 推理
crops = results.crop(save=True)  # 裁剪的检测字典
```

### Pandas 结果

结果可以作为 [Pandas DataFrames](https://pandas.pydata.org/) 返回：

```python
results = model(im)  # 推理
results.pandas().xyxy[0]  # Pandas DataFrame
```

<details>
  <summary>Pandas 输出（点击展开）</summary>

```python
print(results.pandas().xyxy[0])
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
```

</details>

### 排序结果

结果可以按列排序，例如按从左到右（x 轴）排序车牌数字检测：

```python
results = model(im)  # 推理
results.pandas().xyxy[0].sort_values("xmin")  # 从左到右排序
```

### JSON 结果

使用 `.to_json()` 方法转换为 `.pandas()` 数据帧后，结果可以以 JSON 格式返回。可以使用 `orient` 参数修改 JSON 格式。详情参见 pandas `.to_json()` [文档](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html)。

```python
results = model(ims)  # 推理
results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 预测
```

<details>
  <summary>JSON 输出（点击展开）</summary>

```json
[
    {
        "xmin": 749.5,
        "ymin": 43.5,
        "xmax": 1148.0,
        "ymax": 704.5,
        "confidence": 0.8740234375,
        "class": 0,
        "name": "person"
    },
    {
        "xmin": 433.5,
        "ymin": 433.5,
        "xmax": 517.5,
        "ymax": 714.5,
        "confidence": 0.6879882812,
        "class": 27,
        "name": "tie"
    },
    {
        "xmin": 115.25,
        "ymin": 195.75,
        "xmax": 1096.0,
        "ymax": 708.0,
        "confidence": 0.6254882812,
        "class": 0,
        "name": "person"
    },
    {
        "xmin": 986.0,
        "ymin": 304.0,
        "xmax": 1028.0,
        "ymax": 420.0,
        "confidence": 0.2873535156,
        "class": 27,
        "name": "tie"
    }
]
```

</details>

## 自定义模型

此示例使用 PyTorch Hub 加载自定义 20 类 [VOC](https://github.com/ultralytics/yolov5/blob/master/data/VOC.yaml) 训练的 YOLOv5s 模型 `'best.pt'`。

```python
import torch

model = torch.hub.load("ultralytics/yolov5", "custom", path="path/to/best.pt")  # 本地模型
model = torch.hub.load("path/to/yolov5", "custom", path="path/to/best.pt", source="local")  # 本地仓库
```

## TensorRT、ONNX 和 OpenVINO 模型

PyTorch Hub 支持大多数 YOLOv5 导出格式的推理，包括自定义训练的模型。有关导出模型的详细信息，请参阅 [TFLite、ONNX、CoreML、TensorRT 导出教程](./model_export.md)。

💡 专业提示：**TensorRT** 在 [**GPU 基准测试**](https://github.com/ultralytics/yolov5/pull/6963)上可能比 PyTorch 快 2-5 倍
💡 专业提示：**ONNX** 和 **OpenVINO** 在 [**CPU 基准测试**](https://github.com/ultralytics/yolov5/pull/6613)上可能比 PyTorch 快 2-3 倍

```python
import torch

model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt")  # PyTorch
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.torchscript")  # TorchScript
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.onnx")  # ONNX
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s_openvino_model/")  # OpenVINO
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.engine")  # TensorRT
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.mlmodel")  # CoreML（仅 macOS）
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.tflite")  # TFLite
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s_paddle_model/")  # PaddlePaddle
```

## 支持的环境

Ultralytics 提供一系列开箱即用的环境，每个环境都预装了 [CUDA](https://developer.nvidia.com/cuda)、[CUDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/) 等基本依赖项，以便快速启动您的项目。

- **免费 GPU Notebook**：<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="在 Gradient 上运行"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>
- **Google Cloud**：[GCP 快速入门指南](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**：[AWS 快速入门指南](../environments/aws_quickstart_tutorial.md)
- **Azure**：[AzureML 快速入门指南](../environments/azureml_quickstart_tutorial.md)
- **Docker**：[Docker 快速入门指南](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## 项目状态

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

此徽章表示所有 [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) 持续集成（CI）测试均成功通过。这些 CI 测试严格检查 YOLOv5 在各个关键方面的功能和性能：[训练](https://github.com/ultralytics/yolov5/blob/master/train.py)、[验证](https://github.com/ultralytics/yolov5/blob/master/val.py)、[推理](https://github.com/ultralytics/yolov5/blob/master/detect.py)、[导出](https://github.com/ultralytics/yolov5/blob/master/export.py)和[基准测试](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py)。它们确保在 macOS、Windows 和 Ubuntu 上的一致可靠运行，测试每 24 小时进行一次，并在每次新提交时进行。
