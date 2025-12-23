---
comments: true
description: 学习如何使用 Neural Magic 的 DeepSparse 部署 YOLOv5，在 CPU 上实现 GPU 级性能。探索简单集成、灵活部署等功能。
keywords: YOLOv5, DeepSparse, Neural Magic, YOLO 部署, 稀疏推理, 深度学习, 模型稀疏性, CPU 优化, 无硬件加速器, AI 部署
---

# 使用 Neural Magic 的 DeepSparse 部署 YOLOv5

欢迎来到软件交付的 AI。

本指南介绍如何使用 Neural Magic 的 DeepSparse 部署 YOLOv5。

DeepSparse 是一个在 CPU 上具有卓越性能的推理运行时。例如，与 ONNX Runtime 基线相比，DeepSparse 为 YOLOv5s 提供了 5.8 倍的加速，在同一台机器上运行！

<p align="center">
  <img width="60%" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-speed-improvement.avif" alt="YOLOv5 速度提升">
</p>

首次，您的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)工作负载可以满足生产的性能需求，而无需硬件加速器的复杂性和成本。简而言之，DeepSparse 为您提供 GPU 的性能和软件的简单性：

- **灵活部署**：在云、数据中心和边缘一致运行，支持从 Intel 到 AMD 到 ARM 的任何硬件提供商
- **无限可扩展性**：垂直扩展到数百个核心，使用标准 Kubernetes 横向扩展，或使用 Serverless 完全抽象
- **轻松集成**：干净的 API 用于将模型集成到应用程序中并在生产中监控它

## DeepSparse 如何实现 GPU 级性能？

DeepSparse 利用模型稀疏性来获得性能加速。

通过剪枝和量化进行稀疏化是一种广泛研究的技术，可以将执行网络所需的大小和计算量减少一个数量级，同时保持高[准确率](https://www.ultralytics.com/glossary/accuracy)。DeepSparse 是稀疏感知的，这意味着它跳过归零的参数，减少前向传递中的计算量。由于稀疏计算现在是内存受限的，DeepSparse 深度执行网络，将问题分解为张量列（Tensor Columns），即适合缓存的垂直计算条带。

<p align="center">
  <img width="60%" src="https://github.com/ultralytics/docs/releases/download/0/tensor-columns.avif" alt="YOLO 模型剪枝">
</p>

具有压缩计算的稀疏网络，在缓存中深度执行，使 DeepSparse 能够在 CPU 上提供 GPU 级性能！

## 如何创建在我的数据上训练的稀疏版 YOLOv5？

Neural Magic 的开源模型仓库 [SparseZoo](https://github.com/neuralmagic/sparsezoo/blob/main/README.md) 包含每个 YOLOv5 模型的预稀疏化检查点。使用与 Ultralytics 集成的 [SparseML](https://github.com/neuralmagic/sparseml)，您可以通过单个 CLI 命令将稀疏检查点微调到您的数据上。

[查看 Neural Magic 的 YOLOv5 文档了解更多详情](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud)。

## DeepSparse 使用

我们将通过一个示例来演示使用 DeepSparse 对稀疏版 YOLOv5s 进行基准测试和部署。

### 安装 DeepSparse

运行以下命令安装 DeepSparse。我们建议您使用带有 Python 的虚拟环境。

```bash
pip install "deepsparse[server,yolo,onnxruntime]"
```

### 获取 ONNX 文件

DeepSparse 接受 ONNX 格式的模型，可以通过以下方式传递：

- SparseZoo stub，用于标识 SparseZoo 中的 ONNX 文件
- 文件系统中 ONNX 模型的本地路径

以下示例使用标准密集和剪枝量化的 YOLOv5s 检查点，由以下 SparseZoo stub 标识：

```bash
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
```


### 部署模型

DeepSparse 提供便捷的 API 用于将模型集成到应用程序中。

要尝试以下部署示例，请下载示例图像并将其保存为 `basilica.jpg`：

```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

#### Python API

`Pipelines` 在运行时周围包装预处理和输出后处理，为将 DeepSparse 添加到应用程序提供干净的接口。DeepSparse-Ultralytics 集成包含一个开箱即用的 `Pipeline`，接受原始图像并输出边界框。

创建 `Pipeline` 并运行推理：

```python
from deepsparse import Pipeline

# 本地文件系统中的图像列表
images = ["basilica.jpg"]

# 创建 Pipeline
model_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none"
yolo_pipeline = Pipeline.create(
    task="yolo",
    model_path=model_stub,
)

# 对图像运行推理，接收边界框 + 类别
pipeline_outputs = yolo_pipeline(images=images, iou_thres=0.6, conf_thres=0.001)
print(pipeline_outputs)
```

如果您在云中运行，可能会收到 open-cv 找不到 `libGL.so.1` 的错误。在 Ubuntu 上运行以下命令安装它：

```
apt-get install libgl1
```

#### HTTP 服务器

DeepSparse Server 运行在流行的 [FastAPI](https://fastapi.tiangolo.com/) Web 框架和 [Uvicorn](https://www.uvicorn.org/) Web 服务器之上。只需一个 CLI 命令，您就可以轻松设置带有 DeepSparse 的模型服务端点。Server 支持 DeepSparse 的任何 Pipeline，包括使用 YOLOv5 的[目标检测](https://www.ultralytics.com/glossary/object-detection)，使您能够向端点发送原始图像并接收边界框。

使用剪枝量化的 YOLOv5s 启动 Server：

```bash
deepsparse.server \
  --task yolo \
  --model_path zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
```

使用 Python 的 `requests` 包的示例请求：

```python
import json

import requests

# 用于推理的图像列表（客户端的本地文件）
path = ["basilica.jpg"]
files = [("request", open(img, "rb")) for img in path]

# 通过 HTTP 向 /predict/from_files 端点发送请求
url = "http://0.0.0.0:5543/predict/from_files"
resp = requests.post(url=url, files=files)

# 响应以 JSON 格式返回
annotations = json.loads(resp.text)  # 注释结果字典
bounding_boxes = annotations["boxes"]
labels = annotations["labels"]
```

#### 注释 CLI

您还可以使用 annotate 命令让引擎在磁盘上保存注释后的照片。尝试 `--source 0` 来注释您的实时网络摄像头画面！

```bash
deepsparse.object_detection.annotate --model_filepath zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none --source basilica.jpg
```

运行上述命令将创建一个 `annotation-results` 文件夹并将注释后的图像保存在其中。

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/basilica-annotated.avif" alt="已注释" width="60%">
</p>

## 性能基准测试

我们将使用 DeepSparse 的基准测试脚本比较 DeepSparse 和 ONNX Runtime 在 YOLOv5s 上的吞吐量。

基准测试在 AWS `c6i.8xlarge` 实例（16 核）上运行。

### 批次 32 性能比较

#### ONNX Runtime 基线

在批次 32 时，ONNX Runtime 使用标准密集 YOLOv5s 达到 42 图像/秒：

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 32 -nstreams 1 -e onnxruntime

# 原始模型路径: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
# 批次大小: 32
# 场景: sync
# 吞吐量 (items/sec): 41.9025
```

#### DeepSparse 密集性能

虽然 DeepSparse 在优化的稀疏模型上提供最佳性能，但它在标准密集 YOLOv5s 上也表现良好。

在批次 32 时，DeepSparse 使用标准密集 YOLOv5s 达到 70 图像/秒，**比 ORT 性能提升 1.7 倍**！

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 32 -nstreams 1

# 原始模型路径: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
# 批次大小: 32
# 场景: sync
# 吞吐量 (items/sec): 69.5546
```

#### DeepSparse 稀疏性能

当对模型应用稀疏性时，DeepSparse 相对于 ONNX Runtime 的性能提升更加显著。

在批次 32 时，DeepSparse 使用剪枝量化的 YOLOv5s 达到 241 图像/秒，**比 ORT 性能提升 5.8 倍**！

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none -s sync -b 32 -nstreams 1

# 原始模型路径: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
# 批次大小: 32
# 场景: sync
# 吞吐量 (items/sec): 241.2452
```

### 批次 1 性能比较

DeepSparse 还能够在延迟敏感的批次 1 场景中获得相对于 ONNX Runtime 的加速。

#### ONNX Runtime 基线

在批次 1 时，ONNX Runtime 使用标准密集 YOLOv5s 达到 48 图像/秒。

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1 -nstreams 1 -e onnxruntime

# 原始模型路径: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
# 批次大小: 1
# 场景: sync
# 吞吐量 (items/sec): 48.0921
```

#### DeepSparse 稀疏性能

在批次 1 时，DeepSparse 使用剪枝量化的 YOLOv5s 达到 135 项/秒，**比 ONNX Runtime 性能提升 2.8 倍！**

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none -s sync -b 1 -nstreams 1

# 原始模型路径: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
# 批次大小: 1
# 场景: sync
# 吞吐量 (items/sec): 134.9468
```

由于 `c6i.8xlarge` 实例具有 VNNI 指令，如果权重以 4 块为单位剪枝，DeepSparse 的吞吐量可以进一步提高。

在批次 1 时，DeepSparse 使用 4 块剪枝量化的 YOLOv5s 达到 180 项/秒，**比 ONNX Runtime 性能提升 3.7 倍！**

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned35_quant-none-vnni -s sync -b 1 -nstreams 1

# 原始模型路径: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned35_quant-none-vnni
# 批次大小: 1
# 场景: sync
# 吞吐量 (items/sec): 179.7375
```

## 开始使用 DeepSparse

**研究或测试？** DeepSparse Community 对研究和测试免费。从他们的[文档](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud)开始。

有关使用 DeepSparse 部署 YOLOv5 的更多信息，请查看 [Neural Magic 的 DeepSparse 文档](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud)和 [Ultralytics 关于 DeepSparse 集成的博客文章](https://www.ultralytics.com/blog/deploy-yolov5-with-neural-magics-deepsparse-for-gpu-class-performance-on-cpus)。
