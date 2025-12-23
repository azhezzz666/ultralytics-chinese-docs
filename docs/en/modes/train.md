---
comments: true
description: 了解如何使用 YOLO11 高效训练目标检测模型，包括设置、数据增强和硬件利用的全面说明。
keywords: Ultralytics, YOLO11, 模型训练, 深度学习, 目标检测, GPU 训练, 数据集增强, 超参数调优, 模型性能, Apple Silicon 训练
---

# 使用 Ultralytics YOLO 进行模型训练

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统和集成">

## 简介

训练[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型涉及向其提供数据并调整其参数，使其能够做出准确的预测。Ultralytics YOLO11 中的训练模式专为有效和高效地训练目标检测模型而设计，充分利用现代硬件能力。本指南旨在涵盖使用 YOLO11 强大功能集开始训练自己模型所需的所有细节。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何在 Google Colab 中使用自定义数据集训练 YOLO 模型。
</p>

## 为什么选择 Ultralytics YOLO 进行训练？

以下是选择 YOLO11 训练模式的一些令人信服的理由：

- **效率：** 充分利用您的硬件，无论是单 GPU 设置还是跨多个 GPU 扩展。
- **多功能性：** 除了 COCO、VOC 和 ImageNet 等现成数据集外，还可以在自定义数据集上训练。
- **用户友好：** 简单但强大的 CLI 和 Python 接口，提供直接的训练体验。
- **超参数灵活性：** 广泛的可自定义超参数，用于微调模型性能。

### 训练模式的关键特性

以下是 YOLO11 训练模式的一些显著特性：

- **自动数据集下载：** COCO、VOC 和 ImageNet 等标准数据集在首次使用时自动下载。
- **多 GPU 支持：** 无缝地跨多个 GPU 扩展训练工作以加快进程。
- **超参数配置：** 通过 YAML 配置文件或 CLI 参数修改超参数的选项。
- **可视化和监控：** 实时跟踪训练指标并可视化学习过程以获得更好的见解。

!!! tip

    * YOLO11 数据集如 COCO、VOC、ImageNet 等在首次使用时自动下载，即 `yolo train data=coco.yaml`

## 使用示例

在 COCO8 数据集上以图像大小 640 训练 YOLO11n 100 个[轮次](https://www.ultralytics.com/glossary/epoch)。可以使用 `device` 参数指定训练设备。如果未传递参数，将在可用时使用 GPU `device=0`；否则将使用 `device='cpu'`。有关训练参数的完整列表，请参阅下面的参数部分。

!!! warning "Windows 多进程错误"

    在 Windows 上，作为脚本启动训练时可能会收到 `RuntimeError`。在训练代码之前添加 `if __name__ == "__main__":` 块以解决此问题。

!!! example "单 GPU 和 CPU 训练示例"

    设备自动确定。如果 GPU 可用，将使用它（默认 CUDA 设备 0）；否则训练将在 CPU 上开始。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # 从 YAML 构建并转移权重

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640

        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640

        # 从 YAML 构建新模型，转移预训练权重并开始训练
        yolo detect train data=coco8.yaml model=yolo11n.yaml pretrained=yolo11n.pt epochs=100 imgsz=640
        ```

### 多 GPU 训练

多 GPU 训练通过将训练负载分布到多个 GPU 上，可以更有效地利用可用硬件资源。此功能可通过 Python API 和命令行界面使用。要启用多 GPU 训练，请指定要使用的 GPU 设备 ID。

!!! example "多 GPU 训练示例"

    要使用 2 个 GPU（CUDA 设备 0 和 1）进行训练，请使用以下命令。根据需要扩展到更多 GPU。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 使用 2 个 GPU 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])

        # 使用两个最空闲的 GPU 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[-1, -1])
        ```

    === "CLI"

        ```bash
        # 使用 GPU 0 和 1 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=0,1

        # 使用两个最空闲的 GPU
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=-1,-1
        ```

### Apple Silicon MPS 训练

通过在 Ultralytics YOLO 模型中集成对 Apple Silicon 芯片的支持，现在可以在使用强大的 Metal Performance Shaders (MPS) 框架的设备上训练模型。MPS 提供了一种在 Apple 定制芯片上执行计算和图像处理任务的高性能方式。

要在 Apple Silicon 芯片上启用训练，请在启动训练过程时将 'mps' 指定为您的设备。以下是如何在 Python 和命令行中执行此操作的示例：

!!! example "MPS 训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 使用 MPS 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        # 使用 MPS 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=mps
        ```

### 恢复中断的训练

从先前保存的状态恢复训练是处理深度学习模型时的一个关键功能。这在各种场景中都很有用，例如当训练过程意外中断时，或者当您希望使用新数据或更多轮次继续训练模型时。

恢复训练时，Ultralytics YOLO 会加载上次保存模型的权重，并恢复优化器状态、[学习率](https://www.ultralytics.com/glossary/learning-rate)调度器和轮次编号。这允许您从中断的地方无缝继续训练过程。

!!! example "恢复训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/last.pt")  # 加载部分训练的模型

        # 恢复训练
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        # 恢复中断的训练
        yolo train resume model=path/to/last.pt
        ```

## 训练设置

YOLO 模型的训练设置包括训练过程中使用的各种超参数和配置。这些设置影响模型的性能、速度和[精度](https://www.ultralytics.com/glossary/accuracy)。关键训练设置包括批次大小、学习率、动量和权重衰减。此外，优化器的选择、[损失函数](https://www.ultralytics.com/glossary/loss-function)和训练数据集组成也会影响训练过程。仔细调整和实验这些设置对于优化性能至关重要。

{% include "macros/train-args.md" %}

## 数据增强设置和超参数

数据增强技术对于通过向[训练数据](https://www.ultralytics.com/glossary/training-data)引入变化来提高 YOLO 模型的鲁棒性和性能至关重要，帮助模型更好地泛化到未见过的数据。下表概述了每个增强参数的目的和效果：

{% include "macros/augmentation-args.md" %}

## 日志记录

在训练 YOLO11 模型时，您可能会发现跟踪模型随时间的性能很有价值。这就是日志记录发挥作用的地方。Ultralytics YOLO 支持三种类型的日志记录器 - [Comet](../integrations/comet.md)、[ClearML](../integrations/clearml.md) 和 [TensorBoard](../integrations/tensorboard.md)。

## 常见问题

### 如何使用 Ultralytics YOLO11 训练[目标检测](https://www.ultralytics.com/glossary/object-detection)模型？

要使用 Ultralytics YOLO11 训练目标检测模型，您可以使用 Python API 或 CLI。以下是两者的示例：

!!! example "单 GPU 和 CPU 训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关更多详细信息，请参阅[训练设置](#训练设置)部分。

### 我可以在 Apple Silicon 芯片上训练 YOLO11 模型吗？

是的，Ultralytics YOLO11 支持使用 Metal Performance Shaders (MPS) 框架在 Apple Silicon 芯片上训练。将 'mps' 指定为您的训练设备。

!!! example "MPS 训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 在 Apple Silicon 芯片 (M1/M2/M3/M4) 上训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=mps
        ```

有关更多详细信息，请参阅 [Apple Silicon MPS 训练](#apple-silicon-mps-训练)部分。
