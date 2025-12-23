---
comments: true
description: 探索 Ultralytics YOLO11 的多种模式，包括训练、验证、预测、导出、跟踪和基准测试。最大化模型性能和效率。
keywords: Ultralytics, YOLO11, 机器学习, 模型训练, 验证, 预测, 导出, 跟踪, 基准测试, 目标检测
---

# Ultralytics YOLO11 模式

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统和集成">

## 简介

Ultralytics YOLO11 不仅仅是另一个目标检测模型；它是一个多功能框架，旨在覆盖[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型的整个生命周期——从数据摄取和模型训练到验证、部署和实际跟踪。每种模式都有特定的用途，旨在为您提供不同任务和用例所需的灵活性和效率。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics 模式教程：训练、验证、预测、导出和基准测试。
</p>

### 模式概览

了解 Ultralytics YOLO11 支持的不同**模式**对于充分利用您的模型至关重要：

- **训练**模式：在自定义或预加载的数据集上微调您的模型。
- **验证**模式：训练后的检查点，用于验证模型性能。
- **预测**模式：在真实世界数据上释放模型的预测能力。
- **导出**模式：以各种格式使您的[模型部署](https://www.ultralytics.com/glossary/model-deployment)就绪。
- **跟踪**模式：将您的目标检测模型扩展到实时跟踪应用。
- **基准测试**模式：在不同部署环境中分析模型的速度和精度。

本综合指南旨在为您提供每种模式的概述和实用见解，帮助您充分发挥 YOLO11 的潜力。

## [训练](train.md)

训练模式用于在自定义数据集上训练 YOLO11 模型。在此模式下，模型使用指定的数据集和超参数进行训练。训练过程涉及优化模型的参数，使其能够准确预测图像中目标的类别和位置。训练对于创建能够识别与您的应用相关的特定目标的模型至关重要。

[训练示例](train.md){ .md-button }

## [验证](val.md)

验证模式用于在训练后验证 YOLO11 模型。在此模式下，模型在验证集上进行评估，以衡量其精度和泛化性能。验证有助于识别潜在问题，如[过拟合](https://www.ultralytics.com/glossary/overfitting)，并提供[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) 等指标来量化模型性能。此模式对于调整超参数和提高整体模型效果至关重要。

[验证示例](val.md){ .md-button }

## [预测](predict.md)

预测模式用于使用训练好的 YOLO11 模型对新图像或视频进行预测。在此模式下，模型从检查点文件加载，用户可以提供图像或视频进行推理。模型识别并定位输入媒体中的目标，使其准备好用于实际应用。预测模式是将训练好的模型应用于解决实际问题的入口。

[预测示例](predict.md){ .md-button }

## [导出](export.md)

导出模式用于将 YOLO11 模型转换为适合在不同平台和设备上部署的格式。此模式将您的 PyTorch 模型转换为优化格式，如 ONNX、TensorRT 或 CoreML，从而能够在生产环境中部署。导出对于将模型与各种软件应用程序或硬件设备集成至关重要，通常会带来显著的性能改进。

[导出示例](export.md){ .md-button }

## [跟踪](track.md)

跟踪模式将 YOLO11 的目标检测能力扩展到跨视频帧或实时流跟踪目标。此模式对于需要持续目标识别的应用特别有价值，如[监控系统](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai)或[自动驾驶汽车](https://www.ultralytics.com/solutions/ai-in-automotive)。跟踪模式实现了 ByteTrack 等复杂算法，以在帧之间保持目标身份，即使目标暂时从视野中消失。

[跟踪示例](track.md){ .md-button }

## [基准测试](benchmark.md)

基准测试模式分析 YOLO11 各种导出格式的速度和精度。此模式提供关于模型大小、精度（检测任务的 mAP50-95 或分类的 accuracy_top5）以及不同格式（如 ONNX、[OpenVINO](https://docs.ultralytics.com/integrations/openvino/) 和 TensorRT）推理时间的综合指标。基准测试帮助您根据部署环境中对速度和精度的具体要求选择最佳导出格式。

[基准测试示例](benchmark.md){ .md-button }

## 常见问题

### 如何使用 Ultralytics YOLO11 训练自定义[目标检测](https://www.ultralytics.com/glossary/object-detection)模型？

使用 Ultralytics YOLO11 训练自定义目标检测模型涉及使用训练模式。您需要一个 YOLO 格式的数据集，包含图像和相应的标注文件。使用以下命令开始训练过程：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO 模型（可以选择 n、s、m、l 或 x 版本）
        model = YOLO("yolo11n.pt")

        # 在自定义数据集上开始训练
        model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从命令行训练 YOLO 模型
        yolo detect train data=path/to/dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关更详细的说明，您可以参考 [Ultralytics 训练指南](../modes/train.md)。

### Ultralytics YOLO11 使用哪些指标来验证模型性能？

Ultralytics YOLO11 在验证过程中使用各种指标来评估模型性能。这些包括：

- **mAP（平均精度均值）**：评估目标检测的精度。
- **IOU（交并比）**：衡量预测边界框和真实边界框之间的重叠。
- **[精确率](https://www.ultralytics.com/glossary/precision)和[召回率](https://www.ultralytics.com/glossary/recall)**：精确率衡量真正检测与总检测正例的比率，而召回率衡量真正检测与总实际正例的比率。

您可以运行以下命令开始验证：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练或自定义 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 在数据集上运行验证
        model.val(data="path/to/validation.yaml")
        ```

    === "CLI"

        ```bash
        # 从命令行验证 YOLO 模型
        yolo val model=yolo11n.pt data=path/to/validation.yaml
        ```

有关更多详细信息，请参阅[验证指南](../modes/val.md)。

### 如何导出我的 YOLO11 模型进行部署？

Ultralytics YOLO11 提供导出功能，可将训练好的模型转换为各种部署格式，如 ONNX、TensorRT、CoreML 等。使用以下示例导出您的模型：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载训练好的 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 ONNX 格式（可以根据需要指定其他格式）
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        # 从命令行将 YOLO 模型导出为 ONNX 格式
        yolo export model=yolo11n.pt format=onnx
        ```

每种导出格式的详细步骤可在[导出指南](../modes/export.md)中找到。

### Ultralytics YOLO11 中基准测试模式的目的是什么？

Ultralytics YOLO11 中的基准测试模式用于分析各种导出格式（如 ONNX、TensorRT 和 OpenVINO）的速度和[精度](https://www.ultralytics.com/glossary/accuracy)。它提供模型大小、目标检测的 `mAP50-95` 以及不同硬件设置下推理时间等指标，帮助您选择最适合部署需求的格式。

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在 GPU 上运行基准测试（设备 0）
        # 您可以根据需要调整模型、数据集、图像大小和精度等参数
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        # 从命令行对 YOLO 模型进行基准测试
        # 根据您的具体用例调整参数
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

有关更多详细信息，请参阅[基准测试指南](../modes/benchmark.md)。

### 如何使用 Ultralytics YOLO11 执行实时目标跟踪？

可以使用 Ultralytics YOLO11 中的跟踪模式实现实时目标跟踪。此模式将目标检测能力扩展到跨视频帧或实时流跟踪目标。使用以下示例启用跟踪：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 开始跟踪视频中的目标
        # 您也可以使用实时视频流或网络摄像头输入
        model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # 从命令行对视频执行目标跟踪
        # 您可以指定不同的源，如网络摄像头 (0) 或 RTSP 流
        yolo track model=yolo11n.pt source=path/to/video.mp4
        ```

有关深入说明，请访问[跟踪指南](../modes/track.md)。
