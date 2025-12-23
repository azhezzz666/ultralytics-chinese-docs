---
comments: true
description: 使用 Neural Magic 的 DeepSparse 引擎增强 YOLO11 性能。学习如何在 CPU 上部署和基准测试 YOLO11 模型以实现高效目标检测。
keywords: YOLO11, DeepSparse, Neural Magic, 模型优化, 目标检测, 推理速度, CPU 性能, 稀疏性, 剪枝, 量化
---

# 使用 Neural Magic 的 DeepSparse 引擎优化 YOLO11 推理

在各种硬件上部署[目标检测](https://www.ultralytics.com/glossary/object-detection)模型（如 [Ultralytics YOLO11](https://www.ultralytics.com/)）时，你可能会遇到独特的问题，如优化。这就是 YOLO11 与 Neural Magic 的 DeepSparse 引擎集成发挥作用的地方。它改变了 YOLO11 模型的执行方式，并直接在 CPU 上实现 GPU 级性能。

本指南向你展示如何使用 Neural Magic 的 DeepSparse 部署 YOLO11、如何运行推理以及如何基准测试性能以确保其得到优化。

!!! danger "SparseML 生命周期结束"

    Neural Magic 于 [2025 年 1 月被 Red Hat 收购](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud)，并正在弃用其 `deepsparse`、`sparseml`、`sparsezoo` 和 `sparsify` 库的社区版本。有关更多信息，请参阅 [`sparseml` GitHub 仓库 Readme 中发布的通知](https://github.com/neuralmagic/sparsify/blob/5eb26a4e21b497ce573d10024e318a5ce48a7f9c/README.md#-2025-end-of-life-announcement-deepsparse-sparseml-sparsezoo-and-sparsify)。

## Neural Magic 的 DeepSparse

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/neural-magic-deepsparse-overview.avif" alt="Neural Magic 的 DeepSparse 概览">
</p>

[Neural Magic 的 DeepSparse](https://github.com/neuralmagic/deepsparse/blob/main/README.md) 是一个推理运行时，旨在优化 CPU 上神经网络的执行。它应用稀疏性、剪枝和量化等高级技术来大幅减少计算需求，同时保持准确率。DeepSparse 为跨各种设备的高效和可扩展[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)执行提供了敏捷的解决方案。

## 将 Neural Magic 的 DeepSparse 与 YOLO11 集成的好处

在深入了解如何使用 DeepSparse 部署 YOLO11 之前，让我们了解使用 DeepSparse 的好处。一些关键优势包括：

- **增强的推理速度**：在 YOLO11n 上实现高达 525 FPS，与传统方法相比显著加速 YOLO11 的推理能力。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/enhanced-inference-speed.avif" alt="增强的推理速度">
</p>

- **优化的模型效率**：使用剪枝和量化来增强 YOLO11 的效率，减少模型大小和计算需求，同时保持[准确率](https://www.ultralytics.com/glossary/accuracy)。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/optimized-model-efficiency.avif" alt="优化的模型效率">
</p>

- **标准 CPU 上的高性能**：在 CPU 上提供 GPU 级性能，为各种应用提供更易访问和更具成本效益的选项。

- **简化的集成和部署**：提供用户友好的工具，便于将 YOLO11 集成到应用中，包括图像和视频标注功能。

- **支持各种模型类型**：与标准和稀疏性优化的 YOLO11 模型兼容，增加部署灵活性。

- **成本效益和可扩展的解决方案**：降低运营费用并提供高级目标检测模型的可扩展部署。

## Neural Magic 的 DeepSparse 技术如何工作？

Neural Magic 的 DeepSparse 技术受人脑神经网络计算效率的启发。它采用了大脑的两个关键原则：

- **稀疏性**：稀疏化过程涉及从[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)网络中剪枝冗余信息，从而产生更小更快的模型而不影响准确率。此技术显著减少了网络的大小和计算需求。

- **引用局部性**：DeepSparse 使用独特的执行方法，将网络分解为张量列。这些列按深度执行，完全适合 CPU 的缓存。这种方法模仿了大脑的效率，最小化数据移动并最大化 CPU 缓存的使用。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/neural-magic-deepsparse-technology.avif" alt="Neural Magic 的 DeepSparse 技术如何工作">
</p>

## 创建在自定义数据集上训练的 YOLO11 稀疏版本

[SparseZoo](https://github.com/neuralmagic/sparsezoo/blob/main/README.md) 是 Neural Magic 的开源模型仓库，提供[预稀疏化 YOLO11 模型检查点集合](https://github.com/neuralmagic/sparsezoo/blob/main/README.md)。使用与 Ultralytics 无缝集成的 [SparseML](https://github.com/neuralmagic/sparseml)，用户可以使用简单的命令行界面轻松地在其特定数据集上微调这些稀疏检查点。

查看 [Neural Magic 的 SparseML YOLO11 文档](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov8)了解更多详情。

## 用法：使用 DeepSparse 部署 YOLO11

使用 Neural Magic 的 DeepSparse 部署 YOLO11 涉及几个简单的步骤。在深入使用说明之前，请务必查看 [Ultralytics 提供的 YOLO11 模型范围](../models/index.md)。这将帮助你为项目需求选择最合适的模型。以下是如何开始。

### 步骤 1：安装

要安装所需的包，运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装所需的包
        pip install deepsparse[yolov8]
        ```

### 步骤 2：将 YOLO11 导出为 ONNX 格式

DeepSparse 引擎需要 [ONNX 格式](../integrations/onnx.md)的 YOLO11 模型。将模型导出为此格式对于与 DeepSparse 的兼容性至关重要。使用以下命令导出 YOLO11 模型：

!!! tip "模型导出"

    === "CLI"

        ```bash
        # 将 YOLO11 模型导出为 ONNX 格式
        yolo task=detect mode=export model=yolo11n.pt format=onnx opset=13
        ```

此命令将 `yolo11n.onnx` 模型保存到你的磁盘。

### 步骤 3：部署和运行推理

使用 ONNX 格式的 YOLO11 模型，你可以使用 DeepSparse 部署和运行推理。这可以通过其直观的 Python API 轻松完成：

!!! tip "部署和运行推理"

    === "Python"

        ```python
        from deepsparse import Pipeline

        # 指定 YOLO11 ONNX 模型的路径
        model_path = "path/to/yolo11n.onnx"

        # 设置 DeepSparse Pipeline
        yolo_pipeline = Pipeline.create(task="yolov8", model_path=model_path)

        # 在图像上运行模型
        images = ["path/to/image.jpg"]
        pipeline_outputs = yolo_pipeline(images=images)
        ```

### 步骤 4：基准测试性能

检查 YOLO11 模型在 DeepSparse 上是否以最佳性能运行很重要。你可以[基准测试](../modes/benchmark.md)模型的性能以分析吞吐量和延迟：

!!! tip "基准测试"

    === "CLI"

        ```bash
        # 基准测试性能
        deepsparse.benchmark model_path="path/to/yolo11n.onnx" --scenario=sync --input_shapes="[1,3,640,640]"
        ```

### 步骤 5：附加功能

DeepSparse 提供了用于将 YOLO11 实际集成到应用中的附加功能，如图像标注和数据集评估。

!!! tip "附加功能"

    === "CLI"

        ```bash
        # 用于图像标注
        deepsparse.yolov8.annotate --source "path/to/image.jpg" --model_filepath "path/to/yolo11n.onnx"

        # 用于评估数据集上的模型性能
        deepsparse.yolov8.eval --model_path "path/to/yolo11n.onnx"
        ```

运行 annotate 命令处理你指定的图像，检测对象，并保存带有边界框和分类的标注图像。标注图像将存储在 annotation-results 文件夹中。这有助于提供模型检测能力的可视化表示。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/image-annotation-feature.avif" alt="图像标注功能">
</p>

运行 eval 命令后，你将收到详细的输出指标，如[精确率](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和 [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)（平均精度均值）。这提供了模型在数据集上性能的全面视图，对于微调和优化特定用例的 YOLO11 模型特别有用，确保高准确率和效率。

## 总结

本指南探索了将 Ultralytics 的 YOLO11 与 Neural Magic 的 DeepSparse 引擎集成。它强调了此集成如何增强 YOLO11 在 CPU 平台上的性能，提供 GPU 级效率和高级神经网络稀疏性技术。

有关更详细的信息和高级用法，请访问 [Neural Magic 的 DeepSparse 文档](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud)。你还可以[探索 YOLO11 集成指南](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/yolov8#yolov8-inference-pipelines)并[观看 YouTube 上的演练会话](https://www.youtube.com/watch?v=qtJ7bdt52x8)。

此外，要更广泛地了解各种 YOLO11 集成，请访问 [Ultralytics 集成指南页面](../integrations/index.md)，在那里你可以发现一系列其他令人兴奋的集成可能性。

## 常见问题

### 什么是 Neural Magic 的 DeepSparse 引擎，它如何优化 YOLO11 性能？

Neural Magic 的 DeepSparse 引擎是一个推理运行时，旨在通过稀疏性、剪枝和量化等高级技术优化 CPU 上神经网络的执行。通过将 DeepSparse 与 YOLO11 集成，你可以在标准 CPU 上实现 GPU 级性能，显著增强推理速度、模型效率和整体性能，同时保持准确率。有关更多详细信息，请查看 [Neural Magic 的 DeepSparse 部分](#neural-magic-的-deepsparse)。

### 如何安装部署 YOLO11 使用 Neural Magic 的 DeepSparse 所需的包？

安装部署 YOLO11 使用 Neural Magic 的 DeepSparse 所需的包非常简单。你可以使用 CLI 轻松安装它们。以下是你需要运行的命令：

```bash
pip install deepsparse[yolov8]
```

安装后，按照[安装部分](#步骤-1安装)中提供的步骤设置环境并开始使用 DeepSparse 与 YOLO11。

### 如何将 YOLO11 模型转换为 ONNX 格式以与 DeepSparse 一起使用？

要将 YOLO11 模型转换为与 DeepSparse 兼容所需的 ONNX 格式，你可以使用以下 CLI 命令：

```bash
yolo task=detect mode=export model=yolo11n.pt format=onnx opset=13
```

此命令将导出你的 YOLO11 模型（`yolo11n.pt`）为可由 DeepSparse 引擎使用的格式（`yolo11n.onnx`）。有关模型导出的更多信息，请参阅[模型导出部分](#步骤-2将-yolo11-导出为-onnx-格式)。

### 如何在 DeepSparse 引擎上基准测试 YOLO11 性能？

在 DeepSparse 上基准测试 YOLO11 性能有助于你分析吞吐量和延迟以确保模型得到优化。你可以使用以下 CLI 命令运行基准测试：

```bash
deepsparse.benchmark model_path="path/to/yolo11n.onnx" --scenario=sync --input_shapes="[1,3,640,640]"
```

此命令将为你提供重要的性能指标。有关更多详细信息，请参阅[基准测试性能部分](#步骤-4基准测试性能)。

### 为什么应该将 Neural Magic 的 DeepSparse 与 YOLO11 一起用于目标检测任务？

将 Neural Magic 的 DeepSparse 与 YOLO11 集成提供了几个好处：

- **增强的推理速度**：实现高达 525 FPS，显著加速 YOLO11 的能力。
- **优化的模型效率**：使用稀疏性、剪枝和量化技术来减少模型大小和计算需求，同时保持准确率。
- **标准 CPU 上的高性能**：在成本效益高的 CPU 硬件上提供 GPU 级性能。
- **简化的集成**：用于轻松部署和集成的用户友好工具。
- **灵活性**：支持标准和稀疏性优化的 YOLO11 模型。
- **成本效益**：通过高效的资源利用降低运营费用。

要深入了解这些优势，请访问[将 Neural Magic 的 DeepSparse 与 YOLO11 集成的好处部分](#将-neural-magic-的-deepsparse-与-yolo11-集成的好处)。
