---
comments: true
description: 了解如何将 YOLO11 模型导出为各种格式，如 ONNX、TensorRT 和 CoreML。实现最大兼容性和性能。
keywords: YOLO11, 模型导出, ONNX, TensorRT, CoreML, Ultralytics, AI, 机器学习, 推理, 部署
---

# 使用 Ultralytics YOLO 进行模型导出

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统和集成">

## 简介

训练模型的最终目标是将其部署到实际应用中。Ultralytics YOLO11 中的导出模式提供了多种选项，可将训练好的模型导出为不同格式，使其可在各种平台和设备上部署。本综合指南旨在引导您了解模型导出的细节，展示如何实现最大兼容性和性能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何导出自定义训练的 Ultralytics YOLO 模型并在网络摄像头上运行实时推理。
</p>

## 为什么选择 YOLO11 的导出模式？

- **多功能性：** 导出为多种格式，包括 [ONNX](../integrations/onnx.md)、[TensorRT](../integrations/tensorrt.md)、[CoreML](../integrations/coreml.md) 等。
- **性能：** 使用 TensorRT 可获得高达 5 倍的 GPU 加速，使用 ONNX 或 [OpenVINO](../integrations/openvino.md) 可获得高达 3 倍的 CPU 加速。
- **兼容性：** 使您的模型可在众多硬件和软件环境中通用部署。
- **易用性：** 简单的 CLI 和 Python API，实现快速直接的模型导出。

### 导出模式的关键特性

以下是一些突出的功能：

- **一键导出：** 简单的命令即可导出为不同格式。
- **批量导出：** 导出支持批量推理的模型。
- **优化推理：** 导出的模型针对更快的推理时间进行了优化。
- **教程视频：** 深入的指南和教程，提供流畅的导出体验。

!!! tip

    * 导出到 [ONNX](../integrations/onnx.md) 或 [OpenVINO](../integrations/openvino.md) 可获得高达 3 倍的 CPU 加速。
    * 导出到 [TensorRT](../integrations/tensorrt.md) 可获得高达 5 倍的 GPU 加速。

## 使用示例

将 YOLO11n 模型导出为不同格式，如 ONNX 或 TensorRT。有关导出参数的完整列表，请参阅下面的参数部分。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx      # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

## 参数

此表详细说明了将 YOLO 模型导出为不同格式的配置和选项。这些设置对于优化导出模型在各种平台和环境中的性能、大小和兼容性至关重要。正确的配置确保模型以最佳效率准备好在预期应用中部署。

{% include "macros/export-args.md" %}

调整这些参数允许自定义导出过程以适应特定要求，如部署环境、硬件约束和性能目标。选择适当的格式和设置对于在模型大小、速度和[精度](https://www.ultralytics.com/glossary/accuracy)之间实现最佳平衡至关重要。

## 导出格式

可用的 YOLO11 导出格式在下表中。您可以使用 `format` 参数导出为任何格式，即 `format='onnx'` 或 `format='engine'`。您可以直接在导出的模型上进行预测或验证，即 `yolo predict model=yolo11n.onnx`。导出完成后会显示模型的使用示例。

{% include "macros/export-table.md" %}

## 常见问题

### 如何将 YOLO11 模型导出为 ONNX 格式？

使用 Ultralytics 将 YOLO11 模型导出为 ONNX 格式非常简单。它提供了 Python 和 CLI 两种导出模型的方法。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx      # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

有关该过程的更多详细信息，包括处理不同输入大小的高级选项，请参阅 [ONNX 集成指南](../integrations/onnx.md)。

### 使用 TensorRT 进行模型导出有什么好处？

使用 TensorRT 进行模型导出可提供显著的性能改进。导出到 TensorRT 的 YOLO11 模型可实现高达 5 倍的 GPU 加速，非常适合实时推理应用。

- **多功能性：** 针对特定硬件设置优化模型。
- **速度：** 通过高级优化实现更快的推理。
- **兼容性：** 与 NVIDIA 硬件顺畅集成。

要了解有关集成 TensorRT 的更多信息，请参阅 [TensorRT 集成指南](../integrations/tensorrt.md)。

### 导出 YOLO11 模型时如何启用 INT8 量化？

INT8 量化是压缩模型和加速推理的绝佳方法，特别是在边缘设备上。以下是启用 INT8 量化的方法：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # 加载模型
        model.export(format="engine", int8=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=engine int8=True # 使用 INT8 量化导出 TensorRT 模型
        ```

INT8 量化可应用于各种格式，如 [TensorRT](../integrations/tensorrt.md)、[OpenVINO](../integrations/openvino.md) 和 [CoreML](../integrations/coreml.md)。为获得最佳量化结果，请使用 `data` 参数提供代表性[数据集](https://docs.ultralytics.com/datasets/)。

### 导出模型时为什么动态输入大小很重要？

动态输入大小允许导出的模型处理不同的图像尺寸，为不同用例提供灵活性并优化处理效率。当导出为 [ONNX](../integrations/onnx.md) 或 [TensorRT](../integrations/tensorrt.md) 等格式时，启用动态输入大小可确保模型能够无缝适应不同的输入形状。

要启用此功能，请在导出时使用 `dynamic=True` 标志：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx dynamic=True
        ```

动态输入大小对于输入尺寸可能变化的应用特别有用，如视频处理或处理来自不同来源的图像。

### 优化模型性能需要考虑哪些关键导出参数？

理解和配置导出参数对于优化模型性能至关重要：

- **`format:`** 导出模型的目标格式（例如 `onnx`、`torchscript`、`tensorflow`）。
- **`imgsz:`** 模型输入所需的图像大小（例如 `640` 或 `(height, width)`）。
- **`half:`** 启用 FP16 量化，减小模型大小并可能加速推理。
- **`optimize:`** 应用针对移动或受限环境的特定优化。
- **`int8:`** 启用 INT8 量化，对[边缘 AI](https://www.ultralytics.com/blog/deploying-computer-vision-applications-on-edge-ai-devices) 部署非常有益。

对于在特定硬件平台上部署，请考虑使用专门的导出格式，如用于 NVIDIA GPU 的 [TensorRT](../integrations/tensorrt.md)、用于 Apple 设备的 [CoreML](../integrations/coreml.md) 或用于 Google Coral 设备的 [Edge TPU](../integrations/edge-tpu.md)。
