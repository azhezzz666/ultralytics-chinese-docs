---
comments: true
description: 学习如何将 YOLO11 模型导出为 TFLite Edge TPU 格式，以便在移动和嵌入式设备上进行高速、低功耗推理。
keywords: YOLO11, TFLite Edge TPU, TensorFlow Lite, 模型导出, 机器学习, 边缘计算, 神经网络, Ultralytics
---

# 学习从 YOLO11 模型导出为 TFLite Edge TPU 格式

在计算能力有限的设备（如移动或嵌入式系统）上部署计算机视觉模型可能很棘手。使用针对更快性能优化的模型格式可以简化这个过程。[TensorFlow Lite](https://ai.google.dev/edge/litert) [Edge TPU](https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/models-intro/) 或 TFLite Edge TPU 模型格式旨在使用最少的功耗，同时为神经网络提供快速性能。

导出为 TFLite Edge TPU 格式的功能允许您优化 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型，以便在各种移动和嵌入式设备上进行高速、低功耗推理。在本指南中，我们将引导您将模型转换为 TFLite Edge TPU 格式，使您的模型更容易在各种移动和嵌入式设备上表现良好。

## 为什么应该导出为 TFLite Edge TPU？

将模型导出为 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Edge TPU 使[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)任务快速高效。这项技术适合功耗、计算资源和连接性有限的应用。Edge TPU 是 Google 的硬件加速器。它加速边缘设备上的 TensorFlow Lite 模型。下图显示了所涉及过程的示例。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/tflite-edge-tpu-compile-workflow.avif" alt="TFLite Edge TPU">
</p>

Edge TPU 与量化模型配合使用。量化使模型更小更快，同时不会损失太多[准确率](https://www.ultralytics.com/glossary/accuracy)。它非常适合边缘计算的有限资源，通过减少延迟允许应用程序快速响应，并允许在本地快速处理数据，无需依赖云。本地处理还可以保护用户数据的隐私和安全，因为数据不会发送到远程服务器。

## TFLite Edge TPU 的主要特性

以下是使 TFLite Edge TPU 成为开发人员绝佳模型格式选择的主要特性：

- **边缘设备上的优化性能**：TFLite Edge TPU 通过量化、模型优化、硬件加速和编译器优化实现高速神经网络性能。其简约架构有助于其更小的尺寸和成本效益。

- **高计算吞吐量**：TFLite Edge TPU 结合专用硬件加速和高效运行时执行，实现高计算吞吐量。它非常适合在边缘设备上部署具有严格性能要求的机器学习模型。

- **高效矩阵计算**：TensorFlow Edge TPU 针对矩阵运算进行了优化，这对于[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)计算至关重要。这种效率是机器学习模型的关键，特别是那些需要大量复杂矩阵乘法和变换的模型。

## TFLite Edge TPU 的部署选项

在我们深入了解如何将 YOLO11 模型导出为 TFLite Edge TPU 格式之前，让我们了解 TFLite Edge TPU 模型通常在哪里使用。

TFLite Edge TPU 为机器学习模型提供各种部署选项，包括：

- **设备端部署**：TensorFlow Edge TPU 模型可以直接部署在移动和嵌入式设备上。设备端部署允许模型直接在硬件上执行，无需云连接，可以通过将模型嵌入应用程序包或按需下载。

- **使用云 TensorFlow TPU 的边缘计算**：在边缘设备处理能力有限的情况下，TensorFlow Edge TPU 可以将推理任务卸载到配备 TPU 的云服务器。

- **混合部署**：混合方法结合了设备端和云部署，为部署机器学习模型提供了一个多功能且可扩展的解决方案。优势包括设备端处理以实现快速响应，以及[云计算](https://www.ultralytics.com/glossary/cloud-computing)用于更复杂的计算。

## 将 YOLO11 模型导出为 TFLite Edge TPU

您可以通过将 YOLO11 模型转换为 TensorFlow Edge TPU 来扩展模型兼容性和部署灵活性。

### 安装

要安装所需的包，请运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 所需的包
        pip install ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [Ultralytics 安装指南](../quickstart.md)。在安装 YOLO11 所需的包时，如果遇到任何困难，请查阅我们的[常见问题指南](../guides/yolo-common-issues.md)以获取解决方案和提示。

### 使用

所有 [Ultralytics YOLO11 模型](../models/index.md)都设计为开箱即用支持导出，使其易于集成到您首选的部署工作流程中。您可以[查看支持的导出格式和配置选项的完整列表](../modes/export.md)，以选择最适合您应用程序的设置。

!!! example "使用"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 TFLite Edge TPU 格式
        model.export(format="edgetpu")  # 创建 'yolo11n_full_integer_quant_edgetpu.tflite'

        # 加载导出的 TFLite Edge TPU 模型
        edgetpu_model = YOLO("yolo11n_full_integer_quant_edgetpu.tflite")

        # 运行推理
        results = edgetpu_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TFLite Edge TPU 格式
        yolo export model=yolo11n.pt format=edgetpu # 创建 'yolo11n_full_integer_quant_edgetpu.tflite'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n_full_integer_quant_edgetpu.tflite source='https://ultralytics.com/images/bus.jpg'
        ```

### 导出参数

| 参数     | 类型             | 默认值      | 描述                                                                                              |
| -------- | ---------------- | ----------- | ------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'edgetpu'` | 导出模型的目标格式，定义与各种部署环境的兼容性。                                                  |
| `imgsz`  | `int` 或 `tuple` | `640`       | 模型输入所需的图像尺寸。可以是正方形图像的整数或特定尺寸的元组 `(height, width)`。                |
| `device` | `str`            | `None`      | 指定导出设备：CPU（`device=cpu`）。                                                               |

!!! tip

    导出为 EdgeTPU 时，请确保使用 x86 Linux 机器。

有关导出过程的更多详细信息，请访问 [Ultralytics 导出文档页面](../modes/export.md)。


## 部署导出的 YOLO11 TFLite Edge TPU 模型

成功将 Ultralytics YOLO11 模型导出为 TFLite Edge TPU 格式后，您现在可以部署它们。运行 TFLite Edge TPU 模型的主要和推荐的第一步是使用 YOLO("model_edgetpu.tflite") 方法，如前面的使用代码片段所述。

但是，有关部署 TFLite Edge TPU 模型的深入说明，请查看以下资源：

- **[在 Raspberry Pi 上使用 Coral Edge TPU 与 Ultralytics YOLO11](../guides/coral-edge-tpu-on-raspberry-pi.md)**：了解如何将 Coral Edge TPU 与 Raspberry Pi 集成以增强机器学习能力。

- **[代码示例](https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/compiler/)**：访问实用的 TensorFlow Edge TPU 部署示例以启动您的项目。

- **[使用 Python 在 Edge TPU 上运行推理](https://gweb-coral-full.uc.r.appspot.com/docs/edgetpu/tflite-python/#overview)**：探索如何使用 TensorFlow Lite Python API 进行 Edge TPU 应用程序，包括设置和使用指南。

## 总结

在本指南中，我们学习了如何将 Ultralytics YOLO11 模型导出为 TFLite Edge TPU 格式。通过遵循上述步骤，您可以提高[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用程序的速度和功率。

有关使用的更多详细信息，请访问 [Edge TPU 官方网站](https://cloud.google.com/tpu)。

另外，有关其他 Ultralytics YOLO11 集成的更多信息，请访问我们的[集成指南页面](index.md)。在那里，您将发现有价值的资源和见解。

## 常见问题

### 如何将 YOLO11 模型导出为 TFLite Edge TPU 格式？

要将 YOLO11 模型导出为 TFLite Edge TPU 格式，您可以按照以下步骤操作：

!!! example "使用"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 TFLite Edge TPU 格式
        model.export(format="edgetpu")  # 创建 'yolo11n_full_integer_quant_edgetpu.tflite'

        # 加载导出的 TFLite Edge TPU 模型
        edgetpu_model = YOLO("yolo11n_full_integer_quant_edgetpu.tflite")

        # 运行推理
        results = edgetpu_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TFLite Edge TPU 格式
        yolo export model=yolo11n.pt format=edgetpu # 创建 'yolo11n_full_integer_quant_edgetpu.tflite'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n_full_integer_quant_edgetpu.tflite source='https://ultralytics.com/images/bus.jpg'
        ```

有关将模型导出为其他格式的完整详细信息，请参阅我们的[导出指南](../modes/export.md)。

### 将 YOLO11 模型导出为 TFLite Edge TPU 有什么好处？

将 YOLO11 模型导出为 TFLite Edge TPU 提供了几个好处：

- **优化性能**：以最小的功耗实现高速神经网络性能。
- **减少延迟**：无需依赖云即可快速本地数据处理。
- **增强隐私**：本地处理保护用户数据的隐私和安全。

这使其非常适合[边缘计算](https://www.ultralytics.com/glossary/edge-computing)中的应用，其中设备的功率和计算资源有限。了解更多关于[为什么应该导出](#为什么应该导出为-tflite-edge-tpu)的信息。

### 我可以在移动和嵌入式设备上部署 TFLite Edge TPU 模型吗？

是的，TensorFlow Lite Edge TPU 模型可以直接部署在移动和嵌入式设备上。这种部署方法允许模型直接在硬件上执行，提供更快、更高效的推理。有关集成示例，请查看我们关于[在 Raspberry Pi 上部署 Coral Edge TPU](../guides/coral-edge-tpu-on-raspberry-pi.md) 的指南。

### TFLite Edge TPU 模型有哪些常见用例？

TFLite Edge TPU 模型的常见用例包括：

- **智能摄像头**：增强实时图像和视频分析。
- **物联网设备**：实现智能家居和工业自动化。
- **医疗保健**：加速医学成像和诊断。
- **零售**：改善库存管理和客户行为分析。

这些应用受益于 TFLite Edge TPU 模型的高性能和低功耗。了解更多关于[使用场景](#tflite-edge-tpu-的部署选项)的信息。

### 如何解决导出或部署 TFLite Edge TPU 模型时遇到的问题？

如果您在导出或部署 TFLite Edge TPU 模型时遇到问题，请参阅我们的[常见问题指南](../guides/yolo-common-issues.md)获取故障排除提示。本指南涵盖常见问题和解决方案，帮助您确保顺利运行。如需更多支持，请访问我们的[帮助中心](https://docs.ultralytics.com/help/)。
