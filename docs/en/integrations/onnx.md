---
comments: true
description: 学习如何将 YOLO11 模型导出为 ONNX 格式，以便在各种平台上灵活部署并增强性能。
keywords: YOLO11, ONNX, 模型导出, Ultralytics, ONNX Runtime, 机器学习, 模型部署, 计算机视觉, 深度学习
---

# YOLO11 模型的 ONNX 导出

在部署[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型时，你通常需要一种既灵活又与多个平台兼容的模型格式。

将 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型导出为 ONNX 格式可以简化部署并确保在各种环境中的最佳性能。本指南将向你展示如何轻松地将 YOLO11 模型转换为 ONNX 并增强其在实际应用中的可扩展性和有效性。

## ONNX 和 ONNX Runtime

[ONNX](https://onnx.ai/) 代表开放神经网络交换，是一个由 Facebook 和 Microsoft 最初开发的社区项目。ONNX 的持续开发是由 IBM、Amazon（通过 AWS）和 Google 等各种组织支持的协作努力。该项目旨在创建一种开放的文件格式，用于以允许在不同 AI 框架和硬件之间使用的方式表示[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型。

ONNX 模型可用于在不同框架之间无缝转换。例如，在 PyTorch 中训练的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型可以导出为 ONNX 格式，然后轻松导入到 TensorFlow 中。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/onnx-model-portability.avif" alt="ONNX">
</p>

或者，ONNX 模型可以与 ONNX Runtime 一起使用。[ONNX Runtime](https://onnxruntime.ai/) 是一个多功能的跨平台机器学习模型加速器，与 PyTorch、[TensorFlow](https://www.ultralytics.com/glossary/tensorflow)、TFLite、scikit-learn 等框架兼容。

ONNX Runtime 通过利用硬件特定的功能来优化 ONNX 模型的执行。这种优化允许模型在各种硬件平台上高效且高性能地运行，包括 CPU、GPU 和专用加速器。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/onnx-and-onnx-runtime.avif" alt="ONNX 与 ONNX Runtime">
</p>

无论是独立使用还是与 ONNX Runtime 结合使用，ONNX 都为机器学习[模型部署](https://www.ultralytics.com/glossary/model-deployment)和兼容性提供了灵活的解决方案。

## ONNX 模型的关键功能

ONNX 处理各种格式的能力可归因于以下关键功能：

- **通用模型表示**：ONNX 定义了一组通用的运算符（如卷积、层等）和标准数据格式。当模型转换为 ONNX 格式时，其架构和权重被转换为这种通用表示。这种统一性确保模型可以被任何支持 ONNX 的框架理解。

- **版本控制和向后兼容性**：ONNX 为其运算符维护版本控制系统。这确保即使标准演进，在旧版本中创建的模型仍然可用。向后兼容性是一个关键功能，可防止模型快速过时。

- **基于图的模型表示**：ONNX 将模型表示为计算图。这种基于图的结构是表示机器学习模型的通用方式，其中节点表示操作或计算，边表示它们之间流动的张量。这种格式很容易适应各种也将模型表示为图的框架。

- **工具和生态系统**：围绕 ONNX 有丰富的工具生态系统，可帮助进行模型转换、可视化和优化。这些工具使开发人员更容易使用 ONNX 模型并在不同框架之间无缝转换模型。

## ONNX 的常见用法

在我们深入了解如何将 YOLO11 模型导出为 ONNX 格式之前，让我们看看 ONNX 模型通常在哪里使用。

### CPU 部署

ONNX 模型通常部署在 CPU 上，因为它们与 ONNX Runtime 兼容。此运行时针对 CPU 执行进行了优化。它显著提高了推理速度，使实时 CPU 部署变得可行。

### 支持的部署选项

虽然 ONNX 模型通常用于 CPU，但它们也可以部署在以下平台上：

- **GPU 加速**：ONNX 完全支持 GPU 加速，特别是 NVIDIA CUDA。这使得在 NVIDIA GPU 上高效执行需要高计算能力的任务成为可能。

- **边缘和移动设备**：ONNX 扩展到边缘和移动设备，非常适合设备端和实时推理场景。它轻量且与边缘硬件兼容。

- **Web 浏览器**：ONNX 可以直接在 Web 浏览器中运行，为交互式和动态的基于 Web 的 AI 应用提供支持。

## 将 YOLO11 模型导出为 ONNX

你可以通过将 YOLO11 模型转换为 ONNX 格式来扩展模型兼容性和部署灵活性。[Ultralytics YOLO11](../models/yolo11.md) 提供了一个简单的导出过程，可以显著增强模型在不同平台上的性能。

### 安装

要安装所需的包，运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 所需的包
        pip install ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [YOLO11 安装指南](../quickstart.md)。在为 YOLO11 安装所需包时，如果遇到任何困难，请查阅我们的[常见问题指南](../guides/yolo-common-issues.md)获取解决方案和提示。

### 用法

在深入使用说明之前，请务必查看 [Ultralytics 提供的 YOLO11 模型范围](../models/index.md)。这将帮助你为项目需求选择最合适的模型。

!!! example "用法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 ONNX 格式
        model.export(format="onnx")  # 创建 'yolo11n.onnx'

        # 加载导出的 ONNX 模型
        onnx_model = YOLO("yolo11n.onnx")

        # 运行推理
        results = onnx_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 ONNX 格式
        yolo export model=yolo11n.pt format=onnx # 创建 'yolo11n.onnx'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

### 导出参数

将 YOLO11 模型导出为 ONNX 格式时，你可以使用各种参数自定义过程以优化特定部署需求：

| 参数 | 类型 | 默认值 | 描述 |
| ---------- | ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str` | `'onnx'` | 导出模型的目标格式，定义与各种部署环境的兼容性。 |
| `imgsz` | `int` 或 `tuple` | `640` | 模型输入所需的图像大小。可以是整数（用于正方形图像）或元组 `(height, width)`（用于特定尺寸）。 |
| `half` | `bool` | `False` | 启用 FP16（半精度）量化，减小模型大小并可能在支持的硬件上加速推理。 |
| `dynamic` | `bool` | `False` | 允许动态输入大小，增强处理不同图像尺寸的灵活性。 |
| `simplify` | `bool` | `True` | 使用 `onnxslim` 简化模型图，可能提高性能和兼容性。 |
| `opset` | `int` | `None` | 指定 ONNX opset 版本以与不同的 ONNX 解析器和运行时兼容。如果未设置，使用最新支持的版本。 |
| `nms` | `bool` | `False` | 添加非极大值抑制（NMS），对于准确高效的检测后处理至关重要。 |
| `batch` | `int` | `1` | 指定导出模型批量推理大小或导出模型在 `predict` 模式下将并发处理的最大图像数量。 |
| `device` | `str` | `None` | 指定导出设备：GPU (`device=0`)、CPU (`device=cpu`)、Apple silicon 的 MPS (`device=mps`)。 |

有关导出过程的更多详细信息，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 ONNX 模型

成功将 Ultralytics YOLO11 模型导出为 ONNX 格式后，下一步是在各种环境中部署这些模型。有关部署 ONNX 模型的详细说明，请查看以下资源：

- **[ONNX Runtime Python API 文档](https://onnxruntime.ai/docs/api/python/api_summary.html)**：本指南提供了使用 ONNX Runtime 加载和运行 ONNX 模型的基本信息。

- **[在边缘设备上部署](https://onnxruntime.ai/docs/tutorials/iot-edge/)**：查看此文档页面了解在边缘部署 ONNX 模型的不同示例。

- **[GitHub 上的 ONNX 教程](https://github.com/onnx/tutorials)**：涵盖在不同场景中使用和实现 ONNX 模型各个方面的综合教程集合。

- **[Triton 推理服务器](../guides/triton-inference-server.md)**：学习如何使用 NVIDIA 的 Triton 推理服务器部署 ONNX 模型以实现高性能、可扩展的部署。

## 总结

在本指南中，你学习了如何将 Ultralytics YOLO11 模型导出为 ONNX 格式，以增加其在各种平台上的互操作性和性能。你还了解了 ONNX Runtime 和 ONNX 部署选项。

ONNX 导出只是 Ultralytics YOLO11 支持的众多[导出格式](../modes/export.md)之一，允许你在几乎任何环境中部署模型。根据你的特定需求，你可能还想探索其他导出选项，如 [TensorRT](../integrations/tensorrt.md) 以获得最大 GPU 性能或 [CoreML](../integrations/coreml.md) 用于 Apple 设备。

有关用法的更多详细信息，请访问 [ONNX 官方文档](https://onnx.ai/onnx/intro/)。

此外，如果你想了解更多关于其他 Ultralytics YOLO11 集成的信息，请访问我们的[集成指南页面](../integrations/index.md)。你会在那里找到大量有用的资源和见解。

## 常见问题

### 如何使用 Ultralytics 将 YOLO11 模型导出为 ONNX 格式？

要使用 Ultralytics 将 YOLO11 模型导出为 ONNX 格式，请按照以下步骤操作：

!!! example "用法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 ONNX 格式
        model.export(format="onnx")  # 创建 'yolo11n.onnx'

        # 加载导出的 ONNX 模型
        onnx_model = YOLO("yolo11n.onnx")

        # 运行推理
        results = onnx_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 ONNX 格式
        yolo export model=yolo11n.pt format=onnx # 创建 'yolo11n.onnx'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

有关更多详细信息，请访问[导出文档](../modes/export.md)。

### 使用 ONNX Runtime 部署 YOLO11 模型有什么优势？

使用 ONNX Runtime 部署 YOLO11 模型提供了几个优势：

- **跨平台兼容性**：ONNX Runtime 支持各种平台，如 Windows、macOS 和 Linux，确保你的模型在不同环境中顺利运行。
- **硬件加速**：ONNX Runtime 可以利用 CPU、GPU 和专用加速器的硬件特定优化，提供高性能推理。
- **框架互操作性**：在 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 或 TensorFlow 等流行框架中训练的模型可以轻松转换为 ONNX 格式并使用 ONNX Runtime 运行。
- **性能优化**：与原生 PyTorch 模型相比，ONNX Runtime 可以提供高达 3 倍的 CPU 加速，使其非常适合 GPU 资源有限的部署场景。

通过查看 [ONNX Runtime 文档](https://onnxruntime.ai/docs/api/python/api_summary.html)了解更多。

### 导出到 ONNX 的 YOLO11 模型有哪些部署选项？

导出到 ONNX 的 YOLO11 模型可以部署在各种平台上：

- **CPU**：利用 ONNX Runtime 进行优化的 CPU 推理。
- **GPU**：利用 NVIDIA CUDA 进行高性能 GPU 加速。
- **边缘设备**：在边缘和移动设备上运行轻量级模型，实现实时设备端推理。
- **Web 浏览器**：直接在 Web 浏览器中执行模型，用于交互式基于 Web 的应用。
- **云服务**：在支持 ONNX 格式的云平台上部署，实现可扩展推理。

有关更多信息，请探索我们的[模型部署选项指南](../guides/model-deployment-options.md)。

### 为什么应该为 Ultralytics YOLO11 模型使用 ONNX 格式？

为 Ultralytics YOLO11 模型使用 ONNX 格式提供了众多好处：

- **互操作性**：ONNX 允许模型在不同机器学习框架之间无缝转换。
- **性能优化**：ONNX Runtime 可以通过利用硬件特定优化来增强模型性能。
- **灵活性**：ONNX 支持各种部署环境，使你能够在不同平台上使用相同的模型而无需修改。
- **标准化**：ONNX 提供了一种在行业中广泛支持的标准化格式，确保长期兼容性。

请参阅[将 YOLO11 模型导出为 ONNX](https://www.ultralytics.com/blog/export-and-optimize-a-yolov8-model-for-inference-on-openvino) 的综合指南。

### 将 YOLO11 模型导出为 ONNX 时如何排除问题？

将 YOLO11 模型导出为 ONNX 时，你可能会遇到常见问题，如依赖项不匹配或不支持的操作。要排除这些问题：

1. 验证你已安装正确版本的所需依赖项。
2. 查看官方 [ONNX 文档](https://onnx.ai/onnx/intro/)了解支持的运算符和功能。
3. 查看错误消息以获取线索并查阅 [Ultralytics 常见问题指南](../guides/yolo-common-issues.md)。
4. 尝试使用不同的导出参数，如 `simplify=True` 或调整 `opset` 版本。
5. 对于动态输入大小问题，在导出时设置 `dynamic=True`。

如果问题仍然存在，请联系 Ultralytics 支持以获得进一步帮助。
