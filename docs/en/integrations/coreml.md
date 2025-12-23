---
comments: true
description: 学习如何将 YOLO11 模型导出为 CoreML 格式，以便在 iOS 和 macOS 上进行优化的设备端机器学习。按照分步说明操作。
keywords: CoreML 导出, YOLO11 模型, CoreML 转换, Ultralytics, iOS 目标检测, macOS 机器学习, AI 部署, 机器学习集成
---

# YOLO11 模型的 CoreML 导出

在 iPhone 和 Mac 等 Apple 设备上部署[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型需要一种确保无缝性能的格式。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hfSK3Mk5P0I"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何将 Ultralytics YOLO11 导出为 CoreML 以在 Apple 设备上实现 2 倍快速推理 🚀
</p>

CoreML 导出格式允许您优化 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型，以便在 iOS 和 macOS 应用程序中进行高效的[目标检测](https://www.ultralytics.com/glossary/object-detection)。在本指南中，我们将引导您完成将模型转换为 CoreML 格式的步骤，使您的模型更容易在 Apple 设备上表现良好。

## CoreML

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/coreml-overview.avif" alt="CoreML 概述">
</p>

[CoreML](https://developer.apple.com/documentation/coreml) 是 Apple 的基础机器学习框架，建立在 Accelerate、BNNS 和 Metal Performance Shaders 之上。它提供了一种机器学习模型格式，可无缝集成到 iOS 应用程序中，并支持图像分析、[自然语言处理](https://www.ultralytics.com/glossary/natural-language-processing-nlp)、音频到文本转换和声音分析等任务。

应用程序可以利用 Core ML 而无需网络连接或 API 调用，因为 Core ML 框架使用设备端计算。这意味着模型推理可以在用户设备上本地执行。

## CoreML 模型的主要特性

Apple 的 CoreML 框架为设备端机器学习提供了强大的功能。以下是使 CoreML 成为开发人员强大工具的主要特性：

- **全面的模型支持**：转换并运行来自 TensorFlow、[PyTorch](https://www.ultralytics.com/glossary/pytorch)、scikit-learn、XGBoost 和 LibSVM 等流行框架的模型。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/coreml-supported-models.avif" alt="CoreML 支持的模型">
</p>

- **设备端[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)**：通过直接在用户设备上执行模型来确保数据隐私和快速处理，无需网络连接。

- **性能和优化**：使用设备的 CPU、GPU 和神经引擎实现最佳性能，同时最小化功耗和内存使用。提供模型压缩和优化工具，同时保持[准确率](https://www.ultralytics.com/glossary/accuracy)。

- **易于集成**：为各种模型类型提供统一格式和用户友好的 API，以便无缝集成到应用程序中。通过 Vision 和 Natural Language 等框架支持特定领域的任务。

- **高级功能**：包括用于个性化体验的设备端训练功能、用于交互式 ML 体验的异步预测，以及模型检查和验证工具。

## CoreML 部署选项

在我们查看将 YOLO11 模型导出为 CoreML 格式的代码之前，让我们了解 CoreML 模型通常在哪里使用。

CoreML 为机器学习模型提供各种部署选项，包括：

- **设备端部署**：此方法将 CoreML 模型直接集成到您的 iOS 应用程序中。它特别有利于确保低延迟、增强隐私（因为数据保留在设备上）和离线功能。但是，这种方法可能受到设备硬件能力的限制，特别是对于较大和更复杂的模型，它可以通过以下两种方式执行：
    - **嵌入式模型**：这些模型包含在应用程序包中，可立即访问。它们非常适合不需要频繁更新的小型模型。

    - **下载模型**：这些模型根据需要从服务器获取。这种方法适用于较大的模型或需要定期更新的模型。它有助于保持应用程序包大小较小。

- **基于云的部署**：CoreML 模型托管在服务器上，iOS 应用程序通过 API 请求访问它们。这种可扩展且灵活的选项可以轻松更新模型而无需修改应用程序。它非常适合需要定期更新的复杂模型或大规模应用程序。但是，它确实需要互联网连接，并可能带来延迟和安全问题。


## 将 YOLO11 模型导出为 CoreML

将 YOLO11 导出为 CoreML 可在 Apple 生态系统中实现优化的设备端机器学习性能，在效率、安全性以及与 iOS、macOS、watchOS 和 tvOS 平台的无缝集成方面提供优势。

### 安装

要安装所需的包，请运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 所需的包
        pip install ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [YOLO11 安装指南](../quickstart.md)。在安装 YOLO11 所需的包时，如果遇到任何困难，请查阅我们的[常见问题指南](../guides/yolo-common-issues.md)以获取解决方案和提示。

### 使用

在深入了解使用说明之前，请务必查看 [Ultralytics 提供的 YOLO11 模型系列](../models/index.md)。这将帮助您选择最适合项目需求的模型。

!!! example "使用"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 CoreML 格式
        model.export(format="coreml")  # 创建 'yolo11n.mlpackage'

        # 加载导出的 CoreML 模型
        coreml_model = YOLO("yolo11n.mlpackage")

        # 运行推理
        results = coreml_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 CoreML 格式
        yolo export model=yolo11n.pt format=coreml # 创建 'yolo11n.mlpackage'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n.mlpackage source='https://ultralytics.com/images/bus.jpg'
        ```

### 导出参数

| 参数     | 类型             | 默认值     | 描述                                                                                                                                          |
| -------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'coreml'` | 导出模型的目标格式，定义与各种部署环境的兼容性。                                                                                              |
| `imgsz`  | `int` 或 `tuple` | `640`      | 模型输入所需的图像尺寸。可以是正方形图像的整数或特定尺寸的元组 `(height, width)`。                                                            |
| `half`   | `bool`           | `False`    | 启用 FP16（半精度）量化，减小模型大小并可能在支持的硬件上加速推理。                                                                           |
| `int8`   | `bool`           | `False`    | 激活 INT8 量化，进一步压缩模型并以最小的[准确率](https://www.ultralytics.com/glossary/accuracy)损失加速推理，主要用于边缘设备。               |
| `nms`    | `bool`           | `False`    | 添加非极大值抑制（NMS），对于准确高效的检测后处理至关重要。                                                                                   |
| `batch`  | `int`            | `1`        | 指定导出模型批量推理大小或导出模型在 `predict` 模式下将并发处理的最大图像数量。                                                               |
| `device` | `str`            | `None`     | 指定导出设备：GPU（`device=0`）、CPU（`device=cpu`）、Apple silicon 的 MPS（`device=mps`）。                                                  |

!!! tip

    导出为 CoreML 时，请确保使用 macOS 或 x86 Linux 机器。

有关导出过程的更多详细信息，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 CoreML 模型

成功将 Ultralytics YOLO11 模型导出为 CoreML 后，下一个关键阶段是有效部署这些模型。有关在各种环境中部署 CoreML 模型的详细指导，请查看以下资源：

- **[CoreML Tools](https://apple.github.io/coremltools/docs-guides/)**：本指南包含从 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)、PyTorch 和其他库转换模型到 Core ML 的说明和示例。

- **[ML 和 Vision](https://developer.apple.com/videos/)**：涵盖使用和实现 CoreML 模型各个方面的综合视频集合。

- **[将 Core ML 模型集成到您的应用程序](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)**：关于将 CoreML 模型集成到 iOS 应用程序的综合指南，详细说明了从准备模型到在应用程序中实现各种功能的步骤。

## 总结

在本指南中，我们介绍了如何将 Ultralytics YOLO11 模型导出为 CoreML 格式。通过遵循本指南中概述的步骤，您可以确保在将 YOLO11 模型导出为 CoreML 时获得最大的兼容性和性能。

有关使用的更多详细信息，请访问 [CoreML 官方文档](https://developer.apple.com/documentation/coreml)。

另外，如果您想了解更多关于其他 Ultralytics YOLO11 集成的信息，请访问我们的[集成指南页面](../integrations/index.md)。您将在那里找到大量有价值的资源和见解。

## 常见问题

### 如何将 YOLO11 模型导出为 CoreML 格式？

要将您的 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型导出为 CoreML 格式，您首先需要确保已安装 `ultralytics` 包。您可以使用以下命令安装：

!!! example "安装"

    === "CLI"

        ```bash
        pip install ultralytics
        ```

接下来，您可以使用以下 Python 或 CLI 命令导出模型：

!!! example "使用"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="coreml")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=coreml
        ```

有关更多详细信息，请参阅文档的[将 YOLO11 模型导出为 CoreML](../modes/export.md) 部分。

### 使用 CoreML 部署 YOLO11 模型有什么好处？

CoreML 为在 Apple 设备上部署 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型提供了众多优势：

- **设备端处理**：在设备上启用本地模型推理，确保[数据隐私](https://www.ultralytics.com/glossary/data-privacy)并最小化延迟。
- **性能优化**：充分利用设备的 CPU、GPU 和神经引擎的潜力，优化速度和效率。
- **易于集成**：提供与 Apple 生态系统（包括 iOS、macOS、watchOS 和 tvOS）的无缝集成体验。
- **多功能性**：使用 CoreML 框架支持广泛的机器学习任务，如图像分析、音频处理和自然语言处理。

有关将 CoreML 模型集成到 iOS 应用程序的更多信息，请查看[将 Core ML 模型集成到您的应用程序](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)指南。

### 导出为 CoreML 的 YOLO11 模型有哪些部署选项？

将 YOLO11 模型导出为 CoreML 格式后，您有多种部署选项：

1. **设备端部署**：将 CoreML 模型直接集成到您的应用程序中，以增强隐私和离线功能。这可以通过以下方式完成：
    - **嵌入式模型**：包含在应用程序包中，可立即访问。
    - **下载模型**：根据需要从服务器获取，保持应用程序包大小较小。

2. **基于云的部署**：在服务器上托管 CoreML 模型，并通过 API 请求访问它们。这种方法支持更容易的更新，可以处理更复杂的模型。

有关部署 CoreML 模型的详细指导，请参阅 [CoreML 部署选项](#coreml-部署选项)。

### CoreML 如何确保 YOLO11 模型的优化性能？

CoreML 通过利用各种优化技术确保 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型的优化性能：

- **硬件加速**：使用设备的 CPU、GPU 和神经引擎进行高效计算。
- **模型压缩**：提供压缩模型的工具，以减少其占用空间而不影响准确性。
- **自适应推理**：根据设备的能力调整推理，以保持速度和性能之间的平衡。

有关性能优化的更多信息，请访问 [CoreML 官方文档](https://developer.apple.com/documentation/coreml)。

### 我可以直接使用导出的 CoreML 模型运行推理吗？

是的，您可以直接使用导出的 CoreML 模型运行推理。以下是 Python 和 CLI 的命令：

!!! example "运行推理"

    === "Python"

        ```python
        from ultralytics import YOLO

        coreml_model = YOLO("yolo11n.mlpackage")
        results = coreml_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo11n.mlpackage source='https://ultralytics.com/images/bus.jpg'
        ```

有关更多信息，请参阅 CoreML 导出指南的[使用部分](#使用)。
