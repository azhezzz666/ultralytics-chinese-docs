---
comments: true
description: 通过导出到 NCNN 格式优化 YOLO11 模型，用于移动和嵌入式设备。在资源受限环境中增强性能。
keywords: Ultralytics, YOLO11, NCNN, 模型导出, 机器学习, 部署, 移动, 嵌入式系统, 深度学习, AI 模型
---

# 如何从 YOLO11 导出到 NCNN 以实现流畅部署

在计算能力有限的设备（如移动或嵌入式系统）上部署[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型可能很棘手。你需要确保使用针对最佳性能优化的格式。这确保即使处理能力有限的设备也能很好地处理高级计算机视觉任务。

导出到 NCNN 格式功能允许你优化 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型，用于轻量级基于设备的应用。在本指南中，我们将引导你如何将模型转换为 NCNN 格式，使你的模型更容易在各种移动和嵌入式设备上表现良好。

## 为什么应该导出到 NCNN？

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ncnn-overview.avif" alt="NCNN 概览">
</p>

由腾讯开发的 [NCNN](https://github.com/Tencent/ncnn) 框架是一个高性能[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)推理计算框架，专门针对移动平台进行优化，包括手机、嵌入式设备和物联网设备。NCNN 兼容广泛的平台，包括 Linux、Android、iOS 和 macOS。

NCNN 以其在移动 CPU 上的快速处理速度而闻名，能够快速将[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型部署到移动平台。这使得构建智能应用变得更容易，将 AI 的力量直接放在你的指尖。

## NCNN 模型的关键功能

NCNN 模型提供了广泛的关键功能，通过帮助开发人员在移动、嵌入式和边缘设备上运行模型来实现设备端[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)：

- **高效和高性能**：NCNN 模型被设计为高效和轻量，针对在资源有限的移动和嵌入式设备（如 Raspberry Pi）上运行进行了优化。它们还可以在各种基于计算机视觉的任务上实现高性能和高[准确率](https://www.ultralytics.com/glossary/accuracy)。

- **量化**：NCNN 模型通常支持量化，这是一种降低模型权重和激活[精度](https://www.ultralytics.com/glossary/precision)的技术。这导致性能进一步提高并减少内存占用。

- **兼容性**：NCNN 模型与流行的深度学习框架兼容，如 [TensorFlow](https://www.tensorflow.org/)、[Caffe](https://caffe.berkeleyvision.org/) 和 [ONNX](https://onnx.ai/)。这种兼容性允许开发人员轻松使用现有模型和工作流程。

- **易于使用**：NCNN 模型设计为易于集成到各种应用中，这要归功于它们与流行深度学习框架的兼容性。此外，NCNN 提供用户友好的工具，用于在不同格式之间转换模型，确保在整个开发环境中的平滑互操作性。

## 使用 NCNN 的部署选项

在我们查看将 YOLO11 模型导出到 NCNN 格式的代码之前，让我们了解 NCNN 模型通常如何使用。

NCNN 模型设计为高效和高性能，与各种部署平台兼容：

- **移动部署**：专门针对 Android 和 iOS 进行优化，允许无缝集成到移动应用中，实现高效的设备端推理。

- **嵌入式系统和物联网设备**：如果你发现使用 [Ultralytics 指南](../guides/raspberry-pi.md)在 Raspberry Pi 上运行推理不够快，切换到 NCNN 导出的模型可能有助于加速。NCNN 非常适合 Raspberry Pi 和 NVIDIA Jetson 等设备，特别是在需要直接在设备上快速处理的情况下。

- **桌面和服务器部署**：能够在 Linux、Windows 和 macOS 的桌面和服务器环境中部署，支持具有更高计算能力的开发、训练和评估。

## 导出到 NCNN：转换你的 YOLO11 模型

你可以通过将 YOLO11 模型转换为 NCNN 格式来扩展模型兼容性和部署灵活性。

### 安装

要安装所需的包，运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 所需的包
        pip install ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [Ultralytics 安装指南](../quickstart.md)。在为 YOLO11 安装所需包时，如果遇到任何困难，请查阅我们的[常见问题指南](../guides/yolo-common-issues.md)获取解决方案和提示。

### 用法

所有 [Ultralytics YOLO11 模型](../models/index.md)都设计为开箱即用支持导出，使其易于集成到你首选的部署工作流程中。你可以[查看支持的导出格式和配置选项的完整列表](../modes/export.md)，为你的应用选择最佳设置。

!!! example "用法"

    === "Python"

          ```python
          from ultralytics import YOLO

          # 加载 YOLO11 模型
          model = YOLO("yolo11n.pt")

          # 将模型导出为 NCNN 格式
          model.export(format="ncnn")  # 创建 '/yolo11n_ncnn_model'

          # 加载导出的 NCNN 模型
          ncnn_model = YOLO("./yolo11n_ncnn_model")

          # 运行推理
          results = ncnn_model("https://ultralytics.com/images/bus.jpg")
          ```

    === "CLI"

          ```bash
          # 将 YOLO11n PyTorch 模型导出为 NCNN 格式
          yolo export model=yolo11n.pt format=ncnn # 创建 '/yolo11n_ncnn_model'

          # 使用导出的模型运行推理
          yolo predict model='./yolo11n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
          ```

### 导出参数

| 参数 | 类型 | 默认值 | 描述 |
| -------- | ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str` | `'ncnn'` | 导出模型的目标格式，定义与各种部署环境的兼容性。 |
| `imgsz` | `int` 或 `tuple` | `640` | 模型输入所需的图像大小。可以是整数（用于正方形图像）或元组 `(height, width)`（用于特定尺寸）。 |
| `half` | `bool` | `False` | 启用 FP16（半精度）量化，减小模型大小并可能在支持的硬件上加速推理。 |
| `batch` | `int` | `1` | 指定导出模型批量推理大小或导出模型在 `predict` 模式下将并发处理的最大图像数量。 |
| `device` | `str` | `None` | 指定导出设备：GPU (`device=0`)、CPU (`device=cpu`)、Apple silicon 的 MPS (`device=mps`)。 |

有关导出过程的更多详细信息，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 NCNN 模型

成功将 Ultralytics YOLO11 模型导出为 NCNN 格式后，你现在可以部署它们。运行 NCNN 模型的主要和推荐的第一步是使用 `YOLO("yolo11n_ncnn_model/")` 方法，如前面的用法代码片段所述。但是，有关在各种其他设置中部署 NCNN 模型的深入说明，请查看以下资源：

- **[Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)**：此博客解释了如何使用 NCNN 模型通过 Android 应用执行[目标检测](https://www.ultralytics.com/glossary/object-detection)等任务。

- **[macOS](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macos)**：了解如何使用 NCNN 模型通过 macOS 执行任务。

- **[Linux](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)**：探索此页面了解如何在资源有限的设备（如 Raspberry Pi 和其他类似设备）上部署 NCNN 模型。

- **[Windows x64 使用 VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)**：探索此博客了解如何使用 Visual Studio Community 2017 在 Windows x64 上部署 NCNN 模型。

## 总结

在本指南中，我们介绍了将 Ultralytics YOLO11 模型导出到 NCNN 格式。此转换步骤对于提高 YOLO11 模型的效率和速度至关重要，使它们在资源有限的计算环境中更加有效和适用。

有关用法的详细说明，请参阅[官方 NCNN 文档](https://ncnn.readthedocs.io/en/latest/index.html)。

此外，如果你有兴趣探索 Ultralytics YOLO11 的其他集成选项，请务必访问我们的[集成指南页面](index.md)获取更多见解和信息。

## 常见问题

### 如何将 Ultralytics YOLO11 模型导出为 NCNN 格式？

要将 Ultralytics YOLO11 模型导出为 NCNN 格式，请按照以下步骤操作：

- **Python**：使用 YOLO 类的 `export` 函数。

    ```python
    from ultralytics import YOLO

    # 加载 YOLO11 模型
    model = YOLO("yolo11n.pt")

    # 导出为 NCNN 格式
    model.export(format="ncnn")  # 创建 '/yolo11n_ncnn_model'
    ```

- **CLI**：使用带有 `export` 参数的 `yolo` 命令。
    ```bash
    yolo export model=yolo11n.pt format=ncnn # 创建 '/yolo11n_ncnn_model'
    ```

有关详细的导出选项，请查看文档中的[导出](../modes/export.md)页面。

### 将 YOLO11 模型导出到 NCNN 有什么优势？

将 Ultralytics YOLO11 模型导出到 NCNN 提供了几个好处：

- **效率**：NCNN 模型针对移动和嵌入式设备进行了优化，即使在计算资源有限的情况下也能确保高性能。
- **量化**：NCNN 支持量化等技术，可以提高模型速度并减少内存使用。
- **广泛兼容性**：你可以在多个平台上部署 NCNN 模型，包括 Android、iOS、Linux 和 macOS。

有关更多详细信息，请参阅文档中的[导出到 NCNN](#为什么应该导出到-ncnn)部分。

### 为什么应该在移动 AI 应用中使用 NCNN？

由腾讯开发的 NCNN 专门针对移动平台进行了优化。使用 NCNN 的主要原因包括：

- **高性能**：设计用于在移动 CPU 上进行高效和快速处理。
- **跨平台**：与 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 和 ONNX 等流行框架兼容，使跨不同平台转换和部署模型变得更容易。
- **社区支持**：活跃的社区支持确保持续改进和更新。

要了解更多，请访问文档中的 [NCNN 概览](#ncnn-模型的关键功能)。

### NCNN [模型部署](https://www.ultralytics.com/glossary/model-deployment)支持哪些平台？

NCNN 功能多样，支持各种平台：

- **移动端**：Android、iOS。
- **嵌入式系统和物联网设备**：Raspberry Pi 和 NVIDIA Jetson 等设备。
- **桌面和服务器**：Linux、Windows 和 macOS。

如果在 Raspberry Pi 上运行模型不够快，转换为 NCNN 格式可能会加速，如我们的 [Raspberry Pi 指南](../guides/raspberry-pi.md)中所述。

### 如何在 Android 上部署 Ultralytics YOLO11 NCNN 模型？

要在 Android 上部署 YOLO11 模型：

1. **Android 构建**：按照 [NCNN Android 构建](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)指南。
2. **与应用集成**：使用 NCNN Android SDK 将导出的模型集成到你的应用中，实现高效的设备端推理。

有关分步说明，请参阅我们的[部署 YOLO11 NCNN 模型](#部署导出的-yolo11-ncnn-模型)指南。

有关更多高级指南和用例，请访问 [Ultralytics 文档页面](../guides/model-deployment-options.md)。
