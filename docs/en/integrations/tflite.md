---
comments: true
description: 学习如何将 YOLO11 模型转换为 TFLite 以在边缘设备上部署。优化性能并确保在各种平台上无缝执行。
keywords: YOLO11, TFLite, 模型导出, TensorFlow Lite, 边缘设备, 部署, Ultralytics, 机器学习, 设备上推理, 模型优化
---

# YOLO11 模型导出到 TFLite 进行部署的指南

<p align="center">
  <img width="75%" src="https://github.com/ultralytics/docs/releases/download/0/tflite-logo.avif" alt="TFLite Logo">
</p>

在边缘设备或嵌入式设备上部署[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型需要一种能够确保无缝性能的格式。

TensorFlow Lite 或 TFLite 导出格式允许您优化 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型，用于基于边缘设备的应用程序中的[目标检测](https://www.ultralytics.com/glossary/object-detection)和[图像分类](https://www.ultralytics.com/glossary/image-classification)等任务。在本指南中，我们将逐步介绍将模型转换为 TFLite 格式的步骤，使您的模型更容易在各种边缘设备上表现良好。

## 为什么应该导出到 TFLite？

TensorFlow Lite 由 Google 于 2017 年 5 月作为其 TensorFlow 框架的一部分推出，简称 TFLite，是一个专为设备上推理（也称为[边缘计算](https://www.ultralytics.com/glossary/edge-computing)）设计的开源深度学习框架。它为开发者提供了在移动、嵌入式和物联网设备以及传统计算机上执行训练模型所需的工具。

TensorFlow Lite 与多种平台兼容，包括嵌入式 Linux、Android、iOS 和微控制器 (MCU)。将模型导出到 TFLite 可使您的应用程序更快、更可靠，并能够离线运行。

## TFLite 模型的主要特性

TFLite 模型提供了一系列关键特性，通过帮助开发者在移动、嵌入式和边缘设备上运行模型来实现设备上机器学习：

- **设备上优化**：TFLite 针对设备上 ML 进行优化，通过本地处理数据来减少延迟，通过不传输个人数据来增强隐私，并通过节省空间来最小化模型大小。

- **多平台支持**：TFLite 提供广泛的平台兼容性，支持 Android、iOS、嵌入式 Linux 和微控制器。

- **多语言支持**：TFLite 与各种编程语言兼容，包括 Java、Swift、Objective-C、C++ 和 Python。

- **高性能**：通过硬件加速和模型优化实现卓越性能。

## TFLite 中的部署选项

在查看将 YOLO11 模型导出为 TFLite 格式的代码之前，让我们了解 TFLite 模型通常如何使用。

TFLite 为机器学习模型提供各种设备上部署选项，包括：

- **使用 Android 和 iOS 部署**：带有 TFLite 的 Android 和 iOS 应用程序可以分析基于边缘的摄像头馈送和传感器来检测和识别对象。TFLite 还提供用 [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) 和 [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc) 编写的原生 iOS 库。下面的架构图显示了使用 TensorFlow Lite 在 Android 和 iOS 平台上部署训练模型的过程。

 <p align="center">
  <img width="75%" src="https://github.com/ultralytics/docs/releases/download/0/architecture-diagram-tflite-deployment.avif" alt="架构">
</p>

- **使用嵌入式 Linux 实现**：如果使用 [Ultralytics 指南](../guides/raspberry-pi.md)在 [Raspberry Pi](https://www.raspberrypi.org/) 上运行推理不能满足您用例的速度要求，您可以使用导出的 TFLite 模型来加速推理时间。此外，还可以通过使用 [Coral Edge TPU 设备](https://developers.google.com/coral)进一步提高性能。

- **使用微控制器部署**：TFLite 模型也可以部署在只有几千字节内存的微控制器和其他设备上。核心运行时在 Arm Cortex M3 上仅占用 16 KB，可以运行许多基本模型。它不需要操作系统支持、任何标准 C 或 C++ 库或动态内存分配。

## 导出到 TFLite：转换您的 YOLO11 模型

您可以通过将模型转换为 TFLite 格式来提高设备上模型执行效率并优化性能。

### 安装

要安装所需的包，请运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 所需的包
        pip install ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [Ultralytics 安装指南](../quickstart.md)。在为 YOLO11 安装所需包时，如果遇到任何困难，请参阅我们的[常见问题指南](../guides/yolo-common-issues.md)获取解决方案和提示。

### 使用方法

所有 [Ultralytics YOLO11 模型](../models/index.md)都设计为开箱即用地支持导出，使其易于集成到您首选的部署工作流程中。您可以[查看支持的导出格式和配置选项的完整列表](../modes/export.md)，为您的应用程序选择最佳设置。

!!! example "使用方法"

    === "Python"

          ```python
          from ultralytics import YOLO

          # 加载 YOLO11 模型
          model = YOLO("yolo11n.pt")

          # 将模型导出为 TFLite 格式
          model.export(format="tflite")  # 创建 'yolo11n_float32.tflite'

          # 加载导出的 TFLite 模型
          tflite_model = YOLO("yolo11n_float32.tflite")

          # 运行推理
          results = tflite_model("https://ultralytics.com/images/bus.jpg")
          ```

    === "CLI"

          ```bash
          # 将 YOLO11n PyTorch 模型导出为 TFLite 格式
          yolo export model=yolo11n.pt format=tflite # 创建 'yolo11n_float32.tflite'

          # 使用导出的模型运行推理
          yolo predict model='yolo11n_float32.tflite' source='https://ultralytics.com/images/bus.jpg'
          ```

### 导出参数

| 参数       | 类型             | 默认值         | 描述                                                                                                                                                                                                                                                      |
| ---------- | ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'tflite'`     | 导出模型的目标格式，定义与各种部署环境的兼容性。                                                                                                                                               |
| `imgsz`    | `int` 或 `tuple` | `640`          | 模型输入所需的图像尺寸。可以是整数（正方形图像）或元组 `(height, width)` 指定特定尺寸。                                                                                                                |
| `half`     | `bool`           | `False`        | 启用 FP16（半精度）量化，减少模型大小并可能在支持的硬件上加速推理。                                                                                                                                     |
| `int8`     | `bool`           | `False`        | 激活 INT8 量化，进一步压缩模型并加速推理，[准确率](https://www.ultralytics.com/glossary/accuracy)损失最小，主要用于边缘设备。                                                                    |
| `nms`      | `bool`           | `False`        | 添加非极大值抑制 (NMS)，对于准确高效的检测后处理至关重要。                                                                                                                                              |
| `batch`    | `int`            | `1`            | 指定导出模型的批量推理大小或导出模型在 `predict` 模式下并发处理的最大图像数量。                                                                                                                          |
| `data`     | `str`            | `'coco8.yaml'` | [数据集](https://docs.ultralytics.com/datasets/)配置文件的路径（默认：`coco8.yaml`），对量化至关重要。                                                                                                            |
| `fraction` | `float`          | `1.0`          | 指定用于 INT8 量化校准的数据集比例。允许在完整数据集的子集上进行校准，对于实验或资源有限时很有用。如果启用 INT8 但未指定，将使用完整数据集。 |
| `device`   | `str`            | `None`         | 指定导出设备：CPU (`device=cpu`)，Apple silicon 的 MPS (`device=mps`)。                                                                                                                                    |

有关导出过程的更多详情，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 TFLite 模型

成功将 Ultralytics YOLO11 模型导出为 TFLite 格式后，您现在可以部署它们。运行 TFLite 模型的主要和推荐的第一步是使用 `YOLO("model.tflite")` 方法，如之前的使用代码片段所述。但是，有关在各种其他设置中部署 TFLite 模型的深入说明，请查看以下资源：

- **[Android](https://ai.google.dev/edge/litert/android)**：将 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite 集成到 Android 应用程序的快速入门指南，提供设置和运行[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型的简单步骤。

- **[iOS](https://ai.google.dev/edge/litert/ios/quickstart)**：查看此详细指南，了解开发者如何在 iOS 应用程序中集成和部署 TensorFlow Lite 模型，提供分步说明和资源。

- **[端到端示例](https://github.com/tensorflow/examples/tree/master/lite/examples)**：此页面提供各种 TensorFlow Lite 示例的概述，展示旨在帮助开发者在移动和边缘设备上的机器学习项目中实现 TensorFlow Lite 的实际应用和教程。

## 总结

在本指南中，我们重点介绍了如何导出为 TFLite 格式。通过将 Ultralytics YOLO11 模型转换为 TFLite 模型格式，您可以提高 YOLO11 模型的效率和速度，使其在边缘计算环境中更加有效和适用。

有关使用的更多详情，请访问 [TFLite 官方文档](https://ai.google.dev/edge/litert)。

此外，如果您对其他 Ultralytics YOLO11 集成感到好奇，请查看我们的[集成指南页面](../integrations/index.md)。您将在那里找到大量有用的信息和见解。

## 常见问题

### 如何将 YOLO11 模型导出为 TFLite 格式？

要将 YOLO11 模型导出为 TFLite 格式，您可以使用 Ultralytics 库。首先，使用以下命令安装所需的包：

```bash
pip install ultralytics
```

然后，使用以下代码片段导出您的模型：

```python
from ultralytics import YOLO

# 加载 YOLO11 模型
model = YOLO("yolo11n.pt")

# 将模型导出为 TFLite 格式
model.export(format="tflite")  # 创建 'yolo11n_float32.tflite'
```

对于 CLI 用户，您可以使用以下命令实现：

```bash
yolo export model=yolo11n.pt format=tflite # 创建 'yolo11n_float32.tflite'
```

有关更多详情，请访问 [Ultralytics 导出指南](../modes/export.md)。

### 使用 TensorFlow Lite 进行 YOLO11 模型部署有什么好处？

TensorFlow Lite (TFLite) 是一个开源[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)框架，专为设备上推理设计，非常适合在移动、嵌入式和物联网设备上部署 YOLO11 模型。主要好处包括：

- **设备上优化**：通过本地处理数据来最小化延迟并增强隐私。
- **平台兼容性**：支持 Android、iOS、嵌入式 Linux 和 MCU。
- **性能**：利用硬件加速来优化模型速度和效率。

要了解更多，请查看 [TFLite 指南](https://ai.google.dev/edge/litert)。

### 是否可以在 Raspberry Pi 上运行 YOLO11 TFLite 模型？

是的，您可以在 Raspberry Pi 上运行 YOLO11 TFLite 模型以提高推理速度。首先，如上所述将模型导出为 TFLite 格式。然后，使用 TensorFlow Lite Interpreter 等工具在 Raspberry Pi 上执行模型。

为了进一步优化，您可以考虑使用 [Coral Edge TPU](https://developers.google.com/coral)。有关详细步骤，请参阅我们的 [Raspberry Pi 部署指南](../guides/raspberry-pi.md)和 [Edge TPU 集成指南](../integrations/edge-tpu.md)。

### 我可以在微控制器上使用 TFLite 模型进行 YOLO11 预测吗？

是的，TFLite 支持在资源有限的微控制器上部署。TFLite 的核心运行时在 Arm Cortex M3 上仅需要 16 KB 内存，可以运行基本的 YOLO11 模型。这使其适合在计算能力和内存最小的设备上部署。

要开始使用，请访问 [TFLite Micro 微控制器指南](https://ai.google.dev/edge/litert/microcontrollers/overview)。

### 哪些平台与 TFLite 导出的 YOLO11 模型兼容？

TensorFlow Lite 提供广泛的平台兼容性，允许您在各种设备上部署 YOLO11 模型，包括：

- **Android 和 iOS**：通过 TFLite Android 和 iOS 库提供原生支持。
- **嵌入式 Linux**：非常适合 Raspberry Pi 等单板计算机。
- **微控制器**：适用于资源受限的 MCU。

有关部署选项的更多信息，请参阅我们详细的[部署指南](#部署导出的-yolo11-tflite-模型)。

### 如何排除导出 YOLO11 模型到 TFLite 时的常见问题？

如果您在将 YOLO11 模型导出到 TFLite 时遇到错误，常见解决方案包括：

- **检查包兼容性**：确保您使用的是兼容版本的 Ultralytics 和 TensorFlow。请参阅我们的[安装指南](../quickstart.md)。
- **模型支持**：通过查看 Ultralytics [导出文档页面](../modes/export.md)验证特定 YOLO11 模型是否支持 TFLite 导出。
- **量化问题**：使用 INT8 量化时，确保在 `data` 参数中正确指定数据集路径。

有关更多故障排除提示，请访问我们的[常见问题指南](../guides/yolo-common-issues.md)。
