---
comments: true
description: 学习如何将 YOLO11 模型导出为 TF GraphDef 格式，以便在移动端和 Web 等各种平台上无缝部署。
keywords: YOLO11, 导出, TensorFlow, GraphDef, 模型部署, TensorFlow Serving, TensorFlow Lite, TensorFlow.js, 机器学习, AI, 计算机视觉
---

# 如何从 YOLO11 导出到 TF GraphDef 进行部署

当您在不同环境中部署尖端[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型（如 YOLO11）时，可能会遇到兼容性问题。Google 的 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) GraphDef（或 TF GraphDef）提供了一种解决方案，它提供模型的序列化、平台无关的表示。使用 TF GraphDef 模型格式，您可以在完整 TensorFlow 生态系统可能不可用的环境中部署 YOLO11 模型，例如移动设备或专用硬件。

在本指南中，我们将逐步引导您将 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型导出为 TF GraphDef 模型格式。通过转换模型，您可以简化部署并在更广泛的应用程序和平台上使用 YOLO11 的计算机视觉功能。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/tensorflow-graphdef.avif" alt="TensorFlow GraphDef">
</p>

## 为什么应该导出到 TF GraphDef？

TF GraphDef 是 TensorFlow 生态系统的强大组件，由 Google 开发。它可用于优化和部署像 YOLO11 这样的模型。导出到 TF GraphDef 让您可以将模型从研究转移到实际应用。它允许模型在没有完整 TensorFlow 框架的环境中运行。

GraphDef 格式将模型表示为序列化的计算图。这使得各种优化技术成为可能，如常量折叠、量化和图变换。这些优化确保高效执行、减少内存使用和更快的推理速度。

GraphDef 模型可以使用 GPU、TPU 和 AI 芯片等硬件加速器，为 YOLO11 推理流水线释放显著的性能提升。TF GraphDef 格式创建了一个包含模型及其依赖项的自包含包，简化了在不同系统中的部署和集成。

## TF GraphDef 模型的主要特性

TF GraphDef 格式对于简化[模型部署](https://www.ultralytics.com/glossary/model-deployment)和优化具有独特的功能。

以下是其主要特性：

- **模型序列化**：TF GraphDef 提供了一种序列化和存储 TensorFlow 模型的方式，采用平台无关的格式。这种序列化表示允许您在没有原始 Python 代码库的情况下加载和执行模型，使部署更加容易。

- **图优化**：TF GraphDef 支持计算图的优化。这些优化可以通过简化执行流程、减少冗余和定制操作以适应特定硬件来提高性能。

- **部署灵活性**：导出为 GraphDef 格式的模型可以在各种环境中使用，包括资源受限的设备、Web 浏览器和具有专用硬件的系统。这为 TensorFlow 模型的更广泛部署开辟了可能性。

- **生产导向**：GraphDef 专为生产部署设计。它支持高效执行、序列化功能和与实际用例相符的优化。

## 使用 TF GraphDef 的部署选项

在深入了解将 YOLO11 模型导出为 TF GraphDef 格式的过程之前，让我们探索一些使用此格式的典型部署场景。

以下是如何在各种平台上高效部署 TF GraphDef 的方法。

- **TensorFlow Serving**：此框架专为在生产环境中部署 TensorFlow 模型而设计。TensorFlow Serving 提供模型管理、版本控制和大规模高效模型服务的基础设施。这是将基于 GraphDef 的模型集成到生产 Web 服务或 API 中的无缝方式。

- **移动和嵌入式设备**：使用 [TensorFlow Lite](../integrations/tflite.md) 等工具，您可以将 TF GraphDef 模型转换为针对智能手机、平板电脑和各种嵌入式设备优化的格式。然后，您的模型可以用于设备上推理，其中执行在本地完成，通常提供性能优势和离线功能。

- **Web 浏览器**：[TensorFlow.js](../integrations/tfjs.md) 支持直接在 Web 浏览器中部署 TF GraphDef 模型。它为在客户端运行的实时目标检测应用程序铺平了道路，通过 JavaScript 使用 YOLO11 的功能。

- **专用硬件**：TF GraphDef 的平台无关特性允许它针对自定义硬件，如加速器和 TPU（张量处理单元）。这些设备可以为计算密集型模型提供性能优势。

## 将 YOLO11 模型导出到 TF GraphDef

您可以将 YOLO11 目标检测模型转换为与各种系统兼容的 TF GraphDef 格式，以提高其跨平台性能。

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

        # 将模型导出为 TF GraphDef 格式
        model.export(format="pb")  # 创建 'yolo11n.pb'

        # 加载导出的 TF GraphDef 模型
        tf_graphdef_model = YOLO("yolo11n.pb")

        # 运行推理
        results = tf_graphdef_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TF GraphDef 格式
        yolo export model=yolo11n.pt format=pb # 创建 'yolo11n.pb'

        # 使用导出的模型运行推理
        yolo predict model='yolo11n.pb' source='https://ultralytics.com/images/bus.jpg'
        ```

### 导出参数

| 参数     | 类型             | 默认值  | 描述                                                                                                                             |
| -------- | ---------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'pb'`  | 导出模型的目标格式，定义与各种部署环境的兼容性。                                      |
| `imgsz`  | `int` 或 `tuple` | `640`   | 模型输入所需的图像尺寸。可以是整数（正方形图像）或元组 `(height, width)` 指定特定尺寸。       |
| `batch`  | `int`            | `1`     | 指定导出模型的批量推理大小或导出模型在 `predict` 模式下并发处理的最大图像数量。 |
| `device` | `str`            | `None`  | 指定导出设备：CPU (`device=cpu`)，Apple silicon 的 MPS (`device=mps`)。                                           |

有关导出过程的更多详情，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 TF GraphDef 模型

成功将 YOLO11 模型导出为 TF GraphDef 格式后，下一步是部署。运行 TF GraphDef 模型的主要和推荐的第一步是使用 YOLO("model.pb") 方法，如之前的使用代码片段所示。

但是，有关在各种其他设置中部署 TF GraphDef 模型的更多信息，请查看以下资源：

- **[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)**：关于 TensorFlow Serving 的指南，教您如何在生产环境中高效部署和服务[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型。

- **[TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter)**：此页面描述如何将机器学习模型转换为针对 TensorFlow Lite 设备上推理优化的格式。

- **[TensorFlow.js](https://www.tensorflow.org/js/guide/conversion)**：关于模型转换的指南，教您如何将 TensorFlow 或 Keras 模型转换为 TensorFlow.js 格式以在 Web 应用程序中使用。

## 总结

在本指南中，我们探讨了如何将 Ultralytics YOLO11 模型导出为 TF GraphDef 格式。通过这样做，您可以在不同环境中灵活部署优化的 YOLO11 模型。

有关使用的更多详情，请访问 [TF GraphDef 官方文档](https://www.tensorflow.org/api_docs/python/tf/Graph)。

有关将 Ultralytics YOLO11 与其他平台和框架集成的更多信息，请查看我们的[集成指南页面](index.md)。

## 常见问题

### 如何将 YOLO11 模型导出为 TF GraphDef 格式？

Ultralytics YOLO11 模型可以无缝导出为 TensorFlow GraphDef (TF GraphDef) 格式。此格式提供模型的序列化、平台无关的表示，非常适合在移动和 Web 等各种环境中部署。要将 YOLO11 模型导出为 TF GraphDef，请按照以下步骤操作：

!!! example "使用方法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 TF GraphDef 格式
        model.export(format="pb")  # 创建 'yolo11n.pb'

        # 加载导出的 TF GraphDef 模型
        tf_graphdef_model = YOLO("yolo11n.pb")

        # 运行推理
        results = tf_graphdef_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TF GraphDef 格式
        yolo export model="yolo11n.pt" format="pb" # 创建 'yolo11n.pb'

        # 使用导出的模型运行推理
        yolo predict model="yolo11n.pb" source="https://ultralytics.com/images/bus.jpg"
        ```

有关不同导出选项的更多信息，请访问 [Ultralytics 模型导出文档](../modes/export.md)。

### 使用 TF GraphDef 进行 YOLO11 模型部署有什么好处？

将 YOLO11 模型导出为 TF GraphDef 格式提供多种优势，包括：

1. **平台独立性**：TF GraphDef 提供平台无关的格式，允许模型在包括移动和 Web 浏览器在内的各种环境中部署。
2. **优化**：该格式支持多种优化，如常量折叠、量化和图变换，从而提高执行效率并减少内存使用。
3. **硬件加速**：TF GraphDef 格式的模型可以利用 GPU、TPU 和 AI 芯片等硬件加速器获得性能提升。

在我们文档的 [TF GraphDef 部分](#tf-graphdef-模型的主要特性)中阅读更多关于好处的信息。

### 为什么应该使用 Ultralytics YOLO11 而不是其他目标检测模型？

与 YOLOv5 和 YOLOv7 等其他模型相比，Ultralytics YOLO11 提供了众多优势。一些关键好处包括：

1. **最先进的性能**：YOLO11 为实时目标检测、分割和分类提供卓越的速度和[准确率](https://www.ultralytics.com/glossary/accuracy)。
2. **易于使用**：提供用户友好的 API 用于模型训练、验证、预测和导出，使初学者和专家都能轻松使用。
3. **广泛兼容性**：支持多种导出格式，包括 ONNX、TensorRT、CoreML 和 TensorFlow，提供多样化的部署选项。

在我们的 [YOLO11 介绍](../models/yolo11.md)中探索更多详情。

### 如何使用 TF GraphDef 在专用硬件上部署 YOLO11 模型？

一旦 YOLO11 模型导出为 TF GraphDef 格式，您可以在各种专用硬件平台上部署它。典型的部署场景包括：

- **TensorFlow Serving**：使用 TensorFlow Serving 在生产环境中进行可扩展的模型部署。它支持模型管理和高效服务。
- **移动设备**：将 TF GraphDef 模型转换为 TensorFlow Lite，针对移动和嵌入式设备进行优化，实现设备上推理。
- **Web 浏览器**：使用 TensorFlow.js 在 Web 应用程序中进行客户端推理部署模型。
- **AI 加速器**：利用 TPU 和自定义 AI 芯片进行加速推理。

查看[部署选项](#使用-tf-graphdef-的部署选项)部分获取详细信息。

### 在哪里可以找到导出 YOLO11 模型时常见问题的解决方案？

有关导出 YOLO11 模型时常见问题的故障排除，Ultralytics 提供了全面的指南和资源。如果您在安装或模型导出过程中遇到问题，请参阅：

- **[常见问题指南](../guides/yolo-common-issues.md)**：提供常见问题的解决方案。
- **[安装指南](../quickstart.md)**：安装所需包的分步说明。

这些资源应该可以帮助您解决与 YOLO11 模型导出和部署相关的大多数问题。
