---
comments: true
description: 学习如何将 Ultralytics YOLO11 模型导出为 TensorFlow SavedModel 格式，以便在各种平台和环境中轻松部署。
keywords: YOLO11, TF SavedModel, Ultralytics, TensorFlow, 模型导出, 模型部署, 机器学习, AI
---

# 了解如何从 YOLO11 导出到 TF SavedModel 格式

部署[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型可能具有挑战性。然而，使用高效灵活的模型格式可以使您的工作更轻松。TF SavedModel 是一个开源机器学习框架，TensorFlow 使用它以一致的方式加载机器学习模型。它就像 TensorFlow 模型的手提箱，使它们易于在不同设备和系统上携带和使用。

学习如何从 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型导出到 TF SavedModel 可以帮助您在不同平台和环境中轻松部署模型。在本指南中，我们将逐步介绍如何将模型转换为 TF SavedModel 格式，简化在不同设备上使用模型运行推理的过程。

## 为什么应该导出到 TF SavedModel？

TensorFlow SavedModel 格式是 TensorFlow 生态系统的一部分，由 Google 开发，如下所示。它旨在无缝保存和序列化 TensorFlow 模型。它封装了模型的完整详细信息，如架构、权重，甚至编译信息。这使得在不同环境中共享、部署和继续训练变得简单。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/tf-savedmodel-overview.avif" alt="TF SavedModel">
</p>

TF SavedModel 有一个关键优势：其兼容性。它与 [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)、TensorFlow Lite 和 TensorFlow.js 配合良好。这种兼容性使得在各种平台上共享和部署模型更加容易，包括 Web 和移动应用程序。TF SavedModel 格式对研究和生产都很有用。它提供了一种统一的方式来管理您的模型，确保它们为任何应用程序做好准备。

## TF SavedModel 的主要特性

以下是使 TF SavedModel 成为 AI 开发者绝佳选择的主要特性：

- **可移植性**：TF SavedModel 提供语言中立、可恢复、密封的序列化格式。它们使更高级别的系统和工具能够生成、使用和转换 TensorFlow 模型。SavedModel 可以轻松地在不同平台和环境中共享和部署。

- **易于部署**：TF SavedModel 将计算图、训练参数和必要的元数据捆绑到一个包中。它们可以轻松加载并用于推理，而无需构建模型的原始代码。这使得在各种生产环境中部署 TensorFlow 模型变得简单高效。

- **资产管理**：TF SavedModel 支持包含外部资产，如词汇表、[嵌入](https://www.ultralytics.com/glossary/embeddings)或查找表。这些资产与图定义和变量一起存储，确保在加载模型时它们可用。此功能简化了依赖外部资源的模型的管理和分发。

## 使用 TF SavedModel 的部署选项

在深入了解将 YOLO11 模型导出为 TF SavedModel 格式的过程之前，让我们探索一些使用此格式的典型部署场景。

TF SavedModel 提供了一系列部署机器学习模型的选项：

- **TensorFlow Serving**：TensorFlow Serving 是一个灵活、高性能的服务系统，专为生产环境设计。它原生支持 TF SavedModel，使得在云平台、本地服务器或[边缘设备](https://docs.ultralytics.com/guides/raspberry-pi/)上轻松部署和服务模型。

- **云平台**：主要云提供商如 [Google Cloud Platform (GCP)](https://cloud.google.com/vertex-ai)、[Amazon Web Services (AWS)](https://aws.amazon.com/sagemaker/) 和 [Microsoft Azure](https://azure.microsoft.com/en-us/services/machine-learning/) 提供部署和运行 TensorFlow 模型（包括 TF SavedModel）的服务。这些服务提供可扩展的托管基础设施，允许您轻松部署和扩展模型。

- **移动和嵌入式设备**：[TensorFlow Lite](https://docs.ultralytics.com/integrations/tflite/) 是一个轻量级解决方案，用于在移动、嵌入式和物联网设备上运行机器学习模型，支持将 TF SavedModel 转换为 TensorFlow Lite 格式。这允许您在各种设备上部署模型，从智能手机和平板电脑到微控制器和边缘设备。

- **TensorFlow Runtime**：TensorFlow Runtime (`tfrt`) 是用于执行 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 图的高性能运行时。它提供用于在 C++ 环境中加载和运行 TF SavedModel 的低级 API。与标准 TensorFlow 运行时相比，TensorFlow Runtime 提供更好的性能。它适用于需要低延迟推理和与现有 C++ 代码库紧密集成的部署场景。

## 将 YOLO11 模型导出到 TF SavedModel

通过将 YOLO11 模型导出为 TF SavedModel 格式，您可以增强其在各种平台上的适应性和部署便利性。

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

        # 将模型导出为 TF SavedModel 格式
        model.export(format="saved_model")  # 创建 '/yolo11n_saved_model'

        # 加载导出的 TF SavedModel 模型
        tf_savedmodel_model = YOLO("./yolo11n_saved_model")

        # 运行推理
        results = tf_savedmodel_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TF SavedModel 格式
        yolo export model=yolo11n.pt format=saved_model # 创建 '/yolo11n_saved_model'

        # 使用导出的模型运行推理
        yolo predict model='./yolo11n_saved_model' source='https://ultralytics.com/images/bus.jpg'
        ```

### 导出参数

| 参数     | 类型             | 默认值          | 描述                                                                                                                                                                                   |
| -------- | ---------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'saved_model'` | 导出模型的目标格式，定义与各种部署环境的兼容性。                                                                                            |
| `imgsz`  | `int` 或 `tuple` | `640`           | 模型输入所需的图像尺寸。可以是整数（正方形图像）或元组 `(height, width)` 指定特定尺寸。                                                             |
| `keras`  | `bool`           | `False`         | 启用导出为 Keras 格式，提供与 TensorFlow serving 和 API 的兼容性。                                                                                                     |
| `int8`   | `bool`           | `False`         | 激活 INT8 量化，进一步压缩模型并加速推理，[准确率](https://www.ultralytics.com/glossary/accuracy)损失最小，主要用于边缘设备。 |
| `nms`    | `bool`           | `False`         | 添加非极大值抑制 (NMS)，对于准确高效的检测后处理至关重要。                                                                                           |
| `batch`  | `int`            | `1`             | 指定导出模型的批量推理大小或导出模型在 `predict` 模式下并发处理的最大图像数量。                                                       |
| `device` | `str`            | `None`          | 指定导出设备：CPU (`device=cpu`)，Apple silicon 的 MPS (`device=mps`)。                                                                                                 |

有关导出过程的更多详情，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 TF SavedModel 模型

现在您已将 YOLO11 模型导出为 TF SavedModel 格式，下一步是部署它。运行 TF SavedModel 模型的主要和推荐的第一步是使用 `YOLO("yolo11n_saved_model/")` 方法，如之前的使用代码片段所示。

但是，有关深入部署 TF SavedModel 模型的说明，请查看以下资源：

- **[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)**：这是关于如何使用 TensorFlow Serving 部署 TF SavedModel 模型的开发者文档。

- **[在 Node.js 中运行 TensorFlow SavedModel](https://blog.tensorflow.org/2020/01/run-tensorflow-savedmodel-in-nodejs-directly-without-conversion.html)**：一篇关于直接在 Node.js 中运行 TensorFlow SavedModel 而无需转换的 TensorFlow 博客文章。

- **[在云上部署](https://blog.tensorflow.org/2020/04/how-to-deploy-tensorflow-2-models-on-cloud-ai-platform.html)**：一篇关于在 Cloud AI Platform 上部署 TensorFlow SavedModel 模型的 TensorFlow 博客文章。

## 总结

在本指南中，我们探讨了如何将 Ultralytics YOLO11 模型导出为 TF SavedModel 格式。通过导出到 TF SavedModel，您可以灵活地在各种平台上优化、部署和扩展 YOLO11 模型。

有关使用的更多详情，请访问 [TF SavedModel 官方文档](https://www.tensorflow.org/guide/saved_model)。

有关将 Ultralytics YOLO11 与其他平台和框架集成的更多信息，请务必查看我们的[集成指南页面](index.md)。它包含大量有用的资源，帮助您在项目中充分利用 YOLO11。

## 常见问题

### 如何将 Ultralytics YOLO 模型导出为 TensorFlow SavedModel 格式？

将 Ultralytics YOLO 模型导出为 TensorFlow SavedModel 格式非常简单。您可以使用 Python 或 CLI 来实现：

!!! example "将 YOLO11 导出到 TF SavedModel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 TF SavedModel 格式
        model.export(format="saved_model")  # 创建 '/yolo11n_saved_model'

        # 加载导出的 TF SavedModel 进行推理
        tf_savedmodel_model = YOLO("./yolo11n_saved_model")
        results = tf_savedmodel_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11 模型导出为 TF SavedModel 格式
        yolo export model=yolo11n.pt format=saved_model # 创建 '/yolo11n_saved_model'

        # 使用导出的模型运行推理
        yolo predict model='./yolo11n_saved_model' source='https://ultralytics.com/images/bus.jpg'
        ```

有关更多详情，请参阅 [Ultralytics 导出文档](../modes/export.md)。

### 为什么应该使用 TensorFlow SavedModel 格式？

TensorFlow SavedModel 格式为[模型部署](https://www.ultralytics.com/glossary/model-deployment)提供了多种优势：

- **可移植性**：它提供语言中立的格式，使得在不同环境中轻松共享和部署模型。
- **兼容性**：与 TensorFlow Serving、TensorFlow Lite 和 TensorFlow.js 等工具无缝集成，这些工具对于在包括 Web 和移动应用程序在内的各种平台上部署模型至关重要。
- **完整封装**：编码模型架构、权重和编译信息，允许直接共享和继续训练。

有关更多好处和部署选项，请查看 [Ultralytics YOLO 模型部署选项](../guides/model-deployment-options.md)。

### TF SavedModel 的典型部署场景有哪些？

TF SavedModel 可以在各种环境中部署，包括：

- **TensorFlow Serving**：非常适合需要可扩展、高性能模型服务的生产环境。
- **云平台**：支持 Google Cloud Platform (GCP)、Amazon Web Services (AWS) 和 Microsoft Azure 等主要云服务进行可扩展的模型部署。
- **移动和嵌入式设备**：使用 [TensorFlow Lite](https://docs.ultralytics.com/integrations/tflite/) 转换 TF SavedModel 允许在移动设备、物联网设备和微控制器上部署。
- **TensorFlow Runtime**：适用于需要低延迟推理和更好性能的 C++ 环境。

有关详细的部署选项，请访问关于[部署 TensorFlow 模型](https://www.tensorflow.org/tfx/guide/serving)的官方指南。

### 如何安装导出 YOLO11 模型所需的包？

要导出 YOLO11 模型，请使用以下命令：

```bash
pip install ultralytics
```

有关详细的安装说明和最佳实践，请参阅 [Ultralytics 安装指南](../quickstart.md)。如果在安装过程中出现任何问题，请参阅[常见问题指南](../guides/yolo-common-issues.md)。

### TensorFlow SavedModel 格式的主要特性是什么？

由于以下特性，TF SavedModel 格式对 AI 开发者非常有益：

- **可移植性**：允许在各种环境中轻松共享和部署。
- **易于部署**：将计算图、训练参数和元数据封装到单个包中，简化加载和推理。
- **资产管理**：支持词汇表等外部资产，确保在模型加载时它们可用。

有关更多详情，请探索 [TensorFlow 官方文档](https://www.tensorflow.org/guide/saved_model)。
