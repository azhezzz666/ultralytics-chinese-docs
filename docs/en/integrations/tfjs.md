---
comments: true
description: 将 Ultralytics YOLO11 模型转换为 TensorFlow.js 以实现高速、本地目标检测。学习如何为浏览器和 Node.js 应用优化机器学习模型。
keywords: YOLO11, TensorFlow.js, TF.js, 模型导出, 机器学习, 目标检测, 浏览器 ML, Node.js, Ultralytics, YOLO, 导出模型
---

# 从 YOLO11 模型格式导出到 TF.js 模型格式

直接在浏览器或 Node.js 上部署[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型可能很棘手。您需要确保模型格式针对更快的性能进行了优化，以便模型可以用于在用户设备上本地运行交互式应用程序。TensorFlow.js（或 TF.js）模型格式旨在使用最少的功耗同时提供快速性能。

"导出到 TF.js 模型格式"功能允许您优化 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型，以实现高速和本地运行的[目标检测](https://www.ultralytics.com/glossary/object-detection)推理。在本指南中，我们将引导您将模型转换为 TF.js 格式，使您的模型更容易在各种本地浏览器和 Node.js 应用程序上表现良好。

## 为什么应该导出到 TF.js？

将机器学习模型导出到 TensorFlow.js（由 TensorFlow 团队作为更广泛 TensorFlow 生态系统的一部分开发）为部署机器学习应用程序提供了众多优势。它通过将敏感数据保留在设备上来帮助增强用户隐私和安全性。下图显示了 TensorFlow.js 架构，以及机器学习模型如何在 Web 浏览器和 Node.js 上转换和部署。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/tfjs-architecture.avif" alt="TF.js 架构">
</p>

在本地运行模型还可以减少延迟并提供更响应的用户体验。[TensorFlow.js](https://www.ultralytics.com/glossary/tensorflow) 还具有离线功能，允许用户即使没有互联网连接也能使用您的应用程序。TF.js 专为在资源有限的设备上高效执行复杂模型而设计，因为它针对可扩展性进行了工程设计，并支持 GPU 加速。

## TF.js 的主要特性

以下是使 TF.js 成为开发者强大工具的主要特性：

- **跨平台支持**：TensorFlow.js 可以在浏览器和 Node.js 环境中使用，提供跨不同平台部署的灵活性。它让开发者更容易构建和部署应用程序。

- **支持多种后端**：TensorFlow.js 支持各种计算后端，包括 CPU、用于 GPU 加速的 WebGL、用于接近原生执行速度的 WebAssembly (WASM)，以及用于高级基于浏览器的机器学习功能的 WebGPU。

- **离线功能**：使用 TensorFlow.js，模型可以在浏览器中运行而无需互联网连接，使开发可离线运行的应用程序成为可能。

## 使用 TensorFlow.js 的部署选项

在深入了解将 YOLO11 模型导出为 TF.js 格式的过程之前，让我们探索一些使用此格式的典型部署场景。

TF.js 提供了一系列部署机器学习模型的选项：

- **浏览器内 ML 应用程序**：您可以构建直接在浏览器中运行机器学习模型的 Web 应用程序。消除了服务器端计算的需求并减少了服务器负载。

- **Node.js 应用程序**：TensorFlow.js 还支持在 Node.js 环境中部署，使服务器端机器学习应用程序的开发成为可能。它对于需要服务器处理能力或访问服务器端数据的应用程序特别有用。

- **Chrome 扩展**：一个有趣的部署场景是使用 TensorFlow.js 创建 Chrome 扩展。例如，您可以开发一个扩展，允许用户右键单击任何网页中的图像，使用预训练的 ML 模型对其进行分类。TensorFlow.js 可以集成到日常 Web 浏览体验中，根据机器学习提供即时洞察或增强功能。

## 将 YOLO11 模型导出到 TensorFlow.js

您可以通过将 YOLO11 模型转换为 TF.js 来扩展模型兼容性和部署灵活性。

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

        # 将模型导出为 TF.js 格式
        model.export(format="tfjs")  # 创建 '/yolo11n_web_model'

        # 加载导出的 TF.js 模型
        tfjs_model = YOLO("./yolo11n_web_model")

        # 运行推理
        results = tfjs_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TF.js 格式
        yolo export model=yolo11n.pt format=tfjs # 创建 '/yolo11n_web_model'

        # 使用导出的模型运行推理
        yolo predict model='./yolo11n_web_model' source='https://ultralytics.com/images/bus.jpg'
        ```

### 导出参数

| 参数     | 类型             | 默认值   | 描述                                                                                                                                                                                   |
| -------- | ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'tfjs'` | 导出模型的目标格式，定义与各种部署环境的兼容性。                                                                                            |
| `imgsz`  | `int` 或 `tuple` | `640`    | 模型输入所需的图像尺寸。可以是整数（正方形图像）或元组 `(height, width)` 指定特定尺寸。                                                             |
| `half`   | `bool`           | `False`  | 启用 FP16（半精度）量化，减少模型大小并可能在支持的硬件上加速推理。                                                                  |
| `int8`   | `bool`           | `False`  | 激活 INT8 量化，进一步压缩模型并加速推理，[准确率](https://www.ultralytics.com/glossary/accuracy)损失最小，主要用于边缘设备。 |
| `nms`    | `bool`           | `False`  | 添加非极大值抑制 (NMS)，对于准确高效的检测后处理至关重要。                                                                                           |
| `batch`  | `int`            | `1`      | 指定导出模型的批量推理大小或导出模型在 `predict` 模式下并发处理的最大图像数量。                                                       |
| `device` | `str`            | `None`   | 指定导出设备：CPU (`device=cpu`)，Apple silicon 的 MPS (`device=mps`)。                                                                                                 |

有关导出过程的更多详情，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 TensorFlow.js 模型

现在您已将 YOLO11 模型导出为 TF.js 格式，下一步是部署它。运行 TF.js 模型的主要和推荐的第一步是使用 `YOLO("./yolo11n_web_model")` 方法，如之前的使用代码片段所示。

但是，有关深入部署 TF.js 模型的说明，请查看以下资源：

- **[Chrome 扩展](https://www.tensorflow.org/js/tutorials/deployment/web_ml_in_chrome)**：这是关于如何将 TF.js 模型部署到 Chrome 扩展的开发者文档。

- **[在 Node.js 中运行 TensorFlow.js](https://www.tensorflow.org/js/guide/nodejs)**：一篇关于直接在 Node.js 中运行 TensorFlow.js 的 TensorFlow 博客文章。

- **[在云平台上部署 TensorFlow.js - Node 项目](https://www.tensorflow.org/js/guide/node_in_cloud)**：一篇关于在云平台上部署 TensorFlow.js 模型的 TensorFlow 博客文章。

## 总结

在本指南中，我们学习了如何将 Ultralytics YOLO11 模型导出为 TensorFlow.js 格式。通过导出到 TF.js，您可以灵活地在各种平台上优化、部署和扩展 YOLO11 模型。

有关使用的更多详情，请访问 [TensorFlow.js 官方文档](https://www.tensorflow.org/js/guide)。

有关将 Ultralytics YOLO11 与其他平台和框架集成的更多信息，请务必查看我们的[集成指南页面](index.md)。它包含大量有用的资源，帮助您在项目中充分利用 YOLO11。

## 常见问题

### 如何将 Ultralytics YOLO11 模型导出为 TensorFlow.js 格式？

将 Ultralytics YOLO11 模型导出为 TensorFlow.js (TF.js) 格式非常简单。您可以按照以下步骤操作：

!!! example "使用方法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 TF.js 格式
        model.export(format="tfjs")  # 创建 '/yolo11n_web_model'

        # 加载导出的 TF.js 模型
        tfjs_model = YOLO("./yolo11n_web_model")

        # 运行推理
        results = tfjs_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TF.js 格式
        yolo export model=yolo11n.pt format=tfjs # 创建 '/yolo11n_web_model'

        # 使用导出的模型运行推理
        yolo predict model='./yolo11n_web_model' source='https://ultralytics.com/images/bus.jpg'
        ```

有关支持的导出选项的更多详情，请访问 [Ultralytics 部署选项文档页面](../guides/model-deployment-options.md)。

### 为什么应该将 YOLO11 模型导出到 TensorFlow.js？

将 YOLO11 模型导出到 TensorFlow.js 提供了多种优势，包括：

1. **本地执行**：模型可以直接在浏览器或 Node.js 中运行，减少延迟并增强用户体验。
2. **跨平台支持**：TF.js 支持多种环境，允许部署灵活性。
3. **离线功能**：使应用程序无需互联网连接即可运行，确保可靠性和隐私。
4. **GPU 加速**：利用 WebGL 进行 GPU 加速，优化资源有限设备上的性能。

有关全面概述，请参阅我们的[与 TensorFlow.js 集成](../integrations/tf-graphdef.md)。

### TensorFlow.js 如何使基于浏览器的机器学习应用程序受益？

TensorFlow.js 专门设计用于在浏览器和 Node.js 环境中高效执行 ML 模型。以下是它如何使基于浏览器的应用程序受益：

- **减少延迟**：在本地运行机器学习模型，提供即时结果而无需依赖服务器端计算。
- **改善隐私**：将敏感数据保留在用户设备上，最大限度地降低安全风险。
- **支持离线使用**：模型可以在没有互联网连接的情况下运行，确保功能一致。
- **支持多种后端**：提供 CPU、WebGL、WebAssembly (WASM) 和 WebGPU 等后端的灵活性，满足不同的计算需求。

有兴趣了解更多关于 TF.js 的信息吗？查看 [TensorFlow.js 官方指南](https://www.tensorflow.org/js/guide)。

### 部署 YOLO11 模型的 TensorFlow.js 主要特性是什么？

TensorFlow.js 的主要特性包括：

- **跨平台支持**：TF.js 可以在 Web 浏览器和 Node.js 中使用，提供广泛的部署灵活性。
- **多种后端**：支持 CPU、用于 GPU 加速的 WebGL、WebAssembly (WASM) 和用于高级操作的 WebGPU。
- **离线功能**：模型可以直接在浏览器中运行而无需互联网连接，非常适合开发响应式 Web 应用程序。

有关部署场景和更深入的信息，请参阅我们关于[使用 TensorFlow.js 的部署选项](#使用-tensorflowjs-的部署选项)的部分。

### 我可以使用 TensorFlow.js 在服务器端 Node.js 应用程序上部署 YOLO11 模型吗？

是的，TensorFlow.js 允许在 Node.js 环境中部署 YOLO11 模型。这使得受益于服务器处理能力和访问服务器端数据的服务器端机器学习应用程序成为可能。典型用例包括后端服务器上的实时数据处理和机器学习流水线。

要开始 Node.js 部署，请参阅 TensorFlow 的[在 Node.js 中运行 TensorFlow.js](https://www.tensorflow.org/js/guide/nodejs) 指南。
