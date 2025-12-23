---
comments: true
description: 学习如何将 Ultralytics YOLO11 模型导出为 TorchScript 以实现灵活的跨平台部署。提升性能并在各种环境中使用。
keywords: YOLO11, TorchScript, 模型导出, Ultralytics, PyTorch, 深度学习, AI 部署, 跨平台, 性能优化
---

# YOLO11 模型导出到 TorchScript 以实现快速部署

在不同环境中部署[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型，包括嵌入式系统、Web 浏览器或 Python 支持有限的平台，需要一个灵活且可移植的解决方案。TorchScript 专注于可移植性和在整个 Python 框架不可用的环境中运行模型的能力。这使其成为需要在各种设备或平台上部署计算机视觉功能的场景的理想选择。

导出到 TorchScript 以序列化您的 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型，实现跨平台兼容性和简化部署。在本指南中，我们将向您展示如何将 YOLO11 模型导出为 TorchScript 格式，使您更容易在更广泛的应用程序中使用它们。

## 为什么应该导出到 TorchScript？

![TorchScript 概览](https://github.com/ultralytics/docs/releases/download/0/torchscript-overview.avif)

TorchScript 由 PyTorch 的创建者开发，是一个强大的工具，用于在各种平台上优化和部署 PyTorch 模型。将 YOLO11 模型导出到 [TorchScript](https://docs.pytorch.org/docs/stable/jit.html) 对于从研究转向实际应用至关重要。TorchScript 是 PyTorch 框架的一部分，通过允许 PyTorch 模型在不支持 Python 的环境中使用，帮助使这种转换更加顺畅。

该过程涉及两种技术：追踪和脚本化。追踪在模型执行期间记录操作，而脚本化允许使用 Python 的子集定义模型。这些技术确保像 YOLO11 这样的模型即使在其通常的 Python 环境之外也能发挥作用。

![TorchScript 脚本和追踪](https://github.com/ultralytics/docs/releases/download/0/torchscript-script-and-trace.avif)

TorchScript 模型还可以通过算子融合和内存使用优化等技术进行优化，确保高效执行。导出到 TorchScript 的另一个优势是它有可能在各种硬件平台上加速模型执行。它创建了 PyTorch 模型的独立、生产就绪表示，可以集成到 C++ 环境、嵌入式系统中，或部署在 Web 或移动应用程序中。

## TorchScript 模型的主要特性

TorchScript 是 PyTorch 生态系统的关键部分，为优化和部署[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型提供强大功能。

![TorchScript 特性](https://github.com/ultralytics/docs/releases/download/0/torchscript-features.avif)

以下是使 TorchScript 成为开发者宝贵工具的主要特性：

- **静态图执行**：TorchScript 使用模型计算的静态图表示，这与 PyTorch 的动态图执行不同。在静态图执行中，计算图在实际执行之前定义和编译一次，从而在推理期间提高性能。

- **模型序列化**：TorchScript 允许您将 PyTorch 模型序列化为平台无关的格式。序列化的模型可以在不需要原始 Python 代码的情况下加载，从而能够在不同的运行时环境中部署。

- **JIT 编译**：TorchScript 使用即时 (JIT) 编译将 PyTorch 模型转换为优化的中间表示。JIT 编译模型的计算图，实现在目标设备上的高效执行。

- **跨语言集成**：使用 TorchScript，您可以将 PyTorch 模型导出到其他语言，如 C++、Java 和 JavaScript。这使得将 PyTorch 模型集成到用不同语言编写的现有软件系统中变得更加容易。

- **渐进式转换**：TorchScript 提供渐进式转换方法，允许您逐步将 PyTorch 模型的部分转换为 TorchScript。这种灵活性在处理复杂模型或想要优化代码的特定部分时特别有用。

## TorchScript 中的部署选项

在查看将 YOLO11 模型导出为 TorchScript 格式的代码之前，让我们了解 TorchScript 模型通常在哪里使用。

TorchScript 为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型提供各种部署选项，例如：

- **C++ API**：TorchScript 最常见的用例是其 C++ API，它允许您直接在 C++ 应用程序中加载和执行优化的 TorchScript 模型。这对于 Python 可能不适合或不可用的生产环境非常理想。C++ API 提供低开销和高效的 TorchScript 模型执行，最大化性能潜力。

- **移动部署**：TorchScript 提供将模型转换为可在移动设备上轻松部署的格式的工具。PyTorch Mobile 提供在 iOS 和 Android 应用程序中执行这些模型的运行时。这实现了低延迟、离线推理功能，增强用户体验和[数据隐私](https://www.ultralytics.com/glossary/data-privacy)。

- **云部署**：TorchScript 模型可以使用 TorchServe 等解决方案部署到基于云的服务器。它提供模型版本控制、批处理和指标监控等功能，用于生产环境中的可扩展部署。使用 TorchScript 进行云部署可以使您的模型通过 API 或其他 Web 服务访问。

## 导出到 TorchScript：转换您的 YOLO11 模型

将 YOLO11 模型导出到 TorchScript 使其更容易在不同地方使用，并帮助它们运行得更快、更高效。这对于希望在实际应用中更有效地使用深度学习模型的任何人来说都很棒。

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

        # 将模型导出为 TorchScript 格式
        model.export(format="torchscript")  # 创建 'yolo11n.torchscript'

        # 加载导出的 TorchScript 模型
        torchscript_model = YOLO("yolo11n.torchscript")

        # 运行推理
        results = torchscript_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TorchScript 格式
        yolo export model=yolo11n.pt format=torchscript # 创建 'yolo11n.torchscript'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n.torchscript source='https://ultralytics.com/images/bus.jpg'
        ```

### 导出参数

| 参数       | 类型             | 默认值          | 描述                                                                                                                             |
| ---------- | ---------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'torchscript'` | 导出模型的目标格式，定义与各种部署环境的兼容性。                                      |
| `imgsz`    | `int` 或 `tuple` | `640`           | 模型输入所需的图像尺寸。可以是整数（正方形图像）或元组 `(height, width)` 指定特定尺寸。       |
| `dynamic`  | `bool`           | `False`         | 允许动态输入尺寸，增强处理不同图像尺寸的灵活性。                                                 |
| `optimize` | `bool`           | `False`         | 应用移动设备优化，可能减少模型大小并提高性能。                                     |
| `nms`      | `bool`           | `False`         | 添加非极大值抑制 (NMS)，对于准确高效的检测后处理至关重要。                                     |
| `batch`    | `int`            | `1`             | 指定导出模型的批量推理大小或导出模型在 `predict` 模式下并发处理的最大图像数量。 |
| `device`   | `str`            | `None`          | 指定导出设备：GPU (`device=0`)，CPU (`device=cpu`)，Apple silicon 的 MPS (`device=mps`)。                         |

有关导出过程的更多详情，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 TorchScript 模型

成功将 Ultralytics YOLO11 模型导出为 TorchScript 格式后，您现在可以部署它们。运行 TorchScript 模型的主要和推荐的第一步是使用 `YOLO("model.torchscript")` 方法，如之前的使用代码片段所述。有关在其他设置中部署 TorchScript 模型的深入说明，请查看以下资源：

- **[探索移动部署](https://docs.pytorch.org/executorch/)**：[PyTorch](https://www.ultralytics.com/glossary/pytorch) Mobile 文档提供了在移动设备上部署模型的全面指南，确保您的应用程序高效且响应迅速。

- **[掌握服务器端部署](https://docs.pytorch.org/serve/getting_started.html)**：学习如何使用 TorchServe 在服务器端部署模型，提供可扩展、高效模型服务的分步教程。

- **[实现 C++ 部署](https://docs.pytorch.org/tutorials/advanced/cpp_export.html)**：深入了解在 C++ 中加载 TorchScript 模型的教程，便于将 TorchScript 模型集成到 C++ 应用程序中以获得增强的性能和多功能性。

## 总结

在本指南中，我们探讨了将 Ultralytics YOLO11 模型导出为 TorchScript 格式的过程。按照提供的说明，您可以优化 YOLO11 模型的性能，并获得在各种平台和环境中部署它们的灵活性。

有关使用的更多详情，请访问 [TorchScript 官方文档](https://docs.pytorch.org/docs/stable/jit.html)。

此外，如果您想了解更多关于其他 Ultralytics YOLO11 集成的信息，请访问我们的[集成指南页面](../integrations/index.md)。您将在那里找到大量有用的资源和见解。

## 常见问题

### 什么是 Ultralytics YOLO11 模型导出到 TorchScript？

将 Ultralytics YOLO11 模型导出到 TorchScript 允许灵活的跨平台部署。TorchScript 是 PyTorch 生态系统的一部分，通过允许 PyTorch 模型在不支持 Python 的环境中使用，帮助使从研究到实际应用的转换更加顺畅。TorchScript 是 PyTorch 框架的一部分，通过允许 PyTorch 模型在不支持 Python 的环境中使用，帮助使这种转换更加顺畅。导出到 TorchScript 可实现高效性能和更广泛的 YOLO11 模型在不同平台上的适用性。

### 如何使用 Ultralytics 将 YOLO11 模型导出到 TorchScript？

要将 YOLO11 模型导出到 TorchScript，您可以使用以下示例代码：

!!! example "使用方法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 TorchScript 格式
        model.export(format="torchscript")  # 创建 'yolo11n.torchscript'

        # 加载导出的 TorchScript 模型
        torchscript_model = YOLO("yolo11n.torchscript")

        # 运行推理
        results = torchscript_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TorchScript 格式
        yolo export model=yolo11n.pt format=torchscript # 创建 'yolo11n.torchscript'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n.torchscript source='https://ultralytics.com/images/bus.jpg'
        ```

有关导出过程的更多详情，请参阅 [Ultralytics 导出文档](../modes/export.md)。

### 为什么应该使用 TorchScript 部署 YOLO11 模型？

使用 TorchScript 部署 YOLO11 模型提供了多种优势：

- **可移植性**：导出的模型可以在不需要 Python 的环境中运行，如 C++ 应用程序、嵌入式系统或移动设备。
- **优化**：TorchScript 支持静态图执行和即时 (JIT) 编译，可以优化模型性能。
- **跨语言集成**：TorchScript 模型可以集成到其他编程语言中，增强灵活性和可扩展性。
- **序列化**：模型可以被序列化，允许平台无关的加载和推理。

有关部署的更多见解，请访问 [PyTorch Mobile 文档](https://docs.pytorch.org/executorch/)、[TorchServe 文档](https://docs.pytorch.org/serve/getting_started.html)和 [C++ 部署指南](https://docs.pytorch.org/tutorials/advanced/cpp_export.html)。

### 将 YOLO11 模型导出到 TorchScript 的安装步骤是什么？

要安装导出 YOLO11 模型所需的包，请使用以下命令：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 所需的包
        pip install ultralytics
        ```

有关详细说明，请访问 [Ultralytics 安装指南](../quickstart.md)。如果在安装过程中出现任何问题，请参阅[常见问题指南](../guides/yolo-common-issues.md)。

### 如何部署导出的 TorchScript YOLO11 模型？

将 YOLO11 模型导出为 TorchScript 格式后，您可以在各种平台上部署它们：

- **C++ API**：非常适合低开销、高效的生产环境。
- **移动部署**：使用 [PyTorch Mobile](https://docs.pytorch.org/executorch/) 用于 iOS 和 Android 应用程序。
- **云部署**：利用 [TorchServe](https://docs.pytorch.org/serve/getting_started.html) 等服务进行可扩展的服务器端部署。

探索这些设置中部署模型的全面指南，以充分利用 TorchScript 的功能。
