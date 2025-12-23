---
comments: true
description: 学习如何将 YOLO11 模型导出为 PaddlePaddle 格式，以增强性能、灵活性和跨各种平台和设备的部署。
keywords: YOLO11, PaddlePaddle, 导出模型, 计算机视觉, 深度学习, 模型部署, 性能优化
---

# 如何从 YOLO11 模型导出为 PaddlePaddle 格式

在不同条件的实际场景中，弥合开发和部署[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型之间的差距可能很困难。PaddlePaddle 通过其对灵活性、性能和分布式环境中并行处理能力的关注，使这个过程变得更加简单。这意味着你可以在各种设备和平台上使用你的 YOLO11 计算机视觉模型，从智能手机到基于云的服务器。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/c5eFrt2KuzY"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何将 Ultralytics YOLO11 模型导出为 PaddlePaddle 格式 | PaddlePaddle 格式的关键功能
</p>

导出为 PaddlePaddle 模型格式的能力允许你优化 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型，以便在 PaddlePaddle 框架内使用。PaddlePaddle 以促进工业部署而闻名，是在各个领域的实际环境中部署计算机视觉应用的良好选择。

## 为什么应该导出到 PaddlePaddle？

<p align="center">
  <img width="75%" src="https://github.com/PaddlePaddle/Paddle/blob/develop/doc/imgs/logo.png" alt="PaddlePaddle Logo">
</p>

由百度开发的 [PaddlePaddle](https://www.paddlepaddle.org.cn/en)（**PA**rallel **D**istributed **D**eep **LE**arning）是中国第一个开源[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)平台。与一些主要为研究构建的框架不同，PaddlePaddle 优先考虑易用性和跨行业的平滑集成。

它提供与 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 和 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 等流行框架类似的工具和资源，使各种经验水平的开发人员都能使用。从农业和工厂到服务业，PaddlePaddle 拥有超过 477 万的庞大开发者社区，正在帮助创建和部署 AI 应用。

通过将 Ultralytics YOLO11 模型导出为 PaddlePaddle 格式，你可以利用 PaddlePaddle 在性能优化方面的优势。PaddlePaddle 优先考虑高效的模型执行和减少内存使用。因此，你的 YOLO11 模型可能实现更好的性能，在实际场景中提供一流的结果。

## PaddlePaddle 模型的关键功能

PaddlePaddle 模型提供了一系列关键功能，有助于其在各种部署场景中的灵活性、性能和可扩展性：

- **动态到静态图**：PaddlePaddle 支持[动态到静态编译](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/jit/index_en.html)，其中模型可以转换为静态计算图。这使得优化能够减少运行时开销并提高推理性能。

- **算子融合**：PaddlePaddle 像 [TensorRT](../integrations/tensorrt.md) 一样，使用[算子融合](https://developer.nvidia.com/gtc/2020/video/s21436-vid)来简化计算并减少开销。该框架通过合并兼容的操作来最小化内存传输和计算步骤，从而实现更快的推理。

- **量化**：PaddlePaddle 支持[量化技术](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/quantization/PTQ_en.html)，包括训练后量化和量化感知训练。这些技术允许使用较低精度的数据表示，有效提高性能并减小模型大小。

## PaddlePaddle 中的部署选项

在深入了解将 YOLO11 模型导出为 PaddlePaddle 的代码之前，让我们看看 PaddlePaddle 模型擅长的不同部署场景。

PaddlePaddle 提供了一系列选项，每个选项在易用性、灵活性和性能之间提供不同的平衡：

- **Paddle Serving**：此框架简化了将 PaddlePaddle 模型部署为高性能 RESTful API。Paddle Serving 非常适合生产环境，提供模型版本控制、在线 A/B 测试和处理大量请求的可扩展性等功能。

- **Paddle Inference API**：Paddle Inference API 让你对模型执行有低级控制。此选项非常适合需要将模型紧密集成到自定义应用中或针对特定硬件优化性能的场景。

- **Paddle Lite**：Paddle Lite 专为在资源有限的移动和嵌入式设备上部署而设计。它优化模型以在 ARM CPU、GPU 和其他专用硬件上实现更小的尺寸和更快的推理。

- **Paddle.js**：Paddle.js 使你能够直接在 Web 浏览器中部署 PaddlePaddle 模型。Paddle.js 可以加载预训练模型或使用 Paddle.js 提供的模型转换工具从 [paddle-hub](https://github.com/PaddlePaddle/PaddleHub) 转换模型。它可以在支持 WebGL/WebGPU/WebAssembly 的浏览器中运行。

## 导出到 PaddlePaddle：转换你的 YOLO11 模型

将 YOLO11 模型转换为 PaddlePaddle 格式可以提高执行灵活性并优化各种部署场景的性能。

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

所有 [Ultralytics YOLO11 模型](../models/yolo11.md)都支持导出，你可以[浏览导出格式和选项的完整列表](../modes/export.md)，找到最适合你部署需求的设置。

!!! example "用法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 PaddlePaddle 格式
        model.export(format="paddle")  # 创建 '/yolo11n_paddle_model'

        # 加载导出的 PaddlePaddle 模型
        paddle_model = YOLO("./yolo11n_paddle_model")

        # 运行推理
        results = paddle_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 PaddlePaddle 格式
        yolo export model=yolo11n.pt format=paddle # 创建 '/yolo11n_paddle_model'

        # 使用导出的模型运行推理
        yolo predict model='./yolo11n_paddle_model' source='https://ultralytics.com/images/bus.jpg'
        ```

### 导出参数

| 参数 | 类型 | 默认值 | 描述 |
| -------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str` | `'paddle'` | 导出模型的目标格式，定义与各种部署环境的兼容性。 |
| `imgsz` | `int` 或 `tuple` | `640` | 模型输入所需的图像大小。可以是整数（用于正方形图像）或元组 `(height, width)`（用于特定尺寸）。 |
| `batch` | `int` | `1` | 指定导出模型批量推理大小或导出模型在 `predict` 模式下将并发处理的最大图像数量。 |
| `device` | `str` | `None` | 指定导出设备：CPU (`device=cpu`)、Apple silicon 的 MPS (`device=mps`)。 |

有关导出过程的更多详细信息，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 PaddlePaddle 模型

成功将 Ultralytics YOLO11 模型导出为 PaddlePaddle 格式后，你现在可以部署它们。运行 PaddlePaddle 模型的主要和推荐的第一步是使用 YOLO("yolo11n_paddle_model/") 方法，如前面的用法代码片段所述。

但是，有关在各种其他设置中部署 PaddlePaddle 模型的深入说明，请查看以下资源：

- **[Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/v0.9.0/README_CN.md)**：学习如何使用 Paddle Serving 将 PaddlePaddle 模型部署为高性能服务。

- **[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/README_en.md)**：探索如何使用 Paddle Lite 在移动和嵌入式设备上优化和部署模型。

- **[Paddle.js](https://github.com/PaddlePaddle/Paddle.js)**：了解如何使用 Paddle.js 在 Web 浏览器中运行 PaddlePaddle 模型以实现客户端 AI。

## 总结

在本指南中，我们探索了将 Ultralytics YOLO11 模型导出为 PaddlePaddle 格式的过程。通过遵循这些步骤，你可以在各种部署场景中利用 PaddlePaddle 的优势，针对不同的硬件和软件环境优化你的模型。

有关用法的更多详细信息，请访问 [PaddlePaddle 官方文档](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)。

想探索更多集成 Ultralytics YOLO11 模型的方式？我们的[集成指南页面](index.md)探索了各种选项，为你提供有价值的资源和见解。

## 常见问题

### 如何将 Ultralytics YOLO11 模型导出为 PaddlePaddle 格式？

将 Ultralytics YOLO11 模型导出为 PaddlePaddle 格式非常简单。你可以使用 YOLO 类的 `export` 方法执行转换。以下是使用 Python 的示例：

```python
from ultralytics import YOLO

# 加载 YOLO11 模型
model = YOLO("yolo11n.pt")

# 将模型导出为 PaddlePaddle 格式
model.export(format="paddle")  # 创建 '/yolo11n_paddle_model'
```

### 使用 PaddlePaddle 进行模型部署有什么优势？

PaddlePaddle 为模型部署提供了几个关键优势：

- **性能优化**：PaddlePaddle 在高效模型执行和减少内存使用方面表现出色。
- **动态到静态图编译**：它支持动态到静态编译，允许运行时优化。
- **算子融合**：通过合并兼容的操作，减少计算开销。
- **量化技术**：支持训练后量化和量化感知训练，实现较低精度的数据表示以提高性能。

### 为什么应该选择 PaddlePaddle 来部署我的 YOLO11 模型？

由百度开发的 PaddlePaddle 针对工业和商业 AI 部署进行了优化。其庞大的开发者社区和强大的框架提供了与 TensorFlow 和 PyTorch 类似的广泛工具。通过将 YOLO11 模型导出到 PaddlePaddle，你可以利用：

- **增强的性能**：最佳执行速度和减少的内存占用。
- **灵活性**：与从智能手机到云服务器的各种设备广泛兼容。
- **可扩展性**：分布式环境中高效的并行处理能力。

### PaddlePaddle 为 YOLO11 模型提供哪些部署选项？

PaddlePaddle 提供灵活的部署选项：

- **Paddle Serving**：将模型部署为 RESTful API，非常适合生产环境，具有模型版本控制和在线 A/B 测试等功能。
- **Paddle Inference API**：为自定义应用提供对模型执行的低级控制。
- **Paddle Lite**：针对移动和嵌入式设备的有限资源进行优化。
- **Paddle.js**：允许直接在 Web 浏览器中部署模型。

这些选项涵盖了从设备端推理到可扩展云服务的广泛部署场景。
