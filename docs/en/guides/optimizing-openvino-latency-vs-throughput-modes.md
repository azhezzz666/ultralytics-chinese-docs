---
comments: true
description: 了解如何使用 Intel 的 OpenVINO 工具包增强 Ultralytics YOLO 模型性能。高效提升延迟和吞吐量。
keywords: Ultralytics YOLO, OpenVINO 优化, 深度学习, 模型推理, 吞吐量优化, 延迟优化, AI 部署, Intel OpenVINO, 性能调优
---

# YOLO 的 OpenVINO 推理优化

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ecosystem.avif" alt="OpenVINO 生态系统">

## 简介

在部署[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型时，特别是用于[目标检测](https://www.ultralytics.com/glossary/object-detection)的 Ultralytics YOLO 模型，实现最佳性能至关重要。本指南深入探讨如何利用 [Intel 的 OpenVINO 工具包](https://docs.ultralytics.com/integrations/openvino/)优化推理，重点关注延迟和吞吐量。无论您是在消费级应用还是大规模部署中工作，理解和应用这些优化策略将确保您的模型在各种设备上高效运行。

## 延迟优化

延迟优化对于需要单个模型对单个输入立即响应的应用至关重要，这在消费场景中很典型。目标是最小化输入和推理结果之间的延迟。然而，实现低延迟需要仔细考虑，特别是在运行并发推理或管理多个模型时。

### 延迟优化的关键策略：

- **每设备单次推理：** 实现低延迟的最简单方法是限制每个设备一次只进行一次推理。额外的并发通常会导致延迟增加。
- **利用子设备：** 多插槽 CPU 或多瓦片 GPU 等设备可以通过利用其内部子设备执行多个请求，同时将延迟增加降至最低。
- **OpenVINO 性能提示：** 在模型编译期间使用 OpenVINO 的 `ov::hint::PerformanceMode::LATENCY` 作为 `ov::hint::performance_mode` 属性可简化性能调优，提供与设备无关且面向未来的方法。

### 管理首次推理延迟：

- **模型缓存：** 为了减轻模型加载和编译时间对延迟的影响，尽可能使用模型缓存。对于缓存不可行的场景，CPU 通常提供最快的模型加载时间。
- **模型映射与读取：** 为了减少加载时间，OpenVINO 用映射替换了模型读取。但是，如果模型位于可移动或网络驱动器上，请考虑使用 `ov::enable_mmap(false)` 切换回读取模式。
- **AUTO 设备选择：** 此模式在 CPU 上开始推理，一旦加速器准备就绪就切换到加速器，无缝减少首次推理延迟。

## 吞吐量优化

吞吐量优化对于同时处理大量推理请求的场景至关重要，在不显著牺牲单个请求性能的情况下最大化[资源利用率](https://www.ultralytics.com/blog/measuring-ai-performance-to-weigh-the-impact-of-your-innovations)。

### 吞吐量优化方法：

1. **OpenVINO 性能提示：** 一种高级、面向未来的方法，使用性能提示跨设备增强吞吐量。

    ```python
    import openvino.properties.hint as hints

    config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
    compiled_model = core.compile_model(model, "GPU", config)
    ```

2. **显式批处理和流：** 一种更细粒度的方法，涉及显式批处理和使用流进行高级性能调优。

### 设计面向吞吐量的应用：

为了最大化吞吐量，应用应该：

- 并行处理输入，充分利用设备的能力。
- 将数据流分解为并发推理请求，安排并行执行。
- 使用带回调的异步 API 来保持效率并避免设备饥饿。

### 多设备执行：

OpenVINO 的多设备模式通过自动在设备之间平衡推理请求来简化吞吐量扩展，无需应用级设备管理。

## 实际性能提升

在 Ultralytics YOLO 模型上实施 OpenVINO 优化可以带来显著的性能改进。如[基准测试](https://docs.ultralytics.com/integrations/openvino/#openvino-yolov8-benchmarks)所示，用户可以在 Intel CPU 上体验高达 3 倍的推理速度提升，在 Intel 的硬件系列（包括集成 GPU、独立 GPU 和 VPU）上可能实现更大的加速。

例如，在 Intel Xeon CPU 上运行 YOLOv8 模型时，OpenVINO 优化版本在每张图像的推理时间方面始终优于其 PyTorch 对应版本，同时不影响[准确性](https://www.ultralytics.com/glossary/accuracy)。

## 实际实现

要导出并优化您的 Ultralytics YOLO 模型以用于 OpenVINO，您可以使用[导出](https://docs.ultralytics.com/modes/export/)功能：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")

# 将模型导出为 OpenVINO 格式
model.export(format="openvino", half=True)  # 使用 FP16 精度导出
```

导出后，您可以使用优化后的模型运行推理：

```python
# 加载 OpenVINO 模型
ov_model = YOLO("yolov8n_openvino_model/")

# 使用延迟性能提示运行推理
results = ov_model("path/to/image.jpg", verbose=True)
```

## 结论

使用 OpenVINO 优化 Ultralytics YOLO 模型的延迟和吞吐量可以显著增强应用程序的性能。通过仔细应用本指南中概述的策略，开发人员可以确保其模型高效运行，满足各种部署场景的需求。请记住，选择优化延迟还是吞吐量取决于您的特定应用需求和部署环境的特性。

有关更详细的技术信息和最新更新，请参阅 [OpenVINO 文档](https://docs.openvino.ai/2024/index.html)和 [Ultralytics YOLO 仓库](https://github.com/ultralytics/ultralytics)。这些资源提供深入的指南、教程和社区支持，帮助您充分利用深度学习模型。

---

确保模型实现最佳性能不仅仅是调整配置；而是要了解应用程序的需求并做出明智的决策。无论您是优化[实时响应](https://www.ultralytics.com/blog/real-time-inferences-in-vision-ai-solutions-are-making-an-impact)还是最大化大规模处理的吞吐量，Ultralytics YOLO 模型和 OpenVINO 的组合为开发人员部署高性能 AI 解决方案提供了强大的工具包。

## 常见问题

### 如何使用 OpenVINO 优化 Ultralytics YOLO 模型以实现低延迟？

优化 Ultralytics YOLO 模型以实现低延迟涉及几个关键策略：

1. **每设备单次推理：** 限制每个设备一次只进行一次推理以最小化延迟。
2. **利用子设备：** 利用多插槽 CPU 或多瓦片 GPU 等设备，它们可以处理多个请求而延迟增加最小。
3. **OpenVINO 性能提示：** 在模型编译期间使用 OpenVINO 的 `ov::hint::PerformanceMode::LATENCY` 进行简化的、与设备无关的调优。

有关优化延迟的更多实用技巧，请查看我们指南的[延迟优化部分](#延迟优化)。

### 为什么应该使用 OpenVINO 优化 Ultralytics YOLO 吞吐量？

OpenVINO 通过最大化设备资源利用率而不牺牲性能来增强 Ultralytics YOLO 模型吞吐量。主要优势包括：

- **性能提示：** 跨设备的简单、高级性能调优。
- **显式批处理和流：** 用于高级性能的微调。
- **多设备执行：** 自动推理负载平衡，简化应用级管理。

示例配置：

```python
import openvino.properties.hint as hints

config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
compiled_model = core.compile_model(model, "GPU", config)
```

在我们详细指南的[吞吐量优化部分](#吞吐量优化)中了解更多关于吞吐量优化的信息。

### 在 OpenVINO 中减少首次推理延迟的最佳实践是什么？

要减少首次推理延迟，请考虑以下实践：

1. **模型缓存：** 使用模型缓存来减少加载和编译时间。
2. **模型映射与读取：** 默认使用映射（`ov::enable_mmap(true)`），但如果模型位于可移动或网络驱动器上，则切换到读取（`ov::enable_mmap(false)`）。
3. **AUTO 设备选择：** 利用 AUTO 模式从 CPU 推理开始，无缝过渡到加速器。

有关管理首次推理延迟的详细策略，请参阅[管理首次推理延迟部分](#管理首次推理延迟)。

### 如何使用 Ultralytics YOLO 和 OpenVINO 平衡延迟和吞吐量优化？

平衡延迟和吞吐量优化需要了解您的应用需求：

- **延迟优化：** 适用于需要立即响应的实时应用（例如消费级应用）。
- **吞吐量优化：** 最适合具有许多并发推理的场景，最大化资源使用（例如大规模部署）。

使用 OpenVINO 的高级性能提示和多设备模式可以帮助找到正确的平衡。根据您的具体需求选择适当的 [OpenVINO 性能提示](https://docs.ultralytics.com/integrations/openvino/#openvino-performance-hints)。

### 除了 OpenVINO 之外，我可以将 Ultralytics YOLO 模型与其他 AI 框架一起使用吗？

是的，Ultralytics YOLO 模型非常灵活，可以与各种 AI 框架集成。选项包括：

- **TensorRT：** 用于 NVIDIA GPU 优化，请遵循 [TensorRT 集成指南](https://docs.ultralytics.com/integrations/tensorrt/)。
- **CoreML：** 用于 Apple 设备，请参阅我们的 [CoreML 导出说明](https://docs.ultralytics.com/integrations/coreml/)。
- **[TensorFlow](https://www.ultralytics.com/glossary/tensorflow).js：** 用于 Web 和 Node.js 应用，请参阅 [TF.js 转换指南](https://docs.ultralytics.com/integrations/tfjs/)。

在 [Ultralytics 集成页面](https://docs.ultralytics.com/integrations/)上探索更多集成。
