---
comments: true
description: 了解如何使用基准测试模式评估 YOLO11 模型在实际场景中的性能。优化速度、精度和跨导出格式的资源分配。
keywords: 模型基准测试, YOLO11, Ultralytics, 性能评估, 导出格式, ONNX, TensorRT, OpenVINO, CoreML, TensorFlow, 优化, mAP50-95, 推理时间
---

# 使用 Ultralytics YOLO 进行模型基准测试

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统和集成">

## 基准测试可视化

!!! tip "刷新浏览器"

    由于潜在的 cookie 问题，您可能需要刷新页面才能正确查看图表。

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400"></canvas>

## 简介

一旦您的模型经过训练和验证，下一个合理的步骤是评估其在各种实际场景中的性能。Ultralytics YOLO11 中的基准测试模式通过提供一个强大的框架来评估模型在各种导出格式中的速度和[精度](https://www.ultralytics.com/glossary/accuracy)，从而实现这一目的。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rEQlAaevEFc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics YOLO11 模型基准测试 | 如何在不同硬件上比较模型性能？
</p>

## 为什么基准测试至关重要？

- **明智决策：** 深入了解速度和精度之间的权衡。
- **资源分配：** 了解不同导出格式在不同硬件上的表现。
- **优化：** 了解哪种导出格式为您的特定用例提供最佳性能。
- **成本效益：** 根据基准测试结果更有效地利用硬件资源。

### 基准测试模式的关键指标

- **mAP50-95：** 用于[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和姿态估计。
- **accuracy_top5：** 用于[图像分类](https://www.ultralytics.com/glossary/image-classification)。
- **推理时间：** 每张图像的处理时间（毫秒）。

### 支持的导出格式

- **ONNX：** 最佳 CPU 性能
- **TensorRT：** 最大 GPU 效率
- **OpenVINO：** Intel 硬件优化
- **CoreML、TensorFlow SavedModel 等：** 满足多样化部署需求。

!!! tip

    * 导出到 ONNX 或 OpenVINO 可获得高达 3 倍的 CPU 加速。
    * 导出到 TensorRT 可获得高达 5 倍的 GPU 加速。

## 使用示例

在所有支持的导出格式（ONNX、TensorRT 等）上运行 YOLO11n 基准测试。有关导出选项的完整列表，请参阅下面的参数部分。

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在 GPU 上进行基准测试
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)

        # 对特定导出格式进行基准测试
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format="onnx")
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 half=False device=0

        # 对特定导出格式进行基准测试
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 format=onnx
        ```

## 参数

`model`、`data`、`imgsz`、`half`、`device`、`verbose` 和 `format` 等参数为用户提供了灵活性，可以根据特定需求微调基准测试，并轻松比较不同导出格式的性能。

| 键        | 默认值 | 描述                                                                                                                                                                                                    |
| --------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | `None` | 指定模型文件的路径。接受 `.pt` 和 `.yaml` 格式，例如 `"yolo11n.pt"` 用于预训练模型或配置文件。                                                                                                          |
| `data`    | `None` | 定义用于基准测试的数据集的 YAML 文件路径，通常包括[验证数据](https://www.ultralytics.com/glossary/validation-data)的路径和设置。示例：`"coco8.yaml"`。                                                  |
| `imgsz`   | `640`  | 模型的输入图像大小。可以是单个整数（用于正方形图像）或元组 `(width, height)`（用于非正方形），例如 `(640, 480)`。                                                                                        |
| `half`    | `False`| 启用 FP16（半精度）推理，减少内存使用并可能在兼容硬件上提高速度。使用 `half=True` 启用。                                                                                                                |
| `int8`    | `False`| 激活 INT8 量化以在支持的设备上进一步优化性能，对边缘设备特别有用。设置 `int8=True` 使用。                                                                                                               |
| `device`  | `None` | 定义用于基准测试的计算设备，如 `"cpu"` 或 `"cuda:0"`。                                                                                                                                                  |
| `verbose` | `False`| 控制日志输出的详细程度。设置 `verbose=True` 获取详细日志。                                                                                                                                              |
| `format`  | `''`   | 仅对指定的导出格式进行基准测试（例如 `format=onnx`）。留空则自动测试所有支持的格式。                                                                                                                    |

## 导出格式

基准测试将尝试自动在下面列出的所有可能的导出格式上运行。或者，您可以使用 `format` 参数对特定格式运行基准测试，该参数接受下面提到的任何格式。

{% include "macros/export-table.md" %}

有关完整的 `export` 详细信息，请参阅[导出](../modes/export.md)页面。

## 常见问题

### 如何使用 Ultralytics 对我的 YOLO11 模型性能进行基准测试？

Ultralytics YOLO11 提供基准测试模式来评估模型在不同导出格式中的性能。此模式提供关于[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP50-95)、精度和推理时间（毫秒）等关键指标的见解。要运行基准测试，您可以使用 Python 或 CLI 命令。例如，在 GPU 上进行基准测试：

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在 GPU 上进行基准测试
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

有关基准测试参数的更多详细信息，请访问[参数](#参数)部分。

### 将 YOLO11 模型导出到不同格式有什么好处？

将 YOLO11 模型导出到不同格式（如 [ONNX](https://docs.ultralytics.com/integrations/onnx/)、[TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) 和 [OpenVINO](https://docs.ultralytics.com/integrations/openvino/)）允许您根据部署环境优化性能。例如：

- **ONNX：** 提供高达 3 倍的 CPU 加速。
- **TensorRT：** 提供高达 5 倍的 GPU 加速。
- **OpenVINO：** 专门针对 Intel 硬件优化。

这些格式增强了模型的速度和精度，使其更适合各种实际应用。访问[导出](../modes/export.md)页面获取完整详细信息。

### 为什么基准测试在评估 YOLO11 模型时至关重要？

对 YOLO11 模型进行基准测试至关重要，原因如下：

- **明智决策：** 了解速度和精度之间的权衡。
- **资源分配：** 评估不同硬件选项上的性能。
- **优化：** 确定哪种导出格式为特定用例提供最佳性能。
- **成本效益：** 根据基准测试结果优化硬件使用。

mAP50-95、Top-5 精度和推理时间等关键指标有助于进行这些评估。有关更多信息，请参阅[关键指标](#基准测试模式的关键指标)部分。

### YOLO11 支持哪些导出格式，它们的优势是什么？

YOLO11 支持多种导出格式，每种格式都针对特定硬件和用例量身定制：

- **ONNX：** 最适合 CPU 性能。
- **TensorRT：** 理想的 GPU 效率。
- **OpenVINO：** 针对 Intel 硬件优化。
- **CoreML 和 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)：** 适用于 iOS 和通用 ML 应用。

有关支持格式及其各自优势的完整列表，请查看[支持的导出格式](#支持的导出格式)部分。

### 我可以使用哪些参数来微调我的 YOLO11 基准测试？

运行基准测试时，可以自定义多个参数以满足特定需求：

- **model：** 模型文件的路径（例如 "yolo11n.pt"）。
- **data：** 定义数据集的 YAML 文件路径（例如 "coco8.yaml"）。
- **imgsz：** 输入图像大小，可以是单个整数或元组。
- **half：** 启用 FP16 推理以获得更好的性能。
- **int8：** 为边缘设备激活 INT8 量化。
- **device：** 指定计算设备（例如 "cpu"、"cuda:0"）。
- **verbose：** 控制日志详细程度。

有关参数的完整列表，请参阅[参数](#参数)部分。
