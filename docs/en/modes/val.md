---
comments: true
description: 了解如何使用精确指标、易用工具和自定义设置验证 YOLO11 模型以获得最佳性能。
keywords: Ultralytics, YOLO11, 模型验证, 机器学习, 目标检测, mAP 指标, Python API, CLI
---

# 使用 Ultralytics YOLO 进行模型验证

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统和集成">

## 简介

验证是[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)流程中的关键步骤，允许您评估训练模型的质量。Ultralytics YOLO11 中的验证模式提供了一套强大的工具和指标，用于评估[目标检测](https://www.ultralytics.com/glossary/object-detection)模型的性能。本指南是了解如何有效使用验证模式以确保模型既准确又可靠的完整资源。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=47"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics 模式教程：验证
</p>

## 为什么使用 Ultralytics YOLO 进行验证？

以下是使用 YOLO11 验证模式的优势：

- **精度：** 获取 mAP50、mAP75 和 mAP50-95 等准确指标，全面评估您的模型。
- **便利性：** 利用记住训练设置的内置功能，简化验证过程。
- **灵活性：** 使用相同或不同的数据集和图像大小验证您的模型。
- **[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)：** 使用验证指标微调模型以获得更好的性能。

### 验证模式的关键特性

以下是 YOLO11 验证模式提供的显著功能：

- **自动设置：** 模型记住其训练配置，便于直接验证。
- **多指标支持：** 基于一系列精度指标评估您的模型。
- **CLI 和 Python API：** 根据您的验证偏好选择命令行界面或 Python API。
- **数据兼容性：** 与训练阶段使用的数据集以及自定义数据集无缝配合。

!!! tip

    * YOLO11 模型自动记住其训练设置，因此您可以轻松地以相同的图像大小和原始数据集验证模型，只需 `yolo val model=yolo11n.pt` 或 `YOLO("yolo11n.pt").val()`

## 使用示例

在 COCO8 数据集上验证训练好的 YOLO11n 模型[精度](https://www.ultralytics.com/glossary/accuracy)。不需要参数，因为 `model` 保留其训练 `data` 和参数作为模型属性。有关验证参数的完整列表，请参阅下面的参数部分。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 不需要参数，数据集和设置已记住
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # 包含每个类别 mAP50-95 的列表
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolo11n.pt      # 验证官方模型
        yolo detect val model=path/to/best.pt # 验证自定义模型
        ```

## YOLO 模型验证的参数

验证 YOLO 模型时，可以微调多个参数以优化评估过程。这些参数控制输入图像大小、批处理和性能阈值等方面。以下是每个参数的详细说明，帮助您有效自定义验证设置。

{% include "macros/validation-args.md" %}

## 常见问题

### 如何使用 Ultralytics 验证我的 YOLO11 模型？

要验证您的 YOLO11 模型，您可以使用 Ultralytics 提供的验证模式。例如，使用 Python API，您可以加载模型并运行验证：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 验证模型
metrics = model.val()
print(metrics.box.map)  # map50-95
```

或者，您可以使用命令行界面 (CLI)：

```bash
yolo val model=yolo11n.pt
```

有关进一步自定义，您可以在 Python 和 CLI 模式下调整各种参数，如 `imgsz`、`batch` 和 `conf`。查看 [YOLO 模型验证的参数](#yolo-模型验证的参数)部分获取完整的参数列表。

### 我可以从 YOLO11 模型验证中获得哪些指标？

YOLO11 模型验证提供了几个关键指标来评估模型性能。这些包括：

- mAP50（IoU 阈值 0.5 时的平均精度均值）
- mAP75（IoU 阈值 0.75 时的平均精度均值）
- mAP50-95（从 0.5 到 0.95 的多个 IoU 阈值的平均精度均值）

使用 Python API，您可以按如下方式访问这些指标：

```python
metrics = model.val()  # 假设 `model` 已加载
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps)  # 每个类别的 mAP50-95 列表
```

### 我可以使用自定义数据集验证我的 YOLO11 模型吗？

是的，您可以使用[自定义数据集](https://docs.ultralytics.com/datasets/)验证您的 YOLO11 模型。使用数据集配置文件的路径指定 `data` 参数。此文件应包含[验证数据](https://www.ultralytics.com/glossary/validation-data)的路径。

Python 示例：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 使用自定义数据集验证
metrics = model.val(data="path/to/your/custom_dataset.yaml")
print(metrics.box.map)  # map50-95
```

使用 CLI 的示例：

```bash
yolo val model=yolo11n.pt data=path/to/your/custom_dataset.yaml
```
