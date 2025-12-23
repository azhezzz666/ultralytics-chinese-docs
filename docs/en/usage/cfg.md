---
comments: true
description: 使用正确的设置和超参数优化 Ultralytics YOLO 模型的性能。了解训练、验证和预测配置。
keywords: YOLO, 超参数, 配置, 训练, 验证, 预测, 模型设置, Ultralytics, 性能优化, 机器学习
---

# 配置

YOLO 设置和超参数在模型的性能、速度和[准确率](https://www.ultralytics.com/glossary/accuracy)中起着关键作用。这些设置可以影响模型在各个阶段的行为，包括训练、验证和预测。

**观看：**掌握 Ultralytics YOLO：配置

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=87"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>掌握 Ultralytics YOLO：配置
</p>

Ultralytics 命令使用以下语法：

!!! example "示例"

    === "CLI"

        ```bash
        yolo TASK MODE ARGS
        ```

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从预训练权重文件加载 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 使用自定义参数在指定模式下运行模型
        MODE = "predict"
        ARGS = {"source": "image.jpg", "imgsz": 640}
        getattr(model, MODE)(**ARGS)
        ```

其中：

- `TASK`（可选）是以下之一（[detect](../tasks/detect.md)、[segment](../tasks/segment.md)、[classify](../tasks/classify.md)、[pose](../tasks/pose.md)、[obb](../tasks/obb.md)）
- `MODE`（必需）是以下之一（[train](../modes/train.md)、[val](../modes/val.md)、[predict](../modes/predict.md)、[export](../modes/export.md)、[track](../modes/track.md)、[benchmark](../modes/benchmark.md)）
- `ARGS`（可选）是 `arg=value` 对，如 `imgsz=640`，用于覆盖默认值。

默认 `ARG` 值在本页定义，来自 `cfg/default.yaml` [文件](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)。

## 任务

Ultralytics YOLO 模型可以执行各种计算机视觉任务，包括：

- **检测**：[目标检测](https://docs.ultralytics.com/tasks/detect/)识别和定位图像或视频中的目标。
- **分割**：[实例分割](https://docs.ultralytics.com/tasks/segment/)将图像或视频划分为对应不同目标或类别的区域。
- **分类**：[图像分类](https://docs.ultralytics.com/tasks/classify/)预测输入图像的类别标签。
- **姿态**：[姿态估计](https://docs.ultralytics.com/tasks/pose/)识别目标并估计图像或视频中的关键点。
- **旋转边界框**：[旋转边界框](https://docs.ultralytics.com/tasks/obb/)使用旋转边界框，适用于卫星或医学图像。

| 参数 | 默认值 | 描述 |
| -------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`   | `'detect'` | 指定 YOLO 任务：`detect` 用于[目标检测](https://www.ultralytics.com/glossary/object-detection)，`segment` 用于分割，`classify` 用于分类，`pose` 用于姿态估计，`obb` 用于旋转边界框。每个任务针对图像和视频分析中的特定输出和问题进行定制。 |

[任务指南](../tasks/index.md){ .md-button }

## 模式

Ultralytics YOLO 模型在不同模式下运行，每种模式针对模型生命周期的特定阶段设计：

- **训练**：在自定义数据集上训练 YOLO 模型。
- **验证**：验证训练好的 YOLO 模型。
- **预测**：使用训练好的 YOLO 模型对新图像或视频进行预测。
- **导出**：导出 YOLO 模型以进行部署。
- **跟踪**：使用 YOLO 模型实时跟踪目标。
- **基准测试**：对 YOLO 导出（ONNX、TensorRT 等）的速度和准确率进行基准测试。

| 参数 | 默认值 | 描述 |
| -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mode`   | `'train'` | 指定 YOLO 模型的运行模式：`train` 用于模型训练，`val` 用于验证，`predict` 用于推理，`export` 用于转换为部署格式，`track` 用于目标跟踪，`benchmark` 用于性能评估。每种模式支持从开发到部署的不同阶段。 |

[模式指南](../modes/index.md){ .md-button }

## 训练设置

YOLO 模型的训练设置包括影响模型性能、速度和[准确率](https://www.ultralytics.com/glossary/accuracy)的超参数和配置。关键设置包括[批次大小](https://www.ultralytics.com/glossary/batch-size)、[学习率](https://www.ultralytics.com/glossary/learning-rate)、动量和权重衰减。优化器、[损失函数](https://www.ultralytics.com/glossary/loss-function)和数据集组成的选择也会影响训练。调优和实验对于获得最佳性能至关重要。更多详情请参阅 [Ultralytics 入口函数](../reference/cfg/__init__.md)。

{% include "macros/train-args.md" %}

!!! info "批次大小设置说明"

    `batch` 参数提供三种配置选项：

    - **固定批次大小**：使用整数指定每批图像数量（例如 `batch=16`）。
    - **自动模式（60% GPU 内存）**：使用 `batch=-1` 自动调整到约 60% 的 CUDA 内存利用率。
    - **带利用率分数的自动模式**：设置分数（例如 `batch=0.70`）根据指定的 GPU 内存使用率进行调整。

[训练指南](../modes/train.md){ .md-button }

## 预测设置

YOLO 模型的预测设置包括影响推理期间性能、速度和[准确率](https://www.ultralytics.com/glossary/accuracy)的超参数和配置。关键设置包括置信度阈值、[非极大值抑制（NMS）](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)阈值和类别数量。输入数据大小、格式以及掩码等补充功能也会影响预测。调整这些设置对于获得最佳性能至关重要。

推理参数：

{% include "macros/predict-args.md" %}

可视化参数：

{% from "macros/visualization-args.md" import param_table %} {{ param_table() }}

[预测指南](../modes/predict.md){ .md-button }

## 验证设置

YOLO 模型的验证设置涉及在[验证数据集](https://www.ultralytics.com/glossary/validation-data)上评估性能的超参数和配置。这些设置影响性能、速度和[准确率](https://www.ultralytics.com/glossary/accuracy)。常见设置包括批次大小、验证频率和性能指标。验证数据集的大小和组成以及具体任务也会影响过程。

{% include "macros/validation-args.md" %}

仔细调优和实验对于确保最佳性能以及检测和防止[过拟合](https://www.ultralytics.com/glossary/overfitting)至关重要。

[验证指南](../modes/val.md){ .md-button }

## 导出设置

YOLO 模型的导出设置包括保存或导出模型以在不同环境中使用的配置。这些设置影响性能、大小和兼容性。关键设置包括导出文件格式（例如 ONNX、TensorFlow SavedModel）、目标设备（例如 CPU、GPU）以及掩码等功能。模型的任务和目标环境的约束也会影响导出过程。

{% include "macros/export-args.md" %}

周密的配置确保导出的模型针对其用例进行优化，并在目标环境中有效运行。

[导出指南](../modes/export.md){ .md-button }

## 解决方案设置

Ultralytics 解决方案配置设置提供了灵活性，可以为目标计数、热力图创建、健身跟踪、数据分析、区域跟踪、队列管理和基于区域的计数等任务自定义模型。这些选项允许轻松调整以获得针对特定需求的准确和有用的结果。

{% from "macros/solutions-args.md" import param_table %} {{ param_table() }}

[解决方案指南](../solutions/index.md){ .md-button }

## 数据增强设置

[数据增强](https://www.ultralytics.com/glossary/data-augmentation)技术对于通过向[训练数据](https://www.ultralytics.com/glossary/training-data)引入变化来提高 YOLO 模型的鲁棒性和性能至关重要，帮助模型更好地泛化到未见过的数据。下表概述了每个增强参数的目的和效果：

{% include "macros/augmentation-args.md" %}

根据数据集和任务要求调整这些设置。尝试不同的值可以帮助找到最佳模型性能的最优增强策略。

[数据增强指南](../guides/yolo-data-augmentation.md){ .md-button }

## 日志、检查点和绘图设置

训练 YOLO 模型时，日志、检查点、绘图和文件管理非常重要：

- **日志**：使用 [TensorBoard](https://docs.ultralytics.com/integrations/tensorboard/) 等库或写入文件来跟踪模型进度和诊断问题。
- **检查点**：定期保存模型以恢复训练或尝试不同配置。
- **绘图**：使用 matplotlib 或 TensorBoard 等库可视化性能和训练进度。
- **文件管理**：组织训练期间生成的文件，如检查点、日志文件和绘图，以便于访问和分析。

有效管理这些方面有助于跟踪进度，使调试和优化更加容易。

| 参数 | 默认值 | 描述 |
| ---------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project`  | `'runs'` | 指定保存训练运行的根目录。每次运行保存在单独的子目录中。 |
| `name`     | `'exp'`  | 定义实验名称。如果未指定，YOLO 会为每次运行递增此名称（例如 `exp`、`exp2`）以避免覆盖。 |
| `exist_ok` | `False`  | 确定是否覆盖现有实验目录。`True` 允许覆盖；`False` 阻止覆盖。 |
| `plots`    | `False`  | 控制训练和验证图的生成和保存。设置为 `True` 可创建损失曲线、[精确率](https://www.ultralytics.com/glossary/precision)-[召回率](https://www.ultralytics.com/glossary/recall)曲线和样本预测等图表，用于可视化跟踪性能。 |
| `save`     | `False`  | 启用保存训练检查点和最终模型权重。设置为 `True` 可定期保存模型状态，允许恢复训练或模型部署。 |

## 常见问题

### 如何在训练期间提高 YOLO 模型的性能？

通过调整超参数来提高性能，如[批次大小](https://www.ultralytics.com/glossary/batch-size)、[学习率](https://www.ultralytics.com/glossary/learning-rate)、动量和权重衰减。调整[数据增强](https://www.ultralytics.com/glossary/data-augmentation)设置，选择合适的优化器，并使用早停或[混合精度](https://www.ultralytics.com/glossary/mixed-precision)等技术。详情请参阅[训练指南](../modes/train.md)。

### YOLO 模型准确率的关键超参数有哪些？

影响准确率的关键超参数包括：

- **批次大小（`batch`）**：较大的大小可以稳定训练，但需要更多内存。
- **学习率（`lr0`）**：较小的学习率提供精细调整但收敛较慢。
- **动量（`momentum`）**：加速梯度向量，抑制振荡。
- **图像大小（`imgsz`）**：较大的尺寸提高准确率但增加计算负载。

根据您的数据集和硬件调整这些参数。在[训练设置](#训练设置)中了解更多。

### 如何设置训练 YOLO 模型的学习率？

学习率（`lr0`）至关重要；对于 SGD 从 `0.01` 开始，对于 [Adam 优化器](https://www.ultralytics.com/glossary/adam-optimizer)从 `0.001` 开始。监控指标并根据需要调整。使用余弦学习率调度器（`cos_lr`）或预热（`warmup_epochs`、`warmup_momentum`）。详情请参阅[训练指南](../modes/train.md)。

### YOLO 模型的默认推理设置是什么？

默认设置包括：

- **置信度阈值（`conf=0.25`）**：检测的最小置信度。
- **IoU 阈值（`iou=0.7`）**：用于[非极大值抑制（NMS）](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)。
- **图像大小（`imgsz=640`）**：调整输入图像大小。
- **设备（`device=None`）**：选择 CPU 或 GPU。

有关完整概述，请参阅[预测设置](#预测设置)和[预测指南](../modes/predict.md)。

### 为什么在 YOLO 模型中使用混合精度训练？

[混合精度](https://www.ultralytics.com/glossary/mixed-precision)训练（`amp=True`）使用 FP16 和 FP32 减少内存使用并加速训练。它对现代 GPU 有益，允许使用更大的模型和更快的计算，而不会显著损失准确率。在[训练指南](../modes/train.md)中了解更多。
