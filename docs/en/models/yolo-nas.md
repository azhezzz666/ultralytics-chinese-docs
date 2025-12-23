---
comments: true
description: 探索 Deci AI 的 YOLO-NAS - 一个具有量化支持的最先进目标检测模型。了解特性、预训练模型和实现示例。
keywords: YOLO-NAS, Deci AI, 目标检测, 深度学习, 神经架构搜索, Ultralytics, Python API, YOLO 模型, SuperGradients, 预训练模型, 量化, AutoNAC
---

# YOLO-NAS

!!! note "重要更新"

    请注意，YOLO-NAS 的原始创建者 [Deci](https://www.linkedin.com/company/deciai/) 已被 NVIDIA 收购。因此，这些模型不再由 Deci 积极维护。Ultralytics 继续支持这些模型的使用，但预计不会有来自原始团队的进一步更新。

## 概述

由 Deci AI 开发的 YOLO-NAS 是一个开创性的目标检测基础模型。它是先进[神经架构搜索](https://www.ultralytics.com/glossary/neural-architecture-search-nas)技术的产物，经过精心设计以解决之前 YOLO 模型的局限性。通过在量化支持和[精度](https://www.ultralytics.com/glossary/accuracy)-延迟权衡方面的显著改进，YOLO-NAS 代表了目标检测领域的重大飞跃。

![模型示例图像](https://github.com/ultralytics/docs/releases/download/0/yolo-nas-coco-map-metrics.avif) **YOLO-NAS 概述。** YOLO-NAS 采用量化感知块和选择性量化以获得最佳性能。当模型转换为 INT8 量化版本时，精度下降极小，这是相对于其他模型的显著改进。这些进步最终形成了具有前所未有的目标检测能力和出色性能的卓越架构。

### 关键特性

- **量化友好的基本块：** YOLO-NAS 引入了一种对量化友好的新基本块，解决了之前 YOLO 模型的重大局限性之一。
- **复杂的训练和量化：** YOLO-NAS 利用先进的训练方案和训练后量化来增强性能。
- **AutoNAC 优化和预训练：** YOLO-NAS 利用 AutoNAC 优化，并在 COCO、Objects365 和 Roboflow 100 等著名数据集上进行预训练。这种预训练使其非常适合生产环境中的下游目标检测任务。

## 预训练模型

使用 Ultralytics 提供的预训练 YOLO-NAS 模型体验下一代目标检测的强大功能。这些模型旨在在速度和精度方面提供一流的性能。根据您的特定需求从多种选项中选择：

!!! tip "性能"

    === "检测 (COCO)"

        | 模型             | mAP   | 延迟 (ms) |
        | ---------------- | ----- | --------- |
        | YOLO-NAS S       | 47.5  | 3.21      |
        | YOLO-NAS M       | 51.55 | 5.85      |
        | YOLO-NAS L       | 52.22 | 7.87      |
        | YOLO-NAS S INT-8 | 47.03 | 2.36      |
        | YOLO-NAS M INT-8 | 51.0  | 3.78      |
        | YOLO-NAS L INT-8 | 52.1  | 4.78      |

每个模型变体都旨在提供[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) 和延迟之间的平衡，帮助您在性能和速度方面优化目标检测任务。

## 使用示例

Ultralytics 通过我们的 `ultralytics` Python 包使 YOLO-NAS 模型易于集成到您的 Python 应用程序中。该包提供用户友好的 Python API 以简化流程。

以下示例展示了如何使用 `ultralytics` 包与 YOLO-NAS 模型进行推理和验证：

### 推理和验证示例

在此示例中，我们在 COCO8 数据集上验证 YOLO-NAS-s。

!!! example

    此示例提供了 YOLO-NAS 的简单推理和验证代码。有关处理推理结果，请参阅[预测](../modes/predict.md)模式。有关使用 YOLO-NAS 的其他模式，请参阅[验证](../modes/val.md)和[导出](../modes/export.md)。`ultralytics` 包中的 YOLO-NAS 不支持训练。

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) 预训练的 `*.pt` 模型文件可以传递给 `NAS()` 类以在 Python 中创建模型实例：

        ```python
        from ultralytics import NAS

        # 加载 COCO 预训练的 YOLO-NAS-s 模型
        model = NAS("yolo_nas_s.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上验证模型
        results = model.val(data="coco8.yaml")

        # 使用 YOLO-NAS-s 模型在 'bus.jpg' 图像上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI 命令可用于直接运行模型：

        ```bash
        # 加载 COCO 预训练的 YOLO-NAS-s 模型并在 COCO8 示例数据集上验证其性能
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # 加载 COCO 预训练的 YOLO-NAS-s 模型并在 'bus.jpg' 图像上运行推理
        yolo predict model=yolo_nas_s.pt source=path/to/bus.jpg
        ```

## 支持的任务和模式

我们提供三种 YOLO-NAS 模型变体：Small (s)、Medium (m) 和 Large (l)。每个变体都旨在满足不同的计算和性能需求：

- **YOLO-NAS-s**：针对计算资源有限但效率是关键的环境进行优化。
- **YOLO-NAS-m**：提供平衡的方法，适用于具有更高精度的通用[目标检测](https://www.ultralytics.com/glossary/object-detection)。
- **YOLO-NAS-l**：针对需要最高精度、计算资源不是主要约束的场景量身定制。

以下是每个模型的详细概述，包括其预训练权重的链接、支持的任务以及与不同操作模式的兼容性。

| 模型类型   | 预训练权重                                                                                    | 支持的任务                             | 推理 | 验证 | 训练 | 导出 |
| ---------- | --------------------------------------------------------------------------------------------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| YOLO-NAS-s | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo_nas_s.pt) | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ❌   | ✅   |
| YOLO-NAS-m | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo_nas_m.pt) | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ❌   | ✅   |
| YOLO-NAS-l | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo_nas_l.pt) | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ❌   | ✅   |


## 引用和致谢

如果您在研究或开发工作中使用 YOLO-NAS，请引用 SuperGradients：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/records/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

我们感谢 Deci AI 的 [SuperGradients](https://github.com/Deci-AI/super-gradients/) 团队为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源所做的努力。我们相信 YOLO-NAS 凭借其创新架构和卓越的目标检测能力，将成为开发人员和研究人员的关键工具。

## 常见问题

### YOLO-NAS 是什么，它如何改进之前的 YOLO 模型？

YOLO-NAS 由 Deci AI 开发，是一个利用先进神经架构搜索 (NAS) 技术的最先进目标检测模型。它通过引入量化友好的基本块和复杂的训练方案等特性来解决之前 YOLO 模型的局限性。这导致性能显著提升，特别是在计算资源有限的环境中。YOLO-NAS 还支持量化，即使转换为 INT8 版本也能保持高精度，增强了其在生产环境中的适用性。有关更多详细信息，请参阅[概述](#概述)部分。

### 如何将 YOLO-NAS 模型集成到我的 Python 应用程序中？

您可以使用 `ultralytics` 包轻松将 YOLO-NAS 模型集成到您的 Python 应用程序中。以下是如何加载预训练的 YOLO-NAS 模型并执行推理的简单示例：

```python
from ultralytics import NAS

# 加载 COCO 预训练的 YOLO-NAS-s 模型
model = NAS("yolo_nas_s.pt")

# 在 COCO8 示例数据集上验证模型
results = model.val(data="coco8.yaml")

# 使用 YOLO-NAS-s 模型在 'bus.jpg' 图像上运行推理
results = model("path/to/bus.jpg")
```

有关更多信息，请参阅[推理和验证示例](#推理和验证示例)。

### YOLO-NAS 的关键特性是什么，为什么我应该考虑使用它？

YOLO-NAS 引入了几个关键特性，使其成为目标检测任务的卓越选择：

- **量化友好的基本块：** 增强的架构，在量化后[精度](https://www.ultralytics.com/glossary/precision)下降最小。
- **复杂的训练和量化：** 采用先进的训练方案和训练后量化技术。
- **AutoNAC 优化和预训练：** 利用 AutoNAC 优化，并在 COCO、Objects365 和 Roboflow 100 等著名数据集上进行预训练。

这些特性有助于其高精度、高效性能以及适合在生产环境中部署。在[关键特性](#关键特性)部分了解更多。

### YOLO-NAS 模型支持哪些任务和模式？

YOLO-NAS 模型支持各种目标检测任务和模式，如推理、验证和导出。它们不支持训练。支持的模型包括 YOLO-NAS-s、YOLO-NAS-m 和 YOLO-NAS-l，每个都针对不同的计算能力和性能需求量身定制。有关详细概述，请参阅[支持的任务和模式](#支持的任务和模式)部分。

### 是否有预训练的 YOLO-NAS 模型可用，如何访问它们？

是的，Ultralytics 提供您可以直接访问的预训练 YOLO-NAS 模型。这些模型在 COCO 等数据集上进行预训练，确保在速度和精度方面的高性能。您可以使用[预训练模型](#预训练模型)部分提供的链接下载这些模型。以下是一些示例：

- [YOLO-NAS-s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo_nas_s.pt)
- [YOLO-NAS-m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo_nas_m.pt)
- [YOLO-NAS-l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo_nas_l.pt)
