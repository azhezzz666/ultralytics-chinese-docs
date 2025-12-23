---
comments: true
description: 探索美团 YOLOv6，一种在速度和精度之间取得卓越平衡的顶级目标检测器。在 Ultralytics 文档中了解其独特功能和性能指标。
keywords: 美团 YOLOv6, 目标检测, 实时应用, BiC 模块, 锚点辅助训练, COCO 数据集, 高性能模型, Ultralytics 文档
---

# 美团 YOLOv6

## 概述

[美团](https://www.meituan.com/) YOLOv6 是一种尖端的目标检测器，在速度和精度之间提供了卓越的平衡，使其成为实时应用的热门选择。该模型在其架构和训练方案上引入了几项显著的增强，包括双向连接（BiC）模块的实现、锚点辅助训练（AAT）策略，以及改进的[骨干网络](https://www.ultralytics.com/glossary/backbone)和颈部设计，以在 [COCO 数据集](https://docs.ultralytics.com/datasets/detect/coco/)上实现先进的精度。

![美团 YOLOv6](https://github.com/ultralytics/docs/releases/download/0/meituan-yolov6.avif)

### 关键特性

- **双向连接（BiC）模块：** YOLOv6 在检测器的颈部引入了 BiC 模块，增强了定位信号并以可忽略的速度下降提供了性能提升。
- **锚点辅助训练（AAT）策略：** 该模型提出了 AAT，以享受基于锚点和无锚点范式的优势，而不影响推理效率。
- **增强的骨干网络和颈部设计：** 通过在骨干网络和颈部中加深 YOLOv6 以包含另一个阶段，该模型在高分辨率输入上在 COCO 数据集上实现了先进的性能。
- **自蒸馏策略：** 实施了一种新的自蒸馏策略来提升 YOLOv6 较小模型的性能，在训练期间增强辅助回归分支并在推理时移除它以避免明显的速度下降。

## 性能指标

YOLOv6 提供了各种不同规模的预训练模型：

- YOLOv6-N：在 NVIDIA T4 GPU 上以 1187 FPS 在 COCO val2017 上达到 37.5% AP。
- YOLOv6-S：以 484 FPS 达到 45.0% AP。
- YOLOv6-M：以 226 FPS 达到 50.0% AP。
- YOLOv6-L：以 116 FPS 达到 52.8% AP。
- YOLOv6-L6：实时场景中的先进精度。

## 使用示例

本示例提供简单的 YOLOv6 训练和推理示例。有关这些和其他模式的完整文档，请参阅预测、训练、验证和导出文档页面。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从头构建 YOLOv6n 模型
        model = YOLO("yolov6n.yaml")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个 epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLOv6n 模型在 'bus.jpg' 图像上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 从头构建 YOLOv6n 模型并在 COCO8 示例数据集上训练 100 个 epoch
        yolo train model=yolov6n.yaml data=coco8.yaml epochs=100 imgsz=640

        # 从头构建 YOLOv6n 模型并在 'bus.jpg' 图像上运行推理
        yolo predict model=yolov6n.yaml source=path/to/bus.jpg
        ```

## 支持的任务和模式

YOLOv6 系列提供了一系列模型，每个模型都针对高性能目标检测进行了优化。

| 模型     | 文件名         | 任务                           | 推理 | 验证 | 训练 | 导出 |
| -------- | -------------- | ------------------------------ | ---- | ---- | ---- | ---- |
| YOLOv6-N | `yolov6n.yaml` | [目标检测](../tasks/detect.md) | ✅   | ✅   | ✅   | ✅   |
| YOLOv6-S | `yolov6s.yaml` | [目标检测](../tasks/detect.md) | ✅   | ✅   | ✅   | ✅   |
| YOLOv6-M | `yolov6m.yaml` | [目标检测](../tasks/detect.md) | ✅   | ✅   | ✅   | ✅   |
| YOLOv6-L | `yolov6l.yaml` | [目标检测](../tasks/detect.md) | ✅   | ✅   | ✅   | ✅   |
| YOLOv6-X | `yolov6x.yaml` | [目标检测](../tasks/detect.md) | ✅   | ✅   | ✅   | ✅   |

## 引用和致谢

我们要感谢作者在实时目标检测领域的重大贡献：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{li2023yolov6,
              title={YOLOv6 v3.0: A Full-Scale Reloading},
              author={Chuyi Li and Lulu Li and Yifei Geng and Hongliang Jiang and Meng Cheng and Bo Zhang and Zaidan Ke and Xiaoming Xu and Xiangxiang Chu},
              year={2023},
              eprint={2301.05586},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

原始 YOLOv6 论文可以在 [arXiv](https://arxiv.org/abs/2301.05586) 上找到。作者已公开其工作，代码库可以在 [GitHub](https://github.com/meituan/YOLOv6) 上访问。

## 常见问题

### 什么是美团 YOLOv6，它有什么独特之处？

美团 YOLOv6 是一种先进的目标检测器，在速度和精度之间取得平衡，非常适合实时应用。它具有显著的架构增强，如双向连接（BiC）模块和锚点辅助训练（AAT）策略。

### YOLOv6 中的双向连接（BiC）模块如何提高性能？

YOLOv6 中的双向连接（BiC）模块增强了检测器颈部的定位信号，以可忽略的速度影响提供了性能改进。该模块有效地组合了不同的特征图，提高了模型准确检测对象的能力。

### 如何使用 Ultralytics 训练 YOLOv6 模型？

您可以使用简单的 Python 或 CLI 命令通过 Ultralytics 训练 YOLOv6 模型：

```python
from ultralytics import YOLO

model = YOLO("yolov6n.yaml")
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

### 锚点辅助训练（AAT）策略如何使 YOLOv6 受益？

YOLOv6 中的锚点辅助训练（AAT）结合了基于锚点和无锚点方法的元素，增强了模型的检测能力而不影响推理效率。
