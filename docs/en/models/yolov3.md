---
comments: true
description: 探索 YOLOv3 及其变体 YOLOv3-Ultralytics 和 YOLOv3u。了解它们的特性、实现和对目标检测任务的支持。
keywords: YOLOv3, YOLOv3-Ultralytics, YOLOv3u, 目标检测, Ultralytics, 计算机视觉, AI 模型, 深度学习
---

# YOLOv3 和 YOLOv3u

## 概述

本文档概述了三个密切相关的目标检测模型，即 [YOLOv3](https://pjreddie.com/darknet/yolo/)、[YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3) 和 [YOLOv3u](https://github.com/ultralytics/ultralytics)。

1. **YOLOv3：** 这是 You Only Look Once (YOLO) 目标检测算法的第三个版本。最初由 Joseph Redmon 开发，YOLOv3 通过引入多尺度预测和三种不同大小的检测核等特性改进了其前身。

2. **YOLOv3u：** 这是 YOLOv3-Ultralytics 的更新版本，集成了 YOLOv8 模型中使用的无锚点、无目标性分离头部。YOLOv3u 保持了与 YOLOv3 相同的[骨干网络](https://www.ultralytics.com/glossary/backbone)和颈部架构，但使用了来自 YOLOv8 的更新[检测头部](https://www.ultralytics.com/glossary/detection-head)。

![Ultralytics YOLOv3](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov3-banner.avif)

## 关键特性

- **YOLOv3：** 引入了使用三种不同尺度进行检测，利用三种不同大小的检测核：13x13、26x26 和 52x52。这显著提高了不同大小对象的检测精度。此外，YOLOv3 添加了每个[边界框](https://www.ultralytics.com/glossary/bounding-box)的多标签预测和更好的特征提取网络等功能。

- **YOLOv3u：** 这个更新的模型集成了来自 YOLOv8 的无锚点、无目标性分离头部。通过消除对预定义锚框和目标性分数的需求，这种检测头部设计可以提高模型检测不同大小和形状对象的能力。这使得 YOLOv3u 在目标检测任务中更加稳健和准确。

## 支持的任务和模式

YOLOv3 专门设计用于[目标检测](https://www.ultralytics.com/glossary/object-detection)任务。Ultralytics 支持三种 YOLOv3 变体：`yolov3u`、`yolov3-tinyu` 和 `yolov3-sppu`。名称中的 `u` 表示这些模型使用 YOLOv8 的无锚点头部，与其原始基于锚点的架构不同。这些模型以其在各种实际场景中的有效性而闻名，平衡了精度和速度。每个变体都提供独特的功能和优化，使其适用于各种应用。

所有三个模型都支持全面的模式集，确保在[模型部署](https://www.ultralytics.com/glossary/model-deployment)和开发的各个阶段具有多功能性。这些模式包括[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)，为用户提供有效目标检测的完整工具包。

| 模型类型       | 预训练权重        | 支持的任务                             | 推理 | 验证 | 训练 | 导出 |
| -------------- | ----------------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| YOLOv3(u)      | `yolov3u.pt`      | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ✅   |
| YOLOv3-Tiny(u) | `yolov3-tinyu.pt` | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ✅   |
| YOLOv3u-SPP(u) | `yolov3-sppu.pt`  | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ✅   |

此表提供了每个 YOLOv3 变体功能的概览，突显了它们在目标检测工作流程中各种任务和操作模式的多功能性和适用性。

## 使用示例

本示例提供简单的 YOLOv3 训练和推理示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参阅[预测](../modes/predict.md)、[训练](../modes/train.md)、[验证](../modes/val.md)和[导出](../modes/export.md)文档页面。

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) 预训练的 `*.pt` 模型以及配置 `*.yaml` 文件可以传递给 `YOLO()` 类以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLOv3u 模型
        model = YOLO("yolov3u.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个 epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLOv3u 模型在 'bus.jpg' 图像上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI 命令可直接运行模型：

        ```bash
        # 加载 COCO 预训练的 YOLOv3u 模型并在 COCO8 示例数据集上训练 100 个 epoch
        yolo train model=yolov3u.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载 COCO 预训练的 YOLOv3u 模型并在 'bus.jpg' 图像上运行推理
        yolo predict model=yolov3u.pt source=path/to/bus.jpg
        ```

## 引用和致谢

如果您在研究中使用 YOLOv3，请引用原始 YOLO 论文和 Ultralytics YOLOv3 仓库：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

感谢 Joseph Redmon 和 Ali Farhadi 开发了原始 YOLOv3。

## 常见问题

### YOLOv3、YOLOv3-Ultralytics 和 YOLOv3u 之间有什么区别？

YOLOv3 是由 Joseph Redmon 开发的 YOLO（You Only Look Once）[目标检测](https://www.ultralytics.com/glossary/object-detection)算法的第三次迭代，以其[精度](https://www.ultralytics.com/glossary/accuracy)和速度的平衡而闻名，使用三种不同的尺度（13x13、26x26 和 52x52）进行检测。YOLOv3-Ultralytics 是 Ultralytics 对 YOLOv3 的改编，增加了对更多预训练模型的支持并便于更轻松的模型自定义。YOLOv3u 是 YOLOv3-Ultralytics 的升级变体，集成了来自 YOLOv8 的无锚点、无目标性分离头部，提高了各种对象大小的检测稳健性和精度。

### 如何使用 Ultralytics 训练 YOLOv3 模型？

使用 Ultralytics 训练 YOLOv3 模型很简单。您可以使用 Python 或 CLI 训练模型：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLOv3u 模型
        model = YOLO("yolov3u.pt")

        # 在 COCO8 示例数据集上训练模型 100 个 epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 加载 COCO 预训练的 YOLOv3u 模型并在 COCO8 示例数据集上训练 100 个 epoch
        yolo train model=yolov3u.pt data=coco8.yaml epochs=100 imgsz=640
        ```

有关更全面的训练选项和指南，请访问我们的[训练模式文档](../modes/train.md)。

### 是什么使 YOLOv3u 在目标检测任务中更准确？

YOLOv3u 通过集成 YOLOv8 模型中使用的无锚点、无目标性分离头部改进了 YOLOv3 和 YOLOv3-Ultralytics。此升级消除了对预定义锚框和目标性分数的需求，增强了其更精确地检测不同大小和形状对象的能力。这使得 YOLOv3u 成为复杂和多样化目标检测任务的更好选择。

### 如何使用 YOLOv3 模型进行推理？

您可以使用 Python 脚本或 CLI 命令使用 YOLOv3 模型执行推理：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLOv3u 模型
        model = YOLO("yolov3u.pt")

        # 使用 YOLOv3u 模型在 'bus.jpg' 图像上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 加载 COCO 预训练的 YOLOv3u 模型并在 'bus.jpg' 图像上运行推理
        yolo predict model=yolov3u.pt source=path/to/bus.jpg
        ```

有关运行 YOLO 模型的更多详细信息，请参阅[推理模式文档](../modes/predict.md)。

### YOLOv3 及其变体支持哪些任务？

YOLOv3、YOLOv3-Tiny 和 YOLOv3-SPP 主要支持目标检测任务。这些模型可用于模型部署和开发的各个阶段，如推理、验证、训练和导出。有关支持的任务的全面集合和更深入的详细信息，请访问我们的[目标检测任务文档](../tasks/detect.md)。

### 在哪里可以找到在研究中引用 YOLOv3 的资源？

如果您在研究中使用 YOLOv3，请引用原始 YOLO 论文和 Ultralytics YOLOv3 仓库。示例 BibTeX 引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

有关更多引用详细信息，请参阅[引用和致谢](#引用和致谢)部分。
