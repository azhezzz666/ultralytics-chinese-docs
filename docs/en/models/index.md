---
comments: true
description: 探索 Ultralytics 支持的各种模型，包括 YOLOv3 到 YOLO11、NAS、SAM 和 RT-DETR，用于检测、分割等任务。
keywords: Ultralytics, 支持的模型, YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, SAM, SAM2, SAM3, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, YOLO-World, 目标检测, 图像分割, 分类, 姿态估计, 多目标跟踪
---

# Ultralytics 支持的模型

欢迎来到 Ultralytics 的模型文档！我们支持广泛的模型，每个模型都针对特定任务进行了优化，如[目标检测](../tasks/detect.md)、[实例分割](../tasks/segment.md)、[图像分类](../tasks/classify.md)、[姿态估计](../tasks/pose.md)和[多目标跟踪](../modes/track.md)。如果您有兴趣将您的模型架构贡献给 Ultralytics，请查看我们的[贡献指南](../help/contributing.md)。

![Ultralytics YOLO11 比较图](https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png)

## 特色模型

以下是一些支持的关键模型：

1. **[YOLOv3](yolov3.md)**：YOLO 模型系列的第三代，最初由 Joseph Redmon 开发，以其高效的实时目标检测能力而闻名。
2. **[YOLOv4](yolov4.md)**：YOLOv3 的 darknet 原生更新版本，由 Alexey Bochkovskiy 于 2020 年发布。
3. **[YOLOv5](yolov5.md)**：Ultralytics 改进的 YOLO 架构版本，与之前的版本相比提供了更好的性能和速度权衡。
4. **[YOLOv6](yolov6.md)**：由[美团](https://www.meituan.com/)于 2022 年发布，并在该公司的许多自动配送机器人中使用。
5. **[YOLOv7](yolov7.md)**：由 YOLOv4 的作者于 2022 年发布的更新 YOLO 模型。仅支持推理。
6. **[YOLOv8](yolov8.md)**：一个多功能模型，具有增强的功能，如[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)、姿态/关键点估计和分类。
7. **[YOLOv9](yolov9.md)**：一个实验性模型，在 Ultralytics [YOLOv5](yolov5.md) 代码库上训练，实现了可编程梯度信息（PGI）。
8. **[YOLOv10](yolov10.md)**：由清华大学开发，具有无 NMS 训练和效率-精度驱动的架构，提供最先进的性能和延迟。
9. **[YOLO11](yolo11.md) 🚀**：Ultralytics 最新的 YOLO 模型，在检测、分割、姿态估计、跟踪和分类等多项任务中提供最先进（SOTA）的性能。
10. **[YOLO26](yolo26.md) ⚠️ 即将推出**：Ultralytics 的下一代 YOLO 模型，针对边缘部署进行了优化，具有端到端无 NMS 推理。
11. **[Segment Anything Model (SAM)](sam.md)**：Meta 的原始 Segment Anything 模型（SAM）。
12. **[Segment Anything Model 2 (SAM2)](sam-2.md)**：Meta 的下一代 Segment Anything 模型，用于视频和图像。
13. **[Segment Anything Model 3 (SAM3)](sam-3.md) 🚀 新**：Meta 的第三代 Segment Anything 模型，具有可提示概念分割功能，支持基于文本和图像示例的分割。
14. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)**：庆熙大学开发的用于移动应用的 MobileSAM。
15. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)**：中国科学院自动化研究所图像与视频分析组开发的 FastSAM。
16. **[YOLO-NAS](yolo-nas.md)**：YOLO [神经架构搜索](https://www.ultralytics.com/glossary/neural-architecture-search-nas)（NAS）模型。
17. **[Real-Time Detection Transformers (RT-DETR)](rtdetr.md)**：百度 PaddlePaddle 实时检测 [Transformer](https://www.ultralytics.com/glossary/transformer)（RT-DETR）模型。
18. **[YOLO-World](yolo-world.md)**：腾讯 AI Lab 的实时开放词汇目标检测模型。
19. **[YOLOE](yoloe.md)**：一个改进的开放词汇目标检测器，在保持 YOLO 实时性能的同时检测训练数据之外的任意类别。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>只需几行代码即可运行 Ultralytics YOLO 模型。
</p>

## 入门：使用示例

此示例提供了简单的 YOLO 训练和推理示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参阅[预测](../modes/predict.md)、[训练](../modes/train.md)、[验证](../modes/val.md)和[导出](../modes/export.md)文档页面。

请注意，以下示例重点介绍用于[目标检测](https://www.ultralytics.com/glossary/object-detection)的 YOLO11 [检测](../tasks/detect.md)模型。有关其他支持的任务，请参阅[分割](../tasks/segment.md)、[分类](../tasks/classify.md)和[姿态](../tasks/pose.md)文档。

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) 预训练的 `*.pt` 模型以及配置 `*.yaml` 文件可以传递给 `YOLO()`、`SAM()`、`NAS()` 和 `RTDETR()` 类，以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个轮次
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLO11n 模型对 'bus.jpg' 图像运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI 命令可直接运行模型：

        ```bash
        # 加载 COCO 预训练的 YOLO11n 模型并在 COCO8 示例数据集上训练 100 个轮次
        yolo train model=yolo11n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载 COCO 预训练的 YOLO11n 模型并对 'bus.jpg' 图像运行推理
        yolo predict model=yolo11n.pt source=path/to/bus.jpg
        ```

## 贡献新模型

有兴趣将您的模型贡献给 Ultralytics 吗？太好了！我们始终欢迎扩展我们的模型库。

1. **Fork 仓库**：首先 fork [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)。

2. **克隆您的 Fork**：将您的 fork 克隆到本地机器，并创建一个新分支进行工作。

3. **实现您的模型**：按照我们[贡献指南](../help/contributing.md)中提供的编码标准和指南添加您的模型。

4. **彻底测试**：确保对您的模型进行严格测试，包括单独测试和作为流水线的一部分测试。

5. **创建 Pull Request**：一旦您对模型满意，创建一个 pull request 到主仓库进行审核。

6. **代码审核和合并**：审核后，如果您的模型符合我们的标准，它将被合并到主仓库中。

有关详细步骤，请参阅我们的[贡献指南](../help/contributing.md)。

## 常见问题

### 使用 Ultralytics YOLO11 进行目标检测有哪些主要优势？

Ultralytics YOLO11 提供增强的功能，如实时目标检测、实例分割、姿态估计和分类。其优化的架构确保高速性能而不牺牲[精度](https://www.ultralytics.com/glossary/accuracy)，使其成为各种 AI 领域应用的理想选择。YOLO11 在之前版本的基础上进行了改进，具有更好的性能和额外的功能，详见 [YOLO11 文档页面](../models/yolo11.md)。

### 如何在自定义数据上训练 YOLO 模型？

使用 Ultralytics 的库可以轻松在自定义数据上训练 YOLO 模型。以下是一个快速示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO 模型
        model = YOLO("yolo11n.pt")  # 或任何其他 YOLO 模型

        # 在自定义数据集上训练模型
        results = model.train(data="custom_data.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo train model=yolo11n.pt data='custom_data.yaml' epochs=100 imgsz=640
        ```

有关更详细的说明，请访问[训练](../modes/train.md)文档页面。

### Ultralytics 支持哪些 YOLO 版本？

Ultralytics 支持从 YOLOv3 到 YOLO11 的全面 YOLO（You Only Look Once）版本，以及 YOLO-NAS、SAM 和 RT-DETR 等模型。每个版本都针对检测、分割和分类等各种任务进行了优化。有关每个模型的详细信息，请参阅 [Ultralytics 支持的模型](../models/index.md)文档。

### 为什么应该使用 Ultralytics HUB 进行[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)项目？

[Ultralytics HUB](../hub/index.md) 提供了一个无代码、端到端的平台，用于训练、部署和管理 YOLO 模型。它简化了复杂的工作流程，使用户能够专注于模型性能和应用。HUB 还提供[云训练功能](../hub/cloud-training.md)、全面的数据集管理，以及对初学者和经验丰富的开发人员都友好的用户界面。

### YOLO11 可以执行哪些类型的任务，它与其他 YOLO 版本相比如何？

YOLO11 是一个多功能模型，能够执行目标检测、实例分割、分类和姿态估计等任务。与早期版本相比，由于其优化的架构和无锚点设计，YOLO11 在速度和精度方面都有显著改进。有关更深入的比较，请参阅 [YOLO11 文档](../models/yolo11.md)和[任务页面](../tasks/index.md)了解特定任务的更多详情。
