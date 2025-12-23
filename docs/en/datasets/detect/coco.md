---
comments: true
description: 探索用于目标检测和分割的 COCO 数据集。了解其结构、使用方法、预训练模型和主要特点。
keywords: COCO 数据集, 目标检测, 分割, 基准测试, 计算机视觉, 姿态估计, YOLO 模型, COCO 标注
---

# COCO 数据集

[COCO](https://cocodataset.org/#home)（Common Objects in Context）数据集是一个大规模的目标检测、分割和图像描述数据集。它旨在鼓励对各种目标类别的研究，通常用于[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型的基准测试。对于从事目标检测、分割和姿态估计任务的研究人员和开发者来说，这是一个必不可少的数据集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uDrn9QZJ2lk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics COCO 数据集概述
</p>

## COCO 预训练模型

{% include "macros/yolo-det-perf.md" %}

## 主要特点

- COCO 包含 33 万张图像，其中 20 万张图像具有目标检测、分割和图像描述任务的标注。
- 该数据集包含 80 个目标类别，包括汽车、自行车和动物等常见物体，以及雨伞、手提包和运动器材等更具体的类别。
- 标注包括每张图像的目标边界框、分割掩码和图像描述。
- COCO 提供标准化的评估指标，如目标检测的[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）和分割任务的平均[召回率](https://www.ultralytics.com/glossary/recall)均值（mAR），使其适合比较模型性能。

## 数据集结构

COCO 数据集分为三个子集：

1. **Train2017**：此子集包含 11.8 万张用于训练目标检测、分割和图像描述模型的图像。
2. **Val2017**：此子集包含 5000 张用于模型训练期间验证的图像。
3. **Test2017**：此子集包含 2 万张用于测试和基准测试已训练模型的图像。此子集的真实标注不公开，结果需提交到 [COCO 评估服务器](https://codalab.lisn.upsaclay.fr/competitions/7384)进行性能评估。


## 应用

COCO 数据集广泛用于训练和评估目标检测（如 [Ultralytics YOLO](../../models/yolo11.md)、[Faster R-CNN](https://arxiv.org/abs/1506.01497) 和 [SSD](https://arxiv.org/abs/1512.02325)）、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)（如 [Mask R-CNN](https://arxiv.org/abs/1703.06870)）和关键点检测（如 [OpenPose](https://arxiv.org/abs/1812.08008)）中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。该数据集多样化的目标类别、大量的标注图像和标准化的评估指标使其成为计算机视觉研究人员和从业者的重要资源。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含有关数据集路径、类别和其他相关信息。对于 COCO 数据集，`coco.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)。

!!! example "ultralytics/cfg/datasets/coco.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco.yaml"
    ```

## 使用方法

要在 COCO 数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

COCO 数据集包含各种目标类别和复杂场景的多样化图像集。以下是数据集中的一些图像示例及其相应的标注：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/mosaiced-coco-dataset-sample.avif)

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是一种在训练期间使用的技术，将多张图像合并为一张图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 COCO 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克技术的好处。

## 引用和致谢

如果您在研究或开发工作中使用 COCO 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们感谢 COCO 联盟为计算机视觉社区创建和维护这一宝贵资源。有关 COCO 数据集及其创建者的更多信息，请访问 [COCO 数据集网站](https://cocodataset.org/#home)。

## 常见问题

### 什么是 COCO 数据集，为什么它对计算机视觉很重要？

[COCO 数据集](https://cocodataset.org/#home)（Common Objects in Context）是一个用于[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和图像描述的大规模数据集。它包含 33 万张图像，具有 80 个目标类别的详细标注，使其对于基准测试和训练计算机视觉模型至关重要。研究人员使用 COCO 是因为其多样化的类别和标准化的评估指标，如平均[精确率](https://www.ultralytics.com/glossary/precision)均值（mAP）。

### 如何使用 COCO 数据集训练 YOLO 模型？

要使用 COCO 数据集训练 YOLO11 模型，您可以使用以下代码片段：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关更多详情，请参阅[训练页面](../../modes/train.md)了解可用参数。

### COCO 数据集的主要特点是什么？

COCO 数据集包括：

- 33 万张图像，其中 20 万张标注用于目标检测、分割和图像描述。
- 80 个目标类别，从汽车和动物等常见物品到手提包和运动器材等特定物品。
- 目标检测（mAP）和分割（平均召回率均值，mAR）的标准化评估指标。
- 训练批次中的**马赛克**技术，以增强模型在各种目标尺寸和上下文中的泛化能力。

### 在哪里可以找到在 COCO 数据集上训练的预训练 YOLO11 模型？

在 COCO 数据集上预训练的 YOLO11 模型可以从文档中提供的链接下载。示例包括：

- [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
- [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)
- [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt)
- [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt)
- [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)

这些模型在大小、mAP 和推理速度方面各不相同，为不同的性能和资源需求提供选择。

### COCO 数据集的结构是什么，如何使用它？

COCO 数据集分为三个子集：

1. **Train2017**：11.8 万张用于训练的图像。
2. **Val2017**：5000 张用于训练期间验证的图像。
3. **Test2017**：2 万张用于基准测试已训练模型的图像。结果需要提交到 [COCO 评估服务器](https://codalab.lisn.upsaclay.fr/competitions/7384)进行性能评估。

数据集的 YAML 配置文件可在 [coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 找到，其中定义了路径、类别和数据集详情。
