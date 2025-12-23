---
comments: true
description: 探索紧凑、多功能的 COCO8-Pose 数据集，用于测试和调试目标检测模型。非常适合使用 YOLO11 进行快速实验。
keywords: COCO8-Pose, Ultralytics, 姿态检测数据集, 目标检测, YOLO11, 机器学习, 计算机视觉, 训练数据
---

# COCO8-Pose 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO8-Pose 是一个小型但多功能的姿态检测数据集，由 COCO train 2017 集的前 8 张图像组成，4 张用于训练，4 张用于验证。此数据集非常适合测试和调试[目标检测](https://www.ultralytics.com/glossary/object-detection)模型，或用于实验新的检测方法。只有 8 张图像，它足够小以便于管理，但又足够多样化以测试训练流程中的错误，并在训练更大数据集之前作为健全性检查。

## 数据集结构

- **总图像数**：8（4 张训练 / 4 张验证）。
- **类别**：1（人），每个标注有 17 个关键点。
- **推荐目录布局**：`datasets/coco8-pose/images/{train,val}` 和 `datasets/coco8-pose/labels/{train,val}`，YOLO 格式的关键点存储为 `.txt` 文件。

此数据集旨在与 Ultralytics [HUB](https://hub.ultralytics.com/) 和 [YOLO11](https://github.com/ultralytics/ultralytics) 一起使用。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 COCO8-Pose 数据集，`coco8-pose.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml)。

!!! example "ultralytics/cfg/datasets/coco8-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-pose.yaml"
    ```

## 使用方法

要在 COCO8-Pose 数据集上训练 YOLO11n-pose 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

以下是 COCO8-Pose 数据集中的一些图像示例及其对应的标注：

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-5.avif" alt="数据集示例图像" width="800">

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是训练期间使用的一种技术，将多个图像组合成单个图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 COCO8-Pose 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

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

我们感谢 COCO 联盟为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源。有关 COCO 数据集及其创建者的更多信息，请访问 [COCO 数据集网站](https://cocodataset.org/#home)。

## 常见问题

### 什么是 COCO8-Pose 数据集，如何与 Ultralytics YOLO11 一起使用？

COCO8-Pose 数据集是一个小型、多功能的姿态检测数据集，包含 COCO train 2017 集的前 8 张图像，4 张用于训练，4 张用于验证。它专为测试和调试目标检测模型以及实验新的检测方法而设计。此数据集非常适合使用 [Ultralytics YOLO11](../../models/yolo11.md) 进行快速实验。有关数据集配置的更多详细信息，请查看[数据集 YAML 文件](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml)。

### 如何在 Ultralytics 中使用 COCO8-Pose 数据集训练 YOLO11 模型？

要在 COCO8-Pose 数据集上训练 YOLO11n-pose 模型 100 个轮次、图像尺寸为 640，请按照以下示例操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")

        # 训练模型
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

有关训练参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### 使用 COCO8-Pose 数据集有什么好处？

COCO8-Pose 数据集提供多项好处：

- **紧凑尺寸**：只有 8 张图像，易于管理，非常适合快速实验。
- **多样化数据**：尽管规模小，但包含各种场景，可用于全面的流程测试。
- **错误调试**：非常适合在扩展到更大数据集之前识别训练错误和执行健全性检查。

有关其功能和用法的更多信息，请参阅[数据集简介](#简介)部分。

### 马赛克如何有益于使用 COCO8-Pose 数据集的 YOLO11 训练过程？

COCO8-Pose 数据集示例图像中展示的马赛克将多个图像组合成一个，增加每个训练批次中目标和场景的多样性。此技术有助于提高模型对各种目标尺寸、宽高比和上下文的泛化能力，最终增强模型性能。有关示例图像，请参阅[示例图像和标注](#示例图像和标注)部分。

### 在哪里可以找到 COCO8-Pose 数据集 YAML 文件，如何使用它？

COCO8-Pose 数据集 YAML 文件可以在 <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml> 找到。此文件定义了数据集配置，包括路径、类别和其他相关信息。将此文件与 YOLO11 训练脚本一起使用，如[训练示例](#如何在-ultralytics-中使用-coco8-pose-数据集训练-yolo11-模型)部分所示。

有关更多常见问题和详细文档，请访问 [Ultralytics 文档](https://docs.ultralytics.com/)。
