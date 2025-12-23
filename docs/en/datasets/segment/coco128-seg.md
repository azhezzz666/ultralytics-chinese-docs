---
comments: true
description: 探索 Ultralytics 的 COCO128-Seg 数据集，这是一个紧凑但多样化的分割数据集，非常适合测试和训练 YOLO11 模型。
keywords: COCO128-Seg, Ultralytics, 分割数据集, YOLO11, COCO 2017, 模型训练, 计算机视觉, 数据集配置
---

# COCO128-Seg 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO128-Seg 是一个小型但多功能的[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)数据集，由 COCO train 2017 集的前 128 张图像组成。该数据集非常适合测试和调试分割模型，或尝试新的检测方法。只有 128 张图像，它足够小以便于管理，同时又足够多样化，可以测试训练流程中的错误，并在训练更大数据集之前作为健全性检查。

## 数据集结构

- **图像**：共 128 张。默认 YAML 配置将同一目录用于训练和验证，以便快速迭代，但您可以根据需要复制或自定义拆分。
- **类别**：与 COCO 相同的 80 个目标类别。
- **标签**：YOLO 格式的多边形，保存在每张图像旁边的 `labels/{train,val}` 目录中。

该数据集旨在与 Ultralytics [HUB](https://hub.ultralytics.com/) 和 [YOLO11](https://github.com/ultralytics/ultralytics) 配合使用。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 COCO128-Seg 数据集，`coco128-seg.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128-seg.yaml)。

!!! example "ultralytics/cfg/datasets/coco128-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco128-seg.yaml"
    ```

## 使用方法

要在 COCO128-Seg 数据集上训练 YOLO11n-seg 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo segment train data=coco128-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

以下是 COCO128-Seg 数据集中的一些图像示例及其对应的标注：

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-2.avif" alt="数据集示例图像" width="800">

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是训练过程中使用的一种技术，将多张图像组合成单张图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 COCO128-Seg 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

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

### 什么是 COCO128-Seg 数据集，它在 Ultralytics YOLO11 中如何使用？

**COCO128-Seg 数据集**是 Ultralytics 提供的紧凑型实例分割数据集，由 COCO train 2017 集的前 128 张图像组成。该数据集专为测试和调试分割模型或尝试新的检测方法而设计。它特别适合与 Ultralytics [YOLO11](https://github.com/ultralytics/ultralytics) 和 [HUB](https://hub.ultralytics.com/) 配合使用，用于快速迭代和流程错误检查，然后再扩展到更大的数据集。有关详细用法，请参阅模型[训练](../../modes/train.md)页面。

### 如何使用 COCO128-Seg 数据集训练 YOLO11n-seg 模型？

要在 COCO128-Seg 数据集上训练 **YOLO11n-seg** 模型 100 个轮次，图像尺寸为 640，可以使用 Python 或 CLI 命令。以下是快速示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo segment train data=coco128-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

有关可用参数和配置选项的详细说明，请查看[训练](../../modes/train.md)文档。

### 为什么 COCO128-Seg 数据集对模型开发和调试很重要？

**COCO128-Seg 数据集**提供了 128 张图像的平衡组合，兼具可管理性和多样性，非常适合快速测试和调试分割模型或尝试新的检测技术。其适中的规模允许快速训练迭代，同时提供足够的多样性来验证训练流程，然后再扩展到更大的数据集。在 [Ultralytics 分割数据集指南](https://docs.ultralytics.com/datasets/segment/)中了解更多支持的数据集格式。

### 在哪里可以找到 COCO128-Seg 数据集的 YAML 配置文件？

**COCO128-Seg 数据集**的 YAML 配置文件可在 Ultralytics 仓库中获取。您可以直接访问 <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128-seg.yaml>。YAML 文件包含模型训练和验证所需的数据集路径、类别和配置设置等基本信息。

### 在 COCO128-Seg 数据集训练中使用马赛克有什么好处？

在训练过程中使用**马赛克**有助于增加每个训练批次中目标和场景的多样性和变化。这种技术将多张图像组合成单个复合图像，增强模型对场景中不同目标尺寸、宽高比和上下文的泛化能力。马赛克有助于提高模型的鲁棒性和[准确性](https://www.ultralytics.com/glossary/accuracy)，特别是在处理像 COCO128-Seg 这样的中等规模数据集时。有关马赛克图像的示例，请参阅[示例图像和标注](#示例图像和标注)部分。
