---
comments: true
description: 探索 Ultralytics COCO128 数据集，这是一个包含 128 张图像的多功能且易于管理的数据集，非常适合测试目标检测模型和训练流水线。
keywords: COCO128, Ultralytics, 数据集, 目标检测, YOLO11, 训练, 验证, 机器学习, 计算机视觉
---

# COCO128 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO128 是一个小型但多功能的[目标检测](https://www.ultralytics.com/glossary/object-detection)数据集，由 COCO train 2017 集的前 128 张图像组成。该数据集非常适合测试和调试目标检测模型，或用于实验新的检测方法。包含 128 张图像，它足够小以便于管理，同时又足够多样化以测试训练流水线的错误，并在训练更大数据集之前作为健全性检查。

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

该数据集旨在与 Ultralytics [HUB](https://hub.ultralytics.com/) 和 [YOLO11](https://github.com/ultralytics/ultralytics) 一起使用。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含有关数据集路径、类别和其他相关信息。对于 COCO128 数据集，`coco128.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml)。

!!! example "ultralytics/cfg/datasets/coco128.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco128.yaml"
    ```


## 使用方法

要在 COCO128 数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco128.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

以下是 COCO128 数据集中的一些图像示例及其相应的标注：

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-1.avif" alt="数据集示例图像" width="800">

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是一种在训练期间使用的技术，将多张图像合并为一张图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 COCO128 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克技术的好处。

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

### Ultralytics COCO128 数据集用于什么？

Ultralytics COCO128 数据集是一个紧凑的子集，包含 COCO train 2017 数据集的前 128 张图像。它主要用于测试和调试[目标检测](https://www.ultralytics.com/glossary/object-detection)模型、实验新的检测方法以及在扩展到更大数据集之前验证训练流水线。其易于管理的规模使其非常适合快速迭代，同时仍提供足够的多样性作为有意义的测试用例。

### 如何使用 COCO128 数据集训练 YOLO11 模型？

要在 COCO128 数据集上训练 YOLO11 模型，您可以使用 Python 或 CLI 命令。以下是方法：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 训练模型
        results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        yolo detect train data=coco128.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关更多训练选项和参数，请参阅[训练](../../modes/train.md)文档。

### 在 COCO128 上使用马赛克增强有什么好处？

如示例图像所示，马赛克增强在训练期间将多张训练图像组合成一张复合图像。这种技术在使用 COCO128 训练时提供了几个好处：

- 增加每个训练批次中目标和上下文的多样性
- 提高模型在不同目标尺寸和宽高比上的泛化能力
- 增强对各种尺度目标的检测性能
- 通过创建更多样化的训练样本最大化小型数据集的效用

这种技术对于像 COCO128 这样的小型数据集特别有价值，帮助模型从有限的数据中学习更鲁棒的特征。

### COCO128 与其他 COCO 数据集变体相比如何？

COCO128（128 张图像）在规模上介于 [COCO8](../detect/coco8.md)（8 张图像）和完整 [COCO](../detect/coco.md) 数据集（11.8 万张以上图像）之间：

- **COCO8**：仅包含 8 张图像（4 张训练，4 张验证）——非常适合快速测试和调试
- **COCO128**：包含 128 张图像——在规模和多样性之间取得平衡
- **完整 COCO**：包含 11.8 万张以上训练图像——全面但资源密集

COCO128 提供了一个很好的中间地带，比 COCO8 提供更多多样性，同时比完整 COCO 数据集更易于管理，适合实验和初始模型开发。

### 我可以将 COCO128 用于目标检测以外的任务吗？

虽然 COCO128 主要为目标检测设计，但数据集的标注可以适应其他计算机视觉任务：

- **实例分割**：使用标注中提供的分割掩码
- **关键点检测**：用于包含带有关键点标注的人物图像
- **迁移学习**：作为微调自定义任务模型的起点

对于像[分割](../../tasks/segment.md)这样的专门任务，请考虑使用专门构建的变体，如 [COCO8-seg](../segment/coco8-seg.md)，其中包含适当的标注。
