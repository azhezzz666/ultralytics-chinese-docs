---
comments: true
description: 探索 Ultralytics COCO8 数据集，这是一个包含 8 张图像的多功能且易于管理的数据集，非常适合测试目标检测模型和训练流水线。
keywords: COCO8, Ultralytics, 数据集, 目标检测, YOLO11, 训练, 验证, 机器学习, 计算机视觉
---

# COCO8 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO8 数据集是一个紧凑但功能强大的[目标检测](https://www.ultralytics.com/glossary/object-detection)数据集，由 COCO train 2017 集的前 8 张图像组成——4 张用于训练，4 张用于验证。该数据集专为快速测试、调试和使用 [YOLO](https://docs.ultralytics.com/models/yolo11/) 模型和训练流水线进行实验而设计。其小巧的规模使其高度易于管理，同时其多样性确保它可以作为扩展到更大数据集之前的有效健全性检查。

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

COCO8 与 [Ultralytics HUB](https://hub.ultralytics.com/) 和 [YOLO11](../../models/yolo11.md) 完全兼容，可无缝集成到您的计算机视觉工作流程中。

## 数据集 YAML

COCO8 数据集配置在 YAML（Yet Another Markup Language）文件中定义，该文件指定数据集路径、类名和其他基本元数据。您可以在 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml)中查看官方 `coco8.yaml` 文件。

!!! example "ultralytics/cfg/datasets/coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```


## 使用方法

要在 COCO8 数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，请使用以下示例。有关训练选项的完整列表，请参阅 [YOLO 训练文档](../../modes/train.md)。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 在 COCO8 上训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 使用命令行在 COCO8 上训练 YOLO11n
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

以下是 COCO8 数据集中马赛克训练批次的示例：

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-1.avif" alt="数据集示例图像" width="800">

- **马赛克图像**：此图像展示了使用马赛克增强将多个数据集图像组合在一起的训练批次。马赛克增强增加了每个批次中目标和场景的多样性，帮助模型更好地泛化到各种目标尺寸、宽高比和背景。

这种技术对于像 COCO8 这样的小型数据集特别有用，因为它在训练期间最大化了每张图像的价值。

## 引用和致谢

如果您在研究或开发中使用 COCO 数据集，请引用以下论文：

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

特别感谢 [COCO 联盟](https://cocodataset.org/#home)对[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区的持续贡献。

## 常见问题

### Ultralytics COCO8 数据集用于什么？

Ultralytics COCO8 数据集专为快速测试和调试[目标检测](https://www.ultralytics.com/glossary/object-detection)模型而设计。仅包含 8 张图像（4 张用于训练，4 张用于验证），它非常适合验证您的 [YOLO](https://docs.ultralytics.com/models/yolo11/) 训练流水线，并确保在扩展到更大数据集之前一切正常工作。探索 [COCO8 YAML 配置](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml)了解更多详情。

### 如何使用 COCO8 数据集训练 YOLO11 模型？

您可以使用 Python 或 CLI 在 COCO8 上训练 YOLO11 模型：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 在 COCO8 上训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关其他训练选项，请参阅 [YOLO 训练文档](../../modes/train.md)。

### 为什么应该使用 Ultralytics HUB 管理我的 COCO8 训练？

[Ultralytics HUB](https://hub.ultralytics.com/) 简化了 [YOLO](https://docs.ultralytics.com/models/yolo11/) 模型（包括 COCO8）的数据集管理、训练和部署。借助云训练、实时监控和直观的数据集处理等功能，HUB 使您能够一键启动实验，消除手动设置的麻烦。了解更多关于 [Ultralytics HUB](https://hub.ultralytics.com/) 以及它如何加速您的计算机视觉项目。

### 在 COCO8 数据集训练中使用马赛克增强有什么好处？

COCO8 训练中使用的马赛克增强在每个批次中将多张图像组合成一张。这增加了目标和背景的多样性，帮助您的 [YOLO](https://docs.ultralytics.com/models/yolo11/) 模型更好地泛化到新场景。马赛克增强对于小型数据集特别有价值，因为它最大化了每个训练步骤中可用的信息。有关更多信息，请参阅[使用方法](#使用方法)。

### 如何验证在 COCO8 数据集上训练的 YOLO11 模型？

要在 COCO8 上训练后验证您的 YOLO11 模型，请在 Python 或 CLI 中使用模型的验证命令。这将使用标准指标评估模型的性能。有关分步说明，请访问 [YOLO 验证文档](../../modes/val.md)。
