---
comments: true
description: 探索 Ultralytics COCO8-Multispectral 数据集，这是 COCO8 的增强版本，具有插值光谱通道，非常适合测试多光谱目标检测模型和训练流水线。
keywords: COCO8-Multispectral, Ultralytics, 数据集, 多光谱, 目标检测, YOLO11, 训练, 验证, 机器学习, 计算机视觉
---

# COCO8-Multispectral 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO8-Multispectral 数据集是原始 COCO8 数据集的高级变体，旨在促进多光谱目标检测模型的实验。它由 COCO train 2017 集中相同的 8 张图像组成——4 张用于训练，4 张用于验证——但每张图像都转换为 10 通道多光谱格式。通过扩展超越标准 RGB 通道，COCO8-Multispectral 使开发和评估能够利用更丰富光谱信息的模型成为可能。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/coco8-multispectral-overview.avif" alt="多光谱图像概述">
</p>

COCO8-Multispectral 与 [Ultralytics HUB](https://hub.ultralytics.com/) 和 [YOLO11](../../models/yolo11.md) 完全兼容，确保无缝集成到您的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)工作流程中。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yw2Fo6qjJU4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在多光谱数据集上训练 Ultralytics YOLO11 | 多通道视觉 AI 🚀
</p>

## 数据集生成

COCO8-Multispectral 中的多光谱图像是通过在可见光谱内 10 个均匀间隔的光谱通道上插值原始 RGB 图像创建的。该过程包括：

- **波长分配**：为 RGB 通道分配标称波长——红色：650 nm，绿色：510 nm，蓝色：475 nm。
- **插值**：使用线性插值估计 450 nm 和 700 nm 之间中间波长的像素值，产生 10 个光谱通道。
- **外推**：使用 SciPy 的 `interp1d` 函数进行外推，以估计超出原始 RGB 波长的值，确保完整的光谱表示。

这种方法模拟了多光谱成像过程，为模型训练和评估提供了更多样化的数据集。有关多光谱成像的更多阅读，请参阅[多光谱成像维基百科文章](https://en.wikipedia.org/wiki/Multispectral_imaging)。


## 数据集 YAML

COCO8-Multispectral 数据集使用 YAML 文件配置，该文件定义数据集路径、类名和基本元数据。您可以在 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-multispectral.yaml)中查看官方 `coco8-multispectral.yaml` 文件。

!!! example "ultralytics/cfg/datasets/coco8-multispectral.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-multispectral.yaml"
    ```

!!! note

    以 `(channel, height, width)` 顺序准备您的 TIFF 图像，使用 `.tiff` 或 `.tif` 扩展名保存，并确保它们是 `uint8` 格式以便与 Ultralytics 一起使用：

    ```python
    import cv2
    import numpy as np

    # 创建并写入 10 通道 TIFF
    image = np.ones((10, 640, 640), dtype=np.uint8)  # CHW 顺序
    cv2.imwritemulti("example.tiff", image)

    # 读取 TIFF
    success, frames_list = cv2.imreadmulti("example.tiff")
    image = np.stack(frames_list, axis=2)
    print(image.shape)  # (640, 640, 10)  HWC 顺序用于训练和推理
    ```

## 使用方法

要在 COCO8-Multispectral 数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，请使用以下示例。有关训练选项的完整列表，请参阅 [YOLO 训练文档](../../modes/train.md)。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 在 COCO8-Multispectral 上训练模型
        results = model.train(data="coco8-multispectral.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 使用命令行在 COCO8-Multispectral 上训练 YOLO11n
        yolo detect train data=coco8-multispectral.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关模型选择和最佳实践的更多详情，请探索 [Ultralytics YOLO 模型文档](../../models/yolo11.md)和 [YOLO 模型训练技巧指南](https://docs.ultralytics.com/guides/model-training-tips/)。

## 示例图像和标注

以下是 COCO8-Multispectral 数据集中马赛克训练批次的示例：

<img src="https://github.com/ultralytics/docs/releases/download/0/coco8-multispectral-mosaic-batch.avif" alt="数据集示例图像" width="800">

- **马赛克图像**：此图像展示了使用[马赛克增强](https://docs.ultralytics.com/reference/data/augment/)将多个数据集图像组合在一起的训练批次。马赛克增强增加了每个批次中目标和场景的多样性，帮助模型更好地泛化到各种目标尺寸、宽高比和背景。

这种技术对于像 COCO8-Multispectral 这样的小型数据集特别有价值，因为它在训练期间最大化了每张图像的效用。

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

特别感谢 [COCO 联盟](https://cocodataset.org/#home)对[计算机视觉社区](https://www.ultralytics.com/blog/a-history-of-vision-models)的持续贡献。

## 常见问题

### Ultralytics COCO8-Multispectral 数据集用于什么？

Ultralytics COCO8-Multispectral 数据集专为快速测试和调试[多光谱目标检测](https://www.ultralytics.com/glossary/object-detection)模型而设计。仅包含 8 张图像（4 张用于训练，4 张用于验证），它非常适合验证您的 [YOLO](../../models/yolo11.md) 训练流水线，并确保在扩展到更大数据集之前一切正常工作。有关更多可实验的数据集，请访问 [Ultralytics 数据集目录](https://docs.ultralytics.com/datasets/)。

### 多光谱数据如何改进目标检测？

多光谱数据提供超越标准 RGB 的额外光谱信息，使模型能够根据不同波长的反射率细微差异来区分目标。这可以提高检测准确性，特别是在具有挑战性的场景中。了解更多关于[多光谱成像](https://en.wikipedia.org/wiki/Multispectral_imaging)及其在[高级计算机视觉](https://www.ultralytics.com/blog/ai-in-aviation-a-runway-to-smarter-airports)中的应用。

### COCO8-Multispectral 与 Ultralytics HUB 和 YOLO 模型兼容吗？

是的，COCO8-Multispectral 与 [Ultralytics HUB](https://hub.ultralytics.com/) 和所有 [YOLO 模型](../../models/yolo11.md)（包括最新的 YOLO11）完全兼容。这使您可以轻松地将数据集集成到训练和验证工作流程中。

### 在哪里可以找到有关数据增强技术的更多信息？

要深入了解马赛克等数据增强方法及其对模型性能的影响，请参阅 [YOLO 数据增强指南](https://docs.ultralytics.com/guides/yolo-data-augmentation/)和 [Ultralytics 数据增强博客](https://www.ultralytics.com/blog/the-ultimate-guide-to-data-augmentation-in-2025)。

### 我可以将 COCO8-Multispectral 用于基准测试或教育目的吗？

当然可以！COCO8-Multispectral 的小巧规模和多光谱特性使其非常适合基准测试、教育演示和原型设计新模型架构。有关更多基准测试数据集，请参阅 [Ultralytics 基准数据集集合](https://docs.ultralytics.com/datasets/)。
