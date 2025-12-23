---
comments: true
description: 探索 SKU-110k 数据集，包含密集排列的零售货架图像，非常适合训练和评估目标检测任务中的深度学习模型。
keywords: SKU-110k, 数据集, 目标检测, 零售货架图像, 深度学习, 计算机视觉, 模型训练
---

# SKU-110k 数据集

[SKU-110k](https://github.com/eg4000/SKU110K_CVPR19) 数据集是一个密集排列的零售货架图像集合，旨在支持[目标检测](https://www.ultralytics.com/glossary/object-detection)任务的研究。该数据集由 Eran Goldman 等人开发，包含超过 110,000 个独特的库存单位（SKU）类别，其中密集排列的物体通常外观相似甚至相同，且位置相邻。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_gRqR-miFPE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 在 SKU-110k 数据集上训练 YOLOv10 | 零售数据集
</p>

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/densely-packed-retail-shelf.avif)

## 主要特点

- SKU-110k 包含来自世界各地商店货架的图像，其中密集排列的物体对最先进的目标检测器构成挑战。
- 数据集包含超过 110,000 个独特的 SKU 类别，提供多样化的物体外观。
- 标注包括物体的边界框和 SKU 类别标签。

## 数据集结构

SKU-110k 数据集分为三个主要子集：

1. **训练集**：该子集包含 8,219 张图像和标注，用于训练目标检测模型。
2. **验证集**：该子集包含 588 张图像和标注，用于训练过程中的模型验证。
3. **测试集**：该子集包含 2,936 张图像，用于训练后目标检测模型的最终评估。

## 应用场景

SKU-110k 数据集广泛用于训练和评估目标检测任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，特别是在零售货架展示等密集场景中。其应用包括：

- 零售库存管理和自动化
- 电商平台的产品识别
- 货架陈列合规性验证
- 商店自助结账系统
- 仓库机器人拣选和分拣

数据集多样化的 SKU 类别和密集排列的物体布局使其成为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究人员和从业者的宝贵资源。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 SKU-110K 数据集，`SKU-110K.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml)。

!!! example "ultralytics/cfg/datasets/SKU-110K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/SKU-110K.yaml"
    ```

## 使用方法

要在 SKU-110K 数据集上训练 YOLO11n 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=SKU-110K.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例数据和标注

SKU-110k 数据集包含多样化的零售货架图像，其中物体密集排列，为目标检测任务提供丰富的上下文。以下是数据集中的一些数据示例及其对应的标注：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/densely-packed-retail-shelf-1.avif)

- **密集排列的零售货架图像**：此图像展示了零售货架环境中密集排列物体的示例。物体使用边界框和 SKU 类别标签进行标注。

该示例展示了 SKU-110k 数据集中数据的多样性和复杂性，并强调了高质量数据对目标检测任务的重要性。产品的密集排列对检测算法提出了独特的挑战，使该数据集对于开发强大的零售专用计算机视觉解决方案特别有价值。

## 引用和致谢

如果您在研究或开发工作中使用 SKU-110k 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{goldman2019dense,
          author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
          title     = {Precise Detection in Densely Packed Scenes},
          booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
          year      = {2019}
        }
        ```

我们感谢 Eran Goldman 等人创建和维护 SKU-110k 数据集，为计算机视觉研究社区提供了宝贵的资源。有关 SKU-110k 数据集及其创建者的更多信息，请访问 [SKU-110k 数据集 GitHub 仓库](https://github.com/eg4000/SKU110K_CVPR19)。

## 常见问题

### 什么是 SKU-110k 数据集，为什么它对目标检测很重要？

SKU-110k 数据集由密集排列的零售货架图像组成，旨在帮助目标检测任务的研究。该数据集由 Eran Goldman 等人开发，包含超过 110,000 个独特的 SKU 类别。其重要性在于它能够以多样化的物体外观和相邻位置挑战最先进的目标检测器，使其成为计算机视觉研究人员和从业者的宝贵资源。在我们的 [SKU-110k 数据集](#sku-110k-数据集)部分了解更多关于数据集结构和应用的信息。

### 如何使用 SKU-110k 数据集训练 YOLO11 模型？

在 SKU-110k 数据集上训练 YOLO11 模型非常简单。以下是训练 YOLO11n 模型 100 个轮次、图像尺寸为 640 的示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=SKU-110K.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### SKU-110k 数据集的主要子集有哪些？

SKU-110k 数据集分为三个主要子集：

1. **训练集**：包含 8,219 张图像和标注，用于训练目标检测模型。
2. **验证集**：包含 588 张图像和标注，用于训练过程中的模型验证。
3. **测试集**：包含 2,936 张图像，用于训练后目标检测模型的最终评估。

有关更多详细信息，请参阅[数据集结构](#数据集结构)部分。

### 如何配置 SKU-110k 数据集进行训练？

SKU-110k 数据集配置在 YAML 文件中定义，其中包含数据集路径、类别和其他相关信息的详细信息。`SKU-110K.yaml` 文件维护在 [SKU-110K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml)。例如，您可以使用此配置进行模型训练，如我们的[使用方法](#使用方法)部分所示。

### 在[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)背景下，SKU-110k 数据集的主要特点是什么？

SKU-110k 数据集的特点是包含来自世界各地商店货架的图像，展示了对目标检测器构成重大挑战的密集排列物体：

- 超过 110,000 个独特的 SKU 类别
- 多样化的物体外观
- 标注包括边界框和 SKU 类别标签

这些特点使 SKU-110k 数据集对于训练和评估目标检测任务中的深度学习模型特别有价值。有关更多详细信息，请参阅[主要特点](#主要特点)部分。

### 如何在研究中引用 SKU-110k 数据集？

如果您在研究或开发工作中使用 SKU-110k 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{goldman2019dense,
          author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
          title     = {Precise Detection in Densely Packed Scenes},
          booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
          year      = {2019}
        }
        ```

有关数据集的更多信息，请参阅[引用和致谢](#引用和致谢)部分。
