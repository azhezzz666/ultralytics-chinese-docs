---
comments: true
description: 探索我们的非洲野生动物数据集，包含水牛、大象、犀牛和斑马的图像，用于训练计算机视觉模型。非常适合研究和保护工作。
keywords: 非洲野生动物数据集, 南非动物, 目标检测, 计算机视觉, YOLO11, 野生动物研究, 保护, 数据集
---

# 非洲野生动物数据集

该数据集展示了南非自然保护区中常见的四种动物类别。它包含非洲野生动物的图像，如水牛、大象、犀牛和斑马，提供了关于它们特征的宝贵见解。该数据集对于训练[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)算法至关重要，有助于在从动物园到森林的各种栖息地中识别动物，并支持野生动物研究。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/biIW5Z6GYl0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO11 进行非洲野生动物检测
</p>

## 数据集结构

非洲野生动物目标检测数据集分为三个子集：

- **训练集**：包含 1052 张图像，每张图像都有相应的标注。
- **验证集**：包含 225 张图像，每张图像都有配对的标注。
- **测试集**：包含 227 张图像，每张图像都有配对的标注。

## 应用

该数据集可应用于各种计算机视觉任务，如[目标检测](https://www.ultralytics.com/glossary/object-detection)、目标跟踪和研究。具体来说，它可用于训练和评估识别图像中非洲野生动物目标的模型，这在野生动物保护、生态研究以及自然保护区和保护区的监测工作中具有应用价值。此外，它还可以作为教育目的的宝贵资源，使学生和研究人员能够研究和了解不同动物物种的特征和行为。

## 数据集 YAML

YAML（Yet Another Markup Language）文件定义了数据集配置，包括路径、类别和其他相关详细信息。对于非洲野生动物数据集，`african-wildlife.yaml` 文件位于 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml)。

!!! example "ultralytics/cfg/datasets/african-wildlife.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/african-wildlife.yaml"
    ```

## 使用方法

要在非洲野生动物数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，请使用提供的代码示例。有关可用参数的完整列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=african-wildlife.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载非洲野生动物微调模型

        # 使用模型进行推理
        results = model.predict("https://ultralytics.com/assets/african-wildlife-sample.jpg")
        ```

    === "CLI"

        ```bash
        # 使用微调的 *.pt 模型开始预测
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/african-wildlife-sample.jpg"
        ```

## 示例图像和标注

非洲野生动物数据集包含各种展示不同动物物种及其自然栖息地的图像。以下是数据集中的图像示例，每张图像都附有相应的标注。

![非洲野生动物数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/african-wildlife-dataset-sample.avif)

- **马赛克图像**：这里展示的是由马赛克数据集图像组成的训练批次。马赛克是一种训练技术，将多张图像合并为一张，丰富批次多样性。这种方法有助于增强模型在不同目标尺寸、宽高比和上下文中的泛化能力。

此示例说明了非洲野生动物数据集中图像的多样性和复杂性，强调了在训练过程中包含马赛克技术的好处。

## 引用、许可和致谢

我们要感谢原始数据集作者 [Bianca Ferreira](https://www.kaggle.com/biancaferreira/datasets) 向社区发布此数据集。Ultralytics 团队已在内部更新和调整了该数据集，使其能够与 [Ultralytics YOLO](https://www.ultralytics.com/yolo) 模型无缝配合使用。该数据集在 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)下提供。

如果您在研究中使用此数据集，请使用以下详细信息进行引用：

!!! quote ""

    === "BibTeX"

        ```bibtex

        @dataset{Ferreira_African_Wildlife_Ultralytics_Adaptation_2024,
            author  = {Ferreira, Bianca},
            title   = {African Wildlife Detection Dataset (Ultralytics YOLO Adaptation)},
            url     = {https://docs.ultralytics.com/datasets/detect/african-wildlife/},
            note    = {Original dataset by Bianca Ferreira; adapted for Ultralytics YOLO by Glenn Jocher and Muhammad Rizwan Munawar},
            license = {AGPL-3.0},
            version = {1.0.0},
            year    = {2024}
        }
        ```

## 常见问题

### 什么是非洲野生动物数据集，如何在计算机视觉项目中使用？

非洲野生动物数据集包含南非自然保护区中常见的四种动物物种的图像：水牛、大象、犀牛和斑马。它是训练目标检测和动物识别计算机视觉算法的宝贵资源。该数据集支持各种任务，如目标跟踪、研究和保护工作。有关其结构和应用的更多信息，请参阅[数据集结构](#数据集结构)部分和数据集的[应用](#应用)部分。

### 如何使用非洲野生动物数据集训练 YOLO11 模型？

您可以使用 `african-wildlife.yaml` 配置文件在非洲野生动物数据集上训练 YOLO11 模型。以下是如何使用图像尺寸 640 训练 YOLO11n 模型 100 个训练周期的示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=african-wildlife.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关其他训练参数和选项，请参阅[训练](../../modes/train.md)文档。

### 在哪里可以找到非洲野生动物数据集的 YAML 配置文件？

非洲野生动物数据集的 YAML 配置文件名为 `african-wildlife.yaml`，可在[此 GitHub 链接](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml)找到。此文件定义了数据集配置，包括路径、类别和其他对训练[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型至关重要的详细信息。有关更多详情，请参阅[数据集 YAML](#数据集-yaml) 部分。

### 我可以查看非洲野生动物数据集的示例图像和标注吗？

是的，非洲野生动物数据集包含各种展示不同动物物种在其自然栖息地的图像。您可以在[示例图像和标注](#示例图像和标注)部分查看示例图像及其相应的标注。该部分还说明了使用马赛克技术将多张图像合并为一张以丰富批次多样性的方法，从而增强模型的泛化能力。

### 非洲野生动物数据集如何用于支持野生动物保护和研究？

非洲野生动物数据集非常适合通过训练和评估模型来识别不同栖息地中的非洲野生动物，从而支持野生动物保护和研究。这些模型可以帮助[监测动物种群](https://docs.ultralytics.com/solutions/)、研究它们的行为并识别保护需求。此外，该数据集还可用于教育目的，帮助学生和研究人员了解不同动物物种的特征和行为。更多详情请参阅[应用](#应用)部分。
