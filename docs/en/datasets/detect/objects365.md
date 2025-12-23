---
comments: true
description: 探索 Objects365 数据集，包含 200 万张图像和 3000 万个边界框，涵盖 365 个类别。使用多样化、高质量的数据增强您的目标检测模型。
keywords: Objects365 数据集, 目标检测, 机器学习, 深度学习, 计算机视觉, 标注图像, 边界框, YOLO11, 高分辨率图像, 数据集配置
---

# Objects365 数据集

[Objects365](https://www.objects365.org/) 数据集是一个大规模、高质量的数据集，旨在促进目标检测研究，专注于野外多样化目标。该数据集由 [Megvii](https://en.megvii.com/) 研究团队创建，提供广泛的高分辨率图像，具有涵盖 365 个目标类别的全面标注边界框集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/J-RH22rwx1A"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 在 Objects365 数据集上训练 Ultralytics YOLO11 | 200 万标注 🚀
</p>

## 主要特点

- Objects365 包含 365 个目标类别，200 万张图像和超过 3000 万个边界框。
- 该数据集包含各种场景中的多样化目标，为目标检测任务提供丰富且具有挑战性的基准。
- 标注包括目标的边界框，使其适合训练和评估目标检测模型。
- Objects365 预训练模型显著优于 ImageNet 预训练模型，在各种任务上具有更好的泛化能力。

## 数据集结构

Objects365 数据集组织为单一图像集及相应的标注：

- **图像**：该数据集包含 200 万张高分辨率图像，每张图像包含 365 个类别中的各种目标。
- **标注**：图像标注了超过 3000 万个边界框，为[目标检测](https://docs.ultralytics.com/tasks/detect/)任务提供全面的真实标签信息。

## 应用

Objects365 数据集广泛用于训练和评估目标检测任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。该数据集多样化的目标类别和高质量标注使其成为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究人员和从业者的宝贵资源。


## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含有关数据集路径、类别和其他相关信息。对于 Objects365 数据集，`Objects365.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml)。

!!! example "ultralytics/cfg/datasets/Objects365.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/Objects365.yaml"
    ```

## 使用方法

要在 Objects365 数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="Objects365.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=Objects365.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例数据和标注

Objects365 数据集包含来自 365 个类别目标的多样化高分辨率图像集，为[目标检测](https://www.ultralytics.com/glossary/object-detection)任务提供丰富的上下文。以下是数据集中的一些图像示例：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/objects365-sample-image.avif)

- **Objects365**：此图像展示了目标检测的示例，其中目标用边界框标注。该数据集提供广泛的图像以促进此任务模型的开发。

该示例展示了 Objects365 数据集中数据的多样性和复杂性，并强调了准确目标检测对计算机视觉应用的重要性。

## 引用和致谢

如果您在研究或开发工作中使用 Objects365 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{shao2019objects365,
          title={Objects365: A Large-scale, High-quality Dataset for Object Detection},
          author={Shao, Shuai and Li, Zeming and Zhang, Tianyuan and Peng, Chao and Yu, Gang and Li, Jing and Zhang, Xiangyu and Sun, Jian},
          booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
          pages={8425--8434},
          year={2019}
        }
        ```

我们感谢创建和维护 Objects365 数据集的研究团队，这是计算机视觉研究社区的宝贵资源。有关 Objects365 数据集及其创建者的更多信息，请访问 [Objects365 数据集网站](https://www.objects365.org/)。

## 常见问题

### Objects365 数据集用于什么？

[Objects365 数据集](https://www.objects365.org/)专为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉中的目标检测任务而设计。它提供了一个大规模、高质量的数据集，包含 200 万张标注图像和 365 个类别的 3000 万个边界框。利用如此多样化的数据集有助于提高目标检测模型的性能和泛化能力，使其对该领域的研究和开发非常有价值。

### 如何使用 Objects365 数据集训练 YOLO11 模型？

要使用 Objects365 数据集训练 YOLO11n 模型 100 个训练周期，图像尺寸为 640，请按照以下说明操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="Objects365.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=Objects365.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关可用参数的完整列表，请参阅[训练](../../modes/train.md)页面。

### 为什么应该在目标检测项目中使用 Objects365 数据集？

Objects365 数据集为目标检测任务提供了几个优势：

1. **多样性**：它包含 200 万张图像，涵盖 365 个类别的多样化场景中的目标。
2. **高质量标注**：超过 3000 万个边界框提供全面的真实标签数据。
3. **性能**：在 Objects365 上预训练的模型显著优于在 [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) 等数据集上训练的模型，具有更好的泛化能力。

### 在哪里可以找到 Objects365 数据集的 YAML 配置文件？

Objects365 数据集的 YAML 配置文件可在 [Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml) 获取。此文件包含数据集路径和类别标签等基本信息，对于设置训练环境至关重要。

### Objects365 的数据集结构如何增强目标检测建模？

[Objects365 数据集](https://www.objects365.org/)组织有 200 万张高分辨率图像和超过 3000 万个边界框的全面标注。这种结构确保了用于训练目标检测[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型的鲁棒数据集，提供广泛的目标和场景。这种多样性和数量有助于开发更准确且能够很好地泛化到实际应用的模型。有关数据集结构的更多详情，请参阅[数据集 YAML](#数据集-yaml) 部分。
