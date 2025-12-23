---
comments: true
description: 探索 PASCAL VOC 数据集，这是目标检测、分割和分类的重要数据集。了解主要特点、应用和使用技巧。
keywords: PASCAL VOC, VOC 数据集, 目标检测, 分割, 分类, YOLO, Faster R-CNN, Mask R-CNN, 图像标注, 计算机视觉
---

# VOC 数据集

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)（Visual Object Classes）数据集是一个著名的目标检测、分割和分类数据集。它旨在鼓励对各种目标类别的研究，通常用于计算机视觉模型的基准测试。对于从事目标检测、分割和分类任务的研究人员和开发者来说，这是一个必不可少的数据集。

## 主要特点

- VOC 数据集包括两个主要挑战：VOC2007 和 VOC2012。
- 数据集包含 20 个目标类别，包括汽车、自行车和动物等常见物体，以及船、沙发和餐桌等更具体的类别。
- 标注包括目标检测和分类任务的目标边界框和类别标签，以及分割任务的分割掩码。
- VOC 提供标准化的评估指标，如目标检测和分类的[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP），使其适合比较模型性能。

## 数据集结构

VOC 数据集分为三个子集：

1. **训练集**：该子集包含用于训练目标检测、分割和分类模型的图像。
2. **验证集**：该子集包含用于模型训练期间验证的图像。
3. **测试集**：该子集包含用于测试和基准测试训练模型的图像。该子集的真实标注不公开，结果需提交到 [PASCAL VOC 评估服务器](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php)进行性能评估。

## 应用场景

VOC 数据集广泛用于训练和评估目标检测（如 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/)、[Faster R-CNN](https://arxiv.org/abs/1506.01497) 和 [SSD](https://arxiv.org/abs/1512.02325)）、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)（如 [Mask R-CNN](https://arxiv.org/abs/1703.06870)）和[图像分类](https://www.ultralytics.com/glossary/image-classification)中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。数据集多样化的目标类别、大量标注图像和标准化评估指标使其成为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)研究人员和从业者的重要资源。


## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 VOC 数据集，`VOC.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml)。

!!! example "ultralytics/cfg/datasets/VOC.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/VOC.yaml"
    ```

## 使用方法

要在 VOC 数据集上训练 YOLO11n 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="VOC.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=VOC.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

VOC 数据集包含具有各种目标类别和复杂场景的多样化图像集。以下是数据集中的一些图像示例及其对应的标注：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/mosaiced-voc-dataset-sample.avif)

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是训练期间使用的一种技术，将多个图像组合成单个图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 VOC 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

## 引用和致谢

如果您在研究或开发工作中使用 VOC 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{everingham2010pascal,
              title={The PASCAL Visual Object Classes (VOC) Challenge},
              author={Mark Everingham and Luc Van Gool and Christopher K. I. Williams and John Winn and Andrew Zisserman},
              year={2010},
              eprint={0909.5206},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们感谢 PASCAL VOC 联盟为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源。有关 VOC 数据集及其创建者的更多信息，请访问 [PASCAL VOC 数据集网站](http://host.robots.ox.ac.uk/pascal/VOC/)。

## 常见问题

### 什么是 PASCAL VOC 数据集，为什么它对计算机视觉任务很重要？

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)（Visual Object Classes）数据集是计算机视觉中[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和分类的著名基准数据集。它包含全面的标注，如边界框、类别标签和分割掩码，涵盖 20 个不同的目标类别。研究人员广泛使用它来评估 Faster R-CNN、YOLO 和 Mask R-CNN 等模型的性能，因为它具有标准化的评估指标，如平均精度均值（mAP）。

### 如何使用 VOC 数据集训练 YOLO11 模型？

要使用 VOC 数据集训练 YOLO11 模型，您需要在 YAML 文件中配置数据集。以下是训练 YOLO11n 模型 100 个轮次、图像尺寸为 640 的示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="VOC.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=VOC.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

### VOC 数据集包含哪些主要挑战？

VOC 数据集包括两个主要挑战：VOC2007 和 VOC2012。这些挑战测试 20 个不同目标类别的目标检测、分割和分类。每张图像都经过精心标注，包含边界框、类别标签和分割掩码。这些挑战提供标准化指标（如 mAP），便于比较和基准测试不同的计算机视觉模型。

### PASCAL VOC 数据集如何增强模型基准测试和评估？

PASCAL VOC 数据集通过其详细的标注和标准化指标（如平均[精度](https://www.ultralytics.com/glossary/precision)均值（mAP））增强模型基准测试和评估。这些指标对于评估目标检测和分类模型的性能至关重要。数据集多样化和复杂的图像确保了在各种真实场景中进行全面的模型评估。

### 如何在 YOLO 模型中使用 VOC 数据集进行[语义分割](https://www.ultralytics.com/glossary/semantic-segmentation)？

要将 VOC 数据集用于 YOLO 模型的语义分割任务，您需要在 YAML 文件中正确配置数据集。YAML 文件定义了训练分割模型所需的路径和类别。查看 [VOC.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml) 的 VOC 数据集 YAML 配置文件以获取详细设置。对于分割任务，您需要使用分割专用模型（如 `yolo11n-seg.pt`）而不是检测模型。
