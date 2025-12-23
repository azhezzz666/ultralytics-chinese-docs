---
comments: true
description: 探索 CIFAR-100 数据集，包含 100 个类别的 60,000 张 32x32 彩色图像。适用于机器学习和计算机视觉任务。
keywords: CIFAR-100, 数据集, 机器学习, 计算机视觉, 图像分类, 深度学习, YOLO, 训练, 测试, Alex Krizhevsky
---

# CIFAR-100 数据集

[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)（加拿大高级研究院）数据集是 CIFAR-10 数据集的重要扩展，由 100 个不同类别的 60,000 张 32x32 彩色图像组成。它由 CIFAR 研究所的研究人员开发，为更复杂的机器学习和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务提供了更具挑战性的数据集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/6bZeCs0xwO4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何在 CIFAR-100 上训练 Ultralytics YOLO11 | 分步图像分类教程 🚀
</p>

## 主要特点

- CIFAR-100 数据集包含 60,000 张图像，分为 100 个类别。
- 每个类别包含 600 张图像，其中 500 张用于训练，100 张用于测试。
- 图像为彩色，尺寸为 32x32 像素。
- 100 个不同类别被分组为 20 个粗类别，用于更高级别的分类。
- CIFAR-100 在机器学习和计算机视觉领域被广泛用于训练和测试。

## 数据集结构

CIFAR-100 数据集分为两个子集：

1. **训练集**：该子集包含 50,000 张用于训练机器学习模型的图像。
2. **测试集**：该子集包含 10,000 张用于测试和基准评估已训练模型的图像。

## 应用场景

CIFAR-100 数据集被广泛用于训练和评估[图像分类](https://www.ultralytics.com/glossary/image-classification)任务中的深度学习模型，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）以及各种其他机器学习算法。该数据集在类别方面的多样性以及彩色图像的存在，使其成为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉领域研究和开发的更具挑战性和全面性的数据集。

## 使用方法

要在 CIFAR-100 数据集上使用 32x32 的图像尺寸训练 YOLO 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="cifar100", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=cifar100 model=yolo11n-cls.pt epochs=100 imgsz=32
        ```

## 示例图像和标注

CIFAR-100 数据集包含各种目标的彩色图像，为图像分类任务提供了结构良好的数据集。以下是数据集中的一些图像示例：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/cifar100-sample-image.avif)

该示例展示了 CIFAR-100 数据集中目标的多样性和复杂性，强调了多样化数据集对于训练鲁棒图像分类模型的重要性。

## 引用和致谢

如果您在研究或开发工作中使用 CIFAR-100 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @TECHREPORT{Krizhevsky09learningmultiple,
                    author={Alex Krizhevsky},
                    title={Learning multiple layers of features from tiny images},
                    institution={},
                    year={2009}
        }
        ```

我们感谢 Alex Krizhevsky 创建并维护 CIFAR-100 数据集，为机器学习和计算机视觉研究社区提供了宝贵的资源。有关 CIFAR-100 数据集及其创建者的更多信息，请访问 [CIFAR-100 数据集网站](https://www.cs.toronto.edu/~kriz/cifar.html)。

## 常见问题

### 什么是 CIFAR-100 数据集，为什么它很重要？

[CIFAR-100 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)是一个包含 60,000 张 32x32 彩色图像的大型集合，分为 100 个类别。它由加拿大高级研究院（CIFAR）开发，提供了一个具有挑战性的数据集，非常适合复杂的机器学习和计算机视觉任务。其重要性在于类别的多样性和图像的小尺寸，使其成为训练和测试[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型（如卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)（CNN））的宝贵资源，可使用 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/) 等框架。

### 如何在 CIFAR-100 数据集上训练 YOLO 模型？

您可以使用 Python 或 CLI 命令在 CIFAR-100 数据集上训练 YOLO 模型。以下是方法：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="cifar100", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=cifar100 model=yolo11n-cls.pt epochs=100 imgsz=32
        ```

有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### CIFAR-100 数据集的主要应用是什么？

CIFAR-100 数据集被广泛用于训练和评估图像分类的深度学习模型。其 100 个类别（分组为 20 个粗类别）的多样化集合为测试卷积神经网络（CNN）、支持向量机（SVM）和各种其他机器学习方法等算法提供了具有挑战性的环境。该数据集是机器学习和计算机视觉领域研究和开发的关键资源，特别是[目标识别](https://docs.ultralytics.com/tasks/classify/)和分类任务。

### CIFAR-100 数据集的结构是怎样的？

CIFAR-100 数据集分为两个主要子集：

1. **训练集**：包含 50,000 张用于训练机器学习模型的图像。
2. **测试集**：包含 10,000 张用于测试和基准评估已训练模型的图像。

100 个类别中的每个类别包含 600 张图像，其中 500 张用于训练，100 张用于测试，使其非常适合严格的学术和工业研究。

### 在哪里可以找到 CIFAR-100 数据集的示例图像和标注？

CIFAR-100 数据集包含各种目标的彩色图像，使其成为图像分类任务的结构化数据集。您可以参考文档页面查看[示例图像和标注](#示例图像和标注)。这些示例突出了数据集的多样性和复杂性，这对于训练鲁棒的图像分类模型非常重要。有关更多适合分类任务的数据集，请查看 [Ultralytics 分类数据集概述](https://docs.ultralytics.com/datasets/classify/)。
