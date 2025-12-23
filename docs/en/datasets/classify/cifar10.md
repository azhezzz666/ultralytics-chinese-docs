---
comments: true
description: 探索 CIFAR-10 数据集，包含 10 个类别的 60,000 张彩色图像。了解其结构、应用以及如何使用 YOLO 训练模型。
keywords: CIFAR-10, 数据集, 机器学习, 计算机视觉, 图像分类, YOLO, 深度学习, 神经网络
---

# CIFAR-10 数据集

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)（加拿大高级研究院）数据集是一个广泛用于[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉算法的图像集合。它由 CIFAR 研究所的研究人员开发，包含 10 个不同类别的 60,000 张 32x32 彩色图像。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/fLBbyhPbWzY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics YOLO11 在 CIFAR-10 数据集上训练<a href="https://www.ultralytics.com/glossary/image-classification">图像分类</a>模型
</p>

## 主要特点

- CIFAR-10 数据集包含 60,000 张图像，分为 10 个类别。
- 每个类别包含 6,000 张图像，其中 5,000 张用于训练，1,000 张用于测试。
- 图像为彩色，尺寸为 32x32 像素。
- 10 个不同类别分别代表飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船和卡车。
- CIFAR-10 在机器学习和计算机视觉领域被广泛用于训练和测试。

## 数据集结构

CIFAR-10 数据集分为两个子集：

1. **训练集**：该子集包含 50,000 张用于训练机器学习模型的图像。
2. **测试集**：该子集包含 10,000 张用于测试和基准评估已训练模型的图像。

## 应用场景

CIFAR-10 数据集被广泛用于训练和评估[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型的图像分类任务，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）以及各种其他机器学习算法。该数据集在类别方面的多样性以及彩色图像的存在，使其成为机器学习和计算机视觉领域研究和开发的全面数据集。

## 使用方法

要在 CIFAR-10 数据集上使用 32x32 的图像尺寸训练 YOLO 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="cifar10", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=cifar10 model=yolo11n-cls.pt epochs=100 imgsz=32
        ```

## 示例图像和标注

CIFAR-10 数据集包含各种目标的彩色图像，为图像分类任务提供了结构良好的数据集。以下是数据集中的一些图像示例：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/cifar10-sample-image.avif)

该示例展示了 CIFAR-10 数据集中目标的多样性和复杂性，强调了多样化数据集对于训练鲁棒图像分类模型的重要性。

## 引用和致谢

如果您在研究或开发工作中使用 CIFAR-10 数据集，请引用以下论文：

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

我们感谢 Alex Krizhevsky 创建并维护 CIFAR-10 数据集，为机器学习和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)研究社区提供了宝贵的资源。有关 CIFAR-10 数据集及其创建者的更多信息，请访问 [CIFAR-10 数据集网站](https://www.cs.toronto.edu/~kriz/cifar.html)。

## 常见问题

### 如何在 CIFAR-10 数据集上训练 YOLO 模型？

要使用 Ultralytics 在 CIFAR-10 数据集上训练 YOLO 模型，您可以按照 Python 和 CLI 提供的示例进行操作。以下是使用 32x32 像素图像尺寸训练模型 100 个训练周期的基本示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="cifar10", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=cifar10 model=yolo11n-cls.pt epochs=100 imgsz=32
        ```

有关更多详细信息，请参阅模型[训练](../../modes/train.md)页面。

### CIFAR-10 数据集的主要特点是什么？

CIFAR-10 数据集包含 60,000 张彩色图像，分为 10 个类别。每个类别包含 6,000 张图像，其中 5,000 张用于训练，1,000 张用于测试。图像尺寸为 32x32 像素，涵盖以下类别：

- 飞机
- 汽车
- 鸟类
- 猫
- 鹿
- 狗
- 青蛙
- 马
- 船
- 卡车

这个多样化的数据集对于训练机器学习和计算机视觉等领域的图像分类模型至关重要。有关更多信息，请访问[数据集结构](#数据集结构)和[应用场景](#应用场景)部分。

### 为什么使用 CIFAR-10 数据集进行图像分类任务？

CIFAR-10 数据集因其多样性和结构而成为图像分类的优秀基准。它包含 10 个不同类别的 60,000 张标注图像的均衡组合，有助于训练鲁棒且通用的模型。它被广泛用于评估深度学习模型，包括卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)（CNN）和其他机器学习算法。该数据集相对较小，适合快速实验和算法开发。在[应用场景](#应用场景)部分探索其众多应用。

### CIFAR-10 数据集的结构是怎样的？

CIFAR-10 数据集分为两个主要子集：

1. **训练集**：包含 50,000 张用于训练机器学习模型的图像。
2. **测试集**：包含 10,000 张用于测试和基准评估已训练模型的图像。

每个子集包含分为 10 个类别的图像，其标注可直接用于模型训练和评估。有关更详细的信息，请参阅[数据集结构](#数据集结构)部分。

### 如何在研究中引用 CIFAR-10 数据集？

如果您在研究或开发项目中使用 CIFAR-10 数据集，请确保引用以下论文：

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

致谢数据集的创建者有助于支持该领域的持续研究和开发。有关更多详细信息，请参阅[引用和致谢](#引用和致谢)部分。

### 使用 CIFAR-10 数据集有哪些实际示例？

CIFAR-10 数据集通常用于训练图像分类模型，例如卷积神经网络（CNN）和支持向量机（SVM）。这些模型可用于各种计算机视觉任务，包括[目标检测](https://www.ultralytics.com/glossary/object-detection)、[图像识别](https://www.ultralytics.com/glossary/image-recognition)和自动标注。要查看一些实际示例，请查看[使用方法](#使用方法)部分中的代码片段。
