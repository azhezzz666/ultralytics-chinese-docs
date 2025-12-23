---
comments: true
description: 探索广泛使用的 Caltech-101 数据集，包含 101 个类别的约 9,000 张图像。非常适合机器学习和计算机视觉中的目标识别任务。
keywords: Caltech-101, 数据集, 目标识别, 机器学习, 计算机视觉, YOLO, 深度学习, 研究, 人工智能
---

# Caltech-101 数据集

[Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) 数据集是一个广泛用于目标识别任务的数据集，包含来自 101 个目标类别的约 9,000 张图像。这些类别被选择来反映各种现实世界的目标，图像本身经过精心选择和标注，为目标识别算法提供了具有挑战性的基准。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/isc06_9qnM0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics HUB 和 Caltech-256 数据集训练<a href="https://www.ultralytics.com/glossary/image-classification">图像分类</a>模型
</p>

!!! note "自动数据划分"

    Caltech-101 数据集本身不附带预定义的训练/验证集划分。但是，当您使用下面使用示例中提供的训练命令时，Ultralytics 框架将自动为您划分数据集。默认划分为 80% 用于训练集，20% 用于验证集。

## 主要特点

- Caltech-101 数据集包含约 9,000 张彩色图像，分为 101 个类别。
- 这些类别涵盖各种目标，包括动物、车辆、家居用品和人物。
- 每个类别的图像数量不等，每个类别约有 40 到 800 张图像。
- 图像尺寸可变，大多数图像为中等分辨率。
- Caltech-101 广泛用于机器学习领域的训练和测试，特别是目标识别任务。

## 数据集结构

与许多其他数据集不同，Caltech-101 数据集没有正式划分为训练集和测试集。用户通常根据自己的特定需求创建自己的划分。然而，常见的做法是使用随机图像子集进行训练（例如，每个类别 30 张图像），其余图像用于测试。

## 应用

Caltech-101 数据集广泛用于训练和评估目标识别任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）和各种其他机器学习算法。其丰富的类别和高质量图像使其成为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究和开发的优秀数据集。

## 使用方法

要在 Caltech-101 数据集上训练 YOLO 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=caltech101 model=yolo11n-cls.pt epochs=100 imgsz=416
        ```

## 示例图像和标注

Caltech-101 数据集包含各种目标的高质量彩色图像，为[图像分类](https://www.ultralytics.com/glossary/image-classification)任务提供了结构良好的数据集。以下是数据集中的一些图像示例：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/caltech101-sample-image.avif)

该示例展示了 Caltech-101 数据集中目标的多样性和复杂性，强调了多样化数据集对于训练强大目标识别模型的重要性。

## 引用和致谢

如果您在研究或开发工作中使用 Caltech-101 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{fei2007learning,
          title={Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories},
          author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
          journal={Computer vision and Image understanding},
          volume={106},
          number={1},
          pages={59--70},
          year={2007},
          publisher={Elsevier}
        }
        ```

我们要感谢 Li Fei-Fei、Rob Fergus 和 Pietro Perona 创建和维护 Caltech-101 数据集，作为机器学习和计算机视觉研究社区的宝贵资源。有关 Caltech-101 数据集及其创建者的更多信息，请访问 [Caltech-101 数据集网站](https://data.caltech.edu/records/mzrjq-6wc02)。

## 常见问题

### Caltech-101 数据集在机器学习中有什么用途？

[Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) 数据集广泛用于机器学习中的目标识别任务。它包含 101 个类别的约 9,000 张图像，为评估目标识别算法提供了具有挑战性的基准。研究人员利用它来训练和测试模型，特别是卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)（CNN）和支持向量机（SVM），用于计算机视觉。

### 如何在 Caltech-101 数据集上训练 Ultralytics YOLO 模型？

要在 Caltech-101 数据集上训练 Ultralytics YOLO 模型，您可以使用提供的代码片段。例如，训练 100 个周期：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=caltech101 model=yolo11n-cls.pt epochs=100 imgsz=416
        ```

有关更详细的参数和选项，请参阅模型[训练](../../modes/train.md)页面。

### Caltech-101 数据集的主要特点是什么？

Caltech-101 数据集包括：

- 101 个类别的约 9,000 张彩色图像。
- 涵盖各种目标的类别，包括动物、车辆和家居用品。
- 每个类别的图像数量可变，通常在 40 到 800 之间。
- 图像尺寸可变，大多数为中等分辨率。

这些特点使其成为机器学习和计算机视觉中训练和评估目标识别模型的绝佳选择。

### 为什么应该在研究中引用 Caltech-101 数据集？

在研究中引用 Caltech-101 数据集是对创建者贡献的认可，并为可能使用该数据集的其他人提供参考。推荐的引用格式是：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{fei2007learning,
          title={Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories},
          author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
          journal={Computer vision and Image understanding},
          volume={106},
          number={1},
          pages={59--70},
          year={2007},
          publisher={Elsevier}
        }
        ```

引用有助于维护学术工作的完整性，并帮助同行找到原始资源。

### 我可以使用 Ultralytics HUB 在 Caltech-101 数据集上训练模型吗？

是的，您可以使用 [Ultralytics HUB](https://www.ultralytics.com/hub) 在 Caltech-101 数据集上训练模型。Ultralytics HUB 提供了一个直观的平台，用于管理数据集、训练模型和部署它们，无需大量编码。有关详细指南，请参阅[如何使用 Ultralytics HUB 训练您的自定义模型](https://www.ultralytics.com/blog/how-to-train-your-custom-models-with-ultralytics-hub)博客文章。
