---
comments: true
description: 探索 Caltech-256 数据集，包含 257 个类别的约 30,000 张图像，是训练和测试目标识别算法的理想选择。
keywords: Caltech-256 数据集, 目标分类, 图像数据集, 机器学习, 计算机视觉, 深度学习, YOLO, 训练数据集
---

# Caltech-256 数据集

[Caltech-256](https://data.caltech.edu/records/nyy15-4j048) 数据集是一个用于目标分类任务的大型图像集合。它包含约 30,000 张图像，分为 257 个类别（256 个目标类别和 1 个背景类别）。这些图像经过精心筛选和标注，为目标识别算法提供了一个具有挑战性且多样化的基准测试。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/isc06_9qnM0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics HUB 在 Caltech-256 数据集上训练<a href="https://www.ultralytics.com/glossary/image-classification">图像分类</a>模型
</p>

!!! note "自动数据划分"

    Caltech-256 数据集本身并未提供预定义的训练/验证划分。但是，当您使用下面使用示例中提供的训练命令时，Ultralytics 框架会自动为您划分数据集。默认划分比例为 80% 训练集和 20% 验证集。

## 主要特点

- Caltech-256 数据集包含约 30,000 张彩色图像，分为 257 个类别。
- 每个类别至少包含 80 张图像。
- 类别涵盖各种现实世界的目标，包括动物、车辆、家居用品和人物。
- 图像具有不同的尺寸和分辨率。
- Caltech-256 在机器学习领域被广泛用于训练和测试，特别是目标识别任务。

## 数据集结构

与 [Caltech-101](../classify/caltech101.md) 类似，Caltech-256 数据集没有正式的训练集和测试集划分。用户通常根据自己的具体需求创建划分。常见做法是使用随机子集的图像进行训练，其余图像用于测试。

## 应用场景

Caltech-256 数据集被广泛用于训练和评估[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型的目标识别任务，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）以及各种其他机器学习算法。其多样化的类别和高质量的图像使其成为机器学习和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究和开发的宝贵数据集。

## 使用方法

要在 Caltech-256 数据集上训练 YOLO 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="caltech256", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=caltech256 model=yolo11n-cls.pt epochs=100 imgsz=416
        ```

## 示例图像和标注

Caltech-256 数据集包含各种目标的高质量彩色图像，为目标识别任务提供了全面的数据集。以下是数据集中的一些图像示例（[来源](https://ml4a.github.io/demos/tsne_viewer.html)）：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/caltech256-sample-image.avif)

该示例展示了 Caltech-256 数据集中目标的多样性和复杂性，强调了多样化数据集对于训练鲁棒目标识别模型的重要性。

## 引用和致谢

如果您在研究或开发工作中使用 Caltech-256 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{griffin2007caltech,
                 title={Caltech-256 object category dataset},
                 author={Griffin, Gregory and Holub, Alex and Perona, Pietro},
                 year={2007}
        }
        ```

我们感谢 Gregory Griffin、Alex Holub 和 Pietro Perona 创建并维护 Caltech-256 数据集，为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉研究社区提供了宝贵的资源。有关 Caltech-256 数据集及其创建者的更多信息，请访问 [Caltech-256 数据集网站](https://data.caltech.edu/records/nyy15-4j048)。

## 常见问题

### 什么是 Caltech-256 数据集，为什么它对机器学习很重要？

[Caltech-256](https://data.caltech.edu/records/nyy15-4j048) 数据集是一个主要用于机器学习和计算机视觉中目标分类任务的大型图像数据集。它由约 30,000 张彩色图像组成，分为 257 个类别，涵盖各种现实世界的目标。该数据集多样化的高质量图像使其成为评估目标识别算法的优秀基准，这对于开发鲁棒的机器学习模型至关重要。

### 如何使用 Python 或 CLI 在 Caltech-256 数据集上训练 YOLO 模型？

要在 Caltech-256 数据集上训练 YOLO 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关其他选项，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型

        # 训练模型
        results = model.train(data="caltech256", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=caltech256 model=yolo11n-cls.pt epochs=100 imgsz=416
        ```

### Caltech-256 数据集最常见的用例是什么？

Caltech-256 数据集被广泛用于各种目标识别任务，例如：

- 训练卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)（CNN）
- 评估支持向量机（SVM）的性能
- 对新的深度学习算法进行基准测试
- 使用 Ultralytics YOLO 等框架开发[目标检测](https://www.ultralytics.com/glossary/object-detection)模型

其多样性和全面的标注使其成为机器学习和计算机视觉研究与开发的理想选择。

### Caltech-256 数据集的结构和训练/测试划分是怎样的？

Caltech-256 数据集没有预定义的训练和测试划分。用户通常根据自己的具体需求创建划分。常见方法是随机选择一部分图像用于训练，其余图像用于测试。这种灵活性允许用户根据特定项目需求和实验设置定制数据集。

### 为什么应该使用 Ultralytics YOLO 在 Caltech-256 数据集上训练模型？

Ultralytics YOLO 模型在 Caltech-256 数据集上训练具有以下优势：

- **高精度**：YOLO 模型以其在目标检测任务中的最先进性能而闻名。
- **速度**：它们提供实时推理能力，适用于需要快速预测的应用。
- **易用性**：通过 [Ultralytics HUB](https://www.ultralytics.com/hub)，用户可以无需大量编码即可训练、验证和部署模型。
- **预训练模型**：从预训练模型（如 `yolo11n-cls.pt`）开始可以显著减少训练时间并提高模型[精度](https://www.ultralytics.com/glossary/accuracy)。

有关更多详细信息，请探索我们的[综合训练指南](../../modes/train.md)，并了解使用 Ultralytics YOLO 进行[图像分类](../../tasks/classify.md)。
