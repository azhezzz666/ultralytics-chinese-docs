---
comments: true
description: 探索 ImageNette 数据集，这是 ImageNet 的子集，包含 10 个类别，用于高效训练和评估图像分类模型。非常适合机器学习和计算机视觉项目。
keywords: ImageNette 数据集, ImageNet 子集, 图像分类, 机器学习, 深度学习, YOLO, 卷积神经网络, 机器学习数据集, 教育, 训练
---

# ImageNette 数据集

[ImageNette](https://github.com/fastai/imagenette) 数据集是大型 [ImageNet](https://www.image-net.org/) 数据集的子集，但它只包含 10 个易于区分的类别。它的创建是为了提供一个更快、更易于使用的 ImageNet 版本，用于软件开发和教育。

## 主要特点

- ImageNette 包含来自 10 个不同类别的图像，如丁鲷、英国史宾格犬、卡带播放器、链锯、教堂、法国号、垃圾车、加油泵、高尔夫球、降落伞。
- 数据集包含不同尺寸的彩色图像。
- ImageNette 在机器学习领域被广泛用于训练和测试，特别是图像分类任务。

## 数据集结构

ImageNette 数据集分为两个子集：

1. **训练集**：该子集包含数千张用于训练机器学习模型的图像。每个类别的确切数量各不相同。
2. **验证集**：该子集包含数百张用于验证和基准评估已训练模型的图像。同样，每个类别的确切数量各不相同。

## 应用场景

ImageNette 数据集被广泛用于训练和评估图像分类任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）以及各种其他机器学习算法。该数据集简单的格式和精心选择的类别使其成为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域初学者和经验丰富的从业者的便捷资源。

## 使用方法

要在 ImageNette 数据集上使用标准的 224x224 图像尺寸训练模型 100 个训练周期，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="imagenette", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=imagenette model=yolo11n-cls.pt epochs=100 imgsz=224
        ```

## 示例图像和标注

ImageNette 数据集包含各种目标和场景的彩色图像，为[图像分类](https://www.ultralytics.com/glossary/image-classification)任务提供了多样化的数据集。以下是数据集中的一些图像示例：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/imagenette-sample-image.avif)

该示例展示了 ImageNette 数据集中图像的多样性和复杂性，强调了多样化数据集对于训练鲁棒图像分类模型的重要性。

## ImageNette160 和 ImageNette320

为了更快的原型设计和训练，ImageNette 数据集还提供两种缩小尺寸的版本：[ImageNette160](https://github.com/fastai/imagenette) 和 [ImageNette320](https://github.com/fastai/imagenette)。这些数据集保持与完整 ImageNette 数据集相同的类别和结构，但图像被调整为更小的尺寸。因此，这些版本的数据集特别适用于初步模型测试，或在计算资源有限时使用。

要使用这些数据集，只需在训练命令中将 'imagenette' 替换为 'imagenette160' 或 'imagenette320'。以下代码片段说明了这一点：

!!! example "使用 ImageNette160 的训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 使用 ImageNette160 训练模型
        results = model.train(data="imagenette160", epochs=100, imgsz=160)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始使用 ImageNette160 训练
        yolo classify train data=imagenette160 model=yolo11n-cls.pt epochs=100 imgsz=160
        ```

!!! example "使用 ImageNette320 的训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 使用 ImageNette320 训练模型
        results = model.train(data="imagenette320", epochs=100, imgsz=320)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始使用 ImageNette320 训练
        yolo classify train data=imagenette320 model=yolo11n-cls.pt epochs=100 imgsz=320
        ```

这些较小版本的数据集允许在开发过程中快速迭代，同时仍然提供有价值且真实的图像分类任务。

## 引用和致谢

如果您在研究或开发工作中使用 ImageNette 数据集，请适当致谢。有关 ImageNette 数据集的更多信息，请访问 [ImageNette 数据集 GitHub 页面](https://github.com/fastai/imagenette)。

## 常见问题

### 什么是 ImageNette 数据集？

[ImageNette 数据集](https://github.com/fastai/imagenette)是大型 [ImageNet 数据集](https://www.image-net.org/)的简化子集，仅包含 10 个易于区分的类别，如丁鲷、英国史宾格犬和法国号。它的创建是为了提供一个更易于管理的数据集，用于高效训练和评估图像分类模型。该数据集特别适用于[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉中的快速软件开发和教育目的。

### 如何使用 ImageNette 数据集训练 YOLO 模型？

要在 ImageNette 数据集上训练 YOLO 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，您可以使用以下命令。确保已设置好 Ultralytics YOLO 环境。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="imagenette", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=imagenette model=yolo11n-cls.pt epochs=100 imgsz=224
        ```

有关更多详细信息，请参阅[训练](../../modes/train.md)文档页面。

### 为什么应该使用 ImageNette 进行图像分类任务？

ImageNette 数据集有以下几个优势：

- **快速简单**：它只包含 10 个类别，与较大的数据集相比，复杂性更低，耗时更少。
- **教育用途**：非常适合学习和教授图像分类的基础知识，因为它需要更少的计算能力和时间。
- **多功能性**：广泛用于训练和基准测试各种机器学习模型，特别是图像分类。

有关模型训练和数据集管理的更多详细信息，请探索[数据集结构](#数据集结构)部分。

### ImageNette 数据集可以使用不同的图像尺寸吗？

是的，ImageNette 数据集还提供两种调整大小的版本：ImageNette160 和 ImageNette320。这些版本有助于更快的原型设计，在计算资源有限时特别有用。

!!! example "使用 ImageNette160 的训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")

        # 使用 ImageNette160 训练模型
        results = model.train(data="imagenette160", epochs=100, imgsz=160)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始使用 ImageNette160 训练
        yolo classify train data=imagenette160 model=yolo11n-cls.pt epochs=100 imgsz=160
        ```

有关更多信息，请参阅[使用 ImageNette160 和 ImageNette320 训练](#imagenette160-和-imagenette320)。

### ImageNette 数据集有哪些实际应用？

ImageNette 数据集被广泛用于：

- **教育环境**：教授机器学习和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)初学者。
- **软件开发**：用于图像分类模型的快速原型设计和开发。
- **深度学习研究**：评估和基准测试各种深度学习模型的性能，特别是卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)（CNN）。

探索[应用场景](#应用场景)部分了解详细的用例。
