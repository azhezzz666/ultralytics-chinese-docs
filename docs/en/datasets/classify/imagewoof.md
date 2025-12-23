---
comments: true
description: 探索 ImageWoof 数据集，这是 ImageNet 的一个具有挑战性的子集，专注于 10 种犬类品种，旨在增强图像分类模型。在 Ultralytics 文档中了解更多。
keywords: ImageWoof 数据集, ImageNet 子集, 犬类品种, 图像分类, 深度学习, 机器学习, Ultralytics, 训练数据集, 噪声标签
---

# ImageWoof 数据集

[ImageWoof](https://github.com/fastai/imagenette) 数据集是 [ImageNet](imagenet.md) 的一个子集，包含 10 个难以分类的类别，因为它们都是犬类品种。它的创建是为了给[图像分类](https://www.ultralytics.com/glossary/image-classification)算法提供更困难的任务，旨在鼓励开发更先进的模型。

## 主要特点

- ImageWoof 包含 10 种不同犬类品种的图像：澳大利亚梗、边境梗、萨摩耶、比格犬、西施犬、英国猎狐犬、罗得西亚脊背犬、澳洲野犬、金毛寻回犬和古代英国牧羊犬。
- 数据集提供各种分辨率的图像（全尺寸、320px、160px），以适应不同的计算能力和研究需求。
- 它还包含带有噪声标签的版本，提供更真实的场景，其中标签可能并不总是可靠的。

## 数据集结构

ImageWoof 数据集结构基于犬类品种类别，每个品种都有自己的图像目录。与其他分类数据集类似，它遵循分割目录格式，训练集和验证集有单独的文件夹。

## 应用场景

ImageWoof 数据集被广泛用于训练和评估图像分类任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，特别是涉及更复杂和相似类别时。该数据集的挑战在于犬类品种之间的细微差异，推动了模型性能和泛化能力的极限。它特别适用于：

- 在细粒度类别上对分类模型性能进行基准测试
- 测试模型对相似外观类别的鲁棒性
- 开发能够区分细微视觉差异的算法
- 评估从通用领域到特定领域的迁移学习能力

## 使用方法

要在 ImageWoof 数据集上使用 224x224 的图像尺寸训练 CNN 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="imagewoof", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=imagewoof model=yolo11n-cls.pt epochs=100 imgsz=224
        ```

## 数据集变体

ImageWoof 数据集有三种不同的尺寸，以适应各种研究需求和计算能力：

1. **全尺寸（imagewoof）**：这是 ImageWoof 数据集的原始版本。它包含全尺寸图像，非常适合最终训练和性能基准测试。

2. **中等尺寸（imagewoof320）**：此版本包含调整为最大边长 320 像素的图像。它适合更快的训练，而不会显著牺牲模型性能。

3. **小尺寸（imagewoof160）**：此版本包含调整为最大边长 160 像素的图像。它专为快速原型设计和实验而设计，其中训练速度是优先考虑的。

要使用这些变体进行训练，只需在数据集参数中将 'imagewoof' 替换为 'imagewoof320' 或 'imagewoof160'。例如：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 使用中等尺寸数据集
        model.train(data="imagewoof320", epochs=100, imgsz=224)

        # 使用小尺寸数据集
        model.train(data="imagewoof160", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # 加载预训练模型并在中等尺寸数据集上训练
        yolo classify train model=yolo11n-cls.pt data=imagewoof320 epochs=100 imgsz=224
        ```

需要注意的是，使用较小的图像可能会在分类精度方面产生较低的性能。但是，这是在模型开发和原型设计早期阶段快速迭代的绝佳方式。

## 示例图像和标注

ImageWoof 数据集包含各种犬类品种的彩色图像，为图像分类任务提供了具有挑战性的数据集。以下是数据集中的一些图像示例：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/imagewoof-dataset-sample.avif)

该示例展示了 ImageWoof 数据集中不同犬类品种之间的细微差异和相似之处，突出了分类任务的复杂性和难度。

## 引用和致谢

如果您在研究或开发工作中使用 ImageWoof 数据集，请确保通过链接到[官方数据集仓库](https://github.com/fastai/imagenette)来致谢数据集的创建者。

我们感谢 [FastAI](https://www.fast.ai/) 团队创建并维护 ImageWoof 数据集，为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)研究社区提供了宝贵的资源。有关 ImageWoof 数据集的更多信息，请访问 [ImageWoof 数据集仓库](https://github.com/fastai/imagenette)。

## 常见问题

### Ultralytics 中的 ImageWoof 数据集是什么？

[ImageWoof](https://github.com/fastai/imagenette) 数据集是 ImageNet 的一个具有挑战性的子集，专注于 10 种特定的犬类品种。它的创建是为了推动图像分类模型的极限，包括比格犬、西施犬和金毛寻回犬等品种。该数据集包括各种分辨率（全尺寸、320px、160px）的图像，甚至还有用于更真实训练场景的噪声标签。这种复杂性使 ImageWoof 成为开发更先进深度学习模型的理想选择。

### 如何使用 Ultralytics YOLO 在 ImageWoof 数据集上训练模型？

要使用 Ultralytics YOLO 在 ImageWoof 数据集上以 224x224 的图像尺寸训练[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）模型 100 个训练周期，您可以使用以下代码：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型
        results = model.train(data="imagewoof", epochs=100, imgsz=224)
        ```


    === "CLI"

        ```bash
        yolo classify train data=imagewoof model=yolo11n-cls.pt epochs=100 imgsz=224
        ```

有关可用训练参数的详细信息，请参阅[训练](../../modes/train.md)页面。

### ImageWoof 数据集有哪些版本可用？

ImageWoof 数据集有三种尺寸：

1. **全尺寸（imagewoof）**：非常适合最终训练和基准测试，包含全尺寸图像。
2. **中等尺寸（imagewoof320）**：调整为最大边长 320 像素的图像，适合更快的训练。
3. **小尺寸（imagewoof160）**：调整为最大边长 160 像素的图像，非常适合快速原型设计。

通过相应地替换数据集参数中的 'imagewoof' 来使用这些版本。但请注意，较小的图像可能会产生较低的分类[精度](https://www.ultralytics.com/glossary/accuracy)，但对于更快的迭代很有用。

### ImageWoof 数据集中的噪声标签如何有益于训练？

ImageWoof 数据集中的噪声标签模拟了标签可能并不总是准确的真实世界条件。使用这些数据训练模型有助于在图像分类任务中发展鲁棒性和泛化能力。这使模型能够有效处理模糊或错误标记的数据，这在实际应用中经常遇到。

### 使用 ImageWoof 数据集的主要挑战是什么？

ImageWoof 数据集的主要挑战在于它所包含的犬类品种之间的细微差异。由于它专注于 10 种密切相关的品种，区分它们需要更先进和精细调整的图像分类模型。这使 ImageWoof 成为测试[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型能力和改进的优秀基准。
