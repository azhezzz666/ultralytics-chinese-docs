---
comments: true
description: 了解 ImageNet10，这是 ImageNet 的紧凑版本，用于快速模型测试和 CI 检查。非常适合计算机视觉任务的快速评估。
keywords: ImageNet10, ImageNet, Ultralytics, CI 测试, 健全性检查, 训练流水线, 计算机视觉, 深度学习, 数据集
---

# ImageNet10 数据集

[ImageNet10](https://github.com/ultralytics/assets/releases/download/v0.0.0/imagenet10.zip) 数据集是 [ImageNet](https://www.image-net.org/) 数据库的小规模子集，由 [Ultralytics](https://www.ultralytics.com/) 开发，专为 CI 测试、健全性检查和训练流水线的快速测试而设计。该数据集由 ImageNet 前 10 个类别的训练集第一张图像和验证集第一张图像组成。虽然规模显著缩小，但它保留了原始 ImageNet 数据集的结构和多样性。

## 主要特点

- ImageNet10 是 ImageNet 的紧凑版本，包含 20 张图像，代表原始数据集的前 10 个类别。
- 数据集按照 WordNet 层次结构组织，与完整 ImageNet 数据集的结构相同。
- 它非常适合[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务中的 CI 测试、健全性检查和训练流水线的快速测试。
- 虽然不是为模型基准测试设计的，但它可以快速指示模型的基本功能和正确性。

## 数据集结构

ImageNet10 数据集与原始 [ImageNet](../classify/imagenet.md) 一样，使用 WordNet 层次结构组织。ImageNet10 中的 10 个类别中的每一个都由一个同义词集（一组同义术语）描述。ImageNet10 中的图像用一个或多个同义词集进行标注，为测试模型识别各种目标及其关系提供了紧凑的资源。

## 应用场景

ImageNet10 数据集对于快速测试和调试计算机视觉模型和流水线非常有用。其小规模允许快速迭代，使其非常适合[持续集成](../../help/CI.md)测试和健全性检查。它还可用于在使用完整 [ImageNet 数据集](../classify/imagenet.md)进行全面测试之前，对新模型或现有模型的更改进行快速初步测试。

## 使用方法

要在 ImageNet10 数据集上使用 224x224 的图像尺寸测试深度学习模型，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "测试示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="imagenet10", epochs=5, imgsz=224)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=imagenet10 model=yolo11n-cls.pt epochs=5 imgsz=224
        ```

## 示例图像和标注

ImageNet10 数据集包含原始 ImageNet 数据集的图像子集。这些图像被选择来代表数据集中的前 10 个类别，为快速测试和评估提供了多样化但紧凑的数据集。

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/imagenet10-sample-images.avif)

该示例展示了 ImageNet10 数据集中图像的多样性和复杂性，突出了其在健全性检查和计算机视觉模型快速测试中的实用性。

## 引用和致谢

如果您在研究或开发工作中使用 ImageNet10 数据集，请引用原始 ImageNet 论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{ILSVRC15,
                 author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
                 title={ImageNet Large Scale Visual Recognition Challenge},
                 year={2015},
                 journal={International Journal of Computer Vision (IJCV)},
                 volume={115},
                 number={3},
                 pages={211-252}
        }
        ```

我们感谢由 Olga Russakovsky、Jia Deng 和 Li Fei-Fei 领导的 ImageNet 团队创建并维护 ImageNet 数据集。ImageNet10 数据集虽然是一个紧凑的子集，但对于[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉研究社区的快速测试和调试来说是一个宝贵的资源。有关 ImageNet 数据集及其创建者的更多信息，请访问 [ImageNet 网站](https://www.image-net.org/)。

## 常见问题

### 什么是 ImageNet10 数据集，它与完整 ImageNet 数据集有什么不同？

[ImageNet10](https://github.com/ultralytics/assets/releases/download/v0.0.0/imagenet10.zip) 数据集是原始 [ImageNet](https://www.image-net.org/) 数据库的紧凑子集，由 Ultralytics 创建，用于快速 CI 测试、健全性检查和训练流水线评估。ImageNet10 仅包含 20 张图像，代表 ImageNet 前 10 个类别的训练和验证集的第一张图像。尽管规模小，但它保持了完整数据集的结构和多样性，使其非常适合快速测试，但不适合模型基准测试。

### 如何使用 ImageNet10 数据集测试我的深度学习模型？

要在 ImageNet10 数据集上使用 224x224 的图像尺寸测试您的深度学习模型，请使用以下代码片段。

!!! example "测试示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="imagenet10", epochs=5, imgsz=224)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=imagenet10 model=yolo11n-cls.pt epochs=5 imgsz=224
        ```

有关可用参数的完整列表，请参阅[训练](../../modes/train.md)页面。

### 为什么应该使用 ImageNet10 数据集进行 CI 测试和健全性检查？

ImageNet10 数据集专为[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)流水线中的 CI 测试、健全性检查和快速评估而设计。其小规模允许快速迭代和测试，非常适合速度至关重要的持续集成流程。通过保持原始 ImageNet 数据集的结构复杂性和多样性，ImageNet10 提供了模型基本功能和正确性的可靠指示，而无需处理大型数据集的开销。

### ImageNet10 数据集的主要特点是什么？

ImageNet10 数据集有几个关键特点：

- **紧凑规模**：仅包含 20 张图像，允许快速测试和调试。
- **结构化组织**：遵循 WordNet 层次结构，与完整 ImageNet 数据集类似。
- **CI 和健全性检查**：非常适合持续集成测试和健全性检查。
- **不适合基准测试**：虽然对快速模型评估有用，但不是为广泛基准测试设计的。

### ImageNet10 与 ImageNette 等其他小型数据集相比如何？

虽然 [ImageNet10](imagenet10.md) 和 [ImageNette](imagenette.md) 都是 ImageNet 的子集，但它们服务于不同的目的。ImageNet10 仅包含 20 张图像（每个类别 2 张），来自 ImageNet 的前 10 个类别，使其非常轻量，适合 CI 测试和快速健全性检查。相比之下，ImageNette 包含数千张图像，涵盖 10 个易于区分的类别，更适合实际的模型训练和开发。ImageNet10 旨在验证流水线功能，而 ImageNette 更适合进行有意义但比完整 ImageNet 更快的训练实验。
