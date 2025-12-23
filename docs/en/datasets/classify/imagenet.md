---
comments: true
description: 探索大规模 ImageNet 数据集，了解其在计算机视觉深度学习中的作用。访问预训练模型和训练示例。
keywords: ImageNet, 深度学习, 视觉识别, 计算机视觉, 预训练模型, YOLO, 数据集, 目标检测, 图像分类
---

# ImageNet 数据集

[ImageNet](https://www.image-net.org/) 是一个专为视觉目标识别研究设计的大规模标注图像数据库。它包含超过 1400 万张图像，每张图像都使用 WordNet 同义词集进行标注，使其成为训练[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型进行[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务的最广泛资源之一。

## ImageNet 预训练模型

{% include "macros/yolo-cls-perf.md" %}

## 主要特点

- ImageNet 包含超过 1400 万张高分辨率图像，涵盖数千个目标类别。
- 数据集按照 WordNet 层次结构组织，每个同义词集代表一个类别。
- ImageNet 在计算机视觉领域被广泛用于训练和基准测试，特别是[图像分类](https://www.ultralytics.com/glossary/image-classification)和[目标检测](https://www.ultralytics.com/glossary/object-detection)任务。
- 年度 ImageNet 大规模视觉识别挑战赛（ILSVRC）在推动计算机视觉研究方面发挥了重要作用。

## 数据集结构

ImageNet 数据集使用 WordNet 层次结构组织。层次结构中的每个节点代表一个类别，每个类别由一个同义词集（一组同义术语）描述。ImageNet 中的图像用一个或多个同义词集进行标注，为训练模型识别各种目标及其关系提供了丰富的资源。

## ImageNet 大规模视觉识别挑战赛（ILSVRC）

年度 [ImageNet 大规模视觉识别挑战赛（ILSVRC）](https://image-net.org/challenges/LSVRC/) 是计算机视觉领域的重要赛事。它为研究人员和开发者提供了一个平台，使用标准化评估指标在大规模数据集上评估其算法和模型。ILSVRC 推动了图像分类、目标检测和其他计算机视觉任务深度学习模型的重大进展。

## 应用场景

ImageNet 数据集被广泛用于训练和评估各种计算机视觉任务中的深度学习模型，如图像分类、目标检测和目标定位。一些流行的深度学习架构，如 [AlexNet](https://en.wikipedia.org/wiki/AlexNet)、[VGG](https://arxiv.org/abs/1409.1556) 和 [ResNet](https://arxiv.org/abs/1512.03385)，都是使用 ImageNet 数据集开发和基准测试的。

## 使用方法

要在 ImageNet 数据集上使用 224x224 的图像尺寸训练深度学习模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=imagenet model=yolo11n-cls.pt epochs=100 imgsz=224
        ```

## 示例图像和标注

ImageNet 数据集包含涵盖数千个目标类别的高分辨率图像，为训练和评估计算机视觉模型提供了多样化且广泛的数据集。以下是数据集中的一些图像示例：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/imagenet-sample-images.avif)

该示例展示了 ImageNet 数据集中图像的多样性和复杂性，强调了多样化数据集对于训练鲁棒计算机视觉模型的重要性。

## 引用和致谢

如果您在研究或开发工作中使用 ImageNet 数据集，请引用以下论文：

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

我们感谢由 Olga Russakovsky、Jia Deng 和 Li Fei-Fei 领导的 ImageNet 团队创建并维护 ImageNet 数据集，为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉研究社区提供了宝贵的资源。有关 ImageNet 数据集及其创建者的更多信息，请访问 [ImageNet 网站](https://www.image-net.org/)。

## 常见问题

### 什么是 ImageNet 数据集，它在计算机视觉中如何使用？

[ImageNet 数据集](https://www.image-net.org/)是一个大规模数据库，包含超过 1400 万张使用 WordNet 同义词集分类的高分辨率图像。它被广泛用于视觉目标识别研究，包括图像分类和目标检测。该数据集的标注和庞大的数量为训练深度学习模型提供了丰富的资源。值得注意的是，AlexNet、VGG 和 ResNet 等模型都是使用 ImageNet 训练和基准测试的，展示了其在推动计算机视觉方面的作用。

### 如何使用预训练的 YOLO 模型在 ImageNet 数据集上进行图像分类？

要使用预训练的 Ultralytics YOLO 模型在 ImageNet 数据集上进行图像分类，请按照以下步骤操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-cls.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo classify train data=imagenet model=yolo11n-cls.pt epochs=100 imgsz=224
        ```

有关更深入的训练说明，请参阅我们的[训练页面](../../modes/train.md)。

### 为什么应该在 ImageNet 数据集项目中使用 Ultralytics YOLO11 预训练模型？

Ultralytics YOLO11 预训练模型在各种计算机视觉任务的速度和[精度](https://www.ultralytics.com/glossary/accuracy)方面提供最先进的性能。例如，YOLO11n-cls 模型的 top-1 精度为 70.0%，top-5 精度为 89.4%，针对实时应用进行了优化。预训练模型减少了从头开始训练所需的计算资源，并加速了开发周期。在 [ImageNet 预训练模型部分](#imagenet-预训练模型)了解更多关于 YOLO11 模型性能指标的信息。

### ImageNet 数据集的结构是怎样的，为什么它很重要？

ImageNet 数据集使用 WordNet 层次结构组织，其中层次结构中的每个节点代表一个由同义词集（一组同义术语）描述的类别。这种结构允许详细的标注，使其非常适合训练模型识别各种目标。ImageNet 的多样性和标注丰富性使其成为开发鲁棒且通用深度学习模型的宝贵数据集。有关此组织的更多信息，请参阅[数据集结构](#数据集结构)部分。

### ImageNet 大规模视觉识别挑战赛（ILSVRC）在计算机视觉中扮演什么角色？

年度 [ImageNet 大规模视觉识别挑战赛（ILSVRC）](https://image-net.org/challenges/LSVRC/) 在推动计算机视觉进步方面发挥了关键作用，它提供了一个竞争平台，用于在大规模标准化数据集上评估算法。它提供标准化的评估指标，促进了图像分类、目标检测和[图像分割](https://www.ultralytics.com/glossary/image-segmentation)等领域的创新和发展。该挑战赛不断推动深度学习和计算机视觉技术的边界。
