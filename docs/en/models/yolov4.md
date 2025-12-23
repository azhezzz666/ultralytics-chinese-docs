---
comments: true
description: 探索 YOLOv4，由 Alexey Bochkovskiy 开发的先进实时目标检测模型。了解其架构、特性和性能。
keywords: YOLOv4, 目标检测, 实时检测, Alexey Bochkovskiy, 神经网络, 机器学习, 计算机视觉
---

# YOLOv4：高速精确的目标检测

欢迎来到 Ultralytics YOLOv4 文档页面，YOLOv4 是由 Alexey Bochkovskiy 于 2020 年在 [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) 发布的先进实时目标检测器。YOLOv4 旨在提供速度和精度之间的最佳平衡，使其成为许多应用的绝佳选择。

![YOLOv4 架构图](https://github.com/ultralytics/docs/releases/download/0/yolov4-architecture-diagram.avif) **YOLOv4 架构图**。展示了 YOLOv4 的复杂网络设计，包括骨干网络、颈部和头部组件，以及它们相互连接的层，用于实现最佳的实时目标检测。

## 简介

YOLOv4 代表 You Only Look Once 版本 4。它是一个实时目标检测模型，旨在解决之前 YOLO 版本（如 [YOLOv3](yolov3.md)）和其他目标检测模型的局限性。与其他基于[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）的目标检测器不同，YOLOv4 不仅适用于推荐系统，还适用于独立流程管理和减少人工输入。它在传统图形处理单元（GPU）上的运行允许以实惠的价格大规模使用，并且设计为在传统 GPU 上实时工作，同时只需要一个这样的 GPU 进行训练。

## 架构

YOLOv4 利用了几个协同工作的创新特性来优化其性能。这些包括加权残差连接（WRC）、跨阶段部分连接（CSP）、跨小批量归一化（CmBN）、自对抗训练（SAT）、Mish 激活、马赛克[数据增强](https://www.ultralytics.com/glossary/data-augmentation)、DropBlock [正则化](https://www.ultralytics.com/glossary/regularization)和 CIoU 损失。这些特性结合在一起实现了先进的结果。

典型的目标检测器由几个部分组成，包括输入、[骨干网络](https://www.ultralytics.com/glossary/backbone)、颈部和头部。YOLOv4 的骨干网络在 [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) 上预训练，用于预测对象的类别和[边界框](https://www.ultralytics.com/glossary/bounding-box)。骨干网络可以来自多个模型，包括 VGG、ResNet、ResNeXt 或 DenseNet。检测器的颈部部分用于从不同阶段收集[特征图](https://www.ultralytics.com/glossary/feature-maps)，通常包括几个自下而上的路径和几个自上而下的路径。头部部分用于进行最终的目标检测和分类。

## 免费技巧包

YOLOv4 还使用了称为"免费技巧包"的方法，这些技术在训练期间提高模型的[精度](https://www.ultralytics.com/glossary/accuracy)而不增加推理成本。[数据增强](https://www.ultralytics.com/blog/the-ultimate-guide-to-data-augmentation-in-2025)是[目标检测](https://www.ultralytics.com/glossary/object-detection)中常用的免费技巧技术，它增加输入图像的变化性以提高模型的稳健性。一些数据增强的例子包括光度失真（调整图像的亮度、对比度、色调、饱和度和噪声）和几何失真（添加随机缩放、裁剪、翻转和旋转）。这些技术帮助模型更好地泛化到不同类型的图像。

## 特性和性能

YOLOv4 设计用于目标检测中的最佳速度和精度。YOLOv4 的架构包括 CSPDarknet53 作为骨干网络、PANet 作为颈部和 YOLOv3 作为[检测头部](https://www.ultralytics.com/glossary/detection-head)。这种设计允许 YOLOv4 以令人印象深刻的速度执行目标检测，使其适合实时应用。YOLOv4 在精度方面也表现出色，在 [COCO](https://docs.ultralytics.com/datasets/detect/coco/) 等目标检测基准测试中取得了先进的结果。

## 使用示例

截至撰写本文时，Ultralytics 目前不支持 YOLOv4 模型。因此，任何有兴趣使用 YOLOv4 的用户需要直接参考 YOLOv4 GitHub 仓库获取安装和使用说明。

以下是您可能采取的使用 YOLOv4 的典型步骤的简要概述：

1. 访问 YOLOv4 GitHub 仓库：[https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)。

2. 按照 README 文件中提供的安装说明进行操作。这通常涉及克隆仓库、安装必要的依赖项和设置任何必要的环境变量。

3. 安装完成后，您可以按照仓库中提供的使用说明训练和使用模型。这通常涉及准备数据集、配置模型参数、训练模型，然后使用训练好的模型执行目标检测。

请注意，具体步骤可能因您的特定用例和 YOLOv4 仓库的当前状态而异。因此，强烈建议直接参考 YOLOv4 GitHub 仓库中提供的说明。

对于可能造成的任何不便，我们深表歉意，一旦 Ultralytics 实现对 YOLOv4 的支持，我们将努力更新本文档以包含使用示例。

## 结论

YOLOv4 是一个强大而高效的目标检测模型，在速度和精度之间取得了平衡。它在训练期间使用独特的特性和免费技巧技术，使其能够在实时目标检测任务中表现出色。YOLOv4 可以由任何拥有传统 GPU 的人训练和使用，使其对于广泛的应用（包括[监控系统](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai)、[自动驾驶车辆](https://www.ultralytics.com/solutions/ai-in-automotive)和[工业自动化](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision)）都是可访问和实用的。

## 引用和致谢

我们要感谢 YOLOv4 作者在实时目标检测领域的重大贡献：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{bochkovskiy2020yolov4,
              title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
              author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
              year={2020},
              eprint={2004.10934},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

原始 YOLOv4 论文可以在 [arXiv](https://arxiv.org/abs/2004.10934) 上找到。作者已公开其工作，代码库可以在 [GitHub](https://github.com/AlexeyAB/darknet) 上访问。我们感谢他们在推进该领域并使其工作对更广泛的社区可访问方面所做的努力。

## 常见问题

### 什么是 YOLOv4，为什么我应该将其用于[目标检测](https://www.ultralytics.com/glossary/object-detection)？

YOLOv4，即"You Only Look Once 版本 4"，是由 Alexey Bochkovskiy 于 2020 年开发的先进实时目标检测模型。它在速度和[精度](https://www.ultralytics.com/glossary/accuracy)之间实现了最佳平衡，非常适合实时应用。YOLOv4 的架构结合了几个创新特性，如加权残差连接（WRC）、跨阶段部分连接（CSP）和自对抗训练（SAT）等，以实现先进的结果。如果您正在寻找在传统 GPU 上高效运行的高性能模型，YOLOv4 是一个绝佳的选择。

### YOLOv4 的架构如何增强其性能？

YOLOv4 的架构包括几个关键组件：[骨干网络](https://www.ultralytics.com/glossary/backbone)、颈部和头部。骨干网络可以是 VGG、ResNet 或 CSPDarknet53 等模型，经过预训练以预测类别和边界框。颈部利用 PANet 连接来自不同阶段的[特征图](https://www.ultralytics.com/glossary/feature-maps)以进行全面的数据提取。最后，头部使用 YOLOv3 的配置进行最终的目标检测。YOLOv4 还采用了"免费技巧包"技术，如马赛克数据增强和 DropBlock 正则化，进一步优化其速度和精度。

### 在 YOLOv4 的背景下，什么是"免费技巧包"？

"免费技巧包"是指在不增加推理成本的情况下提高 YOLOv4 训练精度的方法。这些技术包括各种形式的数据增强，如光度失真（调整亮度、对比度等）和几何失真（缩放、裁剪、翻转、旋转）。通过增加输入图像的变化性，这些增强帮助 YOLOv4 更好地泛化到不同类型的图像，从而在不影响其实时性能的情况下提高其稳健性和精度。

### 为什么 YOLOv4 被认为适合在传统 GPU 上进行实时目标检测？

YOLOv4 设计为优化速度和精度，使其非常适合需要快速可靠性能的实时目标检测任务。它在传统 GPU 上高效运行，训练和推理只需要一个 GPU。这使其对于从[推荐系统](https://www.ultralytics.com/glossary/recommendation-system)到独立流程管理的各种应用都是可访问和实用的，从而减少了对大量硬件设置的需求，使其成为实时目标检测的经济高效解决方案。

### 如果 Ultralytics 目前不支持 YOLOv4，我该如何开始使用？

要开始使用 YOLOv4，您应该访问官方 [YOLOv4 GitHub 仓库](https://github.com/AlexeyAB/darknet)。按照 README 文件中提供的安装说明进行操作，通常包括克隆仓库、安装依赖项和设置环境变量。安装完成后，您可以通过准备数据集、配置模型参数并按照仓库中提供的使用说明来训练模型。由于 Ultralytics 目前不支持 YOLOv4，建议直接参考 YOLOv4 GitHub 获取最新和最详细的指导。
