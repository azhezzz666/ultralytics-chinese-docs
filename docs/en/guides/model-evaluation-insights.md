---
comments: true
description: 探索评估和优化 YOLO11 模型以获得更好性能的最有效方法。了解评估指标、微调过程以及如何根据特定需求定制模型。
keywords: 模型评估, 机器学习模型评估, 微调机器学习, 微调模型, 评估模型, 模型微调, 如何微调模型
---

# 模型评估和微调的见解

## 简介

一旦您[训练](./model-training-tips.md)了计算机视觉模型，评估和优化它以实现最佳性能是必不可少的。仅仅训练模型是不够的。您需要确保模型准确、高效，并满足计算机视觉项目的[目标](./defining-project-goals.md)。通过评估和微调模型，您可以识别弱点、提高精度并提升整体性能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/-aYO-6VaDrw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 模型评估和微调的见解 | 提高平均精度的技巧
</p>

在本指南中，我们分享关于模型评估和微调的见解，使[计算机视觉项目的这一步骤](./steps-of-a-cv-project.md)更加易于理解。我们讨论如何理解评估指标和实施微调技术，为您提供提升模型能力的知识。

## 使用指标评估模型性能

评估模型的表现有助于我们了解它的工作效果。各种指标用于衡量性能。这些[性能指标](./yolo-performance-metrics.md)提供清晰的数值见解，可以指导改进，确保模型达到预期目标。让我们仔细看看几个关键指标。

### 置信度分数

置信度分数表示模型对检测到的对象属于特定类别的确定性。它的范围从 0 到 1，分数越高表示置信度越高。置信度分数有助于过滤预测；只有置信度分数高于指定阈值的检测才被视为有效。

_快速提示：_在运行推理时，如果您没有看到任何预测，并且已经检查了其他所有内容，请尝试降低置信度分数。有时，阈值太高，导致模型忽略有效的预测。降低分数允许模型考虑更多可能性。这可能不符合您的项目目标，但这是查看模型能做什么并决定如何微调它的好方法。

### 交并比

[交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou)（IoU）是[目标检测](https://www.ultralytics.com/glossary/object-detection)中的一个指标，用于衡量预测的[边界框](https://www.ultralytics.com/glossary/bounding-box)与真实边界框的重叠程度。IoU 值范围从 0 到 1，其中 1 表示完美匹配。IoU 很重要，因为它衡量预测边界与实际对象边界的匹配程度。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/intersection-over-union-overview.avif" alt="交并比概述">
</p>

### 平均精度均值

[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）是衡量目标检测模型性能的一种方法。它查看检测每个对象类别的精度，对这些分数取平均值，并给出一个总体数字，显示模型识别和分类对象的准确程度。

让我们关注两个特定的 mAP 指标：

- *mAP@.5：*在单个 IoU（交并比）阈值 0.5 处测量平均精度。此指标检查模型是否能以较宽松的精度要求正确找到对象。它关注对象是否大致在正确的位置，不需要完美的放置。它有助于查看模型是否总体上擅长发现对象。
- *mAP@.5:.95：*对从 0.5 到 0.95 以 0.05 为增量的多个 IoU 阈值计算的 mAP 值取平均。此指标更详细和严格。它更全面地展示了模型在不同严格程度下找到对象的准确程度，对于需要精确目标检测的应用特别有用。

其他 mAP 指标包括 mAP@0.75，它使用更严格的 IoU 阈值 0.75，以及 mAP@small、medium 和 large，它们评估不同大小对象的精度。

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/mean-average-precision-overview.avif" alt="平均精度均值概述">
</p>

## 评估 YOLO11 模型性能

关于 YOLO11，您可以使用[验证模式](../modes/val.md)来评估模型。此外，请务必查看我们深入介绍 [YOLO11 性能指标](./yolo-performance-metrics.md)及其解释方式的指南。

### 常见社区问题

在评估 YOLO11 模型时，您可能会遇到一些问题。根据常见的社区问题，以下是一些帮助您充分利用 YOLO11 模型的技巧：

#### 处理可变图像大小

使用不同大小的图像评估 YOLO11 模型可以帮助您了解其在不同数据集上的性能。使用 `rect=true` 验证参数，YOLO11 根据图像大小为每个批次调整网络的步幅，允许模型处理矩形图像而不强制它们为单一大小。

`imgsz` 验证参数设置图像调整大小的最大维度，默认为 640。您可以根据数据集的最大维度和可用的 GPU 内存进行调整。即使设置了 `imgsz`，`rect=true` 也可以通过动态调整步幅让模型有效管理不同的图像大小。

#### 访问 YOLO11 指标

如果您想更深入地了解 YOLO11 模型的性能，可以使用几行 Python 代码轻松访问特定的评估指标。下面的代码片段将让您加载模型、运行评估并打印出各种指标，显示模型的表现。

!!! example "用法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")

        # 运行评估
        results = model.val(data="coco8.yaml")

        # 打印特定指标
        print("具有平均精度的类别索引:", results.ap_class_index)
        print("所有类别的平均精度:", results.box.all_ap)
        print("平均精度:", results.box.ap)
        print("IoU=0.50 时的平均精度:", results.box.ap50)
        print("平均精度的类别索引:", results.box.ap_class_index)
        print("类别特定结果:", results.box.class_result)
        print("F1 分数:", results.box.f1)
        print("F1 分数曲线:", results.box.f1_curve)
        print("整体适应度分数:", results.box.fitness)
        print("平均精度均值:", results.box.map)
        print("IoU=0.50 时的平均精度均值:", results.box.map50)
        print("IoU=0.75 时的平均精度均值:", results.box.map75)
        print("不同 IoU 阈值的平均精度均值:", results.box.maps)
        print("不同指标的平均结果:", results.box.mean_results)
        print("平均精确率:", results.box.mp)
        print("平均召回率:", results.box.mr)
        print("精确率:", results.box.p)
        print("精确率曲线:", results.box.p_curve)
        print("精确率值:", results.box.prec_values)
        print("特定精确率指标:", results.box.px)
        print("召回率:", results.box.r)
        print("召回率曲线:", results.box.r_curve)
        ```

结果对象还包括速度指标，如预处理时间、推理时间、损失和后处理时间。通过分析这些指标，您可以微调和优化 YOLO11 模型以获得更好的性能，使其对您的特定用例更有效。

## 微调是如何工作的？

微调涉及采用预训练模型并调整其参数以提高特定任务或数据集上的性能。这个过程，也称为模型重新训练，允许模型更好地理解和预测它在实际应用中将遇到的特定数据的结果。您可以根据模型评估重新训练模型以获得最佳结果。

## 微调模型的技巧

微调模型意味着要密切关注几个重要的参数和技术以实现最佳性能。以下是一些指导您完成该过程的基本技巧。

### 从较高的学习率开始

通常，在初始训练[轮次](https://www.ultralytics.com/glossary/epoch)期间，学习率从低开始并逐渐增加以稳定训练过程。然而，由于您的模型已经从之前的数据集中学习了一些特征，立即从较高的[学习率](https://www.ultralytics.com/glossary/learning-rate)开始可能更有益。

在评估 YOLO11 模型时，您可以将 `warmup_epochs` 验证参数设置为 `warmup_epochs=0` 以防止学习率从太低开始。通过遵循此过程，训练将从提供的权重继续，调整以适应新数据的细微差别。

### 小对象的图像切片

图像切片可以提高小对象的检测精度。通过将较大的图像分成较小的片段，例如将 1280x1280 图像分成多个 640x640 片段，您可以保持原始分辨率，模型可以从高分辨率片段中学习。使用 YOLO11 时，请确保正确调整这些新片段的标签。

## 参与社区

与其他[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)爱好者分享您的想法和问题可以激发创造性的解决方案来解决项目中的障碍。以下是一些学习、故障排除和联系的绝佳方式。

### 寻找帮助和支持

- **GitHub Issues：**探索 YOLO11 GitHub 仓库并使用 [Issues 标签](https://github.com/ultralytics/ultralytics/issues)提问、报告错误和建议功能。社区和维护者可以帮助您解决遇到的任何问题。
- **Ultralytics Discord 服务器：**加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)与其他用户和开发者联系，获得支持，分享知识和头脑风暴。

### 官方文档

- **Ultralytics YOLO11 文档：**查看[官方 YOLO11 文档](./index.md)获取各种计算机视觉任务和项目的全面指南和有价值的见解。

## 最后的想法

评估和微调计算机视觉模型是成功[模型部署](https://www.ultralytics.com/glossary/model-deployment)的重要步骤。这些步骤有助于确保您的模型准确、高效并适合您的整体应用。训练最佳模型的关键是持续实验和学习。不要犹豫调整参数、尝试新技术和探索不同的数据集。继续实验并突破可能的界限！

## 常见问题

### 评估 YOLO11 模型性能的关键指标是什么？

评估 YOLO11 模型性能的重要指标包括置信度分数、交并比（IoU）和平均精度均值（mAP）。置信度分数衡量模型对每个检测到的对象类别的确定性。IoU 评估预测边界框与真实边界框的重叠程度。平均精度均值（mAP）汇总跨类别的精度分数，mAP@.5 和 mAP@.5:.95 是两种常见类型，用于不同的 IoU 阈值。在我们的 [YOLO11 性能指标指南](./yolo-performance-metrics.md)中了解更多关于这些指标的信息。

### 如何为我的特定数据集微调预训练的 YOLO11 模型？

微调预训练的 YOLO11 模型涉及调整其参数以提高特定任务或数据集上的性能。首先使用指标评估您的模型，然后通过将 `warmup_epochs` 参数调整为 0 来设置较高的初始学习率以获得即时稳定性。使用 `rect=true` 等参数有效处理不同的图像大小。有关更详细的指导，请参阅我们关于[微调 YOLO11 模型](#微调是如何工作的)的部分。

### 在评估 YOLO11 模型时如何处理可变图像大小？

要在评估期间处理可变图像大小，请在 YOLO11 中使用 `rect=true` 参数，它根据图像大小为每个批次调整网络的步幅。`imgsz` 参数设置图像调整大小的最大维度，默认为 640。调整 `imgsz` 以适合您的数据集和 GPU 内存。有关更多详情，请访问我们关于[处理可变图像大小](#处理可变图像大小)的部分。

### 我可以采取哪些实际步骤来提高 YOLO11 模型的平均精度均值？

提高 YOLO11 模型的平均精度均值（mAP）涉及几个步骤：

1. **调整超参数**：尝试不同的学习率、[批量大小](https://www.ultralytics.com/glossary/batch-size)和图像增强。
2. **[数据增强](https://www.ultralytics.com/glossary/data-augmentation)**：使用 Mosaic 和 MixUp 等技术创建多样化的训练样本。
3. **图像切片**：将较大的图像分成较小的切片以提高小对象的检测精度。
   有关具体策略，请参阅我们关于[模型微调](#微调模型的技巧)的详细指南。

### 如何在 Python 中访问 YOLO11 模型评估指标？

您可以使用以下步骤通过 Python 访问 YOLO11 模型评估指标：

!!! example "用法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")

        # 运行评估
        results = model.val(data="coco8.yaml")

        # 打印特定指标
        print("具有平均精度的类别索引:", results.ap_class_index)
        print("所有类别的平均精度:", results.box.all_ap)
        print("IoU=0.50 时的平均精度均值:", results.box.map50)
        print("平均召回率:", results.box.mr)
        ```

分析这些指标有助于微调和优化您的 YOLO11 模型。要深入了解，请查看我们关于 [YOLO11 指标](../modes/val.md)的指南。
