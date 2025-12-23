---
comments: true
description: 探索用于高级姿态估计的 COCO-Pose 数据集。了解数据集、预训练模型、指标和使用 YOLO 训练的应用。
keywords: COCO-Pose, 姿态估计, 数据集, 关键点, COCO Keypoints 2017, YOLO, 深度学习, 计算机视觉
---

# COCO-Pose 数据集

[COCO-Pose](https://cocodataset.org/#keypoints-2017) 数据集是 COCO（Common Objects in Context）数据集的专门版本，专为姿态估计任务设计。它利用 COCO Keypoints 2017 图像和标签来训练像 YOLO 这样的姿态估计模型。

![姿态示例图像](https://github.com/ultralytics/docs/releases/download/0/pose-sample-image.avif)

## COCO-Pose 预训练模型

{% include "macros/yolo-pose-perf.md" %}

## 主要特点

- COCO-Pose 基于 COCO Keypoints 2017 数据集构建，包含 20 万张标注了关键点的图像用于姿态估计任务。
- 数据集支持人体的 17 个关键点，便于详细的姿态估计。
- 与 COCO 一样，它提供标准化的评估指标，包括用于姿态估计任务的目标关键点相似度（OKS），使其适合比较模型性能。

## 数据集结构

COCO-Pose 数据集分为三个子集：

1. **Train2017**：该子集包含 COCO 数据集中的 56599 张图像，标注用于训练姿态估计模型。
2. **Val2017**：该子集包含 2346 张图像，用于模型训练期间的验证。
3. **Test2017**：该子集包含用于测试和基准测试训练模型的图像。该子集的真实标注不公开，结果需提交到 [COCO 评估服务器](https://codalab.lisn.upsaclay.fr/competitions/7384)进行性能评估。

## 应用场景

COCO-Pose 数据集专门用于训练和评估关键点检测和姿态估计任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，如 OpenPose。数据集大量的标注图像和标准化评估指标使其成为专注于姿态估计的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)研究人员和从业者的重要资源。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 COCO-Pose 数据集，`coco-pose.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml)。

!!! example "ultralytics/cfg/datasets/coco-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco-pose.yaml"
    ```

## 使用方法

要在 COCO-Pose 数据集上训练 YOLO11n-pose 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo pose train data=coco-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

COCO-Pose 数据集包含多样化的图像集，其中人体标注了关键点。以下是数据集中的一些图像示例及其对应的标注：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-6.avif)

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是训练期间使用的一种技术，将多个图像组合成单个图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 COCO-Pose 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

## 引用和致谢

如果您在研究或开发工作中使用 COCO-Pose 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们感谢 COCO 联盟为计算机视觉社区创建和维护这一宝贵资源。有关 COCO-Pose 数据集及其创建者的更多信息，请访问 [COCO 数据集网站](https://cocodataset.org/#home)。

## 常见问题

### 什么是 COCO-Pose 数据集，如何与 Ultralytics YOLO 一起用于姿态估计？

[COCO-Pose](https://cocodataset.org/#keypoints-2017) 数据集是 COCO（Common Objects in Context）数据集的专门版本，专为姿态估计任务设计。它基于 COCO Keypoints 2017 图像和标注构建，允许训练像 Ultralytics YOLO 这样的模型进行详细的姿态估计。例如，您可以通过加载预训练模型并使用 YAML 配置进行训练，使用 COCO-Pose 数据集训练 YOLO11n-pose 模型。有关训练示例，请参阅[训练](../../modes/train.md)文档。

### 如何在 COCO-Pose 数据集上训练 YOLO11 模型？

在 COCO-Pose 数据集上训练 YOLO11 模型可以使用 Python 或 CLI 命令完成。例如，要训练 YOLO11n-pose 模型 100 个轮次、图像尺寸为 640，可以按照以下步骤操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo pose train data=coco-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

有关训练过程和可用参数的更多详细信息，请查看[训练页面](../../modes/train.md)。

### COCO-Pose 数据集提供哪些不同的评估模型性能的指标？

COCO-Pose 数据集为姿态估计任务提供了几个标准化的评估指标，类似于原始 COCO 数据集。关键指标包括目标关键点相似度（OKS），它评估预测关键点与真实标注的[准确率](https://www.ultralytics.com/glossary/accuracy)。这些指标允许在不同模型之间进行全面的性能比较。例如，COCO-Pose 预训练模型如 YOLO11n-pose、YOLO11s-pose 等在文档中列出了特定的性能指标，如 mAP<sup>pose</sup>50-95 和 mAP<sup>pose</sup>50。

### COCO-Pose 数据集的数据集结构和分割是怎样的？

COCO-Pose 数据集分为三个子集：

1. **Train2017**：包含 56599 张 COCO 图像，标注用于训练姿态估计模型。
2. **Val2017**：2346 张图像，用于模型训练期间的验证。
3. **Test2017**：用于测试和基准测试训练模型的图像。该子集的真实标注不公开；结果需提交到 [COCO 评估服务器](https://codalab.lisn.upsaclay.fr/competitions/7403)进行性能评估。

这些子集有助于有效组织训练、验证和测试阶段。有关配置详细信息，请探索 [GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) 上的 `coco-pose.yaml` 文件。

### COCO-Pose 数据集的主要特点和应用是什么？

COCO-Pose 数据集扩展了 COCO Keypoints 2017 标注，包含人体的 17 个关键点，实现详细的姿态估计。标准化评估指标（如 OKS）便于在不同模型之间进行比较。COCO-Pose 数据集的应用涵盖各个领域，如体育分析、医疗保健和人机交互，无论何处需要人体的详细姿态估计。对于实际使用，利用文档中提供的预训练模型（如 YOLO11n-pose）可以显著简化流程（[主要特点](#主要特点)）。

如果您在研究或开发工作中使用 COCO-Pose 数据集，请使用以下 [BibTeX 条目](#引用和致谢)引用论文。
