---
comments: true
description: 探索全球小麦穗数据集，用于开发准确的小麦穗检测模型。包括训练图像、标注和作物管理使用方法。
keywords: 全球小麦穗数据集, 小麦穗检测, 小麦表型分析, 作物管理, 深度学习, 目标检测, 训练数据集
---

# 全球小麦穗数据集

[全球小麦穗数据集](https://www.global-wheat.com/)是一个图像集合，旨在支持开发准确的小麦穗检测模型，用于小麦表型分析和作物管理应用。小麦穗，也称为麦穗，是小麦植株的结实部分。准确估计小麦穗密度和大小对于评估作物健康、成熟度和产量潜力至关重要。该数据集由来自七个国家的九个研究机构合作创建，涵盖多个种植区域，以确保模型在不同环境中具有良好的泛化能力。

## 主要特点

- 该数据集包含来自欧洲（法国、英国、瑞士）和北美（加拿大）的 3,000 多张训练图像。
- 它包含来自澳大利亚、日本和中国的约 1,000 张测试图像。
- 图像是户外田间图像，捕捉了小麦穗外观的自然变异性。
- 标注包括小麦穗边界框，以支持[目标检测](https://docs.ultralytics.com/tasks/detect/)任务。

## 数据集结构

全球小麦穗数据集分为两个主要子集：

1. **训练集**：此子集包含来自欧洲和北美的 3,000 多张图像。图像标注了小麦穗边界框，为训练目标检测模型提供真实标签。
2. **测试集**：此子集包含来自澳大利亚、日本和中国的约 1,000 张图像。这些图像用于评估训练模型在未见过的基因型、环境和观察条件下的性能。

## 应用

全球小麦穗数据集广泛用于训练和评估小麦穗检测任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。该数据集多样化的图像集捕捉了广泛的外观、环境和条件，使其成为[植物表型分析](https://www.ultralytics.com/blog/computer-vision-in-agriculture-transforming-fruit-detection-and-precision-farming)和作物管理领域研究人员和从业者的宝贵资源。


## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含有关数据集路径、类别和其他相关信息。对于全球小麦穗数据集，`GlobalWheat2020.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml)。

!!! example "ultralytics/cfg/datasets/GlobalWheat2020.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/GlobalWheat2020.yaml"
    ```

## 使用方法

要在全球小麦穗数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=GlobalWheat2020.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例数据和标注

全球小麦穗数据集包含多样化的户外田间图像集，捕捉了小麦穗外观、环境和条件的自然变异性。以下是数据集中的一些数据示例及其相应的标注：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/wheat-head-detection-sample.avif)

- **小麦穗检测**：此图像展示了小麦穗检测的示例，其中小麦穗用边界框标注。该数据集提供各种图像以促进此任务模型的开发。

该示例展示了全球小麦穗数据集中数据的多样性和复杂性，并强调了准确的小麦穗检测对于小麦表型分析和作物管理应用的重要性。

## 引用和致谢

如果您在研究或开发工作中使用全球小麦穗数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{david2020global,
                 title={Global Wheat Head Detection (GWHD) Dataset: A Large and Diverse Dataset of High-Resolution RGB-Labelled Images to Develop and Benchmark Wheat Head Detection Methods},
                 author={David, Etienne and Madec, Simon and Sadeghi-Tehran, Pouria and Aasen, Helge and Zheng, Bangyou and Liu, Shouyang and Kirchgessner, Norbert and Ishikawa, Goro and Nagasawa, Koichi and Badhon, Minhajul and others},
                 journal={arXiv preprint arXiv:2005.02162},
                 year={2020}
        }
        ```

我们感谢为创建和维护全球小麦穗数据集做出贡献的研究人员和机构，这是植物表型分析和作物管理研究社区的宝贵资源。有关数据集及其创建者的更多信息，请访问[全球小麦穗数据集网站](https://www.global-wheat.com/)。

## 常见问题

### 全球小麦穗数据集用于什么？

全球小麦穗数据集主要用于开发和训练旨在小麦穗检测的深度学习模型。这对于[小麦表型分析](https://www.ultralytics.com/blog/from-farm-to-table-how-ai-drives-innovation-in-agriculture)和作物管理应用至关重要，可以更准确地估计小麦穗密度、大小和整体作物产量潜力。准确的检测方法有助于评估作物健康和成熟度，这对于高效的作物管理至关重要。

### 如何在全球小麦穗数据集上训练 YOLO11n 模型？

要在全球小麦穗数据集上训练 YOLO11n 模型，您可以使用以下代码片段。确保您有指定数据集路径和类别的 `GlobalWheat2020.yaml` 配置文件：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo11n.pt")

        # 训练模型
        results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=GlobalWheat2020.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### 全球小麦穗数据集的主要特点是什么？

全球小麦穗数据集的主要特点包括：

- 来自欧洲（法国、英国、瑞士）和北美（加拿大）的 3,000 多张训练图像。
- 来自澳大利亚、日本和中国的约 1,000 张测试图像。
- 由于不同的生长环境，小麦穗外观具有高度变异性。
- 带有小麦穗边界框的详细标注，以辅助[目标检测](https://www.ultralytics.com/glossary/object-detection)模型。

这些特点促进了能够跨多个区域泛化的鲁棒模型的开发。

### 在哪里可以找到全球小麦穗数据集的配置 YAML 文件？

全球小麦穗数据集的配置 YAML 文件名为 `GlobalWheat2020.yaml`，可在 GitHub 上获取。您可以在 <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml> 访问它。此文件包含有关数据集路径、类别和在 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/) 中进行模型训练所需的其他配置详情的必要信息。

### 为什么小麦穗检测在作物管理中很重要？

小麦穗检测在作物管理中至关重要，因为它能够准确估计小麦穗密度和大小，这对于评估作物健康、成熟度和产量潜力至关重要。通过利用在全球小麦穗数据集等数据集上训练的[深度学习模型](https://docs.ultralytics.com/models/)，农民和研究人员可以更好地监测和管理作物，从而提高生产力并优化农业实践中的资源使用。这一技术进步支持[可持续农业](https://www.ultralytics.com/blog/real-time-crop-health-monitoring-with-ultralytics-yolo11)和粮食安全倡议。

有关 AI 在农业中应用的更多信息，请访问[农业中的 AI](https://www.ultralytics.com/solutions/ai-in-agriculture)。
