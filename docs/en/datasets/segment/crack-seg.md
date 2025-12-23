---
comments: true
description: 探索广泛的裂缝分割数据集，非常适合交通安全、基础设施维护和自动驾驶汽车模型开发，使用 Ultralytics YOLO。
keywords: 裂缝分割数据集, Ultralytics, 交通安全, 公共安全, 自动驾驶汽车, 计算机视觉, 道路安全, 基础设施维护, 数据集, YOLO, 分割, 深度学习
---

# 裂缝分割数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-crack-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开裂缝分割数据集"></a>

裂缝分割数据集可在 Roboflow Universe 上获取，是一个广泛的资源，专为从事交通和公共安全研究的人员设计。它也有助于开发[自动驾驶汽车](https://www.ultralytics.com/blog/ai-in-self-driving-cars)模型或探索各种[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用。该数据集是 Ultralytics [数据集中心](../../datasets/index.md)更广泛集合的一部分。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/C4mc40YKm-g"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLOv9 进行裂缝分割。
</p>

该数据集包含 4029 张从不同道路和墙壁场景拍摄的静态图像，是裂缝分割任务的宝贵资产。无论您是研究交通基础设施还是旨在提高自动驾驶系统的[准确性](https://www.ultralytics.com/glossary/accuracy)，该数据集都提供了丰富的图像集合用于训练[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。

## 数据集结构

裂缝分割数据集分为三个子集：

- **训练集**：3717 张图像及其对应的标注。
- **测试集**：112 张图像及其对应的标注。
- **验证集**：200 张图像及其对应的标注。

## 应用场景

裂缝分割在[基础设施维护](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation)中有实际应用，有助于识别和评估建筑物、桥梁和道路的结构损坏。它还通过使自动化系统能够检测路面裂缝以进行及时维修，在增强[道路安全](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries)方面发挥着关键作用。

在工业环境中，使用 [Ultralytics YOLO11](../../models/yolo11.md) 等深度学习模型进行裂缝检测有助于确保建筑完整性、防止[制造业](https://www.ultralytics.com/solutions/ai-in-manufacturing)中的昂贵停机时间，并使道路检查更安全、更有效。自动识别和分类裂缝使维护团队能够高效地确定维修优先级，有助于更好的[模型评估洞察](../../guides/model-evaluation-insights.md)。

## 数据集 YAML

[YAML](https://www.ultralytics.com/glossary/yaml)（Yet Another Markup Language）文件定义数据集配置。它包含数据集路径、类别和其他相关信息的详细信息。对于裂缝分割数据集，`crack-seg.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml)。

!!! example "ultralytics/cfg/datasets/crack-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/crack-seg.yaml"
    ```


## 使用方法

要在裂缝分割数据集上训练 Ultralytics YOLO11n-seg 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，请使用以下 [Python](https://www.python.org/) 或 CLI 代码片段。有关可用参数和配置（如[超参数调优](../../guides/hyperparameter-tuning.md)）的完整列表，请参阅模型[训练](../../modes/train.md)文档页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        # 推荐使用预训练模型如 yolo11n-seg.pt 以加快收敛
        model = YOLO("yolo11n-seg.pt")

        # 在裂缝分割数据集上训练模型
        # 确保 'crack-seg.yaml' 可访问或提供完整路径
        results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640)

        # 训练后，模型可用于预测或导出
        # results = model.predict(source='path/to/your/images')
        ```

    === "CLI"

        ```bash
        # 使用命令行界面从预训练的 *.pt 模型开始训练
        # 确保数据集 YAML 文件 'crack-seg.yaml' 正确配置且可访问
        yolo segment train data=crack-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

## 示例数据和标注

裂缝分割数据集包含从各种角度拍摄的多样化图像集合，展示了道路和墙壁上不同类型的裂缝。以下是一些示例：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/crack-segmentation-sample.avif)

- 此图像展示了[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)，带有标注[边界框](https://www.ultralytics.com/glossary/bounding-box)和掩码勾勒出识别的裂缝。数据集包含来自不同位置和环境的图像，使其成为开发此任务鲁棒模型的综合资源。[数据增强](https://www.ultralytics.com/glossary/data-augmentation)等技术可以进一步增强数据集多样性。在我们的[指南](../../guides/instance-segmentation-and-tracking.md)中了解更多关于实例分割和跟踪的信息。

- 该示例突出了裂缝分割数据集的多样性，强调了高质量数据对训练有效计算机视觉模型的重要性。

## 引用和致谢

如果您在研究或开发工作中使用裂缝分割数据集，请适当引用来源。该数据集通过 Roboflow 提供：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ crack-bphdr_dataset,
            title = { crack Dataset },
            type = { Open Source Dataset },
            author = { University },
            url = { https://universe.roboflow.com/university-bswxt/crack-bphdr },
            journal = { Roboflow Universe },
            publisher = { Roboflow },
            year = { 2022 },
            month = { dec },
            note = { visited on 2024-01-23 },
        }
        ```

我们感谢 Roboflow 团队提供裂缝分割数据集，为计算机视觉社区提供了宝贵资源，特别是对于与道路安全和基础设施评估相关的项目。

## 常见问题

### 什么是裂缝分割数据集？

裂缝分割数据集是一个包含 4029 张静态图像的集合，专为交通和公共安全研究设计。它适用于[自动驾驶汽车](https://www.ultralytics.com/blog/ai-in-self-driving-cars)模型开发和[基础设施维护](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation)等任务。它包含用于裂缝检测和[分割](../../tasks/segment.md)任务的训练、测试和验证集。

### 如何使用裂缝分割数据集和 Ultralytics YOLO11 训练模型？

要在此数据集上训练 [Ultralytics YOLO11](../../models/yolo11.md) 模型，请使用提供的 Python 或 CLI 示例。详细说明和参数可在模型[训练](../../modes/train.md)页面找到。您可以使用 [Ultralytics HUB](https://www.ultralytics.com/hub) 等工具管理训练过程。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型（推荐）
        model = YOLO("yolo11n-seg.pt")

        # 训练模型
        results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 通过 CLI 从预训练模型开始训练
        yolo segment train data=crack-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

### 为什么将裂缝分割数据集用于自动驾驶汽车项目？

该数据集对自动驾驶汽车项目很有价值，因为它包含涵盖各种真实场景的多样化道路和墙壁图像。这种多样性提高了为裂缝检测训练的模型的鲁棒性，这对道路安全和基础设施评估至关重要。详细的标注有助于[开发模型](../../guides/model-training-tips.md)，能够准确识别潜在的道路危险。

### Ultralytics YOLO 为裂缝分割提供了哪些功能？

Ultralytics YOLO 提供实时[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和分类功能，非常适合裂缝分割任务。它能高效处理大型数据集和复杂场景。该框架包括用于[训练](../../modes/train.md)、[预测](../../modes/predict.md)和[导出](../../modes/export.md)模型的综合模式。YOLO 的[无锚检测](https://www.ultralytics.com/blog/benefits-ultralytics-yolo11-being-anchor-free-detector)方法可以提高对裂缝等不规则形状的性能，性能可以使用标准[指标](../../guides/yolo-performance-metrics.md)进行测量。

### 如何引用裂缝分割数据集？

如果在您的工作中使用此数据集，请使用上面提供的 BibTeX 条目引用它，以适当致谢创建者。
