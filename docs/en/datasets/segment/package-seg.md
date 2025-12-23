---
comments: true
description: 探索包裹分割数据集。优化物流并使用精心策划的图像增强视觉模型，用于包裹识别和分拣。
keywords: 包裹分割数据集, 计算机视觉, 包裹识别, 物流, 仓库自动化, 分割模型, 训练数据, Ultralytics YOLO
---

# 包裹分割数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-package-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开包裹分割数据集"></a>

包裹分割数据集可在 Roboflow Universe 上获取，是一个精心策划的图像集合，专门为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域中与包裹分割相关的任务而设计。该数据集旨在帮助从事包裹识别、分拣和处理项目的研究人员、开发者和爱好者，主要关注[图像分割](https://www.ultralytics.com/glossary/image-segmentation)任务。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/im7xBCnPURg"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO11 训练包裹分割模型 | 工业包裹 🎉
</p>

该数据集包含展示各种环境和场景中不同包裹的多样化图像集，是训练和评估分割模型的宝贵资源。无论您从事物流、仓库自动化还是任何需要精确包裹分析的应用，包裹分割数据集都提供了一组有针对性的综合图像，以增强计算机视觉算法的性能。在我们的[数据集概览页面](https://docs.ultralytics.com/datasets/segment/)探索更多分割任务数据集。

## 数据集结构

包裹分割数据集的数据分布结构如下：

- **训练集**：包含 1920 张图像及其对应的标注。
- **测试集**：包含 89 张图像，每张都配有相应的标注。
- **验证集**：包含 188 张图像，每张都有对应的标注。

## 应用场景

包裹分割由包裹分割数据集支持，对于优化物流、增强最后一公里配送、改善制造质量控制以及为智慧城市解决方案做出贡献至关重要。从电子商务到安全应用，该数据集是一个关键资源，促进计算机视觉在多样化和高效包裹分析应用中的创新。

### 智能仓库和物流

在现代仓库中，[视觉 AI 解决方案](https://www.ultralytics.com/solutions)可以通过自动化包裹识别和分拣来简化操作。在该数据集上训练的计算机视觉模型可以快速实时检测和分割包裹，即使在光线昏暗或杂乱的环境中也能工作。这可以加快处理时间、减少错误并提高[物流运营](https://www.ultralytics.com/blog/ultralytics-yolo11-the-key-to-computer-vision-in-logistics)的整体效率。

### 质量控制和损坏检测

包裹分割模型可以通过分析包裹的形状和外观来识别损坏的包裹。通过检测包裹轮廓中的不规则或变形，这些模型有助于确保只有完好的包裹通过供应链，减少客户投诉和退货率。这是[制造业质量控制](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision)的关键方面，对于保持产品完整性至关重要。

## 数据集 YAML

YAML（Yet Another Markup Language）文件定义数据集配置，包括路径、类别和其他基本详细信息。对于包裹分割数据集，`package-seg.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml)。

!!! example "ultralytics/cfg/datasets/package-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/package-seg.yaml"
    ```


## 使用方法

要在包裹分割数据集上训练 [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/) 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练页面](../../modes/train.md)。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-seg.pt")  # 加载预训练的分割模型（推荐用于训练）

        # 在包裹分割数据集上训练模型
        results = model.train(data="package-seg.yaml", epochs=100, imgsz=640)

        # 验证模型
        results = model.val()

        # 对图像进行推理
        results = model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        # 加载预训练的分割模型并开始训练
        yolo segment train data=package-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640

        # 从上一个检查点恢复训练
        yolo segment train data=package-seg.yaml model=path/to/last.pt resume=True

        # 验证训练好的模型
        yolo segment val data=package-seg.yaml model=path/to/best.pt

        # 使用训练好的模型进行推理
        yolo segment predict model=path/to/best.pt source=path/to/image.jpg
        ```

## 示例数据和标注

包裹分割数据集包含从多个角度拍摄的各种图像集合。以下是数据集中的数据实例及其对应的分割掩码：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/dataset-sample-image-1.avif)

- 此图像展示了包裹分割的实例，带有标注掩码勾勒出识别的包裹目标。数据集包含在不同位置、环境和密度下拍摄的多样化图像集合。它是开发特定于此[分割任务](https://docs.ultralytics.com/tasks/segment/)模型的综合资源。
- 该示例强调了数据集中存在的多样性和复杂性，突出了高质量数据对涉及包裹分割的计算机视觉任务的重要性。

## 使用 YOLO11 进行包裹分割的优势

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) 为包裹分割任务提供了多项优势：

1. **速度和准确性平衡**：YOLO11 实现了高精度和高效率，非常适合快节奏物流环境中的[实时推理](https://www.ultralytics.com/glossary/real-time-inference)。与 [YOLOv8](https://docs.ultralytics.com/models/yolov8/) 等模型相比，它提供了强大的平衡。

2. **适应性**：使用 YOLO11 训练的模型可以适应各种仓库条件，从昏暗的光线到杂乱的空间，确保稳健的性能。

3. **可扩展性**：在假日季节等高峰期，YOLO11 模型可以高效扩展以处理增加的包裹量，而不会影响性能或[准确性](https://www.ultralytics.com/glossary/accuracy)。

4. **集成能力**：YOLO11 可以轻松与现有仓库管理系统集成，并使用 [ONNX](https://docs.ultralytics.com/integrations/onnx/) 或 [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) 等格式跨各种平台部署，促进端到端自动化解决方案。

## 引用和致谢

如果您在研究或开发工作中使用包裹分割数据集，请适当引用来源：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ factory_package_dataset,
            title = { factory_package Dataset },
            type = { Open Source Dataset },
            author = { factorypackage },
            url = { https://universe.roboflow.com/factorypackage/factory_package },
            journal = { Roboflow Universe },
            publisher = { Roboflow },
            year = { 2024 },
            month = { jan },
            note = { visited on 2024-01-24 },
        }
        ```

我们感谢包裹分割数据集的创建者对计算机视觉社区的贡献。有关更多数据集和模型训练的探索，请考虑访问我们的 [Ultralytics 数据集](https://docs.ultralytics.com/datasets/)页面和[模型训练技巧](https://docs.ultralytics.com/guides/model-training-tips/)指南。

## 常见问题

### 什么是包裹分割数据集，它如何帮助计算机视觉项目？

- 包裹分割数据集是一个精心策划的图像集合，专为涉及包裹[图像分割](https://www.ultralytics.com/glossary/image-segmentation)的任务而设计。它包含各种场景中的多样化包裹图像，对于训练和评估分割模型非常有价值。该数据集特别适用于物流、仓库自动化以及任何需要精确包裹分析的项目。

### 如何在包裹分割数据集上训练 Ultralytics YOLO11 模型？

- 您可以使用 Python 和 CLI 方法训练 [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) 模型。使用[使用方法](#使用方法)部分提供的代码片段。有关参数和配置的更多详细信息，请参阅模型[训练页面](../../modes/train.md)。

### 包裹分割数据集的组成部分是什么，它的结构如何？

- 数据集分为三个主要组成部分：
    - **训练集**：包含 1920 张带标注的图像。
    - **测试集**：包含 89 张带对应标注的图像。
    - **验证集**：包含 188 张带标注的图像。
- 这种结构确保了用于全面模型训练、验证和测试的平衡数据集，遵循[模型评估指南](https://docs.ultralytics.com/guides/model-evaluation-insights/)中概述的最佳实践。

### 为什么应该将 Ultralytics YOLO11 与包裹分割数据集一起使用？

- Ultralytics YOLO11 为实时[目标检测](https://www.ultralytics.com/glossary/object-detection)和分割任务提供最先进的[准确性](https://www.ultralytics.com/glossary/accuracy)和速度。将其与包裹分割数据集一起使用，可以利用 YOLO11 的精确包裹分割能力，这对[物流](https://www.ultralytics.com/blog/ultralytics-yolo11-the-key-to-computer-vision-in-logistics)和仓库自动化等行业特别有益。

### 在哪里可以访问包裹分割数据集的 package-seg.yaml 文件？

- `package-seg.yaml` 文件托管在 Ultralytics 的 GitHub 仓库中，包含有关数据集路径、类别和配置的基本信息。您可以在 <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml> 查看或下载它。该文件对于高效配置模型以使用数据集至关重要。有关更多见解和实际示例，请探索我们的 [Python 使用](https://docs.ultralytics.com/usage/python/)部分。
