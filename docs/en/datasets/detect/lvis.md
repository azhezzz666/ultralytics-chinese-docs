---
comments: true
description: 探索 Facebook AI Research 的 LVIS 数据集，这是一个用于目标检测和实例分割的大规模、细粒度词汇级标注基准数据集。了解如何使用它。
keywords: LVIS 数据集, 目标检测, 实例分割, Facebook AI Research, YOLO, 计算机视觉, 模型训练, LVIS 示例
---

# LVIS 数据集

[LVIS 数据集](https://www.lvisdataset.org/)是由 Facebook AI Research（FAIR）开发和发布的大规模、细粒度词汇级标注数据集。它主要用作具有大量类别词汇的目标检测和[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)研究基准，旨在推动计算机视觉领域的进一步发展。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/cfTKj96TjSE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 LVIS 数据集的 YOLO World 训练工作流程
</p>

<p align="center">
    <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/lvis-dataset-example-images.avif" alt="LVIS 数据集示例图像">
</p>

## 主要特点

- LVIS 包含 16 万张图像和 200 万个实例标注，用于目标检测、分割和图像描述任务。
- 该数据集包含 1203 个目标类别，包括汽车、自行车和动物等常见物体，以及雨伞、手提包和运动器材等更具体的类别。
- 标注包括每张图像的目标边界框、分割掩码和图像描述。
- LVIS 提供标准化的评估指标，如目标检测的[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）和分割任务的平均[召回率](https://www.ultralytics.com/glossary/recall)均值（mAR），使其适合比较模型性能。
- LVIS 使用与 [COCO](./coco.md) 数据集完全相同的图像，但具有不同的划分和不同的标注。

## 数据集结构

LVIS 数据集分为三个子集：

1. **Train**：此子集包含 10 万张用于训练目标检测、分割和图像描述模型的图像。
2. **Val**：此子集包含 2 万张用于模型训练期间验证的图像。
3. **Minival**：此子集与 COCO val2017 集完全相同，包含 5000 张用于模型训练期间验证的图像。
4. **Test**：此子集包含 2 万张用于测试和基准测试已训练模型的图像。此子集的真实标注不公开，结果需提交到 [LVIS 评估服务器](https://eval.ai/web/challenges/challenge-page/675/overview)进行性能评估。


## 应用

LVIS 数据集广泛用于训练和评估目标检测（如 [YOLO](../../models/yolo11.md)、[Faster R-CNN](https://arxiv.org/abs/1506.01497) 和 [SSD](https://arxiv.org/abs/1512.02325)）和实例分割（如 [Mask R-CNN](https://arxiv.org/abs/1703.06870)）中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。该数据集多样化的目标类别、大量的标注图像和标准化的评估指标使其成为计算机视觉研究人员和从业者的重要资源。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含有关数据集路径、类别和其他相关信息。对于 LVIS 数据集，`lvis.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml)。

!!! example "ultralytics/cfg/datasets/lvis.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/lvis.yaml"
    ```

## 使用方法

要在 LVIS 数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="lvis.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=lvis.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

LVIS 数据集包含各种目标类别和复杂场景的多样化图像集。以下是数据集中的一些图像示例及其相应的标注：

![LVIS 数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/lvis-mosaiced-training-batch.avif)

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是一种在训练期间使用的技术，将多张图像合并为一张图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 LVIS 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克技术的好处。

## 引用和致谢

如果您在研究或开发工作中使用 LVIS 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{gupta2019lvis,
          title={LVIS: A Dataset for Large Vocabulary Instance Segmentation},
          author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
          booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
          year={2019}
        }
        ```

我们感谢 LVIS 联盟为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源。有关 LVIS 数据集及其创建者的更多信息，请访问 [LVIS 数据集网站](https://www.lvisdataset.org/)。

## 常见问题

### 什么是 LVIS 数据集，它在计算机视觉中如何使用？

[LVIS 数据集](https://www.lvisdataset.org/)是由 Facebook AI Research（FAIR）开发的具有细粒度词汇级标注的大规模数据集。它主要用于目标检测和实例分割，具有超过 1203 个目标类别和 200 万个实例标注。研究人员和从业者使用它来训练和基准测试 Ultralytics YOLO 等模型，用于高级计算机视觉任务。该数据集的广泛规模和多样性使其成为推动检测和分割模型性能边界的重要资源。

### 如何使用 LVIS 数据集训练 YOLO11n 模型？

要在 LVIS 数据集上训练 YOLO11n 模型 100 个训练周期，图像尺寸为 640，请按照以下示例操作。此过程利用 Ultralytics 框架，提供全面的训练功能。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="lvis.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=lvis.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关详细的训练配置，请参阅[训练](../../modes/train.md)文档。

### LVIS 数据集与 COCO 数据集有何不同？

LVIS 数据集中的图像与 [COCO 数据集](./coco.md)中的图像相同，但两者在划分和标注方面有所不同。LVIS 提供了更大、更详细的词汇表，包含 1203 个目标类别，而 COCO 有 80 个类别。此外，LVIS 专注于标注的完整性和多样性，旨在通过提供更细致和全面的数据来推动[目标检测](https://www.ultralytics.com/glossary/object-detection)和实例分割模型的极限。

### 为什么应该使用 Ultralytics YOLO 在 LVIS 数据集上训练？

Ultralytics YOLO 模型（包括最新的 YOLO11）针对实时目标检测进行了优化，具有最先进的[准确率](https://www.ultralytics.com/glossary/accuracy)和速度。它们支持广泛的标注，如 LVIS 数据集提供的细粒度标注，使其非常适合高级计算机视觉应用。此外，Ultralytics 提供与各种[训练](../../modes/train.md)、[验证](../../modes/val.md)和[预测](../../modes/predict.md)模式的无缝集成，确保高效的模型开发和部署。

### 我可以查看 LVIS 数据集的一些示例标注吗？

是的，LVIS 数据集包含各种具有多样化目标类别和复杂场景的图像。以下是示例图像及其标注：

![LVIS 数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/lvis-mosaiced-training-batch.avif)

此马赛克图像展示了由多个数据集图像组合成一张的训练批次。马赛克增加了每个训练批次中目标和场景的多样性，增强了模型在不同上下文中的泛化能力。有关 LVIS 数据集的更多详情，请探索 [LVIS 数据集文档](#主要特点)。
