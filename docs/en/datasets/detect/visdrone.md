---
comments: true
description: 探索 VisDrone 数据集，这是一个用于无人机图像和视频分析的大规模基准数据集，包含超过 260 万个行人和车辆等目标的标注。
keywords: VisDrone, 无人机数据集, 计算机视觉, 目标检测, 目标跟踪, 人群计数, 机器学习, 深度学习
---

# VisDrone 数据集

[VisDrone 数据集](https://github.com/VisDrone/VisDrone-Dataset)是由中国天津大学[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)与数据挖掘实验室 AISKYEYE 团队创建的大规模基准数据集。它包含精心标注的真实数据，用于与无人机图像和视频分析相关的各种计算机视觉任务。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/9ymyH4H1fG4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 VisDrone 数据集上训练 Ultralytics YOLO11 | 航拍检测 | 完整教程 🚀
</p>

VisDrone 由 288 个视频片段（包含 261,908 帧）和 10,209 张静态图像组成，由各种无人机搭载的相机拍摄。数据集涵盖广泛的方面，包括地点（中国 14 个不同城市）、环境（城市和农村）、目标（行人、车辆、自行车等）和密度（稀疏和拥挤场景）。数据集使用各种无人机平台在不同场景、天气和光照条件下收集。这些帧手动标注了超过 260 万个目标边界框，如行人、汽车、自行车和三轮车。还提供了场景可见性、目标类别和遮挡等属性，以便更好地利用数据。

## 数据集结构

VisDrone 数据集分为五个主要子集，每个子集专注于特定任务：

1. **任务 1**：图像中的目标检测
2. **任务 2**：视频中的目标检测
3. **任务 3**：单目标跟踪
4. **任务 4**：[多目标跟踪](../index.md#multi-object-tracking)
5. **任务 5**：人群计数

## 应用场景

VisDrone 数据集广泛用于训练和评估无人机[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务中的深度学习模型，如目标检测、目标跟踪和人群计数。数据集多样化的传感器数据、目标标注和属性使其成为无人机计算机视觉领域研究人员和从业者的宝贵资源。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 VisDrone 数据集，`VisDrone.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml)。

!!! example "ultralytics/cfg/datasets/VisDrone.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/VisDrone.yaml"
    ```

## 使用方法

要在 VisDrone 数据集上训练 YOLO11n 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=VisDrone.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例数据和标注

VisDrone 数据集包含由无人机搭载相机拍摄的多样化图像和视频。以下是数据集中的一些数据示例及其对应的标注：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/visdrone-object-detection-sample.avif)

- **任务 1**：图像中的[目标检测](https://www.ultralytics.com/glossary/object-detection) - 此图像展示了图像中目标检测的示例，其中目标使用[边界框](https://www.ultralytics.com/glossary/bounding-box)进行标注。数据集提供了从不同地点、环境和密度拍摄的各种图像，以促进该任务模型的开发。

该示例展示了 VisDrone 数据集中数据的多样性和复杂性，并强调了高质量传感器数据对无人机计算机视觉任务的重要性。

## 引用和致谢

如果您在研究或开发工作中使用 VisDrone 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @ARTICLE{9573394,
          author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Detection and Tracking Meet Drones Challenge},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3119563}}
        ```

我们感谢中国天津大学机器学习与[数据挖掘](https://www.ultralytics.com/glossary/data-mining)实验室 AISKYEYE 团队创建和维护 VisDrone 数据集，为无人机计算机视觉研究社区提供了宝贵的资源。有关 VisDrone 数据集及其创建者的更多信息，请访问 [VisDrone 数据集 GitHub 仓库](https://github.com/VisDrone/VisDrone-Dataset)。

## 常见问题

### 什么是 VisDrone 数据集，它有哪些主要特点？

[VisDrone 数据集](https://github.com/VisDrone/VisDrone-Dataset)是由中国天津大学 AISKYEYE 团队创建的大规模基准数据集。它专为与无人机图像和视频分析相关的各种计算机视觉任务而设计。主要特点包括：

- **组成**：288 个视频片段（包含 261,908 帧）和 10,209 张静态图像。
- **标注**：超过 260 万个行人、汽车、自行车和三轮车等目标的边界框。
- **多样性**：在 14 个城市收集，涵盖城市和农村环境，在不同天气和光照条件下拍摄。
- **任务**：分为五个主要任务——图像和视频中的目标检测、单目标和多目标跟踪以及人群计数。

### 如何使用 Ultralytics 在 VisDrone 数据集上训练 YOLO11 模型？

要在 VisDrone 数据集上训练 YOLO11 模型 100 个轮次、图像尺寸为 640，可以按照以下步骤操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 训练模型
        results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=VisDrone.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关其他配置选项，请参阅模型[训练](../../modes/train.md)页面。

### VisDrone 数据集的主要子集及其应用是什么？

VisDrone 数据集分为五个主要子集，每个子集针对特定的计算机视觉任务：

1. **任务 1**：图像中的目标检测。
2. **任务 2**：视频中的目标检测。
3. **任务 3**：单目标跟踪。
4. **任务 4**：多目标跟踪。
5. **任务 5**：人群计数。

这些子集广泛用于训练和评估无人机应用中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，如监控、交通监控和公共安全。

### 在 Ultralytics 中哪里可以找到 VisDrone 数据集的配置文件？

VisDrone 数据集的配置文件 `VisDrone.yaml` 可以在 Ultralytics 仓库的以下链接找到：
[VisDrone.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml)。

### 如果在研究中使用 VisDrone 数据集，如何引用？

如果您在研究或开发工作中使用 VisDrone 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @ARTICLE{9573394,
          author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Detection and Tracking Meet Drones Challenge},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3119563}
        }
        ```
