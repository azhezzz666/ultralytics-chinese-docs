---
comments: true
description: 探索 DOTA8 数据集 - 一个小型、多功能的旋转目标检测数据集，非常适合使用 Ultralytics YOLO11 测试和调试目标检测模型。
keywords: DOTA8 数据集, Ultralytics, YOLO11, 目标检测, 调试, 训练模型, 旋转目标检测, 数据集 YAML
---

# DOTA8 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) DOTA8 是一个小型但多功能的旋转[目标检测](https://www.ultralytics.com/glossary/object-detection)数据集，由 DOTAv1 分割集的前 8 张图像组成，4 张用于训练，4 张用于验证。此数据集非常适合测试和调试目标检测模型，或用于实验新的检测方法。只有 8 张图像，它足够小以便于管理，但又足够多样化以测试训练流程中的错误，并在训练更大数据集之前作为健全性检查。

## 数据集结构

- **图像**：8 张航拍图块（4 张训练，4 张验证），来源于 DOTAv1。
- **类别**：继承 DOTAv1 的 15 个类别，如飞机、船舶和大型车辆。
- **标签**：YOLO 格式的旋转边界框，保存为每张图像旁边的 `.txt` 文件。
- **推荐布局**：

    ```
    datasets/dota8/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    ```

此数据集旨在与 Ultralytics [HUB](https://hub.ultralytics.com/) 和 [YOLO11](https://github.com/ultralytics/ultralytics) 一起使用。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 DOTA8 数据集，`dota8.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml)。

!!! example "ultralytics/cfg/datasets/dota8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dota8.yaml"
    ```

## 使用方法

要在 DOTA8 数据集上训练 YOLO11n-obb 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-obb.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo obb train data=dota8.yaml model=yolo11n-obb.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

以下是 DOTA8 数据集中的一些图像示例及其对应的标注：

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch.avif" alt="数据集示例图像" width="800">

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是训练期间使用的一种技术，将多个图像组合成单个图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 DOTA8 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

## 引用和致谢

如果您在研究或开发工作中使用 DOTA 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{9560031,
          author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3117983}
        }
        ```

特别感谢 DOTA 数据集团队在策划此数据集方面的杰出努力。有关数据集及其细节的详尽了解，请访问[官方 DOTA 网站](https://captain-whu.github.io/DOTA/index.html)。

## 常见问题

### 什么是 DOTA8 数据集，如何使用？

DOTA8 数据集是一个小型、多功能的旋转目标检测数据集，由 DOTAv1 分割集的前 8 张图像组成，4 张用于训练，4 张用于验证。它非常适合测试和调试像 Ultralytics YOLO11 这样的目标检测模型。由于其可管理的大小和多样性，它有助于识别流程错误并在部署更大数据集之前运行健全性检查。了解更多关于使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 进行目标检测的信息。

### 如何使用 DOTA8 数据集训练 YOLO11 模型？

要在 DOTA8 数据集上训练 YOLO11n-obb 模型 100 个轮次、图像尺寸为 640，可以使用以下代码片段。有关完整的参数选项，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-obb.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo obb train data=dota8.yaml model=yolo11n-obb.pt epochs=100 imgsz=640
        ```

### DOTA 数据集的主要特点是什么，在哪里可以访问 YAML 文件？

DOTA 数据集以其大规模基准和航拍图像目标检测的挑战而闻名。DOTA8 子集是一个较小、可管理的数据集，非常适合初始测试。您可以在此 [GitHub 链接](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml)访问 `dota8.yaml` 文件，其中包含路径、类别和配置详细信息。

### 马赛克如何增强使用 DOTA8 数据集的模型训练？

马赛克在训练期间将多个图像组合成一个，增加每个批次中目标和上下文的多样性。这提高了模型对不同目标尺寸、宽高比和场景的泛化能力。此技术可以通过由马赛克 DOTA8 数据集图像组成的训练批次进行可视化演示，有助于稳健的模型开发。在我们的[训练](../../modes/train.md)页面上探索更多关于马赛克和训练技术的信息。

### 为什么应该使用 Ultralytics YOLO11 进行目标检测任务？

Ultralytics YOLO11 提供最先进的实时目标检测功能，包括旋转边界框（OBB）、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)和高度通用的训练流程等功能。它适用于各种应用，并提供预训练模型以进行高效的微调。在 [Ultralytics YOLO11 文档](https://github.com/ultralytics/ultralytics)中进一步探索其优势和用法。
