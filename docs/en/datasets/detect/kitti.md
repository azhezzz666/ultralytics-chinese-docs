---
comments: true
description: 探索 Ultralytics KITTI 数据集，这是一个用于 3D 目标检测、深度估计和自动驾驶感知等计算机视觉任务的基准数据集。
keywords: KITTI, Ultralytics, 数据集, 目标检测, 3D 视觉, YOLO11, 训练, 验证, 自动驾驶汽车, 计算机视觉
---

# KITTI 数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-kitti-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开 KITTI 数据集"></a>

KITTI 数据集是自动驾驶和计算机视觉领域最具影响力的基准数据集之一。该数据集由卡尔斯鲁厄理工学院和芝加哥丰田技术研究所发布，包含从真实世界驾驶场景中收集的立体相机、LiDAR 和 GPS/IMU 数据。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NNeDlTbq9pA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 KITTI 数据集上训练 Ultralytics YOLO11 🚀
</p>

它广泛用于评估目标检测、深度估计、光流和视觉里程计算法。该数据集与 Ultralytics YOLO11 完全兼容，可用于 2D 目标检测任务，并可轻松集成到 Ultralytics 平台进行训练和评估。

## 数据集结构

!!! warning

    KITTI 原始测试集在此处被排除，因为它不包含真实标注。

该数据集总共包含 7,481 张图像，每张图像都配有汽车、行人、骑自行车者和其他道路元素等目标的详细标注。数据集分为两个主要子集：

- **训练集**：包含 5,985 张带有标注标签的图像，用于模型训练。
- **验证集**：包含 1,496 张带有相应标注的图像，用于性能评估和基准测试。

## 应用

KITTI 数据集推动了自动驾驶和机器人领域的进步，支持以下任务：

- **自动驾驶车辆感知**：训练模型检测和跟踪车辆、行人和障碍物，以实现自动驾驶系统中的安全导航。
- **3D 场景理解**：支持深度估计、立体视觉和 3D 目标定位，帮助机器理解空间环境。
- **光流和运动预测**：实现运动分析以预测目标的移动，并改进动态环境中的轨迹规划。
- **计算机视觉基准测试**：作为评估多种视觉任务（包括目标检测和跟踪）性能的标准基准。


## 数据集 YAML

Ultralytics 使用 YAML 文件定义 KITTI 数据集配置。此文件指定数据集路径、类别标签和训练所需的元数据。配置文件可在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml) 获取。

!!! example "ultralytics/cfg/datasets/kitti.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/kitti.yaml"
    ```

## 使用方法

要在 KITTI 数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，请使用以下命令。有关更多详情，请参阅[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 在 KITTI 数据集上训练
        results = model.train(data="kitti.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=kitti.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

您还可以使用相同的配置文件直接从命令行或 Python API 执行评估、[推理](../../modes/predict.md)和[导出](../../modes/export.md)任务。

## 示例图像和标注

KITTI 数据集提供多样化的驾驶场景。每张图像都包含用于 2D 目标检测任务的边界框标注。该示例展示了数据集的丰富多样性，使模型能够在各种真实世界条件下实现鲁棒的泛化。

<img src="https://github.com/ultralytics/docs/releases/download/0/kitti-dataset-sample.avif" alt="KITTI 示例图像" width="800">

## 引用和致谢

如果您在研究中使用 KITTI 数据集，请引用以下论文：

!!! quote

    === "BibTeX"

        ```bibtex
        @article{Geiger2013IJRR,
          author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
          title = {Vision meets Robotics: The KITTI Dataset},
          journal = {International Journal of Robotics Research (IJRR)},
          year = {2013}
        }
        ```

我们感谢 KITTI 视觉基准套件提供这个全面的数据集，它继续推动计算机视觉、机器人和自动驾驶系统的进步。访问 [KITTI 网站](https://www.cvlibs.net/datasets/kitti/)了解更多信息。

## 常见问题

### KITTI 数据集用于什么？

KITTI 数据集主要用于自动驾驶领域的计算机视觉研究，支持目标检测、深度估计、光流和 3D 定位等任务。

### KITTI 数据集包含多少张图像？

该数据集包含 5,985 张标注的训练图像和 1,496 张验证图像，捕捉了城市、农村和高速公路场景。原始测试集在此处被排除，因为它不包含真实标注。

### 数据集中标注了哪些目标类别？

KITTI 包含汽车、行人、骑自行车者、卡车、有轨电车和其他道路使用者等目标的标注。

### 我可以使用 KITTI 数据集训练 Ultralytics YOLO11 模型吗？

是的，KITTI 与 Ultralytics YOLO11 完全兼容。您可以使用提供的 YAML 配置文件直接[训练](../../modes/train.md)和[验证](../../modes/val.md)模型。

### 在哪里可以找到 KITTI 数据集配置文件？

您可以在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml) 访问 YAML 文件。
