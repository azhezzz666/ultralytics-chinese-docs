---
comments: true
description: 探索 Ultralytics Tiger-Pose 数据集，包含 263 张多样化图像。非常适合测试、训练和优化姿态估计算法。
keywords: Ultralytics, Tiger-Pose, 数据集, 姿态估计, YOLO11, 训练数据, 机器学习, 神经网络
---

# Tiger-Pose 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) 推出 Tiger-Pose 数据集，这是一个专为姿态估计任务设计的多功能集合。此数据集包含 263 张图像，来源于一个 [YouTube 视频](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0)，其中 210 张用于训练，53 张用于验证。它是测试和排查姿态估计算法的优秀资源。

尽管训练集只有 210 张图像，Tiger-Pose 数据集提供了多样性，使其适合评估训练流程、识别潜在错误，并作为处理更大[姿态估计](https://docs.ultralytics.com/tasks/pose/)数据集之前的宝贵初步步骤。

此数据集旨在与 [Ultralytics HUB](https://hub.ultralytics.com/) 和 [YOLO11](https://github.com/ultralytics/ultralytics) 一起使用。

## 数据集结构

- **总图像数**：263（210 张训练 / 53 张验证）。
- **关键点**：每只老虎 12 个（无可见性标志）。
- **目录布局**：YOLO 格式的关键点存储在 `labels/{train,val}` 下，与 `images/{train,val}` 目录并列。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gc6K5eKrTNQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics HUB 在 Tiger-Pose 数据集上训练 YOLO11 姿态模型
</p>

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于指定数据集的配置详细信息。它包含文件路径、类别定义和其他相关信息等关键数据。具体来说，对于 `tiger-pose.yaml` 文件，您可以查看 [Ultralytics Tiger-Pose 数据集配置文件](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml)。

!!! example "ultralytics/cfg/datasets/tiger-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/tiger-pose.yaml"
    ```

## 使用方法

要在 Tiger-Pose 数据集上训练 YOLO11n-pose 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo pose train data=tiger-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

以下是 Tiger-Pose 数据集中的一些图像示例及其对应的标注：

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-4.avif" alt="数据集示例图像" width="100%">

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是训练期间使用的一种技术，将多个图像组合成单个图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 Tiger-Pose 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

## 推理示例

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载 tiger-pose 训练的模型

        # 运行推理
        results = model.predict(source="https://youtu.be/MIBAT6BGE6U", show=True)
        ```

    === "CLI"

        ```bash
        # 使用 tiger-pose 训练的模型运行推理
        yolo pose predict source="https://youtu.be/MIBAT6BGE6U" show=True model="path/to/best.pt"
        ```

## 引用和致谢

该数据集已在 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)下发布。

## 常见问题

### Ultralytics Tiger-Pose 数据集用于什么？

Ultralytics Tiger-Pose 数据集专为姿态估计任务设计，包含 263 张图像，来源于一个 [YouTube 视频](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0)。数据集分为 210 张训练图像和 53 张验证图像。它特别适用于使用 [Ultralytics HUB](https://hub.ultralytics.com/) 和 [YOLO11](https://github.com/ultralytics/ultralytics) 测试、训练和优化姿态估计算法。

### 如何在 Tiger-Pose 数据集上训练 YOLO11 模型？

要在 Tiger-Pose 数据集上训练 YOLO11n-pose 模型 100 个轮次、图像尺寸为 640，请使用以下代码片段。有关更多详细信息，请访问[训练](../../modes/train.md)页面：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo pose train data=tiger-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

### `tiger-pose.yaml` 文件包含哪些配置？

`tiger-pose.yaml` 文件用于指定 Tiger-Pose 数据集的配置详细信息。它包含文件路径和类别定义等关键数据。要查看确切配置，您可以查看 [Ultralytics Tiger-Pose 数据集配置文件](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml)。

### 如何使用在 Tiger-Pose 数据集上训练的 YOLO11 模型运行推理？

要使用在 Tiger-Pose 数据集上训练的 YOLO11 模型执行推理，可以使用以下代码片段。有关详细指南，请访问[预测](../../modes/predict.md)页面：

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载 tiger-pose 训练的模型

        # 运行推理
        results = model.predict(source="https://youtu.be/MIBAT6BGE6U", show=True)
        ```


    === "CLI"

        ```bash
        # 使用 tiger-pose 训练的模型运行推理
        yolo pose predict source="https://youtu.be/MIBAT6BGE6U" show=True model="path/to/best.pt"
        ```

### 使用 Tiger-Pose 数据集进行姿态估计有什么好处？

Tiger-Pose 数据集尽管训练集只有 210 张图像，但提供了多样化的图像集合，非常适合测试姿态估计流程。该数据集有助于识别潜在错误，并在处理更大数据集之前作为初步步骤。此外，该数据集支持使用 [Ultralytics HUB](https://hub.ultralytics.com/) 和 [YOLO11](https://github.com/ultralytics/ultralytics) 等先进工具训练和优化姿态估计算法，增强模型性能和[准确率](https://www.ultralytics.com/glossary/accuracy)。
