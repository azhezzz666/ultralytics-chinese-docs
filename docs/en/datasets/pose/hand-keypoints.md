---
comments: true
description: 探索用于高级姿态估计的手部关键点估计数据集。了解数据集、预训练模型、指标和使用 YOLO 训练的应用。
keywords: Hand KeyPoints, 姿态估计, 数据集, 关键点, MediaPipe, YOLO, 深度学习, 计算机视觉
---

# Hand Keypoints 数据集

## 简介

手部关键点数据集包含 26,768 张手部标注图像，使其适合训练像 Ultralytics YOLO 这样的姿态估计模型。标注使用 Google MediaPipe 库生成，确保高[准确率](https://www.ultralytics.com/glossary/accuracy)和一致性，数据集与 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 格式兼容。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/fd6u1TW_AGY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO11 进行手部关键点估计 | 人手姿态估计教程
</p>

## 手部标志点

![手部标志点](https://github.com/ultralytics/docs/releases/download/0/hand_landmarks.jpg)

## 关键点

数据集包含用于手部检测的关键点。关键点标注如下：

1. 手腕
2. 拇指（4 个点）
3. 食指（4 个点）
4. 中指（4 个点）
5. 无名指（4 个点）
6. 小指（4 个点）

每只手共有 21 个关键点。

## 主要特点

- **大型数据集**：26,768 张带有手部关键点标注的图像。
- **YOLO11 兼容性**：标签以 YOLO 关键点格式提供，可直接用于 YOLO11 模型。
- **21 个关键点**：详细的手部姿态表示，涵盖手腕和每个手指的四个点。

## 数据集结构

手部关键点数据集分为两个子集：

1. **Train**：该子集包含手部关键点数据集中的 18,776 张图像，标注用于训练姿态估计模型。
2. **Val**：该子集包含 7,992 张图像，可用于模型训练期间的验证。

## 应用场景

手部关键点可用于[手势识别](https://www.ultralytics.com/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11)、[AR/VR 控制](https://docs.ultralytics.com/tasks/pose/)、机器人操作和医疗保健中的手部运动分析。它们还可以应用于动画的动作捕捉和安全的生物识别认证系统。手指位置的详细跟踪实现了与虚拟物体的精确交互和无触摸控制界面。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 Hand Keypoints 数据集，`hand-keypoints.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml)。

!!! example "ultralytics/cfg/datasets/hand-keypoints.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/hand-keypoints.yaml"
    ```

## 使用方法

要在 Hand Keypoints 数据集上训练 YOLO11n-pose 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

Hand keypoints 数据集包含多样化的图像集，其中人手标注了关键点。以下是数据集中的一些图像示例及其对应的标注：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/human-hand-pose.avif)

- **马赛克图像**：此图像展示了由马赛克数据集图像组成的训练批次。马赛克是训练期间使用的一种技术，将多个图像组合成单个图像，以增加每个训练批次中目标和场景的多样性。这有助于提高模型对不同目标尺寸、宽高比和上下文的泛化能力。

该示例展示了 Hand Keypoints 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

## 引用和致谢

如果您在研究或开发工作中使用手部关键点数据集，请致谢以下来源：

!!! quote ""

    === "致谢"

    我们感谢以下来源提供此数据集中使用的图像：

    - [11k Hands](https://sites.google.com/view/11khands)
    - [2000 Hand Gestures](https://www.kaggle.com/datasets/ritikagiridhar/2000-hand-gestures)
    - [Gesture Recognition](https://www.kaggle.com/datasets/imsparsh/gesture-recognition)

    图像根据各平台提供的相应许可证收集和使用，并在[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可证](https://creativecommons.org/licenses/by-nc-sa/4.0/)下分发。

我们还要感谢此数据集的创建者 [Rion Dsilva](https://www.linkedin.com/in/rion-dsilva-043464229/)，感谢他对视觉 AI 研究的巨大贡献。

## 常见问题

### 如何在 Hand Keypoints 数据集上训练 YOLO11 模型？

要在 Hand Keypoints 数据集上训练 YOLO11 模型，可以使用 Python 或命令行界面（CLI）。以下是训练 YOLO11n-pose 模型 100 个轮次、图像尺寸为 640 的示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### Hand Keypoints 数据集的主要特点是什么？

Hand Keypoints 数据集专为高级[姿态估计](https://docs.ultralytics.com/datasets/pose/)任务设计，包含几个主要特点：

- **大型数据集**：包含 26,768 张带有手部关键点标注的图像。
- **YOLO11 兼容性**：可直接用于 YOLO11 模型。
- **21 个关键点**：详细的手部姿态表示，包括手腕和手指关节。

有关更多详细信息，您可以探索 [Hand Keypoints 数据集](#简介)部分。

### 哪些应用可以从使用 Hand Keypoints 数据集中受益？

Hand Keypoints 数据集可应用于各个领域，包括：

- **手势识别**：增强人机交互。
- **AR/VR 控制**：改善增强和虚拟现实中的用户体验。
- **机器人操作**：实现机器人手的精确控制。
- **医疗保健**：分析手部运动用于医学诊断。
- **动画**：捕捉动作用于逼真的动画。
- **生物识别认证**：增强安全系统。

有关更多信息，请参阅[应用场景](#应用场景)部分。

### Hand Keypoints 数据集的结构是怎样的？

Hand Keypoints 数据集分为两个子集：

1. **Train**：包含 18,776 张图像用于训练姿态估计模型。
2. **Val**：包含 7,992 张图像用于模型训练期间的验证。

此结构确保了全面的训练和验证过程。有关更多详细信息，请参阅[数据集结构](#数据集结构)部分。

### 如何使用数据集 YAML 文件进行训练？

数据集配置在 YAML 文件中定义，其中包含路径、类别和其他相关信息。`hand-keypoints.yaml` 文件可以在 [hand-keypoints.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml) 找到。

要使用此 YAML 文件进行训练，请在训练脚本或 CLI 命令中指定它，如上面的训练示例所示。有关更多详细信息，请参阅[数据集 YAML](#数据集-yaml) 部分。
