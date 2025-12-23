---
comments: true
description: 探索 xView 数据集，这是一个包含超过 100 万个目标实例的高分辨率卫星图像丰富资源。增强检测能力、学习效率等。
keywords: xView 数据集, 航拍图像, 卫星图像, 目标检测, 高分辨率, 边界框, 计算机视觉, TensorFlow, PyTorch, 数据集结构
---

# xView 数据集

[xView](http://xviewdataset.org/) 数据集是最大的公开可用航拍图像数据集之一，包含来自世界各地复杂场景的图像，使用边界框进行标注。xView 数据集的目标是加速四个[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)前沿领域的进展：

1. 降低检测的最小分辨率要求。
2. 提高学习效率。
3. 发现更多目标类别。
4. 改进细粒度类别的检测。

xView 建立在 [Common Objects in Context (COCO)](../detect/coco.md) 等挑战赛的成功基础上，旨在利用计算机视觉分析日益增长的太空可用图像，以新的方式理解视觉世界并解决一系列重要应用。

!!! warning "需要手动下载"

    xView 数据集**不会**被 Ultralytics 脚本自动下载。您**必须**首先从官方来源手动下载数据集：

    - **来源：** 美国国家地理空间情报局（NGA）DIUx xView 2018 挑战赛
    - **网址：** [https://challenge.xviewdataset.org](https://challenge.xviewdataset.org)

    **重要提示：** 下载必要文件（如 `train_images.tif`、`val_images.tif`、`xView_train.geojson`）后，您需要解压它们并将其放置到正确的目录结构中，通常位于 `datasets/xView/` 文件夹下，**然后**再运行下面提供的训练命令。请确保按照挑战赛说明正确设置数据集。

## 主要特点

- xView 包含超过 100 万个目标实例，涵盖 60 个类别。
- 数据集分辨率为 0.3 米，提供比大多数公开卫星图像数据集更高分辨率的图像。
- xView 具有多样化的小型、稀有、细粒度和多类型目标集合，带有[边界框](https://www.ultralytics.com/glossary/bounding-box)标注。
- 附带使用 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 目标检测 API 的预训练基线模型和 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 示例。

## 数据集结构

xView 数据集由 WorldView-3 卫星以 0.3 米地面采样距离收集的卫星图像组成。它在超过 1,400 平方公里的图像中包含超过 100 万个目标，涵盖 60 个类别。该数据集对于[遥感](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery)应用和环境监测特别有价值。


## 应用场景

xView 数据集广泛用于训练和评估航拍图像中目标检测的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。数据集多样化的目标类别和高分辨率图像使其成为计算机视觉领域研究人员和从业者的宝贵资源，特别是用于卫星图像分析。应用包括：

- 军事和国防侦察
- 城市规划和发展
- 环境监测
- 灾害响应和评估
- 基础设施测绘和管理

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集路径、类别和其他相关信息。对于 xView 数据集，`xView.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml)。

!!! example "ultralytics/cfg/datasets/xView.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/xView.yaml"
    ```

## 使用方法

要在 xView 数据集上训练模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=xView.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例数据和标注

xView 数据集包含高分辨率卫星图像，其中多样化的目标使用边界框进行标注。以下是数据集中的一些数据示例及其对应的标注：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/overhead-imagery-object-detection.avif)

- **航拍图像**：此图像展示了航拍图像中[目标检测](https://www.ultralytics.com/glossary/object-detection)的示例，其中目标使用边界框进行标注。数据集提供高分辨率卫星图像，以促进该任务模型的开发。

该示例展示了 xView 数据集中数据的多样性和复杂性，并强调了高质量卫星图像对目标检测任务的重要性。

## 相关数据集

如果您正在处理卫星图像，您可能还对探索这些相关数据集感兴趣：

- [DOTA-v2](../obb/dota-v2.md)：用于航拍图像中旋转目标检测的数据集
- [VisDrone](../detect/visdrone.md)：用于无人机拍摄图像中目标检测和跟踪的数据集
- [Argoverse](../detect/argoverse.md)：用于自动驾驶的 3D 跟踪标注数据集

## 引用和致谢

如果您在研究或开发工作中使用 xView 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lam2018xview,
              title={xView: Objects in Context in Overhead Imagery},
              author={Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord},
              year={2018},
              eprint={1802.07856},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们感谢[国防创新单元](https://www.diu.mil/)（DIU）和 xView 数据集的创建者为计算机视觉研究社区做出的宝贵贡献。有关 xView 数据集及其创建者的更多信息，请访问 [xView 数据集网站](http://xviewdataset.org/)。

## 常见问题

### 什么是 xView 数据集，它如何有益于计算机视觉研究？

[xView](http://xviewdataset.org/) 数据集是最大的公开可用高分辨率航拍图像集合之一，包含超过 100 万个目标实例，涵盖 60 个类别。它旨在增强计算机视觉研究的各个方面，如降低检测的最小分辨率要求、提高学习效率、发现更多目标类别以及推进细粒度目标检测。

### 如何使用 Ultralytics YOLO 在 xView 数据集上训练模型？

要使用 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/) 在 xView 数据集上训练模型，请按照以下步骤操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=xView.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关详细参数和设置，请参阅模型[训练](../../modes/train.md)页面。

### xView 数据集的主要特点是什么？

xView 数据集因其全面的特点而脱颖而出：

- 超过 100 万个目标实例，涵盖 60 个不同类别。
- 0.3 米的高分辨率图像。
- 多样化的目标类型，包括小型、稀有和细粒度目标，均带有边界框标注。
- 提供 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 和 PyTorch 的预训练基线模型和示例。

### xView 的数据集结构是什么，它是如何标注的？

xView 数据集包含由 WorldView-3 卫星以 0.3 米地面采样距离拍摄的高分辨率卫星图像，在约 1,400 平方公里的标注图像中涵盖超过 100 万个目标，分布在 60 个不同类别中。每个目标都使用边界框进行标注，使该数据集非常适合训练和评估航拍视图中目标检测的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。有关详细分解，请参阅[数据集结构部分](#数据集结构)。

### 如何在研究中引用 xView 数据集？

如果您在研究中使用 xView 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lam2018xview,
            title={xView: Objects in Context in Overhead Imagery},
            author={Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord},
            year={2018},
            eprint={1802.07856},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }
        ```

有关 xView 数据集的更多信息，请访问官方 [xView 数据集网站](http://xviewdataset.org/)。
