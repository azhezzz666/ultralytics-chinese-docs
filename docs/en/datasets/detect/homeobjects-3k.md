---
comments: true
description: 探索 HomeObjects-3K，一个丰富的室内目标检测数据集，包含床、沙发、电视和笔记本电脑等 12 个类别。非常适合智能家居、机器人和 AR 中的计算机视觉应用。
keywords: HomeObjects-3K, 室内数据集, 家居物品, 目标检测, 计算机视觉, YOLO11, 智能家居 AI, 机器人数据集
---

# HomeObjects-3K 数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-homeobjects-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开 HomeObjects-3K 数据集"></a>

HomeObjects-3K 数据集是一个精心策划的常见家居物品图像集合，专为训练、测试和[基准测试](../../modes/benchmark.md)[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型而设计。该数据集包含约 3,000 张图像和 12 个不同的目标类别，非常适合室内场景理解、智能家居设备、[机器人](https://www.ultralytics.com/glossary/robotics)和增强现实的研究和应用。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/v3iqOYoRBFQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 HomeObjects-3K 数据集上训练 Ultralytics YOLO11 | 检测、验证和 ONNX 导出 🚀
</p>

## 数据集结构

HomeObjects-3K 数据集分为以下子集：

- **训练集**：包含 2,285 张标注图像，展示沙发、椅子、桌子、灯具等物品。
- **验证集**：包含 404 张标注图像，用于评估模型性能。

每张图像都使用与 [Ultralytics YOLO](../detect/index.md/#what-is-the-ultralytics-yolo-dataset-format-and-how-to-structure-it) 格式对齐的边界框进行标注。室内光照、目标尺度和方向的多样性使其对于实际部署场景具有鲁棒性。

## 目标类别

该数据集支持 12 个日常物品类别，涵盖家具、电子产品和装饰物品。这些类别的选择反映了室内家居环境中常见的物品，并支持[目标检测](../../tasks/detect.md)和[目标跟踪](../../modes/track.md)等视觉任务。

!!! Tip "HomeObjects-3K 类别"

    0. 床
    1. 沙发
    2. 椅子
    3. 桌子
    4. 灯
    5. 电视
    6. 笔记本电脑
    7. 衣柜
    8. 窗户
    9. 门
    10. 盆栽植物
    11. 相框


## 应用

HomeObjects-3K 支持室内计算机视觉的广泛应用，涵盖研究和实际产品开发：

- **室内目标检测**：使用 [Ultralytics YOLO11](../../models/yolo11.md) 等模型在图像中查找和定位床、椅子、灯和笔记本电脑等常见家居物品。这有助于实时理解室内场景。

- **场景布局解析**：在机器人和智能家居系统中，这有助于设备了解房间的布局，门、窗和家具的位置，以便它们能够安全导航并与环境正确交互。

- **AR 应用**：为使用增强现实的应用提供[目标识别](http://ultralytics.com/glossary/image-recognition)功能。例如，检测电视或衣柜并在其上显示额外信息或效果。

- **教育和研究**：通过为学生和研究人员提供现成的数据集，支持学习和学术项目，用于使用真实世界示例练习室内目标检测。

- **家庭库存和资产跟踪**：自动检测和列出照片或视频中的家居物品，用于管理物品、组织空间或在房地产中可视化家具。

## 数据集 YAML

HomeObjects-3K 数据集的配置通过 YAML 文件提供。此文件概述了基本信息，如训练和验证目录的图像路径以及目标类别列表。
您可以直接从 Ultralytics 仓库访问 `HomeObjects-3K.yaml` 文件：[https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml)

!!! example "ultralytics/cfg/datasets/HomeObjects-3K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/HomeObjects-3K.yaml"
    ```

## 使用方法

您可以在 HomeObjects-3K 数据集上训练 YOLO11n 模型 100 个训练周期，图像尺寸为 640。以下示例展示如何入门。有关更多训练选项和详细设置，请查看[训练](../../modes/train.md)指南。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 在 HomeObjects-3K 数据集上训练模型
        model.train(data="HomeObjects-3K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=HomeObjects-3K.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

该数据集包含丰富的室内场景图像集合，捕捉了自然家居环境中各种家居物品。以下是数据集中的示例视觉效果，每张图像都配有相应的标注，以说明目标位置、尺度和空间关系。

![HomeObjects-3K 数据集示例图像，突出显示不同物品，如床、椅子、门、沙发和植物](https://github.com/ultralytics/docs/releases/download/0/homeobjects-3k-dataset-sample.avif)

## 许可和归属

HomeObjects-3K 由 **[Ultralytics 团队](https://www.ultralytics.com/about)**开发并在 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)下发布，支持开源研究和具有适当归属的商业使用。

如果您在研究中使用此数据集，请使用以下详细信息进行引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Jocher_Ultralytics_Datasets_2025,
            author = {Jocher, Glenn and Rizwan, Muhammad},
            license = {AGPL-3.0},
            month = {May},
            title = {Ultralytics Datasets: HomeObjects-3K Detection Dataset},
            url = {https://docs.ultralytics.com/datasets/detect/homeobjects-3k/},
            version = {1.0.0},
            year = {2025}
        }
        ```

## 常见问题

### HomeObjects-3K 数据集是为什么设计的？

HomeObjects-3K 旨在推进 AI 对室内场景的理解。它专注于检测日常家居物品——如床、沙发、电视和灯——使其非常适合智能家居、机器人、增强现实和室内监控系统的应用。无论您是为实时边缘设备训练模型还是进行学术研究，该数据集都提供了平衡的基础。

### 包含哪些目标类别，为什么选择它们？

该数据集包含 12 种最常见的家居物品：床、沙发、椅子、桌子、灯、电视、笔记本电脑、衣柜、窗户、门、盆栽植物和相框。选择这些物品是为了反映真实的室内环境，并支持机器人导航或 AR/VR 应用中的场景生成等多用途任务。

### 如何使用 HomeObjects-3K 数据集训练 YOLO 模型？

要训练像 YOLO11n 这样的 YOLO 模型，您只需要 `HomeObjects-3K.yaml` 配置文件和[预训练模型](../../models/index.md)权重。无论您使用 Python 还是 CLI，都可以用一条命令启动训练。您可以根据目标性能和硬件设置自定义训练周期、图像尺寸和批次大小等参数。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 在 HomeObjects-3K 数据集上训练模型
        model.train(data="HomeObjects-3K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=HomeObjects-3K.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

### 此数据集适合初学者级别的项目吗？

当然。凭借干净的标注和标准化的 YOLO 兼容标注，HomeObjects-3K 是想要探索室内场景中真实世界目标检测的学生和爱好者的绝佳入门点。它也可以很好地扩展到商业环境中更复杂的应用。

### 在哪里可以找到标注格式和 YAML？

请参阅[数据集 YAML](#数据集-yaml) 部分。格式是标准 YOLO，使其与大多数目标检测流水线兼容。
