---
comments: true
description: 探索汽车零部件分割数据集，用于汽车 AI 应用。使用 Ultralytics YOLO 通过丰富的标注数据增强您的分割模型。
keywords: 汽车零部件分割数据集, 计算机视觉, 汽车 AI, 车辆维护, Ultralytics, YOLO, 分割模型, 深度学习, 目标分割
---

# 汽车零部件分割数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-carparts-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开汽车零部件分割数据集"></a>

汽车零部件分割数据集可在 Roboflow Universe 上获取，是一个精心策划的图像和视频集合，专为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用设计，特别关注[分割任务](https://docs.ultralytics.com/tasks/segment/)。该数据集托管在 Roboflow Universe 上，提供从多个角度拍摄的多样化视觉素材，为训练和测试分割模型提供有价值的[标注](https://www.ultralytics.com/glossary/data-labeling)示例。

无论您是从事[汽车研究](https://www.ultralytics.com/solutions/ai-in-automotive)、开发车辆维护 AI 解决方案，还是探索计算机视觉应用，汽车零部件分割数据集都是使用 [Ultralytics YOLO](../../models/yolo11.md) 等模型提高项目[准确性](https://www.ultralytics.com/glossary/accuracy)和效率的宝贵资源。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/HATMPgLYAPU"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO11 进行汽车零部件<a href="https://www.ultralytics.com/glossary/instance-segmentation">实例分割</a>。
</p>

## 数据集结构

汽车零部件分割数据集的数据分布组织如下：

- **训练集**：包含 3156 张图像及其对应的标注。该集合用于[训练](https://www.ultralytics.com/glossary/training-data)[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)[模型](https://www.ultralytics.com/glossary/foundation-model)。
- **测试集**：包含 276 张图像，每张都配有相应的标注。该集合用于训练后使用[测试数据](https://www.ultralytics.com/glossary/test-data)评估模型性能。
- **验证集**：包含 401 张图像，每张都有对应的标注。该集合用于训练期间调整[超参数](https://docs.ultralytics.com/guides/hyperparameter-tuning/)并使用[验证数据](https://www.ultralytics.com/glossary/validation-data)防止[过拟合](https://www.ultralytics.com/glossary/overfitting)。

## 应用场景

汽车零部件分割在多个领域都有应用，包括：

- **汽车质量控制**：在制造过程中识别汽车零部件的缺陷或不一致（[制造业 AI](https://www.ultralytics.com/solutions/ai-in-manufacturing)）。
- **汽车维修**：协助技师识别需要维修或更换的零部件。
- **电子商务编目**：在在线商店中自动标记和分类汽车零部件，用于[电子商务](https://en.wikipedia.org/wiki/E-commerce)平台。
- **交通监控**：分析交通监控录像中的车辆组件。
- **自动驾驶汽车**：增强[自动驾驶汽车](https://www.ultralytics.com/blog/ai-in-self-driving-cars)的感知系统，以更好地理解周围车辆。
- **保险处理**：在保险理赔过程中通过识别受影响的汽车零部件自动进行损坏评估。
- **回收利用**：对车辆组件进行分类以实现高效回收流程。
- **智慧城市计划**：为[智慧城市](https://en.wikipedia.org/wiki/Smart_city)内的城市规划和交通管理系统提供数据支持。

通过准确识别和分类不同的车辆组件，汽车零部件分割简化了流程，并有助于提高这些行业的效率和自动化水平。


## 数据集 YAML

[YAML](https://www.ultralytics.com/glossary/yaml)（Yet Another Markup Language）文件定义数据集配置，包括路径、类别名称和其他基本详细信息。对于汽车零部件分割数据集，`carparts-seg.yaml` 文件可在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml) 获取。您可以在 [yaml.org](https://yaml.org/) 了解更多关于 YAML 格式的信息。

!!! example "ultralytics/cfg/datasets/carparts-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/carparts-seg.yaml"
    ```

## 使用方法

要在汽车零部件分割数据集上训练 [Ultralytics YOLO11](../../models/yolo11.md) 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，请使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练指南](../../modes/train.md)，并探索[模型训练技巧](https://docs.ultralytics.com/guides/model-training-tips/)以获取最佳实践。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的分割模型，如 YOLO11n-seg
        model = YOLO("yolo11n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 在汽车零部件分割数据集上训练模型
        results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)

        # 训练后，您可以在验证集上验证模型性能
        results = model.val()

        # 或对新图像或视频进行预测
        results = model.predict("path/to/your/image.jpg")
        ```

    === "CLI"

        ```bash
        # 使用命令行界面从预训练的 *.pt 模型开始训练
        # 指定数据集配置文件、模型、轮次数和图像尺寸
        yolo segment train data=carparts-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640

        # 使用验证集验证训练好的模型
        yolo segment val data=carparts-seg.yaml model=path/to/best.pt

        # 使用训练好的模型对特定图像源进行预测
        yolo segment predict model=path/to/best.pt source=path/to/your/image.jpg
        ```

## 示例数据和标注

汽车零部件分割数据集包含从各种角度拍摄的多样化图像和视频。以下是展示数据及其对应标注的示例：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/dataset-sample-image.avif)

- 该图像展示了汽车图像样本中的[目标分割](https://docs.ultralytics.com/tasks/segment/)。带有掩码的标注[边界框](https://www.ultralytics.com/glossary/bounding-box)突出显示了识别出的汽车零部件（如前灯、格栅）。
- 数据集包含在不同条件下（位置、光照、目标密度）拍摄的各种图像，为训练鲁棒的汽车零部件分割模型提供了全面的资源。
- 此示例强调了数据集的复杂性以及[高质量数据](https://www.ultralytics.com/blog/the-importance-of-high-quality-computer-vision-datasets)对计算机视觉任务的重要性，特别是在汽车组件分析等专业领域。[数据增强](https://www.ultralytics.com/glossary/data-augmentation)等技术可以进一步增强模型的泛化能力。

## 引用和致谢

如果您在研究或开发工作中使用汽车零部件分割数据集，请适当引用原始来源：

!!! quote ""

    === "BibTeX"

        ```bibtex
           @misc{ car-seg-un1pm_dataset,
                title = { car-seg Dataset },
                type = { Open Source Dataset },
                author = { Gianmarco Russo },
                url = { https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm },
                journal = { Roboflow Universe },
                publisher = { Roboflow },
                year = { 2023 },
                month = { nov },
                note = { visited on 2024-01-24 },
            }
        ```

我们感谢 Gianmarco Russo 和 Roboflow 团队为计算机视觉社区创建和维护这一宝贵数据集的贡献。有关更多数据集，请访问 [Ultralytics 数据集集合](https://docs.ultralytics.com/datasets/)。

## 常见问题

### 什么是汽车零部件分割数据集？

汽车零部件分割数据集是一个专门的图像和视频集合，用于训练计算机视觉模型对汽车零部件进行[分割](https://docs.ultralytics.com/tasks/segment/)。它包含各种场景下的多样化汽车图像及详细标注，适用于汽车 AI 应用。

### 如何使用 Ultralytics YOLO11 和汽车零部件分割数据集？

您可以使用此数据集训练 [Ultralytics YOLO11](../../models/yolo11.md) 分割模型。加载预训练模型（如 `yolo11n-seg.pt`）并使用提供的 Python 或 CLI 示例开始训练，引用 `carparts-seg.yaml` 配置文件。查看[训练指南](../../modes/train.md)获取详细说明。

!!! example "训练示例片段"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo segment train data=carparts-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

### 汽车零部件分割有哪些应用？

汽车零部件分割在以下领域很有用：

- **汽车质量控制**：确保零部件符合标准（[制造业 AI](https://www.ultralytics.com/solutions/ai-in-manufacturing)）。
- **汽车维修**：识别需要维修的零部件。
- **电子商务**：在线编目零部件。
- **自动驾驶汽车**：改善车辆感知（[汽车 AI](https://www.ultralytics.com/solutions/ai-in-automotive)）。
- **保险**：自动评估车辆损坏。
- **回收利用**：高效分类零部件。

### 在哪里可以找到汽车零部件分割的数据集配置文件？

数据集配置文件 `carparts-seg.yaml` 包含数据集路径和类别的详细信息，位于 Ultralytics GitHub 仓库：[carparts-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml)。

### 为什么应该使用汽车零部件分割数据集？

该数据集提供丰富的标注数据，对于开发用于汽车应用的准确[分割模型](https://docs.ultralytics.com/tasks/segment/)至关重要。其多样性有助于提高模型在自动化车辆检测等实际场景中的鲁棒性和性能，增强安全系统并支持自动驾驶技术。使用这样的高质量、特定领域数据集可以加速 AI 开发。
