---
comments: true
description: 探索包含 MRI/CT 图像的脑肿瘤检测数据集。对于训练用于早期诊断和治疗规划的 AI 模型至关重要。
keywords: 脑肿瘤数据集, MRI 扫描, CT 扫描, 脑肿瘤检测, 医学影像, 医疗保健中的 AI, 计算机视觉, 早期诊断, 治疗规划
---

# 脑肿瘤数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-brain-tumor-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开脑肿瘤数据集"></a>

脑肿瘤检测数据集由 MRI 或 CT 扫描的医学图像组成，包含有关脑肿瘤存在、位置和特征的信息。该数据集对于训练[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)算法以自动化脑肿瘤识别至关重要，有助于[医疗保健应用](https://www.ultralytics.com/solutions/ai-in-healthcare)中的早期诊断和治疗规划。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ogTBBD8McRk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics HUB 进行脑肿瘤检测
</p>

## 数据集结构

脑肿瘤数据集分为两个子集：

- **训练集**：包含 893 张图像，每张图像都有相应的标注。
- **测试集**：包含 223 张图像，每张图像都有配对的标注。

该数据集包含两个类别：

- **阴性**：没有脑肿瘤的图像
- **阳性**：有脑肿瘤的图像


## 应用

使用计算机视觉进行脑肿瘤检测的应用可实现[早期诊断](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency)、治疗规划和肿瘤进展监测。通过分析 MRI 或 CT 扫描等医学影像数据，[计算机视觉系统](https://docs.ultralytics.com/tasks/detect/)有助于准确识别脑肿瘤，协助及时的医疗干预和个性化治疗策略。

医疗专业人员可以利用这项技术：

- 减少诊断时间并提高准确性
- 通过精确定位肿瘤来协助手术规划
- 随时间监测治疗效果
- 支持肿瘤学和神经学研究

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含有关数据集路径、类别和其他相关信息。对于脑肿瘤数据集，`brain-tumor.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml)。

!!! example "ultralytics/cfg/datasets/brain-tumor.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/brain-tumor.yaml"
    ```

## 使用方法

要在脑肿瘤数据集上训练 [YOLO11](https://docs.ultralytics.com/models/yolo11/) 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，请使用提供的代码片段。有关可用参数的详细列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=brain-tumor.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载脑肿瘤微调模型

        # 使用模型进行推理
        results = model.predict("https://ultralytics.com/assets/brain-tumor-sample.jpg")
        ```

    === "CLI"

        ```bash
        # 使用微调的 *.pt 模型开始预测
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/brain-tumor-sample.jpg"
        ```

## 示例图像和标注

脑肿瘤数据集包含各种医学图像，展示有和没有肿瘤的脑部扫描。以下是数据集中的图像示例及其相应的标注。

![脑肿瘤数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/brain-tumor-dataset-sample-image.avif)

- **马赛克图像**：这里展示的是由马赛克数据集图像组成的训练批次。马赛克是一种训练技术，将多张图像合并为一张，增强批次多样性。这种方法有助于提高模型在脑部扫描中不同肿瘤尺寸、形状和位置上的泛化能力。

此示例突出了脑肿瘤数据集中图像的多样性和复杂性，强调了在[医学图像分析](https://www.ultralytics.com/blog/using-yolo11-for-tumor-detection-in-medical-imaging)训练阶段纳入马赛克技术的优势。

## 引用和致谢

该数据集在 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)下提供。

如果您在研究或开发工作中使用此数据集，请适当引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Ultralytics_Brain_Tumor_Dataset_2023,
            author = {Ultralytics},
            title = {Brain Tumor Detection Dataset},
            year = {2023},
            publisher = {Ultralytics},
            url = {https://docs.ultralytics.com/datasets/detect/brain-tumor/}
        }
        ```


## 常见问题

### Ultralytics 文档中提供的脑肿瘤数据集结构是什么？

脑肿瘤数据集分为两个子集：**训练集**包含 893 张图像及相应的标注，**测试集**包含 223 张图像及配对的标注。这种结构化划分有助于开发用于检测脑肿瘤的鲁棒且准确的计算机视觉模型。有关数据集结构的更多信息，请访问[数据集结构](#数据集结构)部分。

### 如何使用 Ultralytics 在脑肿瘤数据集上训练 YOLO11 模型？

您可以使用 Python 和 CLI 方法在脑肿瘤数据集上训练 YOLO11 模型 100 个训练周期，图像尺寸为 640px。以下是两种方法的示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=brain-tumor.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关可用参数的详细列表，请参阅[训练](../../modes/train.md)页面。

### 在医疗保健 AI 中使用脑肿瘤数据集有什么好处？

在 AI 项目中使用脑肿瘤数据集可实现脑肿瘤的早期诊断和治疗规划。它有助于通过计算机视觉自动化脑肿瘤识别，促进准确及时的医疗干预，并支持个性化治疗策略。这一应用在改善患者预后和医疗效率方面具有重大潜力。有关医疗保健 AI 应用的更多见解，请参阅 [Ultralytics 医疗保健解决方案](https://www.ultralytics.com/solutions/ai-in-healthcare)。

### 如何使用微调的 YOLO11 模型在脑肿瘤数据集上进行推理？

可以使用 Python 或 CLI 方法使用微调的 YOLO11 模型进行推理。以下是示例：

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载脑肿瘤微调模型

        # 使用模型进行推理
        results = model.predict("https://ultralytics.com/assets/brain-tumor-sample.jpg")
        ```

    === "CLI"

        ```bash
        # 使用微调的 *.pt 模型开始预测
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/brain-tumor-sample.jpg"
        ```

### 在哪里可以找到脑肿瘤数据集的 YAML 配置？

脑肿瘤数据集的 YAML 配置文件可在 [brain-tumor.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml) 找到。此文件包含在此数据集上训练和评估模型所需的路径、类别和其他相关信息。
