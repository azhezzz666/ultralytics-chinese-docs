---
comments: true
description: 探索医疗药丸检测数据集，包含标注图像。对于训练用于制药识别和自动化的 AI 模型至关重要。
keywords: 医疗药丸数据集, 药丸检测, 制药成像, 医疗保健中的 AI, 计算机视觉, 目标检测, 医疗自动化, 训练数据集
---

# 医疗药丸数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-medical-pills-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开医疗药丸数据集"></a>

医疗药丸检测数据集是一个概念验证（POC）数据集，经过精心策划以展示 AI 在制药应用中的潜力。它包含专门设计用于训练[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)[模型](https://docs.ultralytics.com/models/)识别医疗药丸的标注图像。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/8gePl_Zcs5c"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 <a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-medical-pills-dataset.ipynb">Google Colab</a> 中在医疗药丸检测数据集上训练 Ultralytics YOLO11 模型
</p>

该数据集作为自动化制药工作流程中质量控制、包装自动化和高效分拣等基本[任务](https://docs.ultralytics.com/tasks/)的基础资源。通过将此数据集集成到项目中，研究人员和开发者可以探索创新[解决方案](https://docs.ultralytics.com/solutions/)，提高[准确性](https://www.ultralytics.com/glossary/accuracy)、简化操作，并最终为改善医疗保健结果做出贡献。

## 数据集结构

医疗药丸数据集分为两个子集：

- **训练集**：包含 92 张图像，每张图像都标注了 `pill` 类别。
- **验证集**：包含 23 张带有相应标注的图像。

## 应用

使用计算机视觉进行医疗药丸检测可实现制药行业的自动化，支持以下任务：

- **制药分拣**：根据大小、形状或颜色自动分拣药丸，以提高生产效率。
- **AI 研究与开发**：作为在制药用例中开发和测试计算机视觉算法的基准。
- **数字库存系统**：通过集成自动药丸识别为实时库存监控和补货计划提供智能库存解决方案。
- **质量控制**：通过识别缺陷、不规则或污染来确保药丸生产的一致性。
- **假药检测**：通过分析视觉特征与已知标准进行比较，帮助识别潜在的假药。


## 数据集 YAML

提供了一个 YAML 配置文件来定义数据集的结构，包括路径和类别。对于医疗药丸数据集，`medical-pills.yaml` 文件可在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml) 访问。

!!! example "ultralytics/cfg/datasets/medical-pills.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/medical-pills.yaml"
    ```

## 使用方法

要在医疗药丸数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，请使用以下示例。有关详细参数，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="medical-pills.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=medical-pills.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载微调模型

        # 使用模型进行推理
        results = model.predict("https://ultralytics.com/assets/medical-pills-sample.jpg")
        ```

    === "CLI"

        ```bash
        # 使用微调的 *.pt 模型开始预测
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/medical-pills-sample.jpg"
        ```

## 示例图像和标注

医疗药丸数据集包含展示药丸多样性的标注图像。以下是数据集中标注图像的示例：

![医疗药丸数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/medical-pills-dataset-sample-image.avif)

- **马赛克图像**：展示的是由马赛克数据集图像组成的训练批次。马赛克通过将多张图像合并为一张来增强训练多样性，提高模型泛化能力。

## 与其他数据集的集成

为了进行更全面的制药分析，请考虑将医疗药丸数据集与其他相关数据集结合，如用于包装识别的 [package-seg](../segment/package-seg.md) 或医学影像数据集如 [brain-tumor](brain-tumor.md)，以开发端到端的医疗保健 AI 解决方案。

## 引用和致谢

该数据集在 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)下提供。

如果您在研究或开发工作中使用医疗药丸数据集，请使用以下详细信息进行引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Jocher_Ultralytics_Datasets_2024,
            author = {Jocher, Glenn and Rizwan, Muhammad},
            license = {AGPL-3.0},
            month = {Dec},
            title = {Ultralytics Datasets: Medical-pills Detection Dataset},
            url = {https://docs.ultralytics.com/datasets/detect/medical-pills/},
            version = {1.0.0},
            year = {2024}
        }
        ```

## 常见问题

### 医疗药丸数据集的结构是什么？

该数据集包含 92 张用于训练的图像和 23 张用于验证的图像。每张图像都标注了 `pill` 类别，可有效训练和评估制药应用模型。

### 如何在医疗药丸数据集上训练 YOLO11 模型？

您可以使用提供的 Python 或 CLI 方法训练 YOLO11 模型 100 个训练周期，图像尺寸为 640px。请参阅[使用方法](#使用方法)部分了解详细说明，并查看 [YOLO11 文档](../../models/yolo11.md)了解更多模型功能信息。

### 在 AI 项目中使用医疗药丸数据集有什么好处？

该数据集支持药丸检测自动化，有助于假药预防、质量保证和制药流程优化。它还是开发可改善药物安全和供应链效率的 AI 解决方案的宝贵资源。

### 如何在医疗药丸数据集上进行推理？

可以使用 Python 或 CLI 方法使用微调的 YOLO11 模型进行推理。请参阅[使用方法](#使用方法)部分了解代码片段，并参阅[预测模式文档](../../modes/predict.md)了解其他选项。

### 在哪里可以找到医疗药丸数据集的 YAML 配置文件？

YAML 文件可在 [medical-pills.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml) 获取，包含数据集路径、类别和在此数据集上训练模型所需的其他配置详情。
