---
comments: true
description: 探索签名检测数据集，用于训练模型识别和验证各种文档中的人类签名。非常适合文档验证和欺诈预防。
keywords: 签名检测数据集, 文档验证, 欺诈检测, 计算机视觉, YOLO11, Ultralytics, 标注签名, 训练数据集
---

# 签名检测数据集

该数据集专注于检测文档中的人类手写签名。它包含各种带有标注签名的文档类型，为文档验证和欺诈检测应用提供有价值的见解。该数据集对于训练[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)算法至关重要，有助于识别各种文档格式中的签名，支持文档分析的研究和实际应用。

## 数据集结构

签名检测数据集分为两个子集：

- **训练集**：包含 143 张图像，每张图像都有相应的标注。
- **验证集**：包含 35 张图像，每张图像都有配对的标注。

## 应用

该数据集可应用于各种计算机视觉任务，如[目标检测](https://www.ultralytics.com/glossary/object-detection)、[目标跟踪](https://docs.ultralytics.com/modes/track/)和文档分析。具体来说，它可用于训练和评估识别文档中签名的模型，在以下方面具有重要应用：

- **文档验证**：自动化法律和金融文档的验证流程
- **欺诈检测**：识别潜在的伪造或未经授权的签名
- **数字文档处理**：简化行政和法律部门的工作流程
- **银行和金融**：增强支票处理和贷款文档验证的安全性
- **档案研究**：支持历史文档分析和编目

此外，它还是教育目的的宝贵资源，使学生和研究人员能够研究不同文档类型中的签名特征。

## 数据集 YAML

YAML（Yet Another Markup Language）文件定义了数据集配置，包括路径和类别信息。对于签名检测数据集，`signature.yaml` 文件位于 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml)。

!!! example "ultralytics/cfg/datasets/signature.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/signature.yaml"
    ```


## 使用方法

要在签名检测数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，请使用提供的代码示例。有关可用参数的完整列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="signature.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=signature.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载签名检测微调模型

        # 使用模型进行推理
        results = model.predict("https://ultralytics.com/assets/signature-s.mp4", conf=0.75)
        ```

    === "CLI"

        ```bash
        # 使用微调的 *.pt 模型开始预测
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/signature-s.mp4" conf=0.75
        ```

## 示例图像和标注

签名检测数据集包含各种展示不同文档类型和标注签名的图像。以下是数据集中的图像示例，每张图像都附有相应的标注。

![签名检测数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/signature-detection-mosaiced-sample.avif)

- **马赛克图像**：这里展示的是由马赛克数据集图像组成的训练批次。马赛克是一种训练技术，将多张图像合并为一张，丰富批次多样性。这种方法有助于增强模型在不同签名尺寸、宽高比和上下文中的泛化能力。

此示例说明了签名检测数据集中图像的多样性和复杂性，强调了在训练过程中包含马赛克技术的好处。

## 引用和致谢

该数据集在 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)下发布。

## 常见问题

### 什么是签名检测数据集，如何使用？

签名检测数据集是一个标注图像集合，旨在检测各种文档类型中的人类签名。它可应用于[目标检测](https://www.ultralytics.com/glossary/object-detection)和跟踪等计算机视觉任务，主要用于文档验证、欺诈检测和档案研究。该数据集有助于训练模型识别不同上下文中的签名，使其对[智能文档分析](https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-smart-document-analysis)的研究和实际应用都很有价值。

### 如何在签名检测数据集上训练 YOLO11n 模型？

要在签名检测数据集上训练 YOLO11n 模型，请按照以下步骤操作：

1. 从 [signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml) 下载数据集配置文件。
2. 使用以下 Python 脚本或 CLI 命令开始训练：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 训练模型
        results = model.train(data="signature.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=signature.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关更多详情，请参阅[训练](../../modes/train.md)页面。

### 签名检测数据集的主要应用是什么？

签名检测数据集可用于：

1. **文档验证**：自动验证文档中人类签名的存在和真实性。
2. **欺诈检测**：识别法律和金融文档中的伪造或欺诈签名。
3. **档案研究**：协助历史学家和档案管理员进行历史文档的数字分析和编目。
4. **教育**：支持计算机视觉和[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)领域的学术研究和教学。
5. **金融服务**：通过验证签名真实性来增强银行交易和贷款处理的安全性。

### 如何使用在签名检测数据集上训练的模型进行推理？

要使用在签名检测数据集上训练的模型进行推理，请按照以下步骤操作：

1. 加载您的微调模型。
2. 使用以下 Python 脚本或 CLI 命令进行推理：

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载微调模型
        model = YOLO("path/to/best.pt")

        # 执行推理
        results = model.predict("https://ultralytics.com/assets/signature-s.mp4", conf=0.75)
        ```

    === "CLI"

        ```bash
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/signature-s.mp4" conf=0.75
        ```

### 签名检测数据集的结构是什么，在哪里可以找到更多信息？

签名检测数据集分为两个子集：

- **训练集**：包含 143 张带有标注的图像。
- **验证集**：包含 35 张带有标注的图像。

有关详细信息，您可以参阅[数据集结构](#数据集结构)部分。此外，在 [signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml) 中查看完整的数据集配置。
