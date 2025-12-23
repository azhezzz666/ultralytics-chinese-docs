---
comments: true
description: 学习如何使用 Comet 简化 YOLO11 训练的日志记录。本指南涵盖安装、设置、实时洞察和自定义日志记录。
keywords: YOLO11, Comet, Comet ML, 日志记录, 机器学习, 训练, 模型检查点, 指标, 安装, 配置, 实时洞察, 自定义日志记录
---

# 提升 YOLO11 训练：使用 Comet 简化您的日志记录过程

记录关键训练细节（如参数、指标、图像预测和模型检查点）在[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)中至关重要——它使您的项目保持透明、进度可衡量、结果可重复。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LPodYpvKkvI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Comet 记录 Ultralytics YOLO 模型训练日志和指标 🚀
</p>

[Ultralytics YOLO11](https://www.ultralytics.com/) 与 Comet（前身为 Comet ML）无缝集成，高效捕获和优化 YOLO11 [目标检测](https://www.ultralytics.com/glossary/object-detection)模型训练过程的各个方面。在本指南中，我们将介绍安装过程、Comet 设置、实时洞察、自定义日志记录和离线使用，确保您的 YOLO11 训练得到全面记录并针对出色结果进行微调。

## Comet

<p align="center">
  <img width="640" src="https://www.comet.com/docs/v2/img/landing/home-hero.svg" alt="Comet 概述">
</p>

[Comet](https://www.comet.com/site/) 是一个用于跟踪、比较、解释和优化机器学习模型和实验的平台。它允许您在模型训练期间记录指标、参数、媒体等，并通过美观的 Web 界面监控您的实验。Comet 帮助数据科学家更快地迭代，增强透明度和可重复性，并有助于开发生产模型。

## 利用 YOLO11 和 Comet 的强大功能

通过将 Ultralytics YOLO11 与 Comet 结合，您可以获得一系列好处。包括简化的实验管理、用于快速调整的实时洞察、灵活且定制的日志记录选项，以及在互联网访问受限时离线记录实验的能力。这种集成使您能够做出数据驱动的决策、分析性能指标并取得卓越成果。

## 安装

要安装所需的包，请运行：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 和 Comet 所需的包
        pip install ultralytics comet_ml torch torchvision
        ```

## 配置 Comet

安装必要的包后，您需要注册、获取 [Comet API 密钥](https://www.comet.com/signup)并进行配置。

!!! tip "配置 Comet"

    === "CLI"

        ```bash
        # 设置您的 Comet API 密钥
        export COMET_API_KEY=YOUR_API_KEY
        ```

然后，您可以初始化您的 Comet 项目。Comet 将自动检测 API 密钥并继续设置。

!!! example "初始化 Comet 项目"

    === "Python"

        ```python
        import comet_ml

        comet_ml.login(project_name="comet-example-yolo11-coco128")
        ```

如果您使用的是 Google Colab 笔记本，上述代码将提示您输入 API 密钥进行初始化。

## 使用

在深入了解使用说明之前，请务必查看 [Ultralytics 提供的 YOLO11 模型系列](../models/yolo11.md)。这将帮助您选择最适合项目需求的模型。


!!! example "使用"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")

        # 训练模型
        results = model.train(
            data="coco8.yaml",
            project="comet-example-yolo11-coco128",
            batch=32,
            save_period=1,
            save_json=True,
            epochs=3,
        )
        ```

运行训练代码后，Comet 将在您的 Comet 工作区中自动创建一个实验来跟踪运行。然后，您将获得一个链接，用于查看 [YOLO11 模型训练](../modes/train.md)过程的详细日志记录。

Comet 无需额外配置即可自动记录以下数据：mAP 和损失等指标、超参数、模型检查点、交互式混淆矩阵和图像[边界框](https://www.ultralytics.com/glossary/bounding-box)预测。

## 使用 Comet 可视化了解模型性能

让我们深入了解 YOLO11 模型开始训练后您将在 Comet 仪表板上看到的内容。仪表板是所有操作发生的地方，通过可视化和统计数据呈现一系列自动记录的信息。以下是快速导览：

**实验面板**

Comet 仪表板的实验面板部分组织和呈现不同的运行及其指标，如分割掩码损失、类别损失、精确度和[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/comet-ml-dashboard-overview.avif" alt="Comet 概述">
</p>

**指标**

在指标部分，您还可以选择以表格格式查看指标，如此处所示的专用窗格中显示的那样。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/comet-ml-metrics-tabular.avif" alt="Comet 概述">
</p>

**交互式[混淆矩阵](https://www.ultralytics.com/glossary/confusion-matrix)**

在混淆矩阵选项卡中找到的混淆矩阵提供了一种交互式方式来评估模型的分类[准确率](https://www.ultralytics.com/glossary/accuracy)。它详细说明了正确和错误的预测，让您了解模型的优势和劣势。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/comet-ml-interactive-confusion-matrix.avif" alt="Comet 概述">
</p>

**系统指标**

Comet 记录系统指标以帮助识别训练过程中的任何瓶颈。它包括 GPU 利用率、GPU 内存使用、CPU 利用率和 RAM 使用等指标。这些对于监控模型训练期间资源使用的效率至关重要。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/comet-ml-system-metrics.avif" alt="Comet 概述">
</p>

## 自定义 Comet 日志记录

Comet 提供了通过设置环境变量来自定义其日志记录行为的灵活性。这些配置允许您根据特定需求和偏好定制 Comet。以下是一些有用的自定义选项：

### 记录图像预测

您可以控制 Comet 在实验期间记录的图像预测数量。默认情况下，Comet 从验证集记录 100 个图像预测。但是，您可以更改此数字以更好地满足您的需求。例如，要记录 200 个图像预测，请使用以下代码：

```python
import os

os.environ["COMET_MAX_IMAGE_PREDICTIONS"] = "200"
```

### 批次日志记录间隔

Comet 允许您指定记录图像预测批次的频率。`COMET_EVAL_BATCH_LOGGING_INTERVAL` 环境变量控制此频率。默认设置为 1，即从每个验证批次记录预测。您可以调整此值以不同的间隔记录预测。例如，将其设置为 4 将从每第四个批次记录预测。

```python
import os

os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "4"
```

### 禁用混淆矩阵日志记录

在某些情况下，您可能不希望在每个[训练周期](https://www.ultralytics.com/glossary/epoch)后从验证集记录混淆矩阵。您可以通过将 `COMET_EVAL_LOG_CONFUSION_MATRIX` 环境变量设置为 "false" 来禁用此功能。混淆矩阵将仅在训练完成后记录一次。

```python
import os

os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "false"
```

### 离线日志记录

如果您发现自己处于互联网访问受限的情况，Comet 提供离线日志记录选项。您可以将 `COMET_MODE` 环境变量设置为 "offline" 以启用此功能。您的实验数据将保存在本地目录中，当互联网连接可用时，您可以稍后将其上传到 Comet。

```python
import os

os.environ["COMET_MODE"] = "offline"
```


## 总结

本指南引导您完成了将 Comet 与 Ultralytics 的 YOLO11 集成的过程。从安装到自定义，您已经学会了如何简化实验管理、获得实时洞察，并根据项目需求调整日志记录。

探索 [Comet 的官方 YOLOv8 集成文档](https://www.comet.com/docs/v2/integrations/third-party-tools/yolov8/)，该文档也适用于 YOLO11 项目。

此外，如果您想深入了解 YOLO11 的实际应用，特别是[图像分割](https://www.ultralytics.com/glossary/image-segmentation)任务，这份关于[使用 Comet 微调 YOLO11](https://www.comet.com/site/blog/fine-tuning-yolov8-for-image-segmentation-with-comet/) 的详细指南提供了宝贵的见解和逐步说明，以增强您的模型性能。

另外，要探索与 Ultralytics 的其他令人兴奋的集成，请查看[集成指南页面](../integrations/index.md)，其中提供了丰富的资源和信息。

## 常见问题

### 如何将 Comet 与 Ultralytics YOLO11 集成进行训练？

要将 Comet 与 Ultralytics YOLO11 集成，请按照以下步骤操作：

1. **安装所需的包**：

    ```bash
    pip install ultralytics comet_ml torch torchvision
    ```

2. **设置您的 Comet API 密钥**：

    ```bash
    export COMET_API_KEY=YOUR_API_KEY
    ```

3. **在 Python 代码中初始化您的 Comet 项目**：

    ```python
    import comet_ml

    comet_ml.login(project_name="comet-example-yolo11-coco128")
    ```

4. **训练您的 YOLO11 模型并记录指标**：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    results = model.train(
        data="coco8.yaml",
        project="comet-example-yolo11-coco128",
        batch=32,
        save_period=1,
        save_json=True,
        epochs=3,
    )
    ```

有关更详细的说明，请参阅 [Comet 配置部分](#配置-comet)。

### 将 Comet 与 YOLO11 一起使用有什么好处？

通过将 Ultralytics YOLO11 与 Comet 集成，您可以：

- **监控实时洞察**：获得训练结果的即时反馈，允许快速调整。
- **记录广泛的指标**：自动捕获 mAP、损失、超参数和模型检查点等基本指标。
- **离线跟踪实验**：在互联网访问不可用时在本地记录训练运行。
- **比较不同的训练运行**：使用交互式 Comet 仪表板分析和比较多个实验。

通过利用这些功能，您可以优化机器学习工作流程以获得更好的性能和可重复性。有关更多信息，请访问 [Comet 集成指南](../integrations/index.md)。

### 如何在 YOLO11 训练期间自定义 Comet 的日志记录行为？

Comet 允许使用环境变量进行广泛的日志记录行为自定义：

- **更改记录的图像预测数量**：

    ```python
    import os

    os.environ["COMET_MAX_IMAGE_PREDICTIONS"] = "200"
    ```

- **调整批次日志记录间隔**：

    ```python
    import os

    os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "4"
    ```

- **禁用混淆矩阵日志记录**：

    ```python
    import os

    os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "false"
    ```

有关更多自定义选项，请参阅[自定义 Comet 日志记录](#自定义-comet-日志记录)部分。

### 如何在 Comet 上查看 YOLO11 训练的详细指标和可视化？

一旦您的 YOLO11 模型开始训练，您可以在 Comet 仪表板上访问各种指标和可视化。主要功能包括：

- **实验面板**：查看不同的运行及其指标，包括分割掩码损失、类别损失和平均[精确度](https://www.ultralytics.com/glossary/precision)。
- **指标**：以表格格式查看指标以进行详细分析。
- **交互式混淆矩阵**：使用交互式混淆矩阵评估分类准确性。
- **系统指标**：监控 GPU 和 CPU 利用率、内存使用和其他系统指标。

有关这些功能的详细概述，请访问[使用 Comet 可视化了解模型性能](#使用-comet-可视化了解模型性能)部分。

### 训练 YOLO11 模型时可以使用 Comet 进行离线日志记录吗？

是的，您可以通过将 `COMET_MODE` 环境变量设置为 "offline" 来启用 Comet 中的离线日志记录：

```python
import os

os.environ["COMET_MODE"] = "offline"
```

此功能允许您在本地记录实验数据，稍后在互联网连接可用时可以上传到 Comet。这在互联网访问受限的环境中工作时特别有用。有关更多详细信息，请参阅[离线日志记录](#离线日志记录)部分。
