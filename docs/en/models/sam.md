---
comments: true
description: 探索革命性的 Segment Anything Model (SAM)，用于可提示图像分割，具有零样本性能。了解关键特性、数据集和使用技巧。
keywords: Segment Anything, SAM, 图像分割, 可提示分割, 零样本性能, SA-1B 数据集, 高级架构, 自动标注, Ultralytics, 预训练模型, 实例分割, 计算机视觉, AI, 机器学习
---

# Segment Anything Model (SAM)

!!! note "SAM 演进"

    这是 Meta 的原始 SAM 模型。如需增强功能，请参阅 [SAM 2](sam-2.md) 了解视频分割，或参阅 [SAM 3](sam-3.md) 了解使用文本和图像示例提示的可提示概念分割。

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/inference-with-meta-sam-and-sam2-using-ultralytics-python-package.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="如何在 Colab 中使用 Segment Anything"></a>

欢迎来到[图像分割](https://www.ultralytics.com/glossary/image-segmentation)的前沿领域——Segment Anything Model，简称 SAM。这个革命性的模型通过引入具有实时性能的可提示图像分割，改变了游戏规则，在该领域树立了新标准。

## SAM 简介：Segment Anything Model

Segment Anything Model，即 SAM，是一个尖端的图像分割模型，支持可提示分割，在图像分析任务中提供无与伦比的多功能性。SAM 是 Segment Anything 计划的核心，这是一个开创性的项目，为图像分割引入了新颖的模型、任务和数据集。

SAM 的先进设计使其能够在没有先验知识的情况下适应新的图像分布和任务，这一特性被称为零样本迁移。SAM 在广泛的 [SA-1B 数据集](https://ai.meta.com/datasets/segment-anything/)上进行训练，该数据集包含超过 10 亿个掩码，分布在 1100 万张精心策划的图像上，SAM 展示了令人印象深刻的零样本性能，在许多情况下超越了之前的完全监督结果。

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/sa-1b-dataset-sample.avif) **SA-1B 示例图像。** 数据集图像叠加了来自新引入的 SA-1B 数据集的掩码。SA-1B 包含 1100 万张多样化、高分辨率、经过许可且保护隐私的图像，以及 11 亿个高质量分割掩码。这些掩码由 SAM 完全自动标注，经人工评级和大量实验验证，具有高质量和多样性。图像按每张图像的掩码数量分组以便可视化（平均每张图像约有 100 个掩码）。

## Segment Anything Model (SAM) 的关键特性

- **可提示分割任务：** SAM 设计时考虑了可提示分割任务，允许它从任何给定的提示生成有效的分割掩码，例如识别对象的空间或文本线索。
- **先进架构：** Segment Anything Model 采用强大的图像编码器、提示编码器和轻量级掩码解码器。这种独特的架构支持灵活的提示、实时掩码计算以及分割任务中的歧义感知。
- **SA-1B 数据集：** 由 Segment Anything 项目引入，SA-1B 数据集在 1100 万张图像上包含超过 10 亿个掩码。作为迄今为止最大的分割数据集，它为 SAM 提供了多样化和大规模的训练数据源。
- **零样本性能：** SAM 在各种分割任务中展示了出色的零样本性能，使其成为各种应用的即用工具，几乎不需要[提示工程](https://www.ultralytics.com/glossary/prompt-engineering)。

要深入了解 Segment Anything Model 和 SA-1B 数据集，请访问 [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything) 并查看研究论文 [Segment Anything](https://arxiv.org/abs/2304.02643)。

## 可用模型、支持的任务和操作模式

此表展示了可用模型及其特定的预训练权重、支持的任务，以及与不同操作模式的兼容性，如[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)，用 ✅ 表示支持的模式，用 ❌ 表示不支持的模式。

| 模型类型   | 预训练权重                                                                              | 支持的任务                                   | 推理 | 验证 | 训练 | 导出 |
| ---------- | --------------------------------------------------------------------------------------- | -------------------------------------------- | ---- | ---- | ---- | ---- |
| SAM base   | [sam_b.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt)     | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |
| SAM large  | [sam_l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_l.pt)     | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |

## 如何使用 SAM：图像分割的多功能性和强大功能

Segment Anything Model 可用于超出其训练数据的众多下游任务。这包括边缘检测、对象提议生成、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)和初步的文本到掩码预测。通过提示工程，SAM 可以以零样本方式快速适应新任务和数据分布，使其成为满足所有图像分割需求的多功能且强大的工具。

### SAM 预测示例

!!! example "使用提示进行分割"

    使用给定的提示分割图像。

    === "Python"

        ```python
        from ultralytics import SAM

        # 加载模型
        model = SAM("sam_b.pt")

        # 显示模型信息（可选）
        model.info()

        # 使用边界框提示运行推理
        results = model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

        # 使用单点运行推理
        results = model(points=[900, 370], labels=[1])

        # 使用多点运行推理
        results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

        # 使用每个对象的多点提示运行推理
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # 使用负点提示运行推理
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

!!! example "分割所有内容"

    分割整个图像。

    === "Python"

        ```python
        from ultralytics import SAM

        # 加载模型
        model = SAM("sam_b.pt")

        # 显示模型信息（可选）
        model.info()

        # 运行推理
        model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        # 使用 SAM 模型运行推理
        yolo predict model=sam_b.pt source=path/to/image.jpg
        ```

- 这里的逻辑是，如果您不传递任何提示（边界框/点/掩码），则分割整个图像。

!!! example "SAMPredictor 示例"

    这种方式可以设置一次图像，然后多次运行提示推理，而无需多次运行图像编码器。

    === "提示推理"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # 创建 SAMPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # 设置图像
        predictor.set_image("ultralytics/assets/zidane.jpg")  # 使用图像文件设置
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # 使用 np.ndarray 设置
        results = predictor(bboxes=[439, 437, 524, 709])

        # 使用单点提示运行推理
        results = predictor(points=[900, 370], labels=[1])

        # 使用多点提示运行推理
        results = predictor(points=[[400, 370], [900, 370]], labels=[1, 1])

        # 使用负点提示运行推理
        results = predictor(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

        # 重置图像
        predictor.reset_image()
        ```

    使用附加参数分割所有内容。

    === "分割所有内容"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # 创建 SAMPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # 使用附加参数进行分割
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

!!! note

    上述示例中返回的所有 `results` 都是 [Results](../modes/predict.md#working-with-results) 对象，可以轻松访问预测的掩码和源图像。

- 有关 `分割所有内容` 的更多附加参数，请参阅 [`Predictor/generate` 参考](../reference/models/sam/predict.md)。

## SAM 与 YOLO 的比较

这里我们比较 Meta 的 SAM-b 模型与 Ultralytics 最小的分割模型 [YOLO11n-seg](../tasks/segment.md)：

| 模型                                                                                           | 大小<br><sup>(MB)</sup> | 参数<br><sup>(M)</sup> | 速度 (CPU)<br><sup>(ms/im)</sup> |
| ---------------------------------------------------------------------------------------------- | ----------------------- | ---------------------- | -------------------------------- |
| [Meta SAM-b](sam.md)                                                                           | 375                     | 93.7                   | 49401                            |
| [MobileSAM](mobile-sam.md)                                                                     | 40.7                    | 10.1                   | 25381                            |
| [FastSAM-s](fast-sam.md) 使用 YOLOv8 [骨干网络](https://www.ultralytics.com/glossary/backbone) | 23.7                    | 11.8                   | 55.9                             |
| Ultralytics [YOLOv8n-seg](yolov8.md)                                                           | **6.7** (小 11.7 倍)    | **3.4** (少 11.4 倍)   | **24.5** (快 1061 倍)            |
| Ultralytics [YOLO11n-seg](yolo11.md)                                                           | **5.9** (小 13.2 倍)    | **2.9** (少 13.4 倍)   | **30.1** (快 864 倍)             |

此比较展示了 SAM 变体和 YOLO 分割模型之间在模型大小和速度方面的显著差异。虽然 SAM 提供独特的自动分割功能，但 YOLO 模型，特别是 YOLOv8n-seg 和 YOLO11n-seg，明显更小、更快且计算效率更高。

测试在配备 24GB RAM 的 2025 Apple M4 Pro 上运行，使用 `torch==2.6.0` 和 `ultralytics==8.3.90`。要重现此测试：

!!! example

    === "Python"

        ```python
        from ultralytics import ASSETS, SAM, YOLO, FastSAM

        # 分析 SAM2-t, SAM2-b, SAM-b, MobileSAM
        for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
            model = SAM(file)
            model.info()
            model(ASSETS)

        # 分析 FastSAM-s
        model = FastSAM("FastSAM-s.pt")
        model.info()
        model(ASSETS)

        # 分析 YOLO 模型
        for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
            model = YOLO(file_name)
            model.info()
            model(ASSETS)
        ```

## 自动标注：快速创建分割数据集的途径

自动标注是 SAM 的一个关键特性，允许用户使用预训练的检测模型生成[分割数据集](../datasets/segment/index.md)。此功能可以快速准确地标注大量图像，无需耗时的手动标注。

### 使用检测模型生成分割数据集

要使用 Ultralytics 框架自动标注您的数据集，请使用如下所示的 `auto_annotate` 函数：

!!! example

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam_b.pt")
        ```

{% include "macros/sam-auto-annotate.md" %}

`auto_annotate` 函数接受图像路径，以及可选参数用于指定预训练的检测和 SAM 分割模型、运行模型的设备以及保存标注结果的输出目录。

使用预训练模型进行自动标注可以大大减少创建高质量分割数据集所需的时间和精力。此功能对于处理大型图像集合的研究人员和开发人员特别有益，因为它允许他们专注于模型开发和评估，而不是手动标注。

## 引用和致谢

如果您在研究或开发工作中发现 SAM 有用，请考虑引用我们的论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们要感谢 Meta AI 为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源。

## 常见问题

### Ultralytics 的 Segment Anything Model (SAM) 是什么？

Ultralytics 的 Segment Anything Model (SAM) 是一个革命性的图像分割模型，专为可提示分割任务设计。它利用先进的架构，包括图像和提示编码器与轻量级掩码解码器的组合，从各种提示（如空间或文本线索）生成高质量的分割掩码。SAM 在广泛的 [SA-1B 数据集](https://ai.meta.com/datasets/segment-anything/)上训练，在零样本性能方面表现出色，能够在没有先验知识的情况下适应新的图像分布和任务。

### 如何使用 Segment Anything Model (SAM) 进行图像分割？

您可以使用 Segment Anything Model (SAM) 通过各种提示（如边界框或点）运行推理来进行图像分割。以下是使用 Python 的示例：

```python
from ultralytics import SAM

# 加载模型
model = SAM("sam_b.pt")

# 使用边界框提示进行分割
model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

# 使用点提示进行分割
model("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

# 使用多点提示进行分割
model("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

# 使用每个对象的多点提示进行分割
model("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# 使用负点提示进行分割
model("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
```

或者，您可以在命令行界面 (CLI) 中使用 SAM 运行推理：

```bash
yolo predict model=sam_b.pt source=path/to/image.jpg
```

有关更详细的使用说明，请访问[分割部分](#sam-预测示例)。

### SAM 和 YOLO 模型在性能方面如何比较？

与 YOLO 模型相比，SAM 变体（如 SAM-b、SAM2-t、MobileSAM 和 FastSAM-s）通常更大且更慢，但提供独特的零样本分割功能。例如，Ultralytics [YOLOv8n-seg](../tasks/segment.md) 比 Meta 的原始 SAM-b 模型**小 11.7 倍**且**快 1069 倍**，突显了 YOLO 在速度和效率方面的显著优势。同样，较新的 [YOLO11n-seg](../tasks/segment.md) 提供更小的尺寸并保持令人印象深刻的推理速度。这使得 YOLO 模型非常适合需要快速、轻量级和计算高效分割的应用，而 SAM 模型在灵活、可提示和零样本分割任务方面表现出色。

### 如何使用 SAM 自动标注我的数据集？

Ultralytics 的 SAM 提供自动标注功能，允许使用预训练的检测模型生成分割数据集。以下是 Python 示例：

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam_b.pt")
```

此函数接受图像路径以及预训练检测和 SAM 分割模型的可选参数，以及设备和输出目录规范。有关完整指南，请参阅[自动标注](#自动标注快速创建分割数据集的途径)。

### Segment Anything Model (SAM) 使用哪些数据集进行训练？

SAM 在广泛的 [SA-1B 数据集](https://ai.meta.com/datasets/segment-anything/)上训练，该数据集包含 1100 万张图像上超过 10 亿个掩码。SA-1B 是迄今为止最大的分割数据集，提供高质量和多样化的[训练数据](https://www.ultralytics.com/glossary/training-data)，确保在各种分割任务中具有令人印象深刻的零样本性能。有关更多详细信息，请访问[数据集部分](#segment-anything-model-sam-的关键特性)。
