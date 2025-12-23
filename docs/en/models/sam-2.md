---
comments: true
description: 探索 SAM 2，Meta 的 Segment Anything Model 的下一代版本，支持图像和视频中的实时可提示分割，具有最先进的性能。了解其关键特性、数据集和使用方法。
keywords: SAM 2, SAM 2.1, Segment Anything, 视频分割, 图像分割, 可提示分割, 零样本性能, SA-V 数据集, Ultralytics, 实时分割, AI, 机器学习
---

# SAM 2: Segment Anything Model 2

!!! note "SAM 演进"

    SAM 2 在原始 [SAM](sam.md) 的基础上增加了视频分割功能。如需使用文本和图像示例提示的可提示概念分割，请参阅 [SAM 3](sam-3.md)。

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/inference-with-meta-sam-and-sam2-using-ultralytics-python-package.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中使用 Segment Anything 2 进行推理"></a>

SAM 2 是 Meta 的 [Segment Anything Model (SAM)](sam.md) 的继任者，是一个尖端工具，专为图像和视频中的综合对象分割而设计。它擅长通过统一的、可提示的模型架构处理复杂的视觉数据，支持实时处理和零样本泛化。

![SAM 2 示例结果](https://github.com/ultralytics/docs/releases/download/0/sa-v-dataset.avif)

## 关键特性

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yXQPLMrNX2s"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics 运行 Meta 的 SAM2 推理 | 分步指南 🎉
</p>

### 统一模型架构

SAM 2 将图像和视频分割的功能结合在单一模型中。这种统一简化了部署，并允许在不同媒体类型之间保持一致的性能。它利用灵活的基于提示的接口，使用户能够通过各种提示类型（如点、边界框或掩码）指定感兴趣的对象。

### 实时性能

该模型实现了实时推理速度，每秒处理约 44 帧。这使得 SAM 2 适用于需要即时反馈的应用，如视频编辑和增强现实。

### 零样本泛化

SAM 2 可以分割它从未遇到过的对象，展示了强大的零样本泛化能力。这在多样化或不断发展的视觉领域特别有用，因为预定义的类别可能无法涵盖所有可能的对象。

### 交互式细化

用户可以通过提供额外的提示来迭代细化分割结果，从而精确控制输出。这种交互性对于视频标注或医学成像等应用中的结果微调至关重要。

### 高级视觉挑战处理

SAM 2 包含管理常见视频分割挑战的机制，如对象遮挡和重新出现。它使用复杂的记忆机制来跨帧跟踪对象，即使对象暂时被遮挡或退出并重新进入场景，也能确保连续性。

要深入了解 SAM 2 的架构和功能，请探索 [SAM 2 研究论文](https://arxiv.org/abs/2408.00714)。

## 性能和技术细节

SAM 2 在该领域树立了新的基准，在各种指标上超越了之前的模型：

| 指标                                                                                       | SAM 2         | 之前的 SOTA   |
| ------------------------------------------------------------------------------------------ | ------------- | ------------- |
| **交互式视频分割**                                                                         | **最佳**      | -             |
| **所需人工交互**                                                                           | **减少 3 倍** | 基线          |
| **[图像分割](https://www.ultralytics.com/glossary/image-segmentation)精度**                | **提升**      | SAM           |
| **推理速度**                                                                               | **快 6 倍**   | SAM           |

## 模型架构

### 核心组件

- **图像和视频编码器**：利用基于 [Transformer](https://www.ultralytics.com/glossary/transformer) 的架构从图像和视频帧中提取高级特征。该组件负责理解每个时间步的视觉内容。
- **提示编码器**：处理用户提供的提示（点、框、掩码）以指导分割任务。这允许 SAM 2 适应用户输入并针对场景中的特定对象。
- **记忆机制**：包括记忆编码器、记忆库和记忆注意力模块。这些组件共同存储和利用过去帧的信息，使模型能够随时间保持一致的[对象跟踪](https://www.ultralytics.com/glossary/object-tracking)。
- **掩码解码器**：根据编码的图像特征和提示生成最终的分割掩码。在视频中，它还使用记忆上下文来确保跨帧的准确跟踪。

![SAM 2 架构图](https://github.com/ultralytics/docs/releases/download/0/sam2-architecture-diagram.avif)

### 记忆机制和遮挡处理

记忆机制允许 SAM 2 处理视频数据中的时间依赖性和遮挡。当对象移动和交互时，SAM 2 将其特征记录在记忆库中。当对象被遮挡时，模型可以依靠此记忆来预测其重新出现时的位置和外观。遮挡头专门处理对象不可见的场景，预测对象被遮挡的可能性。

### 多掩码歧义解决

在存在歧义的情况下（例如重叠对象），SAM 2 可以生成多个掩码预测。此功能对于准确表示复杂场景至关重要，因为单个掩码可能无法充分描述场景的细微差别。


## SA-V 数据集

为 SAM 2 训练开发的 SA-V 数据集是目前可用的最大和最多样化的视频分割数据集之一。它包括：

- **51,000+ 视频**：在 47 个国家拍摄，提供广泛的真实世界场景。
- **600,000+ 掩码标注**：详细的时空掩码标注，称为"masklets"，涵盖整个对象和部分。
- **数据集规模**：它的视频数量是之前最大数据集的 4.5 倍，标注数量是 53 倍，提供了前所未有的多样性和复杂性。

## 基准测试

### 视频对象分割

SAM 2 在主要视频分割基准测试中展示了卓越的性能：

| 数据集          | J&F  | J    | F    |
| --------------- | ---- | ---- | ---- |
| **DAVIS 2017**  | 82.5 | 79.8 | 85.2 |
| **YouTube-VOS** | 81.2 | 78.9 | 83.5 |

### 交互式分割

在交互式分割任务中，SAM 2 展示了显著的效率和准确性：

| 数据集                | NoC@90 | AUC   |
| --------------------- | ------ | ----- |
| **DAVIS Interactive** | 1.54   | 0.872 |

## 安装

要安装 SAM 2，请使用以下命令。所有 SAM 2 模型将在首次使用时自动下载。

```bash
pip install ultralytics
```

## 如何使用 SAM 2：图像和视频分割的多功能性

下表详细说明了可用的 SAM 2 模型、其预训练权重、支持的任务，以及与不同操作模式的兼容性，如[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)。

| 模型类型      | 预训练权重                                                                                | 支持的任务                                   | 推理 | 验证 | 训练 | 导出 |
| ------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------- | ---- | ---- | ---- | ---- |
| SAM 2 tiny    | [sam2_t.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_t.pt)     | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |
| SAM 2 small   | [sam2_s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_s.pt)     | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |
| SAM 2 base    | [sam2_b.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt)     | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |
| SAM 2 large   | [sam2_l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_l.pt)     | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |
| SAM 2.1 tiny  | [sam2.1_t.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt) | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |
| SAM 2.1 small | [sam2.1_s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_s.pt) | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |
| SAM 2.1 base  | [sam2.1_b.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt) | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |
| SAM 2.1 large | [sam2.1_l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt) | [实例分割](../tasks/segment.md)              | ✅   | ❌   | ❌   | ❌   |

### SAM 2 预测示例

SAM 2 可用于广泛的任务，包括实时视频编辑、医学成像和自主系统。其分割静态和动态视觉数据的能力使其成为研究人员和开发人员的多功能工具。

#### 使用提示进行分割

!!! example "使用提示进行分割"

    使用提示分割图像或视频中的特定对象。

    === "Python"

        ```python
        from ultralytics import SAM

        # 加载模型
        model = SAM("sam2.1_b.pt")

        # 显示模型信息（可选）
        model.info()

        # 使用边界框提示运行推理
        results = model("path/to/image.jpg", bboxes=[100, 100, 200, 200])

        # 使用单点运行推理
        results = model(points=[900, 370], labels=[1])

        # 使用多点运行推理
        results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

        # 使用每个对象的多点提示运行推理
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # 使用负点提示运行推理
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

#### 分割所有内容

!!! example "分割所有内容"

    无需特定提示即可分割整个图像或视频内容。

    === "Python"

        ```python
        from ultralytics import SAM

        # 加载模型
        model = SAM("sam2.1_b.pt")

        # 显示模型信息（可选）
        model.info()

        # 运行推理
        model("path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # 使用 SAM 2 模型运行推理
        yolo predict model=sam2.1_b.pt source=path/to/video.mp4
        ```

#### 分割视频并跟踪对象

!!! example "分割视频"

    使用特定提示分割整个视频内容并跟踪对象。

    === "Python"

        ```python
        from ultralytics.models.sam import SAM2VideoPredictor

        # 创建 SAM2VideoPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
        predictor = SAM2VideoPredictor(overrides=overrides)

        # 使用单点运行推理
        results = predictor(source="test.mp4", points=[920, 470], labels=[1])

        # 使用多点运行推理
        results = predictor(source="test.mp4", points=[[920, 470], [909, 138]], labels=[1, 1])

        # 使用每个对象的多点提示运行推理
        results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 1]])

        # 使用负点提示运行推理
        results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 0]])
        ```

- 此示例演示了如果不提供任何提示（边界框/点/掩码），SAM 2 如何用于分割图像或视频的整个内容。


## 动态交互式分割和跟踪

SAM2DynamicInteractivePredictor 是 SAM2 的高级无训练扩展，支持与多帧的动态交互和持续学习功能。此预测器支持实时提示更新和记忆管理，以提高跨图像序列的跟踪性能。与原始 SAM2 相比，SAM2DynamicInteractivePredictor 重建了推理流程，以充分利用预训练的 SAM2 模型，无需额外训练。

![SAM 2 示例结果](https://github.com/ultralytics/assets/releases/download/v0.0.0/sam2-interative-sample.avif)

### 关键特性

它提供三个重要增强功能：

1. **动态交互**：在视频处理过程中随时添加新提示以合并/跟踪后续帧中的新实例
2. **持续学习**：为现有实例添加新提示以随时间提高模型性能
3. **独立多图像支持**：处理多个独立图像（不一定来自视频序列），具有记忆共享和跨图像对象跟踪功能

### 核心功能

- **提示灵活性**：接受边界框、点和掩码作为提示
- **记忆库管理**：维护动态记忆库以存储跨帧的对象状态
- **多对象跟踪**：支持同时跟踪多个对象，具有独立的对象 ID
- **实时更新**：允许在推理期间添加新提示，无需重新处理之前的帧
- **独立图像处理**：处理具有共享记忆上下文的独立图像，以实现跨图像对象一致性

!!! example "动态对象添加"

    === "Python"

        ```python
        from ultralytics.models.sam import SAM2DynamicInteractivePredictor

        # 创建 SAM2DynamicInteractivePredictor
        overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_t.pt", save=False)
        predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

        # 通过框提示定义类别
        predictor(source="image1.jpg", bboxes=[[100, 100, 200, 200]], obj_ids=[0], update_memory=True)

        # 在新图像中检测此特定对象
        results = predictor(source="image2.jpg")

        # 使用新对象 ID 添加新类别
        results = predictor(
            source="image4.jpg",
            bboxes=[[300, 300, 400, 400]],  # 新对象
            obj_ids=[1],  # 新对象 ID
            update_memory=True,  # 添加到记忆
        )
        # 执行推理
        results = predictor(source="image5.jpg")

        # 为同一类别添加细化提示以提高性能
        # 当对象外观显著变化时这很有帮助
        results = predictor(
            source="image6.jpg",
            points=[[150, 150]],  # 细化点
            labels=[1],  # 正点
            obj_ids=[1],  # 相同对象 ID
            update_memory=True,  # 使用新信息更新记忆
        )
        # 在新图像上执行推理
        results = predictor(source="image7.jpg")
        ```

!!! note

    `SAM2DynamicInteractivePredictor` 设计用于与 SAM2 模型配合使用，并支持通过 SAM2 原生支持的所有[框/点/掩码提示](#sam-2-预测示例)添加/细化类别。它特别适用于对象随时间出现或变化的场景，如视频标注或交互式编辑任务。

#### 参数

| 名称            | 默认值  | 数据类型    | 描述                           |
| --------------- | ------- | ----------- | ------------------------------ |
| `max_obj_num`   | `3`     | `int`       | 预设的最大类别数量             |
| `update_memory` | `False` | `bool`      | 是否使用新提示更新记忆         |
| `obj_ids`       | `None`  | `List[int]` | 与提示对应的对象 ID 列表       |

### 使用场景

`SAM2DynamicInteractivePredictor` 非常适合：

- **视频标注工作流**：序列中出现新对象的场景
- **交互式视频编辑**：需要实时对象添加和细化
- **监控应用**：具有动态对象跟踪需求
- **医学成像**：跨时间序列跟踪解剖结构
- **自主系统**：需要自适应对象检测和跟踪
- **多图像数据集**：跨独立图像的一致对象分割
- **图像集合分析**：需要跨不同场景跟踪对象
- **跨域分割**：利用来自不同图像上下文的记忆
- **半自动标注**：以最少的人工干预高效创建数据集

## SAM 2 与 YOLO 的比较

这里我们比较 Meta 的 SAM 2 模型（包括最小的 SAM2-t 变体）与 Ultralytics 最小的分割模型 [YOLO11n-seg](../tasks/segment.md)：

| 模型                                                                                           | 大小<br><sup>(MB)</sup> | 参数<br><sup>(M)</sup> | 速度 (CPU)<br><sup>(ms/im)</sup> |
| ---------------------------------------------------------------------------------------------- | ----------------------- | ---------------------- | -------------------------------- |
| [Meta SAM-b](sam.md)                                                                           | 375                     | 93.7                   | 49401                            |
| Meta SAM2-b                                                                                    | 162                     | 80.8                   | 31901                            |
| Meta SAM2-t                                                                                    | 78.1                    | 38.9                   | 25997                            |
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


## 自动标注：高效创建数据集

自动标注是 SAM 2 的强大功能，使用户能够利用预训练模型快速准确地生成分割数据集。此功能对于创建大型高质量数据集特别有用，无需大量手动工作。

### 如何使用 SAM 2 进行自动标注

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/M7xWw4Iodhg"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 使用 Ultralytics 的 Meta Segment Anything 2 模型进行自动标注 | 数据标注
</p>

要使用 SAM 2 自动标注您的数据集，请按照以下示例操作：

!!! example "自动标注示例"

    ```python
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam2_b.pt")
    ```

{% include "macros/sam-auto-annotate.md" %}

此功能有助于快速创建高质量的分割数据集，非常适合希望加速项目的研究人员和开发人员。

## 局限性

尽管 SAM 2 具有优势，但它也有一些局限性：

- **跟踪稳定性**：SAM 2 可能在长序列或显著视角变化期间丢失对象跟踪。
- **对象混淆**：模型有时会混淆外观相似的对象，特别是在拥挤的场景中。
- **多对象效率**：由于缺乏对象间通信，同时处理多个对象时分割效率会降低。
- **细节[精度](https://www.ultralytics.com/glossary/accuracy)**：可能会遗漏细节，特别是对于快速移动的对象。额外的提示可以部分解决此问题，但不能保证时间平滑性。

## 引用和致谢

如果 SAM 2 是您研究或开发工作的关键部分，请使用以下参考文献引用它：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{ravi2024sam2,
          title={SAM 2: Segment Anything in Images and Videos},
          author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
          journal={arXiv preprint},
          year={2024}
        }
        ```

我们感谢 Meta AI 为 AI 社区贡献了这一开创性的模型和数据集。

## 常见问题

### SAM 2 是什么，它如何改进原始的 Segment Anything Model (SAM)？

SAM 2 是 Meta 的 [Segment Anything Model (SAM)](sam.md) 的继任者，是一个尖端工具，专为图像和视频中的综合对象分割而设计。它擅长通过统一的、可提示的模型架构处理复杂的视觉数据，支持实时处理和零样本泛化。SAM 2 相对于原始 SAM 提供了多项改进，包括：

- **统一模型架构**：将图像和视频分割功能结合在单一模型中。
- **实时性能**：每秒处理约 44 帧，适用于需要即时反馈的应用。
- **零样本泛化**：分割从未遇到过的对象，在多样化的视觉领域中很有用。
- **交互式细化**：允许用户通过提供额外的提示来迭代细化分割结果。
- **高级视觉挑战处理**：管理常见的视频分割挑战，如对象遮挡和重新出现。

有关 SAM 2 架构和功能的更多详细信息，请探索 [SAM 2 研究论文](https://arxiv.org/abs/2408.00714)。

### 如何使用 SAM 2 进行实时视频分割？

SAM 2 可以通过利用其可提示接口和实时推理功能进行实时视频分割。以下是基本示例：

!!! example "使用提示进行分割"

    使用提示分割图像或视频中的特定对象。

    === "Python"

        ```python
        from ultralytics import SAM

        # 加载模型
        model = SAM("sam2_b.pt")

        # 显示模型信息（可选）
        model.info()

        # 使用边界框提示进行分割
        results = model("path/to/image.jpg", bboxes=[100, 100, 200, 200])

        # 使用点提示进行分割
        results = model("path/to/image.jpg", points=[150, 150], labels=[1])
        ```

有关更全面的用法，请参阅[如何使用 SAM 2](#如何使用-sam-2图像和视频分割的多功能性)部分。

### SAM 2 使用哪些数据集进行训练，它们如何增强其性能？

SAM 2 在 SA-V 数据集上训练，这是目前可用的最大和最多样化的视频分割数据集之一。SA-V 数据集包括：

- **51,000+ 视频**：在 47 个国家拍摄，提供广泛的真实世界场景。
- **600,000+ 掩码标注**：详细的时空掩码标注，称为"masklets"，涵盖整个对象和部分。
- **数据集规模**：视频数量是之前最大数据集的 4.5 倍，标注数量是 53 倍，提供了前所未有的多样性和复杂性。

这个广泛的数据集使 SAM 2 能够在主要视频分割基准测试中取得卓越性能，并增强其零样本泛化能力。有关更多信息，请参阅 [SA-V 数据集](#sa-v-数据集)部分。

### SAM 2 如何处理视频分割中的遮挡和对象重新出现？

SAM 2 包含复杂的记忆机制来管理视频数据中的时间依赖性和遮挡。记忆机制包括：

- **记忆编码器和记忆库**：存储过去帧的特征。
- **记忆注意力模块**：利用存储的信息随时间保持一致的对象跟踪。
- **遮挡头**：专门处理对象不可见的场景，预测对象被遮挡的可能性。

此机制确保即使对象暂时被遮挡或退出并重新进入场景，也能保持连续性。有关更多详细信息，请参阅[记忆机制和遮挡处理](#记忆机制和遮挡处理)部分。

### SAM 2 与 YOLO11 等其他分割模型相比如何？

SAM 2 模型（如 Meta 的 SAM2-t 和 SAM2-b）提供强大的零样本分割功能，但与 YOLO11 模型相比明显更大且更慢。例如，YOLO11n-seg 比 SAM2-b **小约 13 倍**且**快 860 倍以上**。虽然 SAM 2 在多功能、基于提示和零样本分割场景中表现出色，但 YOLO11 针对速度、效率和实时应用进行了优化，使其更适合在资源受限的环境中部署。
