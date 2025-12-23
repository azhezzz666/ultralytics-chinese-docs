---
comments: true
description: 探索 MobileSAM，一种用于移动和边缘应用的轻量级快速图像分割模型。比较其与 SAM 和 YOLO 模型的性能。
keywords: MobileSAM, 图像分割, 轻量级模型, 快速分割, 移动应用, SAM, Tiny-ViT, YOLO, Ultralytics
---

![MobileSAM Logo](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/logo2.png)

# Mobile Segment Anything (MobileSAM)

MobileSAM 是一个紧凑、高效的图像分割模型，专为移动和边缘设备而构建。它旨在将 Meta 的 Segment Anything Model ([SAM](sam.md)) 的强大功能带到计算资源有限的环境中，MobileSAM 提供近乎即时的分割，同时保持与原始 SAM 流水线的兼容性。无论您是开发实时应用还是轻量级部署，MobileSAM 都能以其前代产品所需大小和速度的一小部分提供令人印象深刻的分割结果。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yXQPLMrNX2s"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 运行 MobileSAM 推理 | 分步指南 🎉
</p>

MobileSAM 已被多个项目采用，包括 [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)、[AnyLabeling](https://github.com/vietanhdev/anylabeling) 和 [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D)。

MobileSAM 在单个 GPU 上使用 100k 图像数据集（原始图像的 1%）在不到一天的时间内完成训练。训练代码将在未来发布。

## 可用模型、支持的任务和操作模式

下表概述了可用的 MobileSAM 模型、其预训练权重、支持的任务，以及与不同操作模式的兼容性。支持的模式用 ✅ 表示，不支持的模式用 ❌ 表示。

| 模型类型   | 预训练权重                                                                            | 支持的任务                              | 推理 | 验证 | 训练 | 导出 |
| ---------- | --------------------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| MobileSAM  | [mobile_sam.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt) | [实例分割](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |

## MobileSAM 与 YOLO 比较

| 模型                                                                           | 大小<br><sup>(MB)</sup> | 参数<br><sup>(M)</sup> | 速度 (CPU)<br><sup>(ms/im)</sup> |
| ------------------------------------------------------------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| Meta SAM-b                                                                      | 375                     | 93.7                         | 49401                             |
| Meta SAM2-b                                                                     | 162                     | 80.8                         | 31901                             |
| Meta SAM2-t                                                                     | 78.1                    | 38.9                         | 25997                             |
| MobileSAM                                                                       | 40.7                    | 10.1                         | 25381                             |
| FastSAM-s 使用 YOLOv8 骨干网络 | 23.7                    | 11.8                         | 55.9                              |
| Ultralytics YOLOv8n-seg                                                         | **6.7**（小 11.7 倍） | **3.4**（少 11.4 倍）         | **24.5**（快 1061 倍）           |
| Ultralytics YOLO11n-seg                                                         | **5.9**（小 13.2 倍） | **2.9**（少 13.4 倍）         | **30.1**（快 864 倍）            |

## 从 SAM 适配到 MobileSAM

MobileSAM 保留了与原始 [SAM](sam.md) 相同的流水线，包括预处理、后处理和所有接口。这意味着您可以以最小的工作流程更改从 SAM 过渡到 MobileSAM。

关键区别在于图像编码器：MobileSAM 用更小的 Tiny-ViT 编码器（5M 参数）替换了原始的 ViT-H 编码器（632M 参数）。在单个 GPU 上，MobileSAM 处理一张图像大约需要 12ms（编码器 8ms，掩码解码器 4ms）。

### 基于 ViT 的图像编码器比较

| 图像编码器 | 原始 SAM | MobileSAM |
| ------------- | ------------ | --------- |
| 参数    | 611M         | 5M        |
| 速度         | 452ms        | 8ms       |

### 提示引导掩码解码器

| 掩码解码器 | 原始 SAM | MobileSAM |
| ------------ | ------------ | --------- |
| 参数   | 3.876M       | 3.876M    |
| 速度        | 4ms          | 4ms       |

### 完整流水线比较

| 完整流水线 (编码器+解码器) | 原始 SAM | MobileSAM |
| ------------------------ | ------------ | --------- |
| 参数               | 615M         | 9.66M     |
| 速度                    | 456ms        | 12ms      |

MobileSAM 比 FastSAM 小约 7 倍，快约 5 倍。

## 在 Ultralytics 中测试 MobileSAM

与原始 [SAM](sam.md) 一样，Ultralytics 提供了一个简单的接口来测试 MobileSAM，支持点提示和框提示。

### 模型下载

从 [Ultralytics assets](https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt) 下载 MobileSAM 预训练权重。

### 点提示

!!! example

    === "Python"

        ```python
        from ultralytics import SAM

        # 加载模型
        model = SAM("mobile_sam.pt")

        # 基于单点提示预测分割
        model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

        # 基于多点提示预测多个分割
        model.predict("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

        # 基于每个对象的多点提示预测分割
        model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # 使用正负提示预测分割
        model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

### 框提示

!!! example

    === "Python"

        ```python
        from ultralytics import SAM

        # 加载模型
        model = SAM("mobile_sam.pt")

        # 基于单点提示预测分割
        model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

        # 基于多点提示预测多个分割
        model.predict("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

        # 基于每个对象的多点提示预测分割
        model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # 使用正负提示预测分割
        model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

`MobileSAM` 和 `SAM` 共享相同的 API。有关更多使用详情，请参阅 [SAM 文档](sam.md)。

### 使用检测模型自动构建分割数据集

要使用 Ultralytics 框架自动标注您的数据集，请使用如下所示的 `auto_annotate` 函数：

!!! example

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="mobile_sam.pt")
        ```

{% include "macros/sam-auto-annotate.md" %}

## 引用和致谢

如果 MobileSAM 对您的研究或开发有帮助，请考虑引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
        ```

在 [arXiv](https://arxiv.org/pdf/2306.14289) 上阅读完整的 MobileSAM 论文。

## 常见问题

### 什么是 MobileSAM，它与原始 SAM 模型有何不同？

MobileSAM 是一个轻量级、快速的[图像分割](https://www.ultralytics.com/glossary/image-segmentation)模型，针对移动和边缘应用进行了优化。它保持与原始 SAM 相同的流水线，但用紧凑的 Tiny-ViT 编码器（5M 参数）替换了大型 ViT-H 编码器（632M 参数）。这使得 MobileSAM 比原始 SAM 小约 5 倍，快约 7 倍，每张图像运行约 12ms，而 SAM 为 456ms。

### 如何使用 Ultralytics 测试 MobileSAM？

在 Ultralytics 中测试 MobileSAM 非常简单。您可以使用点提示和框提示来预测分割。例如，使用点提示：

```python
from ultralytics import SAM

# 加载模型
model = SAM("mobile_sam.pt")

# 基于点提示预测分割
model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])
```

### 为什么应该在移动应用中使用 MobileSAM？

MobileSAM 由于其轻量级设计和快速推理速度，非常适合移动和边缘应用。与原始 SAM 相比，MobileSAM 小约 5 倍，快约 7 倍，适合在计算资源有限的设备上进行实时分割。

### MobileSAM 的主要用例是什么？

MobileSAM 专为移动和边缘环境中的快速、高效图像分割而设计。主要用例包括：

- 移动应用的实时[目标检测和分割](https://www.ultralytics.com/glossary/object-detection)
- 计算资源有限的设备上的低延迟图像处理
- 集成到 AI 驱动的移动应用中，用于增强现实（AR）、分析等
