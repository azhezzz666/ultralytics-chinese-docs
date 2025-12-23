---
comments: true
description: 探索 FastSAM，一种基于 CNN 的实时解决方案，用于分割图像中的任何对象。高效、具有竞争力，是各种视觉任务的理想选择。
keywords: FastSAM, Fast Segment Anything Model, Ultralytics, 实时分割, CNN, YOLOv8-seg, 目标分割, 图像处理, 计算机视觉
---

# Fast Segment Anything Model (FastSAM)

Fast Segment Anything Model（FastSAM）是一种新颖的、基于 CNN 的实时解决方案，用于 Segment Anything 任务。该任务旨在根据各种可能的用户交互提示分割图像中的任何对象。FastSAM 显著降低了计算需求，同时保持了具有竞争力的性能，使其成为各种视觉任务的实用选择。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/F7db-EHhxss"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics 的 FastSAM 进行目标跟踪
</p>

## 模型架构

![Fast Segment Anything Model (FastSAM) 架构概览](https://github.com/ultralytics/docs/releases/download/0/fastsam-architecture-overview.avif)

## 概述

FastSAM 旨在解决 [Segment Anything Model (SAM)](sam.md) 的局限性，SAM 是一个具有大量计算资源需求的重型 [Transformer](https://www.ultralytics.com/glossary/transformer) 模型。FastSAM 将 segment anything 任务解耦为两个连续阶段：全[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)和提示引导选择。第一阶段使用 [YOLOv8-seg](../tasks/segment.md) 生成图像中所有实例的分割掩码。在第二阶段，它输出与提示对应的感兴趣区域。

## 主要特性

1. **实时解决方案：** 通过利用 CNN 的计算效率，FastSAM 为 segment anything 任务提供了实时解决方案，对于需要快速结果的工业应用非常有价值。

2. **效率和性能：** FastSAM 在不影响性能质量的情况下显著降低了计算和资源需求。它实现了与 SAM 相当的性能，但计算资源大幅减少，使实时应用成为可能。

3. **提示引导分割：** FastSAM 可以根据各种可能的用户交互提示分割图像中的任何对象，在不同场景中提供灵活性和适应性。

4. **基于 YOLOv8-seg：** FastSAM 基于 [YOLOv8-seg](../tasks/segment.md)，这是一个配备实例分割分支的目标检测器。这使其能够有效地生成图像中所有实例的分割掩码。

5. **基准测试中的竞争性结果：** 在 MS COCO 的目标提议任务上，FastSAM 在单个 NVIDIA RTX 3090 上以比 [SAM](sam.md) 快得多的速度获得了高分，展示了其效率和能力。

6. **实际应用：** 所提出的方法为大量视觉任务提供了一种新的、实用的解决方案，速度非常快，比当前方法快数十倍甚至数百倍。

7. **模型压缩可行性：** FastSAM 展示了通过在结构中引入人工先验来显著减少计算工作量的路径的可行性，从而为通用视觉任务的大型模型架构开辟了新的可能性。

## 可用模型、支持的任务和操作模式

此表展示了可用模型及其特定的预训练权重、它们支持的任务，以及它们与不同操作模式的兼容性，用 ✅ 表示支持的模式，用 ❌ 表示不支持的模式。

| 模型类型   | 预训练权重                                                                                  | 支持的任务                              | 推理 | 验证 | 训练 | 导出 |
| ---------- | ------------------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| FastSAM-s  | [FastSAM-s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt) | [实例分割](../tasks/segment.md) | ✅        | ❌         | ❌       | ✅     |
| FastSAM-x  | [FastSAM-x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt) | [实例分割](../tasks/segment.md) | ✅        | ❌         | ❌       | ✅     |

## FastSAM 与 YOLO 比较

| 模型                                                                                          | 大小<br><sup>(MB)</sup> | 参数<br><sup>(M)</sup> | 速度 (CPU)<br><sup>(ms/im)</sup> |
| ---------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| [Meta SAM-b](sam.md)                                                                           | 375                     | 93.7                         | 49401                             |
| [Meta SAM2-b](sam-2.md)                                                                        | 162                     | 80.8                         | 31901                             |
| [Meta SAM2-t](sam-2.md)                                                                        | 78.1                    | 38.9                         | 25997                             |
| [MobileSAM](mobile-sam.md)                                                                     | 40.7                    | 10.1                         | 25381                             |
| [FastSAM-s](fast-sam.md) 使用 YOLOv8 骨干网络 | 23.7                    | 11.8                         | 55.9                              |
| Ultralytics [YOLOv8n-seg](yolov8.md)                                                           | **6.7**（小 11.7 倍） | **3.4**（少 11.4 倍）         | **24.5**（快 1061 倍）           |
| Ultralytics [YOLO11n-seg](yolo11.md)                                                           | **5.9**（小 13.2 倍） | **2.9**（少 13.4 倍）         | **30.1**（快 864 倍）            |

## 使用示例

### 预测使用

!!! example

    === "Python"

        ```python
        from ultralytics import FastSAM

        # 定义推理源
        source = "path/to/bus.jpg"

        # 创建 FastSAM 模型
        model = FastSAM("FastSAM-s.pt")  # 或 FastSAM-x.pt

        # 对图像运行推理
        everything_results = model(source, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # 使用边界框提示运行推理
        results = model(source, bboxes=[439, 437, 524, 709])

        # 使用点提示运行推理
        results = model(source, points=[[200, 200]], labels=[1])

        # 使用文本提示运行推理
        results = model(source, texts="a photo of a dog")
        ```

    === "CLI"

        ```bash
        yolo segment predict model=FastSAM-s.pt source=path/to/bus.jpg imgsz=640
        ```

### 验证使用

!!! example

    === "Python"

        ```python
        from ultralytics import FastSAM

        model = FastSAM("FastSAM-s.pt")
        results = model.val(data="coco8-seg.yaml")
        ```

    === "CLI"

        ```bash
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

## 引用和致谢

!!! quote ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

## 常见问题

### 什么是 FastSAM，它与 SAM 有何不同？

FastSAM 是一种基于 CNN 的实时解决方案，旨在减少计算需求，同时在目标分割任务中保持高性能。与使用较重的基于 Transformer 架构的 SAM 不同，FastSAM 利用 YOLOv8-seg 在两个阶段进行高效的实例分割。

### FastSAM 支持哪些类型的分割任务提示？

FastSAM 支持多种提示类型：Everything 提示、边界框提示、文本提示和点提示，使其在不同场景中具有灵活性和适应性。
