---
comments: true
description: 探索百度的 RT-DETR，一种基于 Vision Transformer 的实时目标检测器，提供高精度和可调节的推理速度。通过 Ultralytics 了解更多。
keywords: RT-DETR, 百度, Vision Transformer, 实时目标检测, PaddlePaddle, Ultralytics, 预训练模型, AI, 机器学习, 计算机视觉
---

# 百度的 RT-DETR：基于 Vision [Transformer](https://www.ultralytics.com/glossary/transformer) 的实时目标检测器

## 概述

Real-Time Detection Transformer（RT-DETR）由百度开发，是一种前沿的端到端目标检测器，在保持高[精度](https://www.ultralytics.com/glossary/accuracy)的同时提供实时性能。它基于 DETR（无 NMS 框架）的理念，同时引入基于卷积的[骨干网络](https://www.ultralytics.com/glossary/backbone)和高效的混合编码器以获得实时速度。RT-DETR 通过解耦尺度内交互和跨尺度融合来高效处理多尺度特征。该模型具有高度适应性，支持使用不同的解码器层灵活调整推理速度而无需重新训练。RT-DETR 在 CUDA 与 TensorRT 等加速后端上表现出色，优于许多其他实时目标检测器。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/i70ecEGB1ro"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用百度的 RT-DETR 进行目标检测 | 使用 Ultralytics 进行推理和基准测试 🚀
</p>

![模型示例图像](https://github.com/ultralytics/docs/releases/download/0/baidu-rtdetr-model-overview.avif) **百度 RT-DETR 概览。** RT-DETR 模型架构图显示骨干网络的最后三个阶段 {S3, S4, S5} 作为编码器的输入。高效的混合编码器通过尺度内特征交互（AIFI）和跨尺度特征融合模块（CCFM）将多尺度特征转换为图像特征序列。采用 IoU 感知查询选择来选择固定数量的图像特征作为解码器的初始目标查询。最后，带有辅助预测头的解码器迭代优化目标查询以生成边界框和置信度分数（[来源](https://arxiv.org/pdf/2304.08069)）。

### 主要特性

- **高效混合编码器：** 百度的 RT-DETR 使用高效的混合编码器，通过解耦尺度内交互和跨尺度融合来处理多尺度特征。这种独特的基于 Vision Transformers 的设计降低了计算成本，实现了实时[目标检测](https://www.ultralytics.com/glossary/object-detection)。
- **IoU 感知查询选择：** 百度的 RT-DETR 通过利用 IoU 感知查询选择来改进目标查询初始化。这使模型能够专注于场景中最相关的对象，提高检测精度。
- **可调节推理速度：** 百度的 RT-DETR 支持通过使用不同的解码器层灵活调整推理速度，无需重新训练。这种适应性便于在各种实时目标检测场景中的实际应用。
- **无 NMS 框架：** 基于 DETR，RT-DETR 消除了对[非极大值抑制](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)后处理的需求，简化了检测流水线并可能提高效率。
- **无锚点检测：** 作为[无锚点检测器](https://www.ultralytics.com/glossary/anchor-free-detectors)，RT-DETR 简化了检测过程，可能提高在不同数据集上的泛化能力。

## 预训练模型

Ultralytics Python API 提供不同规模的预训练 PaddlePaddle RT-DETR 模型：

- RT-DETR-L：在 COCO val2017 上 53.0% AP，在 T4 GPU 上 114 FPS
- RT-DETR-X：在 COCO val2017 上 54.8% AP，在 T4 GPU 上 74 FPS

此外，百度于 2024 年 7 月发布了 RTDETRv2，进一步改进了原始架构，具有更好的性能指标。

## 使用示例

此示例提供简单的 RT-DETR 训练和推理示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参阅[预测](../modes/predict.md)、[训练](../modes/train.md)、[验证](../modes/val.md)和[导出](../modes/export.md)文档页面。

!!! example

    === "Python"

        ```python
        from ultralytics import RTDETR

        # 加载 COCO 预训练的 RT-DETR-l 模型
        model = RTDETR("rtdetr-l.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个轮次
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 RT-DETR-l 模型对 'bus.jpg' 图像运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 加载 COCO 预训练的 RT-DETR-l 模型并在 COCO8 示例数据集上训练 100 个轮次
        yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载 COCO 预训练的 RT-DETR-l 模型并对 'bus.jpg' 图像运行推理
        yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
        ```

## 支持的任务和模式

| 模型类型          | 预训练权重                                                                        | 支持的任务                        | 推理 | 验证 | 训练 | 导出 |
| ------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------- | --------- | ---------- | -------- | ------ |
| RT-DETR Large       | [rtdetr-l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt) | [目标检测](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| RT-DETR Extra-Large | [rtdetr-x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt) | [目标检测](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |

## 理想用例

RT-DETR 特别适合需要高精度和实时性能的应用：

- **自动驾驶**：用于自动驾驶系统中可靠的环境感知，其中速度和精度都至关重要。
- **高级机器人**：使机器人能够在动态环境中执行需要准确目标识别和交互的复杂任务。
- **医学成像**：用于医疗保健应用，其中目标检测的精度对诊断至关重要。
- **监控系统**：用于需要高检测精度实时监控的安全应用。
- **卫星图像分析**：用于高分辨率图像的详细分析，其中全局上下文理解很重要。

## 引用和致谢

如果您在研究或开发工作中使用百度的 RT-DETR，请引用[原始论文](https://arxiv.org/abs/2304.08069)：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lv2023detrs,
              title={DETRs Beat YOLOs on Real-time Object Detection},
              author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
              year={2023},
              eprint={2304.08069},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

## 常见问题

### 什么是百度的 RT-DETR 模型，它是如何工作的？

百度的 RT-DETR（Real-Time Detection Transformer）是一种基于 Vision Transformer 架构构建的先进实时目标检测器。它通过其高效的混合编码器解耦尺度内交互和跨尺度融合来高效处理多尺度特征。通过采用 IoU 感知查询选择，模型专注于最相关的对象，提高检测精度。其可调节的推理速度（通过调整解码器层而无需重新训练实现）使 RT-DETR 适用于各种实时目标检测场景。

### 为什么应该选择百度的 RT-DETR 而不是其他实时目标检测器？

百度的 RT-DETR 因其高效的混合编码器和 IoU 感知查询选择而脱颖而出，这大大降低了计算成本，同时保持了高精度。其独特的能力是通过使用不同的解码器层来调整推理速度而无需重新训练，这增加了显著的灵活性。这使其在 CUDA 与 TensorRT 等加速后端上需要实时性能的应用中特别有优势，优于许多其他实时目标检测器。
