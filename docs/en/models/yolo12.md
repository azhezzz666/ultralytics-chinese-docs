---
comments: true
description: 探索 YOLO12，采用突破性的注意力中心架构，实现先进的目标检测，具有无与伦比的精度和效率。
keywords: YOLO12, 注意力中心目标检测, YOLO 系列, Ultralytics, 计算机视觉, AI, 机器学习, 深度学习
---

# YOLO12：注意力中心目标检测

## 概述

YOLO12 引入了一种注意力中心架构，与之前 YOLO 模型中使用的传统基于 CNN 的方法不同，但保留了许多应用所必需的实时推理速度。该模型通过注意力机制和整体网络架构的新颖方法创新，实现了先进的目标检测精度，同时保持实时性能。尽管有这些优势，YOLO12 仍然是一个社区驱动的版本，由于其重型注意力块，可能会出现训练不稳定、内存消耗增加和 CPU 吞吐量较慢的情况，因此 Ultralytics 仍然建议大多数生产工作负载使用 YOLO11。

!!! note "社区模型"

    YOLO12 主要用于基准测试和研究。如果您需要稳定的训练、可预测的内存使用和优化的 CPU 推理，请选择 [YOLO11](yolo11.md) 或其他 Ultralytics 维护的版本进行部署。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/mcqTxD-FD5M"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics 包进行 YOLO12 目标检测 | YOLO12 是快还是慢？🚀
</p>

## 关键特性

- **区域注意力机制**：一种新的自注意力方法，可高效处理大感受野。它将[特征图](https://www.ultralytics.com/glossary/feature-maps)划分为 _l_ 个等大小的区域（默认为 4），水平或垂直划分，避免复杂操作并保持大的有效感受野。与标准自注意力相比，这显著降低了计算成本。
- **残差高效层聚合网络（R-ELAN）**：基于 ELAN 的改进特征聚合模块，旨在解决优化挑战，特别是在更大规模的注意力中心模型中。R-ELAN 引入了：
    - 带缩放的块级残差连接（类似于层缩放）。
    - 重新设计的特征聚合方法，创建类似瓶颈的结构。
- **优化的注意力架构**：YOLO12 简化了标准注意力机制，以提高效率并与 YOLO 框架兼容。这包括：
    - 使用 FlashAttention 最小化内存访问开销。
    - 移除位置编码以获得更简洁、更快的模型。
    - 调整 MLP 比率（从典型的 4 调整到 1.2 或 2），以更好地平衡注意力和前馈层之间的计算。
    - 减少堆叠块的深度以改进优化。
    - 利用卷积操作（在适当的地方）以提高计算效率。
    - 向注意力机制添加 7x7 可分离卷积（"位置感知器"）以隐式编码位置信息。
- **全面的任务支持**：YOLO12 支持一系列核心计算机视觉任务：目标检测、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)、[图像分类](https://www.ultralytics.com/glossary/image-classification)、姿态估计和旋转目标检测（OBB）。
- **增强的效率**：与许多先前模型相比，以更少的参数实现更高的精度，展示了速度和精度之间的改进平衡。
- **灵活部署**：设计用于跨多种平台部署，从边缘设备到云基础设施。

![YOLO12 对比可视化](https://github.com/ultralytics/docs/releases/download/0/yolo12-comparison-visualization.avif)

## 支持的任务和模式

YOLO12 支持多种计算机视觉任务。下表显示了任务支持以及为每个任务启用的操作模式（推理、验证、训练和导出）：

| 模型类型                                                                                                       | 任务                                   | 推理 | 验证 | 训练 | 导出 |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| [YOLO12](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/12/yolo12.yaml)           | [检测](../tasks/detect.md)             | ✅   | ✅   | ✅   | ✅   |
| [YOLO12-seg](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/12/yolo12-seg.yaml)   | [分割](../tasks/segment.md)            | ✅   | ✅   | ✅   | ✅   |
| [YOLO12-pose](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/12/yolo12-pose.yaml) | [姿态](../tasks/pose.md)               | ✅   | ✅   | ✅   | ✅   |
| [YOLO12-cls](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/12/yolo12-cls.yaml)   | [分类](../tasks/classify.md)           | ✅   | ✅   | ✅   | ✅   |
| [YOLO12-obb](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/12/yolo12-obb.yaml)   | [OBB](../tasks/obb.md)                 | ✅   | ✅   | ✅   | ✅   |

## 性能指标

YOLO12 在所有模型规模上都展示了显著的[精度](https://www.ultralytics.com/glossary/accuracy)改进，与_最快_的先前 YOLO 模型相比在速度上有一些权衡。以下是 COCO 验证数据集上[目标检测](https://www.ultralytics.com/glossary/object-detection)的定量结果：

### 检测性能（COCO val2017）

!!! tip "性能"

    === "检测 (COCO)"

        | 模型                                                                                 | 尺寸<br><sup>(像素)</sup> | mAP<sup>val<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>T4 TensorRT<br>(ms)</sup> | 参数<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> | 对比<br><sup>(mAP/速度)</sup> |
        | ------------------------------------------------------------------------------------ | ------------------------- | -------------------------- | ----------------------------------- | -------------------------------------- | ---------------------- | ----------------------- | ----------------------------- |
        | [YOLO12n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt) | 640                       | 40.6                       | -                                   | 1.64                                   | 2.6                    | 6.5                     | +2.1%/-9% (vs. YOLOv10n)      |
        | [YOLO12s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt) | 640                       | 48.0                       | -                                   | 2.61                                   | 9.3                    | 21.4                    | +0.1%/+42% (vs. RT-DETRv2)    |
        | [YOLO12m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt) | 640                       | 52.5                       | -                                   | 4.86                                   | 20.2                   | 67.5                    | +1.0%/-3% (vs. YOLO11m)       |
        | [YOLO12l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt) | 640                       | 53.7                       | -                                   | 6.77                                   | 26.4                   | 88.9                    | +0.4%/-8% (vs. YOLO11l)       |
        | [YOLO12x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt) | 640                       | 55.2                       | -                                   | 11.79                                  | 59.1                   | 199.0                   | +0.6%/-4% (vs. YOLO11x)       |

- 推理速度在 NVIDIA T4 GPU 上使用 TensorRT FP16 [精度](https://www.ultralytics.com/glossary/precision)测量。
- 对比显示 mAP 的相对改进和速度的百分比变化（正值表示更快；负值表示更慢）。对比基于 YOLOv10、YOLO11 和 RT-DETR 的已发布结果（如有）。


## 使用示例

本节提供 YOLO12 训练和推理的示例。有关这些和其他模式（包括[验证](../modes/val.md)和[导出](../modes/export.md)）的更全面文档，请参阅专门的[预测](../modes/predict.md)和[训练](../modes/train.md)页面。

以下示例专注于 YOLO12 [检测](../tasks/detect.md)模型（用于目标检测）。有关其他支持的任务（分割、分类、旋转目标检测和姿态估计），请参阅相应的任务特定文档：[分割](../tasks/segment.md)、[分类](../tasks/classify.md)、[OBB](../tasks/obb.md) 和[姿态](../tasks/pose.md)。

!!! example

    === "Python"

        预训练的 `*.pt` 模型（使用 [PyTorch](https://pytorch.org/)）和配置 `*.yaml` 文件可以传递给 `YOLO()` 类以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO12n 模型
        model = YOLO("yolo12n.pt")

        # 在 COCO8 示例数据集上训练模型 100 个 epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLO12n 模型在 'bus.jpg' 图像上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        命令行界面（CLI）命令也可用：

        ```bash
        # 加载 COCO 预训练的 YOLO12n 模型并在 COCO8 示例数据集上训练 100 个 epoch
        yolo train model=yolo12n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载 COCO 预训练的 YOLO12n 模型并在 'bus.jpg' 图像上运行推理
        yolo predict model=yolo12n.pt source=path/to/bus.jpg
        ```

## 关键改进

1. **增强的[特征提取](https://www.ultralytics.com/glossary/feature-extraction)**：
    - **区域注意力**：高效处理大[感受野](https://www.ultralytics.com/glossary/receptive-field)，降低计算成本。
    - **优化平衡**：改进了注意力和前馈网络计算之间的平衡。
    - **R-ELAN**：使用 R-ELAN 架构增强特征聚合。

2. **优化创新**：
    - **残差连接**：引入带缩放的残差连接以稳定训练，特别是在较大模型中。
    - **精细化特征集成**：在 R-ELAN 中实现了改进的特征集成方法。
    - **FlashAttention**：集成 FlashAttention 以减少内存访问开销。

3. **架构效率**：
    - **减少参数**：与许多先前模型相比，在保持或提高精度的同时实现更低的参数数量。
    - **简化注意力**：使用简化的注意力实现，避免位置编码。
    - **优化 MLP 比率**：调整 MLP 比率以更有效地分配计算资源。

## 要求

Ultralytics YOLO12 实现默认_不需要_ FlashAttention。但是，FlashAttention 可以选择性地编译并与 YOLO12 一起使用。要编译 FlashAttention，需要以下 NVIDIA GPU 之一：

- [Turing GPU](<https://en.wikipedia.org/wiki/Turing_(microarchitecture)>)（例如 T4、Quadro RTX 系列）
- [Ampere GPU](<https://en.wikipedia.org/wiki/Ampere_(microarchitecture)>)（例如 RTX30 系列、A30/40/100）
- [Ada Lovelace GPU](https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/)（例如 RTX40 系列）
- [Hopper GPU](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)（例如 H100/H200）

## 引用和致谢

如果您在研究中使用 YOLO12，请引用[布法罗大学](https://www.buffalo.edu/)和[中国科学院大学](https://english.ucas.ac.cn/)的原始工作：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{tian2025yolo12,
          title={YOLO12: Attention-Centric Real-Time Object Detectors},
          author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
          journal={arXiv preprint arXiv:2502.12524},
          year={2025}
        }

        @software{yolo12,
          author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
          title = {YOLO12: Attention-Centric Real-Time Object Detectors},
          year = {2025},
          url = {https://github.com/sunsmarterjie/yolov12},
          license = {AGPL-3.0}
        }
        ```

## 常见问题

### YOLO12 如何在保持高精度的同时实现实时目标检测？

YOLO12 结合了几项关键创新来平衡速度和精度。区域[注意力机制](https://www.ultralytics.com/glossary/attention-mechanism)高效处理大感受野，与标准自注意力相比降低了计算成本。残差高效层聚合网络（R-ELAN）改进了特征聚合，解决了较大注意力中心模型中的优化挑战。优化的注意力架构，包括使用 FlashAttention 和移除位置编码，进一步提高了效率。这些特性使 YOLO12 能够实现先进的精度，同时保持对许多应用至关重要的实时推理速度。

### YOLO12 支持哪些[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务？

YOLO12 是一个多功能模型，支持广泛的核心计算机视觉任务。它在目标[检测](../tasks/detect.md)、实例[分割](../tasks/segment.md)、图像[分类](../tasks/classify.md)、[姿态估计](../tasks/pose.md)和旋转目标检测（OBB）（[查看详情](../tasks/obb.md)）方面表现出色。这种全面的任务支持使 YOLO12 成为各种应用的强大工具，从[机器人](https://www.ultralytics.com/glossary/robotics)和自动驾驶到医学成像和工业检测。每个任务都可以在推理、验证、训练和导出模式下执行。

### YOLO12 与其他 YOLO 模型和 RT-DETR 等竞争对手相比如何？

与 YOLOv10 和 YOLO11 等先前 YOLO 模型相比，YOLO12 在所有模型规模上都展示了显著的精度改进，与_最快_的先前模型相比在速度上有一些权衡。例如，YOLO12n 在 COCO val2017 数据集上比 YOLOv10n 实现了 +2.1% 的 mAP 改进，比 YOLO11n 提高了 +1.2%。与 [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) 等模型相比，YOLO12s 提供了 +1.5% 的 mAP 改进和显著的 +42% 速度提升。这些指标突显了 YOLO12 在精度和效率之间的强大平衡。有关详细对比，请参阅[性能指标部分](#性能指标)。

### 运行 YOLO12 的硬件要求是什么，特别是使用 FlashAttention？

默认情况下，Ultralytics YOLO12 实现_不_需要 FlashAttention。但是，FlashAttention 可以选择性地编译并与 YOLO12 一起使用以最小化内存访问开销。要编译 FlashAttention，需要以下 NVIDIA GPU 之一：Turing GPU（例如 T4、Quadro RTX 系列）、Ampere GPU（例如 RTX30 系列、A30/40/100）、Ada Lovelace GPU（例如 RTX40 系列）或 Hopper GPU（例如 H100/H200）。这种灵活性允许用户在硬件资源允许时利用 FlashAttention 的优势。

### 在哪里可以找到 YOLO12 的使用示例和更详细的文档？

本页面提供了基本的[使用示例](#使用示例)用于训练和推理。有关这些和其他模式的全面文档，包括[验证](../modes/val.md)和[导出](../modes/export.md)，请参阅专门的[预测](../modes/predict.md)和[训练](../modes/train.md)页面。有关任务特定信息（分割、分类、旋转目标检测和姿态估计），请参阅相应的文档：[分割](../tasks/segment.md)、[分类](../tasks/classify.md)、[OBB](../tasks/obb.md) 和[姿态](../tasks/pose.md)。这些资源提供了在各种场景中有效利用 YOLO12 的深入指导。
