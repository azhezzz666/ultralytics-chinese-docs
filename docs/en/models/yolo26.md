---
comments: true
description: Ultralytics YOLO26 提供更快、更简单的端到端无 NMS 目标检测，专为边缘和低功耗设备优化。
keywords: YOLO26, Ultralytics YOLO, 目标检测, 端到端无 NMS, 简化架构, 计算机视觉, AI, 机器学习, 边缘 AI, 低功耗设备, 量化, 实时推理
---

# Ultralytics YOLO26

!!! note "即将推出 ⚠️"

    🚧 YOLO26 模型仍在开发中，尚未发布。此处显示的性能数据**仅为预览**。
    最终下载和发布即将推出 — 请通过 [YOLO Vision 2025](https://www.ultralytics.com/events/yolovision) 获取最新动态。

## 概述

[Ultralytics](https://www.ultralytics.com/) YOLO26 是 YOLO 系列实时目标检测器的最新演进，从头开始为**边缘和低功耗设备**设计。它引入了精简设计，移除了不必要的复杂性，同时集成了针对性的创新，以实现更快、更轻量、更易于部署的目标。

YOLO26 的架构遵循三个核心原则：

- **简洁性：** YOLO26 是一个**原生端到端模型**，直接生成预测结果，无需非极大值抑制（NMS）。通过消除这个后处理步骤，推理变得更快、更轻量，更易于在实际系统中部署。这种突破性方法最初由清华大学的 Ao Wang 在 [YOLOv10](../models/yolov10.md) 中首创，并在 YOLO26 中得到进一步发展。
- **部署效率：** 端到端设计省去了整个流程阶段，大大简化了集成，降低了延迟，使部署在各种环境中更加稳健。
- **训练创新：** YOLO26 引入了 **MuSGD 优化器**，这是 [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) 和 [Muon](https://arxiv.org/abs/2502.16982) 的混合体 — 灵感来自 Moonshot AI 的 [Kimi K2](https://www.kimi.com/) 在大语言模型训练方面的突破。该优化器带来了增强的稳定性和更快的收敛速度，将语言模型的优化进展转移到计算机视觉领域。

这些创新共同打造了一个模型系列，在小目标上实现更高精度，提供无缝部署，并且**在 CPU 上运行速度提高多达 43%** — 使 YOLO26 成为迄今为止资源受限环境中最实用、最易部署的 YOLO 模型之一。

![Ultralytics YOLO26 对比图](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo-comparison-plot.png)

## 关键特性

- **移除 DFL**  
  分布焦点损失（DFL）模块虽然有效，但通常会使导出复杂化并限制硬件兼容性。YOLO26 完全移除了 DFL，简化了推理并扩大了对**边缘和低功耗设备**的支持。

- **端到端无 NMS 推理**  
  与依赖 NMS 作为单独后处理步骤的传统检测器不同，YOLO26 是**原生端到端**的。预测直接生成，减少延迟，使集成到生产系统更快、更轻量、更可靠。

- **ProgLoss + STAL**  
  改进的损失函数提高了检测精度，在**小目标识别**方面有显著改进，这是物联网、机器人、航空图像和其他边缘应用的关键需求。

- **MuSGD 优化器**  
  一种新的混合优化器，结合了 [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) 和 [Muon](https://arxiv.org/abs/2502.16982)。受 Moonshot AI 的 [Kimi K2](https://www.kimi.com/) 启发，MuSGD 将大语言模型训练的先进优化方法引入计算机视觉，实现更稳定的训练和更快的收敛。

- **CPU 推理速度提高多达 43%**  
  专门针对边缘计算优化，YOLO26 提供显著更快的 CPU 推理，确保在没有 GPU 的设备上实现实时性能。

---

## 支持的任务和模式

YOLO26 设计为**多任务模型系列**，将 YOLO 的多功能性扩展到各种计算机视觉挑战：

| 模型        | 任务                                   | 推理 | 验证 | 训练 | 导出 |
| ----------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| YOLO26      | [检测](../tasks/detect.md)             | ✅   | ✅   | ✅   | ✅   |
| YOLO26-seg  | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLO26-pose | [姿态/关键点](../tasks/pose.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLO26-obb  | [旋转检测](../tasks/obb.md)            | ✅   | ✅   | ✅   | ✅   |
| YOLO26-cls  | [分类](../tasks/classify.md)           | ✅   | ✅   | ✅   | ✅   |

这个统一框架确保 YOLO26 适用于实时检测、分割、分类、姿态估计和旋转目标检测 — 所有任务都支持训练、验证、推理和导出。

---

## 性能指标

!!! tip "性能预览"

    以下基准测试是**早期预览**。最终数据和可下载权重将在训练完成后发布。

    === "检测 (COCO)"

        在 [COCO](../datasets/detect/coco.md) 上训练，包含 80 个预训练类别。
        模型发布后请参阅[检测文档](../tasks/detect.md)了解使用方法。

        | 模型    | 尺寸<br><sup>(像素)</sup> | mAP<sup>val<br>50-95(e2e)</sup> | mAP<sup>val<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>T4 TensorRT10<br>(ms)</sup> | 参数<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------- | ------------------------- | ------------------------------- | -------------------------- | ----------------------------------- | ---------------------------------------- | ---------------------- | ----------------------- |
        | YOLO26n | 640                       | 39.8                            | 40.3                       | 38.90 ± 0.7                         | 1.7 ± 0.0                                | 2.4                    | 5.4                     |
        | YOLO26s | 640                       | 47.2                            | 47.6                       | 87.16 ± 0.9                         | 2.7 ± 0.0                                | 9.5                    | 20.7                    |
        | YOLO26m | 640                       | 51.5                            | 51.7                       | 220.0 ± 1.4                         | 4.9 ± 0.1                                | 20.4                   | 68.2                    |
        | YOLO26l | 640                       | 53.0*                           | 53.4*                      | 286.17 ± 2.0*                       | 6.5 ± 0.2*                               | 24.8                   | 86.4                    |
        | YOLO26x | 640                       | -                               | -                          | -                                   | -                                        | -                      | -                       |

        *YOLO26l 和 YOLO26x 的指标正在进行中。最终基准测试将在此处添加。

    === "分割 (COCO)"

        性能指标即将推出。

    === "分类 (ImageNet)"

        性能指标即将推出。

    === "姿态 (COCO)"

        性能指标即将推出。

    === "OBB (DOTAv1)"

        性能指标即将推出。

---

## 引用和致谢

!!! tip "Ultralytics YOLO26 出版物"

    由于模型快速发展的特性，Ultralytics 尚未发布 YOLO26 的正式研究论文。相反，我们专注于提供尖端模型并使其易于使用。有关 YOLO 特性、架构和使用的最新更新，请访问我们的 [GitHub 仓库](https://github.com/ultralytics/ultralytics)和[文档](https://docs.ultralytics.com/)。

如果您在工作中使用 YOLO26 或其他 Ultralytics 软件，请按以下方式引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yolo26_ultralytics,
          author = {Glenn Jocher and Jing Qiu},
          title = {Ultralytics YOLO26},
          version = {26.0.0},
          year = {2025},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

DOI 正在申请中。YOLO26 根据 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 和[企业](https://www.ultralytics.com/license)许可证提供。

---

## 常见问题

### 与 YOLO11 相比，YOLO26 有哪些关键改进？

- **移除 DFL**：简化导出并扩展边缘兼容性
- **端到端无 NMS 推理**：消除 NMS 以实现更快、更简单的部署
- **ProgLoss + STAL**：提高精度，特别是在小目标上
- **MuSGD 优化器**：结合 SGD 和 Muon（受 Moonshot 的 Kimi K2 启发）以实现更稳定、更高效的训练
- **CPU 推理速度提高多达 43%**：仅 CPU 设备的重大性能提升

### YOLO26 将支持哪些任务？

YOLO26 设计为**统一模型系列**，为多种计算机视觉任务提供端到端支持：

- [目标检测](../tasks/detect.md)
- [实例分割](../tasks/segment.md)
- [图像分类](../tasks/classify.md)
- [姿态估计](../tasks/pose.md)
- [旋转目标检测（OBB）](../tasks/obb.md)

计划在发布时每个尺寸变体（n、s、m、l、x）都支持所有任务。

### 为什么 YOLO26 针对边缘部署进行了优化？

YOLO26 提供**先进的边缘性能**：

- CPU 推理速度提高多达 43%
- 减少模型大小和内存占用
- 简化架构以提高兼容性（无 DFL、无 NMS）
- 灵活的导出格式，包括 TensorRT、ONNX、CoreML、TFLite 和 OpenVINO

### YOLO26 模型何时可用？

YOLO26 模型仍在训练中，尚未开源。此处显示的是性能预览，官方下载和发布计划在不久的将来。
请参阅 [YOLO Vision 2025](https://www.ultralytics.com/events/yolovision) 了解 YOLO26 相关演讲。
