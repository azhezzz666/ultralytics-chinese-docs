---
comments: true
description: 探索 YOLOv9，实时目标检测的飞跃，具有 PGI 和 GELAN 等创新技术，在效率和精度方面实现新基准。
keywords: YOLOv9, 目标检测, 实时, PGI, GELAN, 深度学习, MS COCO, AI, 神经网络, 模型效率, 精度, Ultralytics
---

# YOLOv9：[目标检测](https://www.ultralytics.com/glossary/object-detection)技术的飞跃

YOLOv9 标志着实时目标检测的重大进步，引入了可编程梯度信息 (PGI) 和通用高效层聚合网络 (GELAN) 等突破性技术。该模型在效率、精度和适应性方面展示了显著改进，在 MS COCO 数据集上设立了新基准。YOLOv9 项目虽然由独立的开源团队开发，但建立在 [Ultralytics](https://www.ultralytics.com/) [YOLOv5](yolov5.md) 提供的强大代码库之上，展示了 AI 研究社区的协作精神。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ZF7EAodHn1U"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 使用 Ultralytics 在自定义数据上训练 YOLOv9 | 工业包装数据集
</p>

![YOLOv9 性能比较](https://github.com/ultralytics/docs/releases/download/0/yolov9-performance-comparison.avif)

## YOLOv9 简介

在追求最佳实时目标检测的过程中，YOLOv9 以其创新方法脱颖而出，解决了深度[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)中固有的信息丢失挑战。通过整合 PGI 和多功能的 GELAN 架构，YOLOv9 不仅增强了模型的学习能力，还确保了在整个检测过程中保留关键信息，从而实现卓越的精度和性能。

## YOLOv9 的核心创新

YOLOv9 的进步深深植根于解决深度神经网络中信息丢失带来的挑战。信息瓶颈原理和可逆函数的创新使用是其设计的核心，确保 YOLOv9 保持高效率和精度。

### 信息瓶颈原理

信息瓶颈原理揭示了深度学习中的一个基本挑战：随着数据通过网络的连续层传递，信息丢失的可能性增加。这一现象用数学表示为：

```python
I(X, X) >= I(X, f_theta(X)) >= I(X, g_phi(f_theta(X)))
```

其中 `I` 表示互信息，`f` 和 `g` 分别表示具有参数 `theta` 和 `phi` 的变换函数。YOLOv9 通过实现可编程梯度信息 (PGI) 来应对这一挑战，该技术有助于在网络深度中保留关键数据，确保更可靠的梯度生成，从而实现更好的模型收敛和性能。

### 可逆函数

可逆函数的概念是 YOLOv9 设计的另一个基石。如果一个函数可以在不丢失任何信息的情况下被反转，则该函数被认为是可逆的，表示为：

```python
X = v_zeta(r_psi(X))
```

其中 `psi` 和 `zeta` 分别是可逆函数及其逆函数的参数。这一特性对于[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)架构至关重要，因为它允许网络保持完整的信息流，从而能够更准确地更新模型参数。YOLOv9 在其架构中融入了可逆函数，以减轻信息退化的风险，特别是在更深层中，确保保留目标检测任务所需的关键数据。

### 对轻量级模型的影响

解决信息丢失对于轻量级模型尤为重要，这些模型通常参数不足，容易在前向传播过程中丢失大量信息。YOLOv9 的架构通过使用 PGI 和可逆函数，确保即使是精简的模型也能保留并有效利用准确目标检测所需的关键信息。

### 可编程梯度信息 (PGI)

PGI 是 YOLOv9 中引入的一个新概念，用于解决信息瓶颈问题，确保在深度网络层中保留关键数据。这允许生成可靠的梯度，促进准确的模型更新并提高整体检测性能。

### 通用高效层聚合网络 (GELAN)

GELAN 代表了一种战略性的架构进步，使 YOLOv9 能够实现卓越的参数利用率和计算效率。其设计允许灵活集成各种计算块，使 YOLOv9 能够适应广泛的应用，而不会牺牲速度或精度。

![YOLOv9 架构比较](https://github.com/ultralytics/docs/releases/download/0/yolov9-architecture-comparison.avif)

## YOLOv9 基准测试

使用 [Ultralytics](https://docs.ultralytics.com/modes/benchmark/) 对 YOLOv9 进行基准测试涉及评估训练和验证模型在实际场景中的性能。此过程包括：

- **性能评估：** 评估模型的速度和精度。
- **导出格式：** 在不同导出格式中测试模型，以确保其满足必要标准并在各种环境中表现良好。
- **框架支持：** 在 Ultralytics YOLOv8 中提供全面的框架以促进这些评估，确保一致和可靠的结果。

通过基准测试，您可以确保模型不仅在受控测试环境中表现良好，而且在实际应用中也能保持高性能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ziJR01lKnio"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics Python 包对 YOLOv9 模型进行基准测试
</p>

## MS COCO 数据集上的性能

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv9"]'></canvas>

YOLOv9 在 [COCO 数据集](../datasets/detect/coco.md)上的性能体现了其在实时目标检测方面的重大进步，在各种模型尺寸上设立了新基准。表 1 展示了最先进实时目标检测器的全面比较，说明了 YOLOv9 的卓越效率和[精度](https://www.ultralytics.com/glossary/accuracy)。

!!! tip "性能"

    === "检测 (COCO)"

        | 模型                                                                                  | 尺寸<br><sup>(像素)</sup> | mAP<sup>val<br>50-95</sup> | mAP<sup>val<br>50</sup> | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        |---------------------------------------------------------------------------------------|-----------------------|----------------------|-------------------|--------------------|-------------------|
        | [YOLOv9t](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9t.pt)  | 640                   | 38.3                 | 53.1              | 2.0                | 7.7               |
        | [YOLOv9s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9s.pt)  | 640                   | 46.8                 | 63.4              | 7.2                | 26.7              |
        | [YOLOv9m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9m.pt)  | 640                   | 51.4                 | 68.1              | 20.1               | 76.8              |
        | [YOLOv9c](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c.pt)  | 640                   | 53.0                 | 70.2              | 25.5               | 102.8             |
        | [YOLOv9e](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9e.pt)  | 640                   | 55.6                 | 72.8              | 58.1               | 192.5             |

    === "分割 (COCO)"

        | 模型                                                                                          | 尺寸<br><sup>(像素)</sup> | mAP<sup>box<br>50-95</sup> | mAP<sup>mask<br>50-95</sup> | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        |-----------------------------------------------------------------------------------------------|-----------------------|----------------------|-----------------------|--------------------|-------------------|
        | [YOLOv9c-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c-seg.pt)  | 640                   | 52.4                 | 42.2                  | 27.9               | 159.4             |
        | [YOLOv9e-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9e-seg.pt)  | 640                   | 55.1                 | 44.3                  | 60.5               | 248.4             |

YOLOv9 的迭代版本，从微型 `t` 变体到大型 `e` 模型，不仅在精度（mAP 指标）方面有所改进，而且在效率方面也有所提升，参数数量和计算需求减少（FLOPs）。此表强调了 YOLOv9 在保持或减少计算开销的同时提供高[精度](https://www.ultralytics.com/glossary/precision)的能力，与之前版本和竞争模型相比。

相比之下，YOLOv9 展示了显著的提升：

- **轻量级模型**：YOLOv9s 在参数效率和计算负载方面超越了 YOLO MS-S，同时 AP 提高了 0.4∼0.6%。
- **中大型模型**：YOLOv9m 和 YOLOv9e 在平衡模型复杂性和检测性能之间的权衡方面展示了显著进步，在提高精度的同时大幅减少了参数和计算量。

YOLOv9c 模型特别突出了架构优化的有效性。它的运行参数比 YOLOv7 AF 少 42%，计算需求少 21%，但实现了相当的精度，展示了 YOLOv9 显著的效率改进。此外，YOLOv9e 模型为大型模型设立了新标准，参数比 [YOLOv8x](yolov8.md) 少 15%，计算需求少 25%，同时 AP 增量提高了 1.7%。

这些结果展示了 YOLOv9 在模型设计方面的战略进步，强调了其在不牺牲实时目标检测任务所需精度的情况下提高效率。该模型不仅推动了性能指标的边界，还强调了计算效率的重要性，使其成为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域的关键发展。

## 结论

YOLOv9 代表了实时目标检测的关键发展，在效率、精度和适应性方面提供了显著改进。通过 PGI 和 GELAN 等创新解决方案解决关键挑战，YOLOv9 为该领域的未来研究和应用设立了新先例。随着 AI 社区的不断发展，YOLOv9 证明了协作和创新在推动技术进步方面的力量。

## 使用示例

此示例提供简单的 YOLOv9 训练和推理示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参阅[预测](../modes/predict.md)、[训练](../modes/train.md)、[验证](../modes/val.md)和[导出](../modes/export.md)文档页面。

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) 预训练的 `*.pt` 模型以及配置 `*.yaml` 文件可以传递给 `YOLO()` 类以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 从头开始构建 YOLOv9c 模型
        model = YOLO("yolov9c.yaml")

        # 从预训练权重构建 YOLOv9c 模型
        model = YOLO("yolov9c.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个轮次
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLOv9c 模型对 'bus.jpg' 图像运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI 命令可直接运行模型：

        ```bash
        # 从头开始构建 YOLOv9c 模型并在 COCO8 示例数据集上训练 100 个轮次
        yolo train model=yolov9c.yaml data=coco8.yaml epochs=100 imgsz=640

        # 从头开始构建 YOLOv9c 模型并对 'bus.jpg' 图像运行推理
        yolo predict model=yolov9c.yaml source=path/to/bus.jpg
        ```

## 支持的任务和模式

YOLOv9 系列提供一系列模型，每个模型都针对高性能[目标检测](../tasks/detect.md)进行了优化。这些模型满足不同的计算需求和精度要求，使其适用于广泛的应用。

| 模型       | 文件名                                                           | 任务                                         | 推理 | 验证 | 训练 | 导出 |
| ---------- | ---------------------------------------------------------------- | -------------------------------------------- | ---- | ---- | ---- | ---- |
| YOLOv9     | `yolov9t.pt` `yolov9s.pt` `yolov9m.pt` `yolov9c.pt` `yolov9e.pt` | [目标检测](../tasks/detect.md)               | ✅   | ✅   | ✅   | ✅   |
| YOLOv9-seg | `yolov9c-seg.pt` `yolov9e-seg.pt`                                | [实例分割](../tasks/segment.md)              | ✅   | ✅   | ✅   | ✅   |

此表详细概述了 YOLOv9 模型变体，突出了它们在目标检测任务中的能力以及与各种操作模式（如[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)）的兼容性。这种全面的支持确保用户可以在广泛的目标检测场景中充分利用 YOLOv9 模型的能力。

!!! note

    训练 YOLOv9 模型将需要比同等大小的 [YOLOv8 模型](yolov8.md)_更多_的资源**和**更长的时间。

## 引用和致谢

我们要感谢 YOLOv9 作者在实时目标检测领域做出的重大贡献：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2024yolov9,
          title={YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information},
          author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
          booktitle={arXiv preprint arXiv:2402.13616},
          year={2024}
        }
        ```

原始 YOLOv9 论文可在 [arXiv](https://arxiv.org/pdf/2402.13616) 上找到。作者已公开其工作，代码库可在 [GitHub](https://github.com/WongKinYiu/yolov9) 上访问。我们感谢他们在推进该领域发展并使其工作对更广泛社区可用方面所做的努力。

## 常见问题

### YOLOv9 为实时目标检测引入了哪些创新？

YOLOv9 引入了可编程梯度信息 (PGI) 和通用高效层聚合网络 (GELAN) 等突破性技术。这些创新解决了深度神经网络中的信息丢失挑战，确保高效率、精度和适应性。PGI 在网络层中保留关键数据，而 GELAN 优化参数利用率和计算效率。了解更多关于 [YOLOv9 核心创新](#yolov9-的核心创新)的信息，这些创新在 MS COCO 数据集上设立了新基准。

### YOLOv9 在 MS COCO 数据集上与其他模型相比表现如何？

YOLOv9 通过实现更高的精度和效率超越了最先进的实时目标检测器。在 [COCO 数据集](../datasets/detect/coco.md)上，YOLOv9 模型在各种尺寸上展示了卓越的 mAP 分数，同时保持或减少了计算开销。例如，YOLOv9c 以比 YOLOv7 AF 少 42% 的参数和 21% 的计算需求实现了相当的精度。探索[性能比较](#ms-coco-数据集上的性能)获取详细指标。

### 如何使用 Python 和 CLI 训练 YOLOv9 模型？

您可以使用 Python 和 CLI 命令训练 YOLOv9 模型。对于 Python，使用 `YOLO` 类实例化模型并调用 `train` 方法：

```python
from ultralytics import YOLO

# 从预训练权重构建 YOLOv9c 模型并训练
model = YOLO("yolov9c.pt")
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

对于 CLI 训练，执行：

```bash
yolo train model=yolov9c.yaml data=coco8.yaml epochs=100 imgsz=640
```

了解更多关于训练和推理的[使用示例](#使用示例)。

### 使用 Ultralytics YOLOv9 进行轻量级模型有什么优势？

YOLOv9 旨在减轻信息丢失，这对于通常容易丢失大量信息的轻量级模型尤为重要。通过整合可编程梯度信息 (PGI) 和可逆函数，YOLOv9 确保保留关键数据，提高模型的精度和效率。这使其非常适合需要高性能紧凑模型的应用。有关更多详细信息，请探索 [YOLOv9 对轻量级模型的影响](#对轻量级模型的影响)部分。

### YOLOv9 支持哪些任务和模式？

YOLOv9 支持各种任务，包括目标检测和[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)。它与多种操作模式兼容，如推理、验证、训练和导出。这种多功能性使 YOLOv9 能够适应各种实时计算机视觉应用。有关更多信息，请参阅[支持的任务和模式](#支持的任务和模式)部分。
