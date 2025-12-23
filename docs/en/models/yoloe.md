---
comments: true
description: YOLOE 是一个实时开放词汇检测和分割模型，通过文本、图像或内部词汇提示扩展 YOLO，能够以先进的零样本性能检测任何对象类别。
keywords: YOLOE, 开放词汇检测, 实时目标检测, 实例分割, YOLO, 文本提示, 视觉提示, 零样本检测
---

# YOLOE：实时万物感知

## 简介

![YOLOE 提示选项](https://github.com/ultralytics/docs/releases/download/0/yoloe-visualization.avif)

[YOLOE（实时万物感知）](https://arxiv.org/html/2503.07465v1)是零样本、可提示 YOLO 模型的新进展，专为**开放词汇**检测和分割设计。与之前仅限于固定类别的 YOLO 模型不同，YOLOE 使用文本、图像或内部词汇提示，实现任何对象类别的实时检测。基于 YOLOv10 构建并受 [YOLO-World](yolo-world.md) 启发，YOLOE 以对速度和精度的最小影响实现了**先进的零样本性能**。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/HMOoM2NwFIQ"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics Python 包使用 YOLOE：开放词汇和实时万物感知 🚀
</p>

与早期 YOLO 模型相比，YOLOE 显著提高了效率和精度。它在 LVIS 上比 YOLO-Worldv2 提高了 **+3.5 AP**，同时仅使用三分之一的训练资源并实现 1.4 倍更快的推理速度。在 COCO 上微调后，YOLOE-v8-large 以近乎 **4 倍更少的训练时间**超越 YOLOv8-L **0.1 mAP**。这展示了 YOLOE 在精度、效率和多功能性之间的卓越平衡。以下部分探讨 YOLOE 的架构、基准对比以及与 [Ultralytics](https://www.ultralytics.com/) 框架的集成。

## 架构概述

<p align="center">
  <img src="https://github.com/THU-MIG/yoloe/raw/main/figures/pipeline.svg" alt="YOLOE 架构" width=90%>
</p>

YOLOE 保留了标准 YOLO 结构——用于特征提取的卷积**骨干网络**（如 CSP-Darknet）、用于多尺度融合的**颈部**（如 PAN-FPN），以及**无锚点、解耦**的检测**头部**（如 YOLOv8/YOLO11），独立预测目标性、类别和边界框。YOLOE 引入了三个新模块实现开放词汇检测：

- **可重参数化区域-文本对齐（RepRTA）**：通过小型辅助网络细化文本[嵌入](https://www.ultralytics.com/glossary/embeddings)（如来自 CLIP），支持**文本提示检测**。在推理时，该网络被折叠到主模型中，确保零开销。因此 YOLOE 可以检测任意文本标记的对象（如未见过的"交通灯"）而无运行时惩罚。

- **语义激活视觉提示编码器（SAVPE）**：通过轻量级嵌入分支实现**视觉提示检测**。给定参考图像，SAVPE 编码语义和激活特征，使模型能够检测视觉相似的对象——这是一种单样本检测能力，适用于标志或特定部件。

- **惰性区域-提示对比（LRPC）**：在**无提示模式**下，YOLOE 使用在大词汇表（来自 LVIS 和 Objects365 的 1200+ 类别）上训练的内部嵌入执行开放集识别。无需外部提示或编码器，YOLOE 通过嵌入相似性查找识别对象，在推理时高效处理大标签空间。

此外，YOLOE 通过扩展检测头部添加掩码预测分支（类似于 YOLACT 或 YOLOv8-Seg）集成实时**实例分割**，增加最小开销。

关键的是，当作为常规闭集 YOLO 使用时，YOLOE 的开放世界模块**不引入推理成本**。训练后，YOLOE 参数可以重参数化为标准 YOLO 头部，保持相同的 FLOPs 和速度（例如，与 [YOLO11](yolo11.md) 完全匹配）。

## 可用模型、支持的任务和操作模式

本节详细介绍了具有特定预训练权重的可用模型、它们支持的任务，以及与各种操作模式的兼容性，如[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)，用 ✅ 表示支持的模式，用 ❌ 表示不支持的模式。

### 文本/视觉提示模型

| 模型类型   | 预训练权重                                                                                          | 支持的任务                             | 推理 | 验证 | 训练 | 导出 |
| ---------- | --------------------------------------------------------------------------------------------------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| YOLOE-11S  | [yoloe-11s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-11M  | [yoloe-11m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11m-seg.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-11L  | [yoloe-11l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-v8S  | [yoloe-v8s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8s-seg.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-v8M  | [yoloe-v8m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8m-seg.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-v8L  | [yoloe-v8l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |

### 无提示模型

| 模型类型     | 预训练权重                                                                                              | 支持的任务                             | 推理 | 验证 | 训练 | 导出 |
| ------------ | ------------------------------------------------------------------------------------------------------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| YOLOE-11S-PF | [yoloe-11s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-11M-PF | [yoloe-11m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11m-seg-pf.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-11L-PF | [yoloe-11l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg-pf.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-v8S-PF | [yoloe-v8s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8s-seg-pf.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-v8M-PF | [yoloe-v8m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8m-seg-pf.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |
| YOLOE-v8L-PF | [yoloe-v8l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg-pf.pt) | [实例分割](../tasks/segment.md)        | ✅   | ✅   | ✅   | ✅   |


## 使用示例

YOLOE 模型易于集成到您的 Python 应用程序中。Ultralytics 提供用户友好的 [Python API](../usage/python.md) 和 [CLI 命令](../usage/cli.md)以简化开发。

### 训练用法

#### 在自定义数据集上微调

您可以在自定义 YOLO 数据集上微调任何[预训练 YOLOE 模型](#文本视觉提示模型)，用于检测和实例分割任务。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/vnn90bEyk0w"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何在汽车零件分割数据集上训练 YOLOE | 开放词汇模型、预测和导出 🚀
</p>

!!! example

    === "微调"

        **实例分割**

        微调 YOLOE 预训练检查点主要遵循[标准 YOLO 训练流程](../modes/train.md)。关键区别是在 `model.train()` 中显式传递 `YOLOEPESegTrainer` 作为 `trainer` 参数：

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

        model = YOLOE("yoloe-11s-seg.pt")

        # 在您的分割数据集上微调
        results = model.train(
            data="coco128-seg.yaml",  # 分割数据集
            epochs=80,
            patience=10,
            trainer=YOLOEPESegTrainer,  # <- 重要：使用分割训练器
        )
        ```

        **目标检测**

        所有[预训练 YOLOE 模型](#文本视觉提示模型)默认执行实例分割。要使用这些预训练检查点训练检测模型，请使用 YAML 配置从头初始化检测模型，然后加载相同规模的预训练分割检查点。注意我们使用 `YOLOEPETrainer` 而不是 `YOLOEPESegTrainer`，因为我们正在训练检测模型：

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPETrainer

        # 从配置初始化检测模型
        model = YOLOE("yoloe-11s.yaml")

        # 从预训练分割检查点加载权重（相同规模）
        model.load("yoloe-11s-seg.pt")

        # 在您的检测数据集上微调
        results = model.train(
            data="coco128.yaml",  # 检测数据集
            epochs=80,
            patience=10,
            trainer=YOLOEPETrainer,  # <- 重要：使用检测训练器
        )
        ```

### 预测用法

YOLOE 支持基于文本和视觉的提示。使用提示很简单——只需通过 `predict` 方法传递它们，如下所示：

!!! example

    === "文本提示"

        文本提示允许您通过文本描述指定要检测的类别。以下代码展示了如何使用 YOLOE 检测图像中的人和公交车：

        ```python
        from ultralytics import YOLOE

        # 初始化 YOLOE 模型
        model = YOLOE("yoloe-11l-seg.pt")  # 或选择 yoloe-11s/m-seg.pt 获取不同大小

        # 设置文本提示以检测人和公交车。加载模型后只需执行一次。
        names = ["person", "bus"]
        model.set_classes(names, model.get_text_pe(names))

        # 在给定图像上运行检测
        results = model.predict("path/to/image.jpg")

        # 显示结果
        results[0].show()
        ```

    === "视觉提示"

        视觉提示允许您通过向模型展示目标类别的视觉示例来引导模型，而不是用文本描述它们。

        `visual_prompts` 参数接受一个包含两个键的字典：`bboxes` 和 `cls`。`bboxes` 中的每个边界框应紧密包围您希望模型检测的对象示例，`cls` 中的相应条目指定该框的类别标签。

        ```python
        import numpy as np

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        # 初始化 YOLOE 模型
        model = YOLOE("yoloe-11l-seg.pt")

        # 使用边界框及其对应的类别 ID 定义视觉提示
        visual_prompts = dict(
            bboxes=np.array(
                [
                    [221.52, 405.8, 344.98, 857.54],  # 包围人的框
                    [120, 425, 160, 445],  # 包围眼镜的框
                ],
            ),
            cls=np.array(
                [
                    0,  # 分配给人的 ID
                    1,  # 分配给眼镜的 ID
                ]
            ),
        )

        # 在图像上运行推理，使用提供的视觉提示作为指导
        results = model.predict(
            "ultralytics/assets/bus.jpg",
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )

        # 显示结果
        results[0].show()
        ```

    === "无提示"

        YOLOE 还包括无提示变体，带有内置词汇表。这些模型不需要任何提示，像传统 YOLO 模型一样工作。它们基于 [Recognize Anything Model Plus (RAM++)](https://arxiv.org/abs/2310.15200) 使用的标签集，从[预定义的 4,585 个类别列表](https://github.com/xinyu1205/recognize-anything/blob/main/ram/data/ram_tag_list.txt)中检测对象。

        ```python
        from ultralytics import YOLOE

        # 初始化 YOLOE 模型
        model = YOLOE("yoloe-11l-seg-pf.pt")

        # 运行预测。无需提示。
        results = model.predict("path/to/image.jpg")

        # 显示结果
        results[0].show()
        ```

### 验证用法

在数据集上进行模型验证的流程如下：

!!! example

    === "文本提示"

        ```python
        from ultralytics import YOLOE

        # 创建 YOLOE 模型
        model = YOLOE("yoloe-11l-seg.pt")  # 或选择 yoloe-11s/m-seg.pt 获取不同大小

        # 在 COCO128-seg 示例数据集上进行模型验证
        metrics = model.val(data="coco128-seg.yaml")
        ```

    === "视觉提示"

        默认使用提供的数据集为每个类别提取视觉嵌入。

        ```python
        from ultralytics import YOLOE

        # 创建 YOLOE 模型
        model = YOLOE("yoloe-11l-seg.pt")  # 或选择 yoloe-11s/m-seg.pt 获取不同大小

        # 在 COCO128-seg 示例数据集上进行模型验证
        metrics = model.val(data="coco128-seg.yaml", load_vp=True)
        ```

    === "无提示"

        ```python
        from ultralytics import YOLOE

        # 创建 YOLOE 模型
        model = YOLOE("yoloe-11l-seg-pf.pt")  # 或选择 yoloe-11s/m-seg-pf.pt 获取不同大小

        # 在 COCO128-seg 示例数据集上进行模型验证
        metrics = model.val(data="coco128-seg.yaml", single_cls=True)
        ```

### 导出用法

导出过程与其他 YOLO 模型类似，增加了处理文本和视觉提示的灵活性：

!!! example

    ```python
    from ultralytics import YOLOE

    # 选择 yoloe-11s/m-seg.pt 获取不同大小
    model = YOLOE("yoloe-11l-seg.pt")

    # 在导出模型之前配置 set_classes()
    names = ["person", "bus"]
    model.set_classes(names, model.get_text_pe(names))

    export_model = model.export(format="onnx")
    model = YOLOE(export_model)

    # 在给定图像上运行检测
    results = model.predict("path/to/image.jpg")

    # 显示结果
    results[0].show()
    ```

## YOLOE 性能对比

YOLOE 在 COCO 等标准基准测试中匹配或超越闭集 YOLO 模型的精度，同时不影响速度或模型大小。下表将 YOLOE-L（基于 YOLO11 构建）与相应的 [YOLOv8](yolov8.md) 和 YOLO11 模型进行比较：

| 模型                      | COCO mAP<sub>50-95</sub> | 推理速度 (T4)         | 参数     | GFLOPs (640px)     |
| ------------------------- | ------------------------ | --------------------- | -------- | ------------------ |
| **YOLOv8-L**（闭集）      | 52.9%                    | **9.06 ms** (110 FPS) | 43.7 M   | 165.2 B            |
| **YOLO11-L**（闭集）      | 53.5%                    | **6.2 ms** (130 FPS)  | 26.2 M   | 86.9 B             |
| **YOLOE-L**（开放词汇）   | 52.6%                    | **6.2 ms** (130 FPS)  | 26.2 M   | 86.9 B<sup>†</sup> |

<sup>†</sup> _YOLO11-L 和 YOLOE-L 具有相同的架构（YOLO11-L 中禁用提示模块），导致相同的推理速度和相似的 GFLOPs 估计。_

YOLOE-L 实现了 **52.6% mAP**，以大约 **40% 更少的参数**（26M vs. 43.7M）超越 YOLOv8-L（**52.9%**）。它以 **6.2 ms（161 FPS）**处理 640×640 图像，而 YOLOv8-L 为 **9.06 ms（110 FPS）**，突显了 YOLO11 的效率。关键的是，YOLOE 的开放词汇模块**不产生推理成本**，展示了**"无免费午餐权衡"**设计。

## 与之前模型的比较

YOLOE 相比之前的 YOLO 模型和开放词汇检测器引入了显著进步：

- **YOLOE vs YOLOv5：**
  [YOLOv5](yolov5.md) 提供了良好的速度-精度平衡，但需要为新类别重新训练并使用基于锚点的头部。相比之下，YOLOE 是**无锚点**的，可以动态检测新类别。

- **YOLOE vs YOLOv8：**
  YOLOE 扩展了 [YOLOv8](yolov8.md) 重新设计的架构，实现了相似或更优的精度。关键进步是 YOLOE 的**开放世界能力**，通过提示检测未见过的对象。

- **YOLOE vs YOLO11：**
  [YOLO11](yolo11.md) 在 YOLOv8 基础上提高了效率并减少了参数。YOLOE 直接继承了这些优势，匹配 YOLO11 的推理速度和参数数量，同时添加了**开放词汇检测和分割**。

- **YOLOE vs 之前的开放词汇检测器：**
  早期的开放词汇模型（GLIP、OWL-ViT、[YOLO-World](yolo-world.md)）严重依赖视觉-语言 [Transformer](https://www.ultralytics.com/glossary/transformer)，导致推理缓慢。YOLOE 在零样本精度上超越这些模型（例如，比 YOLO-Worldv2 **+3.5 AP**），同时运行速度 **1.4 倍更快**，训练资源显著更少。

## 用例和应用

YOLOE 的开放词汇检测和分割支持超越传统固定类别模型的多种应用：

- **开放世界目标检测：** 适用于动态场景，如[机器人](https://www.ultralytics.com/blog/understanding-the-integration-of-computer-vision-in-robotics)，机器人使用提示识别以前未见过的对象。
- **少样本和单样本检测：** 使用视觉提示（SAVPE），YOLOE 可以从单个参考图像快速学习新对象。
- **大词汇表和长尾识别：** 配备 1000+ 类别的词汇表，YOLOE 在生物多样性监测、零售库存等任务中表现出色。
- **交互式检测和分割：** YOLOE 支持实时交互应用，如可搜索的视频/图像检索、增强现实（AR）和直观的图像编辑。
- **任意对象分割：** 通过提示将分割能力扩展到任意对象——特别适用于医学成像、显微镜或卫星图像分析。

## 引用和致谢

如果 YOLOE 对您的研究或项目有所贡献，请引用**清华大学**的 **Ao Wang、Lihao Liu、Hui Chen、Zijia Lin、Jungong Han 和 Guiguang Ding** 的原始论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{wang2025yoloerealtimeseeing,
              title={YOLOE: Real-Time Seeing Anything},
              author={Ao Wang and Lihao Liu and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
              year={2025},
              eprint={2503.07465},
              archivePrefix={arXiv},
              primaryClass={cs.CV},
              url={https://arxiv.org/abs/2503.07465},
        }
        ```

## 常见问题

### YOLOE 与 YOLO-World 有何不同？

虽然 YOLOE 和 [YOLO-World](yolo-world.md) 都支持开放词汇检测，但 YOLOE 提供了几个优势。YOLOE 在 LVIS 上实现了 +3.5 AP 更高的精度，同时使用 3 倍更少的训练资源，运行速度比 YOLO-Worldv2 快 1.4 倍。YOLOE 还支持三种提示模式（文本、视觉和内部词汇），而 YOLO-World 主要专注于文本提示。此外，YOLOE 包含内置的[实例分割](https://www.ultralytics.com/blog/what-is-instance-segmentation-a-quick-guide)功能。

### 我可以将 YOLOE 用作常规 YOLO 模型吗？

是的，YOLOE 可以像标准 YOLO 模型一样工作，没有性能损失。在闭集模式下使用时（不使用提示），YOLOE 的开放词汇模块被重参数化为标准检测头部，导致与等效 YOLO11 模型相同的速度和精度。

### YOLOE 可以使用哪些类型的提示？

YOLOE 支持三种类型的提示：

1. **文本提示**：使用自然语言指定对象类别（例如，"person"、"traffic light"）
2. **视觉提示**：提供您想要检测的对象的参考图像
3. **内部词汇**：使用 YOLOE 内置的 1200+ 类别词汇表，无需外部提示

### YOLOE 如何处理实例分割？

YOLOE 通过扩展检测头部添加掩码预测分支，将实例分割直接集成到其架构中。分割掩码自动包含在推理结果中，可以通过 `results[0].masks` 访问。
