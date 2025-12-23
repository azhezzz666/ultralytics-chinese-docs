---
comments: true
description: 探索 YOLO-World 模型，使用 Ultralytics YOLOv8 进步实现高效、实时的开放词汇目标检测。以最小的计算量实现顶级性能。
keywords: YOLO-World, Ultralytics, 开放词汇检测, YOLOv8, 实时目标检测, 机器学习, 计算机视觉, AI, 深度学习, 模型训练
---

# YOLO-World 模型

YOLO-World 模型引入了一种先进的、基于 [Ultralytics](https://www.ultralytics.com/) [YOLOv8](yolov8.md) 的实时开放词汇检测方法。这一创新使得能够基于描述性文本检测图像中的任何对象。通过显著降低计算需求同时保持竞争性能，YOLO-World 成为众多基于视觉应用的多功能工具。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/cfTKj96TjSE"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> YOLO World 在自定义数据集上的训练工作流
</p>

![YOLO-World 模型架构概述](https://github.com/ultralytics/docs/releases/download/0/yolo-world-model-architecture-overview.avif)

## 概述

YOLO-World 解决了传统开放词汇检测模型面临的挑战，这些模型通常依赖需要大量计算资源的笨重 [Transformer](https://www.ultralytics.com/glossary/transformer) 模型。这些模型对预定义对象类别的依赖也限制了它们在动态场景中的实用性。YOLO-World 通过开放词汇检测功能振兴了 YOLOv8 框架，采用视觉-[语言建模](https://www.ultralytics.com/glossary/language-modeling)并在大规模数据集上进行预训练，以在零样本场景中以无与伦比的效率识别广泛的对象。

## 关键特性

1. **实时解决方案：** 利用 CNN 的计算速度，YOLO-World 提供快速的开放词汇检测解决方案，满足需要即时结果的行业需求。

2. **效率和性能：** YOLO-World 在不牺牲性能的情况下大幅削减计算和资源需求，提供了 SAM 等模型的强大替代方案，但计算成本仅为其一小部分，支持实时应用。

3. **离线词汇推理：** YOLO-World 引入了"提示后检测"策略，采用离线词汇进一步提高效率。这种方法允许使用预先计算的自定义提示（包括标题或类别）进行编码并存储为离线词汇嵌入，简化检测过程。

4. **由 YOLOv8 驱动：** 基于 [Ultralytics YOLOv8](yolov8.md) 构建，YOLO-World 利用实时目标检测的最新进展，以无与伦比的精度和速度促进开放词汇检测。

5. **基准卓越：** YOLO-World 在标准基准测试中在速度和效率方面优于现有的开放词汇检测器，包括 MDETR 和 GLIP 系列，展示了 YOLOv8 在单个 NVIDIA V100 GPU 上的卓越能力。

6. **多功能应用：** YOLO-World 的创新方法为众多视觉任务开辟了新的可能性，相比现有方法提供了数量级的速度提升。

## 可用模型、支持的任务和操作模式

本节详细介绍了具有特定预训练权重的可用模型、它们支持的任务，以及与各种操作模式的兼容性，如[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)，用 ✅ 表示支持的模式，用 ❌ 表示不支持的模式。

!!! note

    所有 YOLOv8-World 权重都直接从官方 [YOLO-World](https://github.com/AILab-CVC/YOLO-World) 仓库迁移，突显了他们的出色贡献。

| 模型类型        | 预训练权重                                                                                              | 支持的任务                             | 推理 | 验证 | 训练 | 导出 |
| --------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| YOLOv8s-world   | [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt)     | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ❌   |
| YOLOv8s-worldv2 | [yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt) | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ✅   |
| YOLOv8m-world   | [yolov8m-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-world.pt)     | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ❌   |
| YOLOv8m-worldv2 | [yolov8m-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-worldv2.pt) | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ✅   |
| YOLOv8l-world   | [yolov8l-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt)     | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ❌   |
| YOLOv8l-worldv2 | [yolov8l-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt) | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ✅   |
| YOLOv8x-world   | [yolov8x-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-world.pt)     | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ❌   |
| YOLOv8x-worldv2 | [yolov8x-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt) | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ✅   |

## COCO 数据集上的零样本迁移

!!! tip "性能"

    === "检测 (COCO)"

        | 模型类型        | mAP  | mAP50 | mAP75 |
        | --------------- | ---- | ----- | ----- |
        | yolov8s-world   | 37.4 | 52.0  | 40.6  |
        | yolov8s-worldv2 | 37.7 | 52.2  | 41.0  |
        | yolov8m-world   | 42.0 | 57.0  | 45.6  |
        | yolov8m-worldv2 | 43.0 | 58.4  | 46.8  |
        | yolov8l-world   | 45.7 | 61.3  | 49.8  |
        | yolov8l-worldv2 | 45.8 | 61.3  | 49.8  |
        | yolov8x-world   | 47.0 | 63.0  | 51.2  |
        | yolov8x-worldv2 | 47.1 | 62.8  | 51.4  |


## 使用示例

YOLO-World 模型易于集成到您的 Python 应用程序中。Ultralytics 提供用户友好的 [Python API](../usage/python.md) 和 [CLI 命令](../usage/cli.md)以简化开发。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/sWEm3dIGKU8"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> YOLO-World 模型使用示例与 Ultralytics | 开放词汇、无提示等 🚀
</p>

### 训练用法

!!! tip

    我们强烈建议使用 `yolov8-worldv2` 模型进行自定义训练，因为它支持确定性训练，并且易于导出其他格式，如 onnx/tensorrt。

使用 `train` 方法进行[目标检测](https://www.ultralytics.com/glossary/object-detection)非常简单，如下所示：

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) 预训练的 `*.pt` 模型以及配置 `*.yaml` 文件可以传递给 `YOLOWorld()` 类以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLOWorld

        # 加载预训练的 YOLOv8s-worldv2 模型
        model = YOLOWorld("yolov8s-worldv2.pt")

        # 在 COCO8 示例数据集上训练模型 100 个 epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLO-World 模型在 'bus.jpg' 图像上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 加载预训练的 YOLOv8s-worldv2 模型并在 COCO8 示例数据集上训练 100 个 epoch
        yolo train model=yolov8s-worldv2.yaml data=coco8.yaml epochs=100 imgsz=640
        ```

### 预测用法

使用 `predict` 方法进行目标检测非常简单，如下所示：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOWorld

        # 初始化 YOLO-World 模型
        model = YOLOWorld("yolov8s-world.pt")  # 或选择 yolov8m/l-world.pt 获取不同大小

        # 使用 YOLOv8s-world 模型在指定图像上执行推理
        results = model.predict("path/to/image.jpg")

        # 显示结果
        results[0].show()
        ```

    === "CLI"

        ```bash
        # 使用 YOLO-World 模型执行目标检测
        yolo predict model=yolov8s-world.pt source=path/to/image.jpg imgsz=640
        ```

此代码片段演示了加载预训练模型并在图像上运行预测的简单性。

### 验证用法

在数据集上进行模型验证的流程如下：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 创建 YOLO-World 模型
        model = YOLO("yolov8s-world.pt")  # 或选择 yolov8m/l-world.pt 获取不同大小

        # 在 COCO8 示例数据集上进行模型验证
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # 在 COCO8 数据集上验证 YOLO-World 模型，指定图像大小
        yolo val model=yolov8s-world.pt data=coco8.yaml imgsz=640
        ```

### 跟踪用法

使用 YOLO-World 模型在视频/图像上进行[对象跟踪](https://www.ultralytics.com/glossary/object-tracking)的流程如下：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 创建 YOLO-World 模型
        model = YOLO("yolov8s-world.pt")  # 或选择 yolov8m/l-world.pt 获取不同大小

        # 使用 YOLO-World 模型在视频上进行跟踪
        results = model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # 使用 YOLO-World 模型在视频上进行跟踪，指定图像大小
        yolo track model=yolov8s-world.pt imgsz=640 source="path/to/video.mp4"
        ```

!!! note

    Ultralytics 提供的 YOLO-World 模型预配置了 [COCO 数据集](../datasets/detect/coco.md)类别作为其离线词汇的一部分，提高了即时应用的效率。这种集成允许 YOLOv8-World 模型直接识别和预测 COCO 数据集中定义的 80 个标准类别，无需额外设置或自定义。

### 设置提示

![YOLO-World 提示类名概述](https://github.com/ultralytics/docs/releases/download/0/yolo-world-prompt-class-names-overview.avif)

YOLO-World 框架允许通过自定义提示动态指定类别，使用户能够根据特定需求定制模型**而无需重新训练**。此功能对于将模型适应到原本不属于[训练数据](https://www.ultralytics.com/glossary/training-data)的新领域或特定任务特别有用。通过设置自定义提示，用户可以基本上引导模型关注感兴趣的对象，提高检测结果的相关性和[精度](https://www.ultralytics.com/glossary/accuracy)。

例如，如果您的应用程序只需要检测"人"和"公交车"对象，您可以直接指定这些类别：

!!! example

    === "自定义推理提示"

        ```python
        from ultralytics import YOLO

        # 初始化 YOLO-World 模型
        model = YOLO("yolov8s-world.pt")  # 或选择 yolov8m/l-world.pt

        # 定义自定义类别
        model.set_classes(["person", "bus"])

        # 在图像上执行指定类别的预测
        results = model.predict("path/to/image.jpg")

        # 显示结果
        results[0].show()
        ```

您还可以在设置自定义类别后保存模型。通过这样做，您可以创建一个专门针对您特定用例的 YOLO-World 模型版本。此过程将您的自定义类别定义直接嵌入到模型文件中，使模型可以直接使用您指定的类别，无需进一步调整。按照以下步骤保存和加载您的自定义 YOLOv8 模型：

!!! example

    === "使用自定义词汇持久化模型"

        首先加载 YOLO-World 模型，为其设置自定义类别并保存：

        ```python
        from ultralytics import YOLO

        # 初始化 YOLO-World 模型
        model = YOLO("yolov8s-world.pt")  # 或选择 yolov8m/l-world.pt

        # 定义自定义类别
        model.set_classes(["person", "bus"])

        # 使用定义的离线词汇保存模型
        model.save("custom_yolov8s.pt")
        ```

        保存后，custom_yolov8s.pt 模型的行为与任何其他预训练 YOLOv8 模型相同，但有一个关键区别：它现在已优化为仅检测您定义的类别。这种自定义可以显著提高您特定应用场景的检测性能和效率。

        ```python
        from ultralytics import YOLO

        # 加载您的自定义模型
        model = YOLO("custom_yolov8s.pt")

        # 运行推理以检测您的自定义类别
        results = model.predict("path/to/image.jpg")

        # 显示结果
        results[0].show()
        ```

### 使用自定义词汇保存的好处

- **效率**：通过专注于相关对象来简化检测过程，减少计算开销并加速推理。
- **灵活性**：允许轻松将模型适应新的或小众检测任务，无需大量重新训练或数据收集。
- **简单性**：通过消除在运行时重复指定自定义类别的需要来简化部署，使模型可以直接使用其嵌入的词汇。
- **性能**：通过将模型的注意力和资源集中在识别定义的对象上，提高指定类别的检测精度。

这种方法提供了一种强大的手段，可以为特定任务自定义最先进的[目标检测](../tasks/detect.md)模型，使高级 AI 更易于访问并适用于更广泛的实际应用。
