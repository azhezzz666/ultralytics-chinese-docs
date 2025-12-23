---
comments: true
description: 探索 YOLO 命令行界面（CLI），无需 Python 环境即可轻松执行检测任务。
keywords: YOLO CLI, 命令行界面, YOLO 命令, 检测任务, Ultralytics, 模型训练, 模型预测
---

# 命令行界面

Ultralytics 命令行界面（CLI）提供了一种简单的方式来使用 Ultralytics YOLO 模型，无需 Python 环境。CLI 支持使用 `yolo` 命令直接从终端运行各种任务，无需自定义或 Python 代码。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=19"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>掌握 Ultralytics YOLO：CLI
</p>

!!! example "示例"

    === "语法"

        Ultralytics `yolo` 命令使用以下语法：
        ```bash
        yolo TASK MODE ARGS
        ```

        其中：
        - `TASK`（可选）是 [detect, segment, classify, pose, obb] 之一
        - `MODE`（必需）是 [train, val, predict, export, track, benchmark] 之一
        - `ARGS`（可选）是任意数量的自定义 `arg=value` 对，如 `imgsz=320`，用于覆盖默认值。

        在完整的[配置指南](cfg.md)中查看所有参数，或使用 `yolo cfg`。

    === "训练"

        以初始[学习率](https://www.ultralytics.com/glossary/learning-rate) 0.01 训练检测模型 10 个[训练周期](https://www.ultralytics.com/glossary/epoch)：

        ```bash
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
        ```

    === "预测"

        使用预训练分割模型在 YouTube 视频上以图像尺寸 320 进行预测：

        ```bash
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "验证"

        以[批次大小](https://www.ultralytics.com/glossary/batch-size) 1 和图像尺寸 640 验证预训练检测模型：

        ```bash
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
        ```

    === "导出"

        以图像尺寸 224x128 将 YOLO 分类模型导出为 ONNX 格式（无需 TASK）：

        ```bash
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128
        ```

    === "特殊命令"

        运行特殊命令查看版本、设置、运行检查等：

        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

其中：

- `TASK`（可选）是 `[detect, segment, classify, pose, obb]` 之一。如果未明确传递，YOLO 将尝试从模型类型推断 `TASK`。
- `MODE`（必需）是 `[train, val, predict, export, track, benchmark]` 之一
- `ARGS`（可选）是任意数量的自定义 `arg=value` 对，如 `imgsz=320`，用于覆盖默认值。有关可用 `ARGS` 的完整列表，请参阅[配置](cfg.md)页面和 `default.yaml`。

!!! warning "警告"

    参数必须以 `arg=val` 对的形式传递，用等号 `=` 分隔，对之间用空格分隔。不要使用 `--` 参数前缀或参数之间的逗号 `,`。

    - `yolo predict model=yolo11n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolo11n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolo11n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

## 训练

在 COCO8 数据集上以图像尺寸 640 训练 YOLO 100 个训练周期。有关可用参数的完整列表，请参阅[配置](cfg.md)页面。

!!! example "示例"

    === "训练"

        在 COCO8 上以图像尺寸 640 开始训练 YOLO11n 100 个训练周期：

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

    === "恢复"

        恢复中断的训练会话：

        ```bash
        yolo detect train resume model=last.pt
        ```

## 验证

在 COCO8 数据集上验证训练模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。不需要参数，因为 `model` 会保留其训练时的 `data` 和参数作为模型属性。

!!! example "示例"

    === "官方模型"

        验证官方 YOLO11n 模型：

        ```bash
        yolo detect val model=yolo11n.pt
        ```

    === "自定义模型"

        验证自定义训练的模型：

        ```bash
        yolo detect val model=path/to/best.pt
        ```

## 预测

使用训练好的模型对图像进行预测。

!!! example "示例"

    === "官方模型"

        使用官方 YOLO11n 模型进行预测：

        ```bash
        yolo detect predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
        ```

    === "自定义模型"

        使用自定义模型进行预测：

        ```bash
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'
        ```

## 导出

将模型导出为不同格式，如 ONNX 或 CoreML。

!!! example "示例"

    === "官方模型"

        将官方 YOLO11n 模型导出为 ONNX 格式：

        ```bash
        yolo export model=yolo11n.pt format=onnx
        ```

    === "自定义模型"

        将自定义训练的模型导出为 ONNX 格式：

        ```bash
        yolo export model=path/to/best.pt format=onnx
        ```

可用的 Ultralytics 导出格式如下表所示。您可以使用 `format` 参数导出为任何格式，例如 `format='onnx'` 或 `format='engine'`。

{% include "macros/export-table.md" %}

有关完整的 `export` 详情，请参阅[导出](../modes/export.md)页面。

## 覆盖默认参数

通过在 CLI 中以 `arg=value` 对的形式传递参数来覆盖默认参数。

!!! tip "提示"

    === "训练"

        以学习率 0.01 训练检测模型 10 个训练周期：

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
        ```

    === "预测"

        使用预训练分割模型在 YouTube 视频上以图像尺寸 320 进行预测：

        ```bash
        yolo segment predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "验证"

        以批次大小 1 和图像尺寸 640 验证预训练检测模型：

        ```bash
        yolo detect val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
        ```

## 覆盖默认配置文件

通过使用 `cfg` 参数传递新文件来完全覆盖 `default.yaml` 配置文件，例如 `cfg=custom.yaml`。

为此，首先使用 `yolo copy-cfg` 命令在当前工作目录中创建 `default.yaml` 的副本，这将创建一个 `default_copy.yaml` 文件。

然后您可以将此文件作为 `cfg=default_copy.yaml` 传递，以及任何其他参数，如本例中的 `imgsz=320`：

!!! example "示例"

    === "CLI"

        ```bash
        yolo copy-cfg
        yolo cfg=default_copy.yaml imgsz=320
        ```

## 解决方案命令

Ultralytics 通过 CLI 为常见的计算机视觉应用提供即用型解决方案。这些解决方案简化了目标计数、健身监控和队列管理等复杂任务的实现。

!!! example "示例"

    === "计数"

        在视频或实时流中计数目标：

        ```bash
        yolo solutions count show=True
        yolo solutions count source="path/to/video.mp4" # 指定视频文件路径
        ```

    === "健身"

        使用姿态模型监控健身运动：

        ```bash
        yolo solutions workout show=True
        yolo solutions workout source="path/to/video.mp4" # 指定视频文件路径

        # 使用关键点进行腹部训练
        yolo solutions workout kpts=[5, 11, 13] # 左侧
        yolo solutions workout kpts=[6, 12, 14] # 右侧
        ```

    === "队列"

        在指定队列或区域中计数目标：

        ```bash
        yolo solutions queue show=True
        yolo solutions queue source="path/to/video.mp4"                                # 指定视频文件路径
        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" # 配置队列坐标
        ```

    === "推理"

        使用 Streamlit 在 Web 浏览器中执行目标检测、实例分割或姿态估计：

        ```bash
        yolo solutions inference
        yolo solutions inference model="path/to/model.pt" # 使用自定义模型
        ```

    === "帮助"

        查看可用的解决方案及其选项：

        ```bash
        yolo solutions help
        ```

有关 Ultralytics 解决方案的更多信息，请访问[解决方案](../solutions/index.md)页面。

## 常见问题

### 如何使用 Ultralytics YOLO 命令行界面（CLI）进行模型训练？

要使用 CLI 训练模型，在终端中执行单行命令。例如，以[学习率](https://www.ultralytics.com/glossary/learning-rate) 0.01 训练检测模型 10 个训练周期，运行：

```bash
yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
```

此命令使用 `train` 模式和特定参数。有关可用参数的完整列表，请参阅[配置指南](cfg.md)。

### 我可以使用 Ultralytics YOLO CLI 执行哪些任务？

Ultralytics YOLO CLI 支持各种任务，包括[检测](../tasks/detect.md)、[分割](../tasks/segment.md)、[分类](../tasks/classify.md)、[姿态估计](../tasks/pose.md)和[旋转边界框检测](../tasks/obb.md)。您还可以执行以下操作：

- **训练模型**：运行 `yolo train data=<data.yaml> model=<model.pt> epochs=<num>`。
- **运行预测**：使用 `yolo predict model=<model.pt> source=<data_source> imgsz=<image_size>`。
- **导出模型**：执行 `yolo export model=<model.pt> format=<export_format>`。
- **使用解决方案**：运行 `yolo solutions <solution_name>` 获取即用型应用。

使用各种参数自定义每个任务。有关详细语法和示例，请参阅相应部分，如[训练](#训练)、[预测](#预测)和[导出](#导出)。

### 如何使用 CLI 验证训练好的 YOLO 模型的准确率？

要验证模型的[准确率](https://www.ultralytics.com/glossary/accuracy)，使用 `val` 模式。例如，以[批次大小](https://www.ultralytics.com/glossary/batch-size) 1 和图像尺寸 640 验证预训练检测模型，运行：

```bash
yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
```

此命令在指定数据集上评估模型，并提供 [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)、[精确率](https://www.ultralytics.com/glossary/precision)和[召回率](https://www.ultralytics.com/glossary/recall)等性能指标。更多详情请参阅[验证](#验证)部分。

### 我可以使用 CLI 将 YOLO 模型导出为哪些格式？

您可以将 YOLO 模型导出为各种格式，包括 ONNX、TensorRT、CoreML、TensorFlow 等。例如，要将模型导出为 ONNX 格式，运行：

```bash
yolo export model=yolo11n.pt format=onnx
```

导出命令支持多种选项来优化您的模型以适应特定的部署环境。有关所有可用导出格式及其特定参数的完整详情，请访问[导出](../modes/export.md)页面。

### 如何使用 Ultralytics CLI 中的预构建解决方案？

Ultralytics 通过 `solutions` 命令提供即用型解决方案。例如，要在视频中计数目标：

```bash
yolo solutions count source="path/to/video.mp4"
```

这些解决方案需要最少的配置，并为常见的计算机视觉任务提供即时功能。要查看所有可用的解决方案，运行 `yolo solutions help`。每个解决方案都有特定的参数，可以根据您的需求进行自定义。
