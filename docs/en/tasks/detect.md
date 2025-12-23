---
comments: true
description: 了解 YOLO11 目标检测。探索预训练模型、训练、验证、预测和导出详情，实现高效的目标识别。
keywords: 目标检测, YOLO11, 预训练模型, 训练, 验证, 预测, 导出, 机器学习, 计算机视觉
---

# 目标检测

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/object-detection-examples.avif" alt="目标检测示例">

[目标检测](https://www.ultralytics.com/glossary/object-detection)是一项涉及识别图像或视频流中目标位置和类别的任务。

目标检测器的输出是一组包围图像中目标的边界框，以及每个框的类别标签和置信度分数。当您需要识别场景中感兴趣的目标，但不需要知道目标的确切位置或精确形状时，目标检测是一个很好的选择。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用预训练 Ultralytics YOLO 模型进行目标检测。
</p>

!!! tip "提示"

    YOLO11 检测模型是默认的 YOLO11 模型，即 `yolo11n.pt`，在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上进行了预训练。

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

此处展示 YOLO11 预训练检测模型。检测、分割和姿态模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上预训练，而分类模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 数据集上预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

{% include "macros/yolo-det-perf.md" %}

- **mAP<sup>val</sup>** 值是在 [COCO val2017](https://cocodataset.org/) 数据集上单模型单尺度的结果。<br>复现命令：`yolo val detect data=coco.yaml device=0`
- **速度**是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 COCO 验证图像取平均值。<br>复现命令：`yolo val detect data=coco.yaml batch=1 device=0|cpu`

## 训练

在 COCO8 数据集上以图像尺寸 640 训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)。有关可用参数的完整列表，请参阅[配置](../usage/cfg.md)页面。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # 从 YAML 构建并迁移权重

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640

        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640

        # 从 YAML 构建新模型，迁移预训练权重并开始训练
        yolo detect train data=coco8.yaml model=yolo11n.yaml pretrained=yolo11n.pt epochs=100 imgsz=640
        ```

### 数据集格式

YOLO 检测数据集格式的详细信息可以在[数据集指南](../datasets/detect/index.md)中找到。要将现有数据集从其他格式（如 COCO 等）转换为 YOLO 格式，请使用 Ultralytics 的 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 工具。

## 验证

在 COCO8 数据集上验证训练好的 YOLO11n 模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。不需要传递任何参数，因为 `model` 会保留其训练时的 `data` 和参数作为模型属性。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置会被记住
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # 包含每个类别 mAP50-95 的列表
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolo11n.pt      # 验证官方模型
        yolo detect val model=path/to/best.pt # 验证自定义模型
        ```

## 预测

使用训练好的 YOLO11n 模型对图像进行预测。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 使用模型进行预测
        results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测

        # 访问结果
        for result in results:
            xywh = result.boxes.xywh  # 中心点x, 中心点y, 宽度, 高度
            xywhn = result.boxes.xywhn  # 归一化
            xyxy = result.boxes.xyxy  # 左上角x, 左上角y, 右下角x, 右下角y
            xyxyn = result.boxes.xyxyn  # 归一化
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # 每个框的类别名称
            confs = result.boxes.conf  # 每个框的置信度分数
        ```

    === "CLI"

        ```bash
        yolo detect predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'      # 使用官方模型预测
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # 使用自定义模型预测
        ```

有关完整的 `predict` 模式详情，请参阅[预测](../modes/predict.md)页面。

## 导出

将 YOLO11n 模型导出为不同格式，如 ONNX、CoreML 等。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx      # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

可用的 YOLO11 导出格式如下表所示。您可以使用 `format` 参数导出为任何格式，例如 `format='onnx'` 或 `format='engine'`。您可以直接对导出的模型进行预测或验证，例如 `yolo predict model=yolo11n.onnx`。导出完成后会显示您模型的使用示例。

{% include "macros/export-table.md" %}

有关完整的 `export` 详情，请参阅[导出](../modes/export.md)页面。

## 常见问题

### 如何在自定义数据集上训练 YOLO11 模型？

在自定义数据集上训练 YOLO11 模型需要几个步骤：

1. **准备数据集**：确保您的数据集是 YOLO 格式。有关指导，请参阅我们的[数据集指南](../datasets/detect/index.md)。
2. **加载模型**：使用 Ultralytics YOLO 库加载预训练模型或从 YAML 文件创建新模型。
3. **训练模型**：在 Python 中执行 `train` 方法或在命令行界面中使用 `yolo detect train` 命令。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 在自定义数据集上训练模型
        model.train(data="my_custom_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=my_custom_dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关详细的配置选项，请访问[配置](../usage/cfg.md)页面。

### YOLO11 中有哪些预训练模型可用？

Ultralytics YOLO11 提供各种用于目标检测、分割和姿态估计的预训练模型。这些模型在 COCO 数据集上预训练，或在 ImageNet 上预训练用于分类任务。以下是一些可用的模型：

- [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
- [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)
- [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt)
- [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt)
- [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)

有关详细列表和性能指标，请参阅[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)部分。

### 如何验证训练好的 YOLO 模型的准确率？

要验证训练好的 YOLO11 模型的准确率，您可以在 Python 中使用 `.val()` 方法或在命令行界面中使用 `yolo detect val` 命令。这将提供 mAP50-95、mAP50 等指标。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")

        # 验证模型
        metrics = model.val()
        print(metrics.box.map)  # mAP50-95
        ```

    === "CLI"

        ```bash
        yolo detect val model=path/to/best.pt
        ```

有关更多验证详情，请访问[验证](../modes/val.md)页面。

### YOLO11 模型可以导出为哪些格式？

Ultralytics YOLO11 允许将模型导出为各种格式，如 [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange)、[TensorRT](https://www.ultralytics.com/glossary/tensorrt)、[CoreML](https://docs.ultralytics.com/integrations/coreml/) 等，以确保在不同平台和设备上的兼容性。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 ONNX 格式
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx
        ```

在[导出](../modes/export.md)页面查看支持格式的完整列表和说明。

### 为什么应该使用 Ultralytics YOLO11 进行目标检测？

Ultralytics YOLO11 旨在为目标检测、分割和姿态估计提供最先进的性能。以下是一些主要优势：

1. **预训练模型**：利用在 [COCO](https://docs.ultralytics.com/datasets/detect/coco/) 和 [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) 等流行数据集上预训练的模型，加速开发。
2. **高准确率**：实现令人印象深刻的 mAP 分数，确保可靠的目标检测。
3. **速度**：针对[实时推理](https://www.ultralytics.com/glossary/real-time-inference)进行优化，非常适合需要快速处理的应用。
4. **灵活性**：将模型导出为 ONNX 和 TensorRT 等各种格式，以便在多个平台上部署。

探索我们的[博客](https://www.ultralytics.com/blog)，了解展示 YOLO11 实际应用的用例和成功案例。
