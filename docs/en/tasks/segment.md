---
comments: true
description: 掌握使用 YOLO11 进行实例分割。学习如何检测、分割和勾勒图像中的目标，包含详细指南和示例。
keywords: 实例分割, YOLO11, 目标检测, 图像分割, 机器学习, 深度学习, 计算机视觉, COCO 数据集, Ultralytics
model_name: yolo11n-seg
---

# 实例分割

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/instance-segmentation-examples.avif" alt="实例分割示例">

[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)比目标检测更进一步，涉及识别图像中的单个目标并将它们从图像的其余部分分割出来。

实例分割模型的输出是一组掩码或轮廓，勾勒出图像中的每个目标，以及每个目标的类别标签和置信度分数。当您不仅需要知道目标在图像中的位置，还需要知道它们的确切形状时，实例分割非常有用。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>在 Python 中使用预训练 Ultralytics YOLO 模型运行分割。
</p>

!!! tip "提示"

    YOLO11 分割模型使用 `-seg` 后缀，即 `yolo11n-seg.pt`，在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上进行了预训练。

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

此处展示 YOLO11 预训练分割模型。检测、分割和姿态模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上预训练，而分类模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 数据集上预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

{% include "macros/yolo-seg-perf.md" %}

- **mAP<sup>val</sup>** 值是在 [COCO val2017](https://cocodataset.org/) 数据集上单模型单尺度的结果。<br>复现命令：`yolo val segment data=coco.yaml device=0`
- **速度**是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 COCO 验证图像取平均值。<br>复现命令：`yolo val segment data=coco.yaml batch=1 device=0|cpu`

## 训练

在 COCO8-seg 数据集上以图像尺寸 640 训练 YOLO11n-seg 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)。有关可用参数的完整列表，请参阅[配置](../usage/cfg.md)页面。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-seg.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo11n-seg.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # 从 YAML 构建并迁移权重

        # 训练模型
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo segment train data=coco8-seg.yaml model=yolo11n-seg.yaml epochs=100 imgsz=640

        # 从预训练的 *.pt 模型开始训练
        yolo segment train data=coco8-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640

        # 从 YAML 构建新模型，迁移预训练权重并开始训练
        yolo segment train data=coco8-seg.yaml model=yolo11n-seg.yaml pretrained=yolo11n-seg.pt epochs=100 imgsz=640
        ```

### 数据集格式

YOLO 分割数据集格式的详细信息可以在[数据集指南](../datasets/segment/index.md)中找到。要将现有数据集从其他格式（如 COCO 等）转换为 YOLO 格式，请使用 Ultralytics 的 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 工具。

## 验证

在 COCO8-seg 数据集上验证训练好的 YOLO11n-seg 模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。不需要传递任何参数，因为 `model` 会保留其训练时的 `data` 和参数作为模型属性。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-seg.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置会被记住
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # 包含每个类别 mAP50-95(B) 的列表
        metrics.seg.map  # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps  # 包含每个类别 mAP50-95(M) 的列表
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolo11n-seg.pt  # 验证官方模型
        yolo segment val model=path/to/best.pt # 验证自定义模型
        ```

## 预测

使用训练好的 YOLO11n-seg 模型对图像进行预测。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-seg.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 使用模型进行预测
        results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测

        # 访问结果
        for result in results:
            xy = result.masks.xy  # 多边形格式的掩码
            xyn = result.masks.xyn  # 归一化
            masks = result.masks.data  # 矩阵格式的掩码 (num_objects x H x W)
        ```

    === "CLI"

        ```bash
        yolo segment predict model=yolo11n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # 使用官方模型预测
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # 使用自定义模型预测
        ```

有关完整的 `predict` 模式详情，请参阅[预测](../modes/predict.md)页面。

## 导出

将 YOLO11n-seg 模型导出为不同格式，如 ONNX、CoreML 等。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-seg.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-seg.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

可用的 YOLO11-seg 导出格式如下表所示。您可以使用 `format` 参数导出为任何格式，例如 `format='onnx'` 或 `format='engine'`。您可以直接对导出的模型进行预测或验证，例如 `yolo predict model=yolo11n-seg.onnx`。导出完成后会显示您模型的使用示例。

{% include "macros/export-table.md" %}

有关完整的 `export` 详情，请参阅[导出](../modes/export.md)页面。

## 常见问题

### 如何在自定义数据集上训练 YOLO11 分割模型？

要在自定义数据集上训练 YOLO11 分割模型，首先需要将数据集准备为 YOLO 分割格式。您可以使用 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 等工具从其他格式转换数据集。数据集准备好后，您可以使用 Python 或命令行界面命令训练模型：

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO11 分割模型
        model = YOLO("yolo11n-seg.pt")

        # 训练模型
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo segment train data=path/to/your_dataset.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

查看[配置](../usage/cfg.md)页面了解更多可用参数。

### YOLO11 中[目标检测](https://www.ultralytics.com/glossary/object-detection)和实例分割有什么区别？

目标检测通过在目标周围绘制边界框来识别和定位图像中的目标，而实例分割不仅识别边界框，还勾勒出每个目标的确切形状。YOLO11 实例分割模型提供掩码或轮廓来勾勒每个检测到的目标，这对于需要知道目标精确形状的任务特别有用，例如医学成像或自动驾驶。

### 为什么使用 YOLO11 进行实例分割？

Ultralytics YOLO11 是一个最先进的模型，以其高准确率和实时性能而闻名，非常适合实例分割任务。YOLO11 分割模型在 [COCO 数据集](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)上预训练，确保在各种目标上具有稳健的性能。此外，YOLO 支持训练、验证、预测和导出功能的无缝集成，使其在研究和工业应用中都非常通用。

### 如何加载和验证预训练的 YOLO 分割模型？

加载和验证预训练的 YOLO 分割模型非常简单。以下是使用 Python 和命令行界面的方法：

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n-seg.pt")

        # 验证模型
        metrics = model.val()
        print("边界框的平均精度均值:", metrics.box.map)
        print("掩码的平均精度均值:", metrics.seg.map)
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolo11n-seg.pt
        ```

这些步骤将为您提供验证指标，如[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP），这对于评估模型性能至关重要。

### 如何将 YOLO 分割模型导出为 ONNX 格式？

将 YOLO 分割模型导出为 ONNX 格式非常简单，可以使用 Python 或命令行界面命令完成：

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n-seg.pt")

        # 将模型导出为 ONNX 格式
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-seg.pt format=onnx
        ```

有关导出为各种格式的更多详情，请参阅[导出](../modes/export.md)页面。
