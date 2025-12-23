---
comments: true
description: 了解如何使用 YOLO11 进行姿态估计任务。学习模型训练、验证、预测以及导出为各种格式。
keywords: 姿态估计, YOLO11, Ultralytics, 关键点, 模型训练, 图像识别, 深度学习, 人体姿态检测, 计算机视觉, 实时跟踪
model_name: yolo11n-pose
---

# 姿态估计

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/pose-estimation-examples.avif" alt="姿态估计示例">

姿态估计是一项涉及识别图像中特定点位置的任务，这些点通常称为关键点。关键点可以表示目标的各个部分，如关节、地标或其他显著特征。关键点的位置通常表示为一组 2D `[x, y]` 或 3D `[x, y, visible]` 坐标。

姿态估计模型的输出是一组表示图像中目标关键点的点，通常还包括每个点的置信度分数。当您需要识别场景中目标的特定部分及其相对位置时，姿态估计是一个很好的选择。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/AAkfToU3nAc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics YOLO11 姿态估计教程 | 实时目标跟踪和人体姿态检测
</p>

!!! tip "提示"

    YOLO11 _pose_ 模型使用 `-pose` 后缀，即 `yolo11n-pose.pt`。这些模型在 [COCO 关键点](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml)数据集上训练，适用于各种姿态估计任务。

    在默认的 YOLO11 姿态模型中，有 17 个关键点，每个代表人体的不同部位。以下是每个索引到相应身体关节的映射：

    0. 鼻子
    1. 左眼
    2. 右眼
    3. 左耳
    4. 右耳
    5. 左肩
    6. 右肩
    7. 左肘
    8. 右肘
    9. 左手腕
    10. 右手腕
    11. 左髋
    12. 右髋
    13. 左膝
    14. 右膝
    15. 左脚踝
    16. 右脚踝

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

此处展示 Ultralytics YOLO11 预训练姿态模型。检测、分割和姿态模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上预训练，而分类模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 数据集上预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

{% include "macros/yolo-pose-perf.md" %}

- **mAP<sup>val</sup>** 值是在 [COCO Keypoints val2017](https://cocodataset.org/) 数据集上单模型单尺度的结果。<br>复现命令：`yolo val pose data=coco-pose.yaml device=0`
- **速度**是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 COCO 验证图像取平均值。<br>复现命令：`yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`

## 训练

在 COCO8-pose 数据集上训练 YOLO11-pose 模型。[COCO8-pose 数据集](https://docs.ultralytics.com/datasets/pose/coco8-pose/)是一个小型示例数据集，非常适合测试和调试您的姿态估计模型。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo11n-pose.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # 从 YAML 构建并迁移权重

        # 训练模型
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.yaml epochs=100 imgsz=640

        # 从预训练的 *.pt 模型开始训练
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640

        # 从 YAML 构建新模型，迁移预训练权重并开始训练
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.yaml pretrained=yolo11n-pose.pt epochs=100 imgsz=640
        ```

### 数据集格式

YOLO 姿态数据集格式的详细信息可以在[数据集指南](../datasets/pose/index.md)中找到。要将现有数据集从其他格式（如 [COCO](https://docs.ultralytics.com/datasets/pose/coco/) 等）转换为 YOLO 格式，请使用 Ultralytics 的 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 工具。

对于自定义姿态估计任务，您还可以探索专门的数据集，如用于动物姿态估计的 [Tiger-Pose](https://docs.ultralytics.com/datasets/pose/tiger-pose/)、用于手部跟踪的 [Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) 或用于犬类姿态分析的 [Dog-Pose](https://docs.ultralytics.com/datasets/pose/dog-pose/)。

## 验证

在 COCO8-pose 数据集上验证训练好的 YOLO11n-pose 模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。不需要传递任何参数，因为 `model` 会保留其训练时的 `data` 和参数作为模型属性。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置会被记住
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # 包含每个类别 mAP50-95 的列表
        metrics.pose.map  # map50-95(P)
        metrics.pose.map50  # map50(P)
        metrics.pose.map75  # map75(P)
        metrics.pose.maps  # 包含每个类别 mAP50-95(P) 的列表
        ```

    === "CLI"

        ```bash
        yolo pose val model=yolo11n-pose.pt # 验证官方模型
        yolo pose val model=path/to/best.pt # 验证自定义模型
        ```

## 预测

使用训练好的 YOLO11n-pose 模型对图像进行预测。[预测模式](https://docs.ultralytics.com/modes/predict/)允许您对图像、视频或实时流进行推理。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 使用模型进行预测
        results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测

        # 访问结果
        for result in results:
            xy = result.keypoints.xy  # x 和 y 坐标
            xyn = result.keypoints.xyn  # 归一化
            kpts = result.keypoints.data  # x, y, 可见性（如果可用）
        ```

    === "CLI"

        ```bash
        yolo pose predict model=yolo11n-pose.pt source='https://ultralytics.com/images/bus.jpg' # 使用官方模型预测
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # 使用自定义模型预测
        ```

有关完整的 `predict` 模式详情，请参阅[预测](../modes/predict.md)页面。

## 导出

将 YOLO11n Pose 模型导出为不同格式，如 ONNX、CoreML 等。这允许您在各种平台和设备上部署模型进行[实时推理](https://www.ultralytics.com/glossary/real-time-inference)。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-pose.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-pose.pt format=onnx # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

可用的 YOLO11-pose 导出格式如下表所示。您可以使用 `format` 参数导出为任何格式，例如 `format='onnx'` 或 `format='engine'`。您可以直接对导出的模型进行预测或验证，例如 `yolo predict model=yolo11n-pose.onnx`。导出完成后会显示您模型的使用示例。

{% include "macros/export-table.md" %}

有关完整的 `export` 详情，请参阅[导出](../modes/export.md)页面。

## 常见问题

### 什么是 Ultralytics YOLO11 姿态估计，它是如何工作的？

使用 Ultralytics YOLO11 进行姿态估计涉及识别图像中的特定点，称为关键点。这些关键点通常代表目标的关节或其他重要特征。输出包括每个点的 `[x, y]` 坐标和置信度分数。YOLO11-pose 模型专门为此任务设计，使用 `-pose` 后缀，如 `yolo11n-pose.pt`。这些模型在 [COCO 关键点](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml)等数据集上预训练，可用于各种姿态估计任务。更多信息请访问[姿态估计页面](#姿态估计)。

### 如何在自定义数据集上训练 YOLO11-pose 模型？

在自定义数据集上训练 YOLO11-pose 模型涉及加载模型，可以是由 YAML 文件定义的新模型或预训练模型。然后您可以使用指定的数据集和参数开始训练过程。

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n-pose.yaml")  # 从 YAML 构建新模型
model = YOLO("yolo11n-pose.pt")  # 加载预训练模型（推荐用于训练）

# 训练模型
results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
```

有关训练的详细信息，请参阅[训练部分](#训练)。您还可以使用 [Ultralytics HUB](https://www.ultralytics.com/hub) 进行无代码方式训练自定义姿态估计模型。

### 如何验证训练好的 YOLO11-pose 模型？

验证 YOLO11-pose 模型涉及使用训练期间保留的相同数据集参数评估其准确率。以下是示例：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n-pose.pt")  # 加载官方模型
model = YOLO("path/to/best.pt")  # 加载自定义模型

# 验证模型
metrics = model.val()  # 无需参数，数据集和设置会被记住
```

更多信息请访问[验证部分](#验证)。

### 我可以将 YOLO11-pose 模型导出为其他格式吗，如何操作？

是的，您可以将 YOLO11-pose 模型导出为各种格式，如 ONNX、CoreML、TensorRT 等。这可以使用 Python 或命令行界面（CLI）完成。

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n-pose.pt")  # 加载官方模型
model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

# 导出模型
model.export(format="onnx")
```

有关更多详情，请参阅[导出部分](#导出)。导出的模型可以部署在边缘设备上用于[实时应用](https://www.ultralytics.com/blog/real-time-inferences-in-vision-ai-solutions-are-making-an-impact)，如健身跟踪、运动分析或[机器人](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics)。

### 有哪些可用的 Ultralytics YOLO11-pose 模型及其性能指标？

Ultralytics YOLO11 提供各种预训练姿态模型，如 YOLO11n-pose、YOLO11s-pose、YOLO11m-pose 等。这些模型在大小、准确率（mAP）和速度方面有所不同。例如，YOLO11n-pose 模型实现了 50.0 的 mAP<sup>pose</sup>50-95 和 81.0 的 mAP<sup>pose</sup>50。有关完整列表和性能详情，请访问[模型部分](#模型)。
