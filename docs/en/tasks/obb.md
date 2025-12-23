---
comments: true
description: 了解如何使用 YOLO11 旋转边界框模型进行更高精度的旋转目标检测。学习、训练、验证和导出旋转边界框模型。
keywords: 旋转边界框, OBB, 目标检测, YOLO11, Ultralytics, DOTAv1, 模型训练, 模型导出, 人工智能, 机器学习
model_name: yolo11n-obb
---

# 旋转边界框[目标检测](https://www.ultralytics.com/glossary/object-detection)

<!-- obb task poster -->

旋转目标检测比标准目标检测更进一步，通过引入额外的角度来更准确地定位图像中的目标。

旋转目标检测器的输出是一组精确包围图像中目标的旋转边界框，以及每个框的类别标签和置信度分数。当目标以各种角度出现时，旋转边界框特别有用，例如在航拍图像中，传统的轴对齐边界框可能会包含不必要的背景。

<!-- youtube video link for obb task -->

!!! tip "提示"

    YOLO11 旋转边界框模型使用 `-obb` 后缀，即 `yolo11n-obb.pt`，在 [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) 数据集上进行了预训练。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Z7Z9pHF8wJc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO 旋转边界框（YOLO-OBB）进行目标检测
</p>

## 可视化示例

|                                              使用旋转边界框检测船只                                               |                                               使用旋转边界框检测车辆                                                |
| :------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: |
| ![使用旋转边界框检测船只](https://github.com/ultralytics/docs/releases/download/0/ships-detection-using-obb.avif) | ![使用旋转边界框检测车辆](https://github.com/ultralytics/docs/releases/download/0/vehicle-detection-using-obb.avif) |

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

此处展示 YOLO11 预训练旋转边界框模型，这些模型在 [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) 数据集上预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

{% include "macros/yolo-obb-perf.md" %}

- **mAP<sup>test</sup>** 值是在 [DOTAv1](https://captain-whu.github.io/DOTA/index.html) 数据集上单模型多尺度的结果。<br>复现命令：`yolo val obb data=DOTAv1.yaml device=0 split=test`，并将合并结果提交到 [DOTA 评估](https://captain-whu.github.io/DOTA/evaluation.html)。
- **速度**是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 DOTAv1 验证图像取平均值。<br>复现命令：`yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu`

## 训练

在 DOTA8 数据集上以图像尺寸 640 训练 YOLO11n-obb 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)。有关可用参数的完整列表，请参阅[配置](../usage/cfg.md)页面。

!!! note "注意"

    旋转边界框角度被限制在 **0-90 度**范围内（不包括 90 度）。不支持 90 度或更大的角度。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-obb.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo11n-obb.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")  # 从 YAML 构建并迁移权重

        # 训练模型
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo obb train data=dota8.yaml model=yolo11n-obb.yaml epochs=100 imgsz=640

        # 从预训练的 *.pt 模型开始训练
        yolo obb train data=dota8.yaml model=yolo11n-obb.pt epochs=100 imgsz=640

        # 从 YAML 构建新模型，迁移预训练权重并开始训练
        yolo obb train data=dota8.yaml model=yolo11n-obb.yaml pretrained=yolo11n-obb.pt epochs=100 imgsz=640
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uZ7SymQfqKI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics HUB 在 DOTA 数据集上训练 Ultralytics YOLO-OBB（旋转边界框）模型
</p>

### 数据集格式

旋转边界框数据集格式的详细信息可以在[数据集指南](../datasets/obb/index.md)中找到。YOLO 旋转边界框格式通过四个角点指定边界框，坐标归一化在 0 到 1 之间，遵循以下结构：

```
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

在内部，YOLO 使用 `xywhr` 格式处理损失和输出，该格式表示[边界框](https://www.ultralytics.com/glossary/bounding-box)的中心点（xy）、宽度、高度和旋转角度。

## 验证

在 DOTA8 数据集上验证训练好的 YOLO11n-obb 模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。不需要传递任何参数，因为 `model` 会保留其训练时的 `data` 和参数作为模型属性。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-obb.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val(data="dota8.yaml")  # 无需参数，数据集和设置会被记住
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # 包含每个类别 mAP50-95(B) 的列表
        ```

    === "CLI"

        ```bash
        yolo obb val model=yolo11n-obb.pt data=dota8.yaml         # 验证官方模型
        yolo obb val model=path/to/best.pt data=path/to/data.yaml # 验证自定义模型
        ```

## 预测

使用训练好的 YOLO11n-obb 模型对图像进行预测。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-obb.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 使用模型进行预测
        results = model("https://ultralytics.com/images/boats.jpg")  # 对图像进行预测

        # 访问结果
        for result in results:
            xywhr = result.obb.xywhr  # 中心点x, 中心点y, 宽度, 高度, 角度（弧度）
            xyxyxyxy = result.obb.xyxyxyxy  # 4点多边形格式
            names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # 每个框的类别名称
            confs = result.obb.conf  # 每个框的置信度分数
        ```

    === "CLI"

        ```bash
        yolo obb predict model=yolo11n-obb.pt source='https://ultralytics.com/images/boats.jpg'  # 使用官方模型预测
        yolo obb predict model=path/to/best.pt source='https://ultralytics.com/images/boats.jpg' # 使用自定义模型预测
        ```

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5XYdm5CYODA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO-OBB 检测和跟踪储油罐 | 旋转边界框 | DOTA
</p>

有关完整的 `predict` 模式详情，请参阅[预测](../modes/predict.md)页面。

## 导出

将 YOLO11n-obb 模型导出为不同格式，如 ONNX、CoreML 等。

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-obb.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-obb.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

可用的 YOLO11-obb 导出格式如下表所示。您可以使用 `format` 参数导出为任何格式，例如 `format='onnx'` 或 `format='engine'`。您可以直接对导出的模型进行预测或验证，例如 `yolo predict model=yolo11n-obb.onnx`。导出完成后会显示您模型的使用示例。

{% include "macros/export-table.md" %}

有关完整的 `export` 详情，请参阅[导出](../modes/export.md)页面。

## 实际应用

使用 YOLO11 进行旋转边界框检测在各行业有众多实际应用：

- **海事和港口管理**：以各种角度检测船只和船舶，用于[船队管理](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-obb-object-detection)和监控。
- **城市规划**：从航拍图像分析建筑物和基础设施。
- **农业**：从无人机画面监控农作物和农业设备。
- **能源领域**：检查不同方向的太阳能电池板和风力涡轮机。
- **交通运输**：从各种角度跟踪道路和停车场中的车辆。

这些应用受益于旋转边界框能够精确拟合任意角度目标的能力，比传统边界框提供更准确的检测。

## 常见问题

### 什么是旋转边界框（OBB），它与普通边界框有什么区别？

旋转边界框（OBB）包含一个额外的角度来提高图像中目标定位的准确性。与普通边界框（轴对齐的矩形）不同，旋转边界框可以旋转以更好地拟合目标的方向。这对于需要精确目标定位的应用特别有用，例如航拍或卫星图像（[数据集指南](../datasets/obb/index.md)）。

### 如何使用自定义数据集训练 YOLO11n-obb 模型？

要使用自定义数据集训练 YOLO11n-obb 模型，请按照以下 Python 或命令行界面示例操作：

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n-obb.pt")

        # 训练模型
        results = model.train(data="path/to/custom_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo obb train data=path/to/custom_dataset.yaml model=yolo11n-obb.pt epochs=100 imgsz=640
        ```

有关更多训练参数，请查看[配置](../usage/cfg.md)部分。

### 可以使用哪些数据集来训练 YOLO11-OBB 模型？

YOLO11-OBB 模型在 [DOTAv1](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DOTAv1.yaml) 等数据集上预训练，但您可以使用任何格式化为旋转边界框的数据集。有关旋转边界框数据集格式的详细信息，请参阅[数据集指南](../datasets/obb/index.md)。

### 如何将 YOLO11-OBB 模型导出为 ONNX 格式？

使用 Python 或命令行界面将 YOLO11-OBB 模型导出为 ONNX 格式非常简单：

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-obb.pt")

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-obb.pt format=onnx
        ```

有关更多导出格式和详情，请参阅[导出](../modes/export.md)页面。

### 如何验证 YOLO11n-obb 模型的准确率？

要验证 YOLO11n-obb 模型，您可以使用如下所示的 Python 或命令行界面命令：

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-obb.pt")

        # 验证模型
        metrics = model.val(data="dota8.yaml")
        ```

    === "CLI"

        ```bash
        yolo obb val model=yolo11n-obb.pt data=dota8.yaml
        ```

有关完整的验证详情，请参阅[验证](../modes/val.md)部分。
