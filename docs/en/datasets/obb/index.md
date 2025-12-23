---
comments: true
description: 探索 Ultralytics YOLO 模型的旋转边界框数据集格式。了解其结构、应用和格式转换，以增强目标检测训练。
keywords: 旋转边界框, OBB 数据集, YOLO, Ultralytics, 目标检测, 数据集格式
---

# 旋转边界框（OBB）数据集概述

使用旋转边界框（OBB）训练精确的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型需要全面的数据集。本指南介绍与 Ultralytics YOLO 模型兼容的各种 OBB 数据集格式，提供其结构、应用和格式转换方法的见解。

## 支持的 OBB 数据集格式

### YOLO OBB 格式

YOLO OBB 格式通过四个角点指定边界框，坐标归一化在 0 到 1 之间。它遵循以下格式：

```bash
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

在内部，YOLO 以 `xywhr` 格式处理损失和输出，该格式表示[边界框](https://www.ultralytics.com/glossary/bounding-box)的中心点（xy）、宽度、高度和旋转角度。

<p align="center"><img width="800" src="https://github.com/ultralytics/docs/releases/download/0/obb-format-examples.avif" alt="OBB 格式示例"></p>

上图的 `*.txt` 标签文件示例，包含一个类别为 `0` 的 OBB 格式目标，可能如下所示：

```bash
0 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749758
```

### 数据集 YAML 格式

Ultralytics 框架使用 YAML 文件格式来定义训练 OBB 模型的数据集和模型配置。以下是用于定义 OBB 数据集的 YAML 格式示例：

!!! example "ultralytics/cfg/datasets/dota8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dota8.yaml"
    ```

## 使用方法

要使用这些 OBB 格式训练模型：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从头创建新的 YOLO11n-OBB 模型
        model = YOLO("yolo11n-obb.yaml")

        # 在 DOTAv1 数据集上训练模型
        results = model.train(data="DOTAv1.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # 在 DOTAv1 数据集上训练新的 YOLO11n-OBB 模型
        yolo obb train data=DOTAv1.yaml model=yolo11n-obb.pt epochs=100 imgsz=1024
        ```

## 支持的数据集

目前，支持以下带有旋转边界框的数据集：

- [DOTA-v1](dota-v2.md#dota-v10)：DOTA 数据集的第一个版本，提供全面的航拍图像集，带有用于目标检测的旋转边界框。
- [DOTA-v1.5](dota-v2.md#dota-v15)：DOTA 数据集的中间版本，提供比 DOTA-v1 更多的标注和改进，用于增强目标检测任务。
- [DOTA-v2](dota-v2.md#dota-v20)：DOTA（航拍图像中的大规模目标检测数据集）版本 2，强调从航拍视角进行检测，包含 170 万个实例和 11,268 张图像的旋转边界框。
- [DOTA8](dota8.md)：完整 DOTA 数据集的小型 8 图像子集，适用于测试工作流程和 `ultralytics` 仓库中 OBB 训练的持续集成（CI）检查。

### 整合您自己的 OBB 数据集

对于希望引入自己带有旋转边界框数据集的用户，请确保与上述"YOLO OBB 格式"兼容。将您的标注转换为此所需格式，并在相应的 YAML 配置文件中详细说明路径、类别和类别名称。

## 转换标签格式

### DOTA 数据集格式转 YOLO OBB 格式

可以使用此脚本将标签从 DOTA 数据集格式转换为 YOLO OBB 格式：

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb("path/to/DOTA")
        ```

此转换机制对于 DOTA 格式的数据集非常有用，确保与 [Ultralytics YOLO](../../models/yolo11.md) OBB 格式对齐。

验证数据集与模型的兼容性并遵守必要的格式约定至关重要。结构正确的数据集对于使用旋转边界框训练高效的目标检测模型至关重要。

## 常见问题

### 什么是旋转边界框（OBB），它们如何在 Ultralytics YOLO 模型中使用？

旋转边界框（OBB）是一种边界框标注类型，其中框可以旋转以更紧密地与被检测目标对齐，而不仅仅是轴对齐。这在航拍或卫星图像中特别有用，因为目标可能不与图像轴对齐。在 [Ultralytics YOLO](../../tasks/obb.md) 模型中，OBB 通过 YOLO OBB 格式的四个角点表示。这允许更准确的目标检测，因为边界框可以旋转以更好地适应目标。

### 如何将现有的 DOTA 数据集标签转换为 YOLO OBB 格式以用于 Ultralytics YOLO11？

您可以使用 Ultralytics 的 [`convert_dota_to_yolo_obb`](../../reference/data/converter.md) 函数将 DOTA 数据集标签转换为 YOLO OBB 格式。此转换确保与 Ultralytics YOLO 模型兼容，使您能够利用 OBB 功能进行增强的目标检测。以下是一个快速示例：

```python
from ultralytics.data.converter import convert_dota_to_yolo_obb

convert_dota_to_yolo_obb("path/to/DOTA")
```

此脚本将把您的 DOTA 标注重新格式化为 YOLO 兼容格式。

### 如何在我的数据集上使用旋转边界框（OBB）训练 YOLO11 模型？

使用 OBB 训练 YOLO11 模型需要确保您的数据集是 YOLO OBB 格式，然后使用 [Ultralytics API](../../usage/python.md) 训练模型。以下是 Python 和 CLI 的示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从头创建新的 YOLO11n-OBB 模型
        model = YOLO("yolo11n-obb.yaml")

        # 在自定义数据集上训练模型
        results = model.train(data="your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 在自定义数据集上训练新的 YOLO11n-OBB 模型
        yolo obb train data=your_dataset.yaml model=yolo11n-obb.yaml epochs=100 imgsz=640
        ```

这确保您的模型利用详细的 OBB 标注来提高检测[准确率](https://www.ultralytics.com/glossary/accuracy)。

### Ultralytics YOLO 模型目前支持哪些数据集进行 OBB 训练？

目前，Ultralytics 支持以下数据集进行 OBB 训练：

- [DOTA-v1](dota-v2.md)：DOTA 数据集的第一个版本，提供全面的航拍图像集，带有用于目标检测的旋转边界框。
- [DOTA-v1.5](dota-v2.md)：DOTA 数据集的中间版本，提供比 DOTA-v1 更多的标注和改进，用于增强目标检测任务。
- [DOTA-v2](dota-v2.md)：此数据集包含 170 万个带有旋转边界框的实例和 11,268 张图像，主要专注于航拍目标检测。
- [DOTA8](dota8.md)：DOTA 数据集的较小 8 图像子集，用于测试和[持续集成](../../help/CI.md)（CI）检查。

这些数据集专为 OBB 提供显著优势的场景量身定制，如航拍和卫星图像分析。

### 我可以使用自己的带有旋转边界框的数据集进行 YOLO11 训练吗？如果可以，如何操作？

是的，您可以使用自己的带有旋转边界框的数据集进行 YOLO11 训练。确保您的数据集标注转换为 YOLO OBB 格式，该格式通过四个角点定义边界框。然后，您可以创建一个 [YAML 配置文件](../../usage/cfg.md)，指定数据集路径、类别和其他必要详细信息。有关创建和配置数据集的更多信息，请参阅[支持的数据集](#支持的数据集)部分。
