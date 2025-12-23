---
comments: true
description: 探索 Ultralytics YOLO 支持的数据集格式，了解如何准备和使用数据集来训练目标分割模型。
keywords: Ultralytics, YOLO, 实例分割, 数据集格式, 自动标注, COCO, 分割模型, 训练数据
---

# 实例分割数据集概述

实例分割是一项计算机视觉任务，涉及识别和描绘图像中的各个目标。本指南概述了 Ultralytics YOLO 支持的实例分割任务数据集格式，以及如何准备、转换和使用这些数据集来训练模型的说明。

## 支持的数据集格式

### Ultralytics YOLO 格式

用于训练 YOLO 分割模型的数据集标签格式如下：

1. 每张图像一个文本文件：数据集中的每张图像都有一个对应的文本文件，文件名与图像文件相同，扩展名为 ".txt"。
2. 每个目标一行：文本文件中的每一行对应图像中的一个目标实例。
3. 每行的目标信息：每行包含以下目标实例信息：
    - 目标类别索引：表示目标类别的整数（例如，0 表示人，1 表示汽车等）。
    - 目标边界坐标：掩码区域周围的边界坐标，归一化到 0 到 1 之间。

分割数据集文件中单行的格式如下：

```
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

在此格式中，`<class-index>` 是目标的类别索引，`<x1> <y1> <x2> <y2> ... <xn> <yn>` 是目标分割掩码的归一化多边形坐标（值在 `[0, 1]` 之间，相对于图像宽度和高度）。坐标之间用空格分隔。

以下是单张图像的 YOLO 数据集格式示例，包含两个目标，分别由 3 点线段和 5 点线段组成。

```
0 0.681 0.485 0.670 0.487 0.676 0.487
1 0.504 0.000 0.501 0.004 0.498 0.004 0.493 0.010 0.492 0.0104
```

!!! tip

    - 每行的长度**不必**相等。
    - 每个分割标签必须至少有 **3 个 `(x, y)` 点**：`<class-index> <x1> <y1> <x2> <y2> <x3> <y3>`

### 数据集 YAML 格式

Ultralytics 框架使用 YAML 文件格式来定义训练分割模型的数据集和模型配置。以下是用于定义分割数据集的 YAML 格式示例：

!!! example "ultralytics/cfg/datasets/coco8-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-seg.yaml"
    ```

`train` 和 `val` 字段分别指定包含训练和验证图像的目录路径。

`names` 是类别名称的字典。名称的顺序应与 YOLO 数据集文件中目标类别索引的顺序匹配。

## 使用方法

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo segment train data=coco8-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

## 支持的数据集

Ultralytics YOLO 支持各种实例分割任务的数据集。以下是最常用的数据集列表：

- [Carparts-seg](carparts-seg.md)：专注于汽车零件分割的专业数据集，非常适合汽车应用。它包含各种车辆，带有各个汽车组件的详细标注。
- [COCO](coco.md)：用于[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和图像描述的综合数据集，包含超过 20 万张标注图像，涵盖广泛的类别。
- [COCO8-seg](coco8-seg.md)：COCO 的紧凑 8 图像子集，专为快速测试分割模型训练而设计，非常适合 `ultralytics` 仓库中的 CI 检查和工作流验证。
- [COCO128-seg](coco128-seg.md)：用于[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)任务的较小数据集，包含 128 张带有分割标注的 COCO 图像子集。
- [Crack-seg](crack-seg.md)：专为各种表面裂缝分割而定制的数据集。对于基础设施维护和质量控制至关重要，它提供详细的图像用于训练模型识别结构弱点。
- [Package-seg](package-seg.md)：专门用于不同类型包装材料和形状分割的数据集。它对物流和仓库自动化特别有用，有助于开发包裹处理和分拣系统。

### 添加您自己的数据集

如果您有自己的数据集并希望使用它来训练 Ultralytics YOLO 格式的分割模型，请确保它遵循上述"Ultralytics YOLO 格式"中指定的格式。将您的标注转换为所需格式，并在 YAML 配置文件中指定路径、类别数量和类别名称。

## 转换标签格式

### COCO 数据集格式转 YOLO 格式

您可以使用以下代码片段轻松将流行的 COCO 数据集格式的标签转换为 YOLO 格式：

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/", use_segments=True)
        ```

此转换工具可用于将 COCO 数据集或任何 COCO 格式的数据集转换为 Ultralytics YOLO 格式。

请记住仔细检查您要使用的数据集是否与您的模型兼容并遵循必要的格式约定。格式正确的数据集对于训练成功的分割模型至关重要。

## 自动标注

自动标注是一项重要功能，允许您使用预训练的检测模型生成分割数据集。它使您能够快速准确地标注大量图像，而无需手动标注，节省时间和精力。

### 使用检测模型生成分割数据集

要使用 Ultralytics 框架自动标注您的数据集，可以使用 `auto_annotate` 函数，如下所示：

!!! example

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam_b.pt")
        ```

{% include "macros/sam-auto-annotate.md" %}

`auto_annotate` 函数接受图像路径，以及可选参数用于指定预训练检测模型（如 [YOLO11](../../models/yolo11.md)、[YOLOv8](../../models/yolov8.md) 或其他[模型](../../models/index.md)）和分割模型（如 [SAM](../../models/sam.md)、[SAM2](../../models/sam-2.md) 或 [MobileSAM](../../models/mobile-sam.md)）、运行模型的设备以及保存标注结果的输出目录。

通过利用预训练模型的强大功能，自动标注可以显著减少创建高质量分割数据集所需的时间和精力。此功能对于处理大型图像集合的研究人员和开发人员特别有用，因为它允许他们专注于模型开发和评估，而不是手动标注。

### 可视化数据集标注

在训练模型之前，可视化数据集标注以确保其正确性通常很有帮助。Ultralytics 为此提供了一个实用函数：

```python
from ultralytics.data.utils import visualize_image_annotations

label_map = {  # 定义包含所有标注类别标签的标签映射。
    0: "person",
    1: "car",
}

# 可视化
visualize_image_annotations(
    "path/to/image.jpg",  # 输入图像路径。
    "path/to/annotations.txt",  # 图像的标注文件路径。
    label_map,
)
```

此函数绘制边界框、用类别名称标注目标，并调整文本颜色以提高可读性，帮助您在训练前识别和纠正任何标注错误。

### 将分割掩码转换为 YOLO 格式

如果您有二进制格式的分割掩码，可以使用以下方法将其转换为 YOLO 分割格式：

```python
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# 对于像 COCO 这样有 80 个类别的数据集
convert_segment_masks_to_yolo_seg(masks_dir="path/to/masks_dir", output_dir="path/to/output_dir", classes=80)
```

此实用程序将二进制掩码图像转换为 YOLO 分割格式，并将其保存在指定的输出目录中。

## 常见问题

### Ultralytics YOLO 支持哪些实例分割数据集格式？

Ultralytics YOLO 支持多种实例分割数据集格式，主要格式是其自己的 Ultralytics YOLO 格式。数据集中的每张图像都需要一个对应的文本文件，其中目标信息分割成多行（每个目标一行），列出类别索引和归一化边界坐标。有关 YOLO 数据集格式的更详细说明，请访问[实例分割数据集概述](#实例分割数据集概述)。

### 如何将 COCO 数据集标注转换为 YOLO 格式？

使用 Ultralytics 工具将 COCO 格式标注转换为 YOLO 格式非常简单。您可以使用 `ultralytics.data.converter` 模块中的 `convert_coco` 函数：

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/coco/annotations/", use_segments=True)
```

此脚本将您的 COCO 数据集标注转换为所需的 YOLO 格式，使其适合训练您的 YOLO 模型。有关更多详细信息，请参阅[转换标签格式](#coco-数据集格式转-yolo-格式)。

### 如何为训练 Ultralytics YOLO 模型准备 YAML 文件？

要为使用 Ultralytics 训练 YOLO 模型准备 YAML 文件，您需要定义数据集路径和类别名称。以下是 YAML 配置示例：

```yaml
--8<-- "ultralytics/cfg/datasets/coco8-seg.yaml"
```

确保根据您的数据集更新路径和类别名称。有关更多信息，请查看[数据集 YAML 格式](#数据集-yaml-格式)部分。

### Ultralytics YOLO 中的自动标注功能是什么？

Ultralytics YOLO 中的自动标注允许您使用预训练的检测模型为数据集生成分割标注。这显著减少了手动标注的需求。您可以使用 `auto_annotate` 函数如下：

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam_b.pt")  # 或 sam_model="mobile_sam.pt"
```

此函数自动化标注过程，使其更快更高效。有关更多详细信息，请探索[自动标注参考](https://docs.ultralytics.com/reference/data/annotator/#ultralytics.data.annotator.auto_annotate)。
