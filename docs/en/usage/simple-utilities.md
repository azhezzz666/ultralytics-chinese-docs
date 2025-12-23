---
comments: true
description: 探索 Ultralytics 包中的基本实用工具，加速和增强您的工作流。了解数据处理、标注、转换等功能。
keywords: Ultralytics, 实用工具, 数据处理, 自动标注, YOLO, 数据集转换, 边界框, 图像压缩, 机器学习工具
---

# 简单实用工具

<p align="center">
  <img src="https://github.com/ultralytics/docs/releases/download/0/code-with-perspective.avif" alt="带透视的代码">
</p>

`ultralytics` 包提供各种实用工具来支持、增强和加速您的工作流。虽然还有更多可用的工具，但本指南重点介绍对开发者最有用的一些，作为使用 Ultralytics 工具进行编程的实用参考。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/1bPY2LRG590"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics 实用工具 | 自动标注、Explorer API 和数据集转换
</p>

## 数据

### 自动标注 / 标注

数据集标注是一个资源密集且耗时的过程。如果您有一个在合理数量数据上训练的 Ultralytics YOLO [目标检测](https://www.ultralytics.com/glossary/object-detection)模型，您可以将其与 [SAM](../models/sam.md) 一起使用，以分割格式自动标注额外数据。

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data="path/to/new/data",
    det_model="yolo11n.pt",
    sam_model="mobile_sam.pt",
    device="cuda",
    output_dir="path/to/save_labels",
)
```

此函数不返回任何值。更多详情：

- 请参阅 [`annotator.auto_annotate` 参考部分](../reference/data/annotator.md#ultralytics.data.annotator.auto_annotate)了解函数如何运作的更多信息。
- 与 [`segments2boxes` 函数](#将分割转换为边界框)结合使用，也可以生成目标检测边界框。

### 可视化数据集标注

此函数在训练前可视化图像上的 YOLO 标注，帮助识别和纠正可能导致错误检测结果的任何错误标注。它绘制边界框，用类别名称标记目标，并根据背景亮度调整文本颜色以提高可读性。

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

### 将分割掩码转换为 YOLO 格式

![分割掩码转 YOLO 格式](https://github.com/ultralytics/docs/releases/download/0/segmentation-masks-to-yolo-format.avif)

使用此功能将分割掩码图像数据集转换为 [Ultralytics YOLO](../models/yolo11.md) 分割格式。此函数获取包含二进制格式掩码图像的目录，并将它们转换为 YOLO 分割格式。

转换后的掩码将保存在指定的输出目录中。

```python
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# 这里的 classes 是数据集中的总类别数。
# 对于 COCO 数据集，我们有 80 个类别。
convert_segment_masks_to_yolo_seg(masks_dir="path/to/masks_dir", output_dir="path/to/output_dir", classes=80)
```

### 将 COCO 转换为 YOLO 格式

使用此功能将 [COCO](https://docs.ultralytics.com/datasets/detect/coco/) JSON 标注转换为 YOLO 格式。对于目标检测（边界框）数据集，将 `use_segments` 和 `use_keypoints` 都设置为 `False`。

```python
from ultralytics.data.converter import convert_coco

convert_coco(
    "coco/annotations/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
)
```

有关 `convert_coco` 函数的更多信息，请[访问参考页面](../reference/data/converter.md#ultralytics.data.converter.convert_coco)。

### 获取边界框尺寸

```python
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("yolo11n.pt")  # 加载预训练或微调模型

# 处理图像
source = cv2.imread("path/to/image.jpg")
results = model(source)

# 提取结果
annotator = Annotator(source, example=model.names)

for box in results[0].boxes.xyxy.cpu():
    width, height, area = annotator.get_bbox_dimension(box)
    print(f"边界框宽度 {width.item()}, 高度 {height.item()}, 面积 {area.item()}")
```

### 将边界框转换为分割

使用现有的 `x y w h` 边界框数据，使用 `yolo_bbox2segment` 函数转换为分割。按如下方式组织图像和标注的文件：

```
data
|__ images
    ├─ 001.jpg
    ├─ 002.jpg
    ├─ ..
    └─ NNN.jpg
|__ labels
    ├─ 001.txt
    ├─ 002.txt
    ├─ ..
    └─ NNN.txt
```

```python
from ultralytics.data.converter import yolo_bbox2segment

yolo_bbox2segment(
    im_dir="path/to/images",
    save_dir=None,  # 保存到 images 目录中的 "labels-segment"
    sam_model="sam_b.pt",
)
```

[访问 `yolo_bbox2segment` 参考页面](../reference/data/converter.md#ultralytics.data.converter.yolo_bbox2segment)了解有关该函数的更多信息。

### 将分割转换为边界框

如果您有使用[分割数据集格式](../datasets/segment/index.md)的数据集，您可以使用此函数轻松将其转换为直立（或水平）边界框（`x y w h` 格式）。

```python
import numpy as np

from ultralytics.utils.ops import segments2boxes

segments = np.array(
    [
        [805, 392, 797, 400, ..., 808, 714, 808, 392],
        [115, 398, 113, 400, ..., 150, 400, 149, 298],
        [267, 412, 265, 413, ..., 300, 413, 299, 412],
    ]
)

segments2boxes([s.reshape(-1, 2) for s in segments])
# >>> array([[ 741.66, 631.12, 133.31, 479.25],
#           [ 146.81, 649.69, 185.62, 502.88],
#           [ 281.81, 636.19, 118.12, 448.88]],
#           dtype=float32) # xywh 边界框
```

要了解此函数的工作原理，请访问[参考页面](../reference/utils/ops.md#ultralytics.utils.ops.segments2boxes)。

## 实用工具

### 图像压缩

将单个图像文件压缩到较小的尺寸，同时保持其宽高比和质量。如果输入图像小于最大尺寸，则不会调整大小。

```python
from pathlib import Path

from ultralytics.data.utils import compress_one_image

for f in Path("path/to/dataset").rglob("*.jpg"):
    compress_one_image(f)
```

### 自动划分数据集

自动将数据集划分为 `train`/`val`/`test` 集，并将结果划分保存到 `autosplit_*.txt` 文件中。此函数使用随机采样，在使用 [`fraction` 参数进行训练](../modes/train.md#train-settings)时会被排除。

```python
from ultralytics.data.split import autosplit

autosplit(
    path="path/to/images",
    weights=(0.9, 0.1, 0.0),  # (训练, 验证, 测试) 分数划分
    annotated_only=False,  # 当为 True 时仅划分有标注文件的图像
)
```

有关此函数的更多详情，请参阅[参考页面](../reference/data/split.md#ultralytics.data.split.autosplit)。

### 分割多边形转二进制掩码

将单个多边形（作为列表）转换为指定图像大小的二进制掩码。多边形应为 `[N, 2]` 形式，其中 `N` 是定义多边形轮廓的 `(x, y)` 点的数量。

!!! warning "警告"

    `N` **必须始终**为偶数。

```python
import numpy as np

from ultralytics.data.utils import polygon2mask

imgsz = (1080, 810)
polygon = np.array([805, 392, 797, 400, ..., 808, 714, 808, 392])  # (238, 2)

mask = polygon2mask(
    imgsz,  # 元组
    [polygon],  # 作为列表输入
    color=255,  # 8位二进制
    downsample_ratio=1,
)
```

## 边界框

### 边界框（水平）实例

要管理边界框数据，`Bboxes` 类帮助在框坐标格式之间转换、缩放框尺寸、计算面积、包含偏移等。

```python
import numpy as np

from ultralytics.utils.instance import Bboxes

boxes = Bboxes(
    bboxes=np.array(
        [
            [22.878, 231.27, 804.98, 756.83],
            [48.552, 398.56, 245.35, 902.71],
            [669.47, 392.19, 809.72, 877.04],
            [221.52, 405.8, 344.98, 857.54],
            [0, 550.53, 63.01, 873.44],
            [0.0584, 254.46, 32.561, 324.87],
        ]
    ),
    format="xyxy",
)

boxes.areas()
# >>> array([ 4.1104e+05,       99216,       68000,       55772,       20347,      2288.5])

boxes.convert("xywh")
print(boxes.bboxes)
# >>> array(
#     [[ 413.93, 494.05,  782.1, 525.56],
#      [ 146.95, 650.63,  196.8, 504.15],
#      [  739.6, 634.62, 140.25, 484.85],
#      [ 283.25, 631.67, 123.46, 451.74],
#      [ 31.505, 711.99,  63.01, 322.91],
#      [  16.31, 289.67, 32.503,  70.41]]
# )
```

有关更多属性和方法，请参阅 [`Bboxes` 参考部分](../reference/utils/instance.md#ultralytics.utils.instance.Bboxes)。

!!! tip "提示"

    以下许多函数（以及更多）可以使用 [`Bboxes` 类](#边界框水平实例)访问，但如果您更喜欢直接使用函数，请参阅下一小节了解如何独立导入它们。

### 缩放边界框

当放大或缩小图像时，您可以使用 `ultralytics.utils.ops.scale_boxes` 适当地缩放相应的边界框坐标以匹配。

```python
import cv2 as cv
import numpy as np

from ultralytics.utils.ops import scale_boxes

image = cv.imread("ultralytics/assets/bus.jpg")
h, w, c = image.shape
resized = cv.resize(image, None, (), fx=1.2, fy=1.2)
new_h, new_w, _ = resized.shape

xyxy_boxes = np.array(
    [
        [22.878, 231.27, 804.98, 756.83],
        [48.552, 398.56, 245.35, 902.71],
        [669.47, 392.19, 809.72, 877.04],
        [221.52, 405.8, 344.98, 857.54],
        [0, 550.53, 63.01, 873.44],
        [0.0584, 254.46, 32.561, 324.87],
    ]
)

new_boxes = scale_boxes(
    img1_shape=(h, w),  # 原始图像尺寸
    boxes=xyxy_boxes,  # 来自原始图像的框
    img0_shape=(new_h, new_w),  # 调整大小后的图像尺寸（缩放到）
    ratio_pad=None,
    padding=False,
    xywh=False,
)

print(new_boxes)
# >>> array(
#     [[  27.454,  277.52,  965.98,   908.2],
#     [   58.262,  478.27,  294.42,  1083.3],
#     [   803.36,  470.63,  971.66,  1052.4],
#     [   265.82,  486.96,  413.98,    1029],
#     [        0,  660.64,  75.612,  1048.1],
#     [   0.0701,  305.35,  39.073,  389.84]]
# )
```

### 边界框格式转换

#### XYXY → XYWH

将边界框坐标从 (x1, y1, x2, y2) 格式转换为 (x, y, width, height) 格式，其中 (x1, y1) 是左上角，(x2, y2) 是右下角。

```python
import numpy as np

from ultralytics.utils.ops import xyxy2xywh

xyxy_boxes = np.array(
    [
        [22.878, 231.27, 804.98, 756.83],
        [48.552, 398.56, 245.35, 902.71],
        [669.47, 392.19, 809.72, 877.04],
        [221.52, 405.8, 344.98, 857.54],
        [0, 550.53, 63.01, 873.44],
        [0.0584, 254.46, 32.561, 324.87],
    ]
)
xywh = xyxy2xywh(xyxy_boxes)

print(xywh)
# >>> array(
#     [[ 413.93,  494.05,   782.1, 525.56],
#     [  146.95,  650.63,   196.8, 504.15],
#     [   739.6,  634.62,  140.25, 484.85],
#     [  283.25,  631.67,  123.46, 451.74],
#     [  31.505,  711.99,   63.01, 322.91],
#     [   16.31,  289.67,  32.503,  70.41]]
# )
```

### 所有边界框转换

```python
from ultralytics.utils.ops import (
    ltwh2xywh,
    ltwh2xyxy,
    xywh2ltwh,  # xywh → 左上角, w, h
    xywh2xyxy,
    xywhn2xyxy,  # 归一化 → 像素
    xyxy2ltwh,  # xyxy → 左上角, w, h
    xyxy2xywhn,  # 像素 → 归一化
)

for func in (ltwh2xywh, ltwh2xyxy, xywh2ltwh, xywh2xyxy, xywhn2xyxy, xyxy2ltwh, xyxy2xywhn):
    print(help(func))  # 打印函数文档字符串
```

查看每个函数的文档字符串或访问 `ultralytics.utils.ops` [参考页面](../reference/utils/ops.md)了解更多。

## 绘图

### 标注实用工具

Ultralytics 包含一个 `Annotator` 类用于标注各种数据类型。它最适合与[目标检测边界框](../modes/predict.md#boxes)、[姿态关键点](../modes/predict.md#keypoints)和[旋转边界框](../modes/predict.md#obb)一起使用。

#### 框标注

!!! example "使用 Ultralytics YOLO 🚀 的 Python 示例"

    === "水平边界框"

        ```python
        import cv2 as cv
        import numpy as np

        from ultralytics.utils.plotting import Annotator, colors

        names = {
            0: "person",
            5: "bus",
            11: "stop sign",
        }

        image = cv.imread("ultralytics/assets/bus.jpg")
        ann = Annotator(
            image,
            line_width=None,  # 默认自动大小
            font_size=None,  # 默认自动大小
            font="Arial.ttf",  # 必须与 ImageFont 兼容
            pil=False,  # 使用 PIL，否则使用 OpenCV
        )

        xyxy_boxes = np.array(
            [
                [5, 22.878, 231.27, 804.98, 756.83],  # 类别索引 x1 y1 x2 y2
                [0, 48.552, 398.56, 245.35, 902.71],
                [0, 669.47, 392.19, 809.72, 877.04],
                [0, 221.52, 405.8, 344.98, 857.54],
                [0, 0, 550.53, 63.01, 873.44],
                [11, 0.0584, 254.46, 32.561, 324.87],
            ]
        )

        for nb, box in enumerate(xyxy_boxes):
            c_idx, *box = box
            label = f"{str(nb).zfill(2)}:{names.get(int(c_idx))}"
            ann.box_label(box, label, color=colors(c_idx, bgr=True))

        image_with_bboxes = ann.result()
        ```

    === "旋转边界框 (OBB)"

        ```python
        import cv2 as cv
        import numpy as np

        from ultralytics.utils.plotting import Annotator, colors

        obb_names = {10: "small vehicle"}
        obb_image = cv.imread("datasets/dota8/images/train/P1142__1024__0___824.jpg")
        obb_boxes = np.array(
            [
                [0, 635, 560, 919, 719, 1087, 420, 803, 261],  # 类别索引 x1 y1 x2 y2 x3 y2 x4 y4
                [0, 331, 19, 493, 260, 776, 70, 613, -171],
                [9, 869, 161, 886, 147, 851, 101, 833, 115],
            ]
        )
        ann = Annotator(
            obb_image,
            line_width=None,  # 默认自动大小
            font_size=None,  # 默认自动大小
            font="Arial.ttf",  # 必须与 ImageFont 兼容
            pil=False,  # 使用 PIL，否则使用 OpenCV
        )
        for obb in obb_boxes:
            c_idx, *obb = obb
            obb = np.array(obb).reshape(-1, 4, 2).squeeze()
            label = f"{obb_names.get(int(c_idx))}"
            ann.box_label(
                obb,
                label,
                color=colors(c_idx, True),
            )

        image_with_obb = ann.result()
        ```

当[处理检测结果](../modes/predict.md#working-with-results)时，可以使用 `model.names` 中的名称。
另请参阅 [`Annotator` 参考页面](../reference/utils/plotting.md/#ultralytics.utils.plotting.Annotator)了解更多信息。

#### 自适应标签标注

!!! warning "警告"

    从 **Ultralytics v8.3.167** 开始，`circle_label` 和 `text_label` 已被统一的 `adaptive_label` 函数替代。您现在可以使用 `shape` 参数指定标注类型：

    * **矩形**: `annotator.adaptive_label(box, label=names[int(cls)], color=colors(cls, True), shape="rect")`
    * **圆形**: `annotator.adaptive_label(box, label=names[int(cls)], color=colors(cls, True), shape="circle")`

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/c-S5M36XWmg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Python 实时演示深入指南文本和圆形标注 | Ultralytics 标注 🚀
</p>

有关更多信息，请参阅 [`SolutionAnnotator` 参考页面](../reference/solutions/solutions.md/#ultralytics.solutions.solutions.SolutionAnnotator.adaptive_label)。

## 杂项

### 代码性能分析

使用 `with` 或作为装饰器检查代码运行/处理的持续时间。

```python
from ultralytics.utils.ops import Profile

with Profile(device="cuda:0") as dt:
    pass  # 要测量的操作

print(dt)
# >>> "Elapsed time is 9.5367431640625e-07 s"
```

### Ultralytics 支持的格式

需要以编程方式使用 Ultralytics 支持的[图像或视频格式](../modes/predict.md#image-and-video-formats)？如果需要，请使用这些常量：

```python
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

print(IMG_FORMATS)
# {'tiff', 'pfm', 'bmp', 'mpo', 'dng', 'jpeg', 'png', 'webp', 'tif', 'jpg'}

print(VID_FORMATS)
# {'avi', 'mpg', 'wmv', 'mpeg', 'm4v', 'mov', 'mp4', 'asf', 'mkv', 'ts', 'gif', 'webm'}
```

### 使可整除

计算最接近 `x` 且能被 `y` 整除的整数。

```python
from ultralytics.utils.ops import make_divisible

make_divisible(7, 3)
# >>> 9
make_divisible(7, 2)
# >>> 8
```

## 常见问题

### Ultralytics 包中包含哪些实用工具来增强机器学习工作流？

Ultralytics 包包含旨在简化和优化机器学习工作流的实用工具。关键实用工具包括用于标注数据集的[自动标注](../reference/data/annotator.md#ultralytics.data.annotator.auto_annotate)、使用 [convert_coco](../reference/data/converter.md#ultralytics.data.converter.convert_coco) 将 [COCO](https://docs.ultralytics.com/datasets/detect/coco/) 转换为 YOLO 格式、压缩图像和数据集自动划分。这些工具减少了手动工作，确保一致性，并提高了数据处理效率。

### 如何使用 Ultralytics 自动标注我的数据集？

如果您有一个预训练的 Ultralytics YOLO 目标检测模型，您可以将其与 [SAM](../models/sam.md) 模型一起使用，以分割格式自动标注您的数据集。以下是示例：

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data="path/to/new/data",
    det_model="yolo11n.pt",
    sam_model="mobile_sam.pt",
    device="cuda",
    output_dir="path/to/save_labels",
)
```

更多详情请查看 [auto_annotate 参考部分](../reference/data/annotator.md#ultralytics.data.annotator.auto_annotate)。

### 如何在 Ultralytics 中将 COCO 数据集标注转换为 YOLO 格式？

要将 COCO JSON 标注转换为 YOLO 格式用于目标检测，您可以使用 `convert_coco` 实用工具。以下是示例代码片段：

```python
from ultralytics.data.converter import convert_coco

convert_coco(
    "coco/annotations/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
)
```

更多信息请访问 [convert_coco 参考页面](../reference/data/converter.md#ultralytics.data.converter.convert_coco)。

### Ultralytics 包中 YOLO Data Explorer 的用途是什么？

[YOLO Explorer](../datasets/explorer/index.md) 是 `8.1.0` 更新中引入的强大工具，用于增强数据集理解。它允许您使用文本查询在数据集中查找目标实例，使分析和管理数据更加容易。此工具提供有关数据集组成和分布的宝贵见解，有助于改进模型训练和性能。

### 如何在 Ultralytics 中将边界框转换为分割？

要将现有的边界框数据（`x y w h` 格式）转换为分割，您可以使用 `yolo_bbox2segment` 函数。确保您的文件按图像和标签的单独目录组织。

```python
from ultralytics.data.converter import yolo_bbox2segment

yolo_bbox2segment(
    im_dir="path/to/images",
    save_dir=None,  # 保存到 images 目录中的 "labels-segment"
    sam_model="sam_b.pt",
)
```

更多信息请访问 [yolo_bbox2segment 参考页面](../reference/data/converter.md#ultralytics.data.converter.yolo_bbox2segment)。
