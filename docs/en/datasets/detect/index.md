---
comments: true
description: 了解与 Ultralytics YOLO 兼容的数据集格式，用于鲁棒的目标检测。探索支持的数据集并学习如何转换格式。
keywords: Ultralytics, YOLO, 目标检测数据集, 数据集格式, COCO, 数据集转换, 训练数据集
---

# 目标检测数据集概述

训练一个鲁棒且准确的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型需要一个全面的数据集。本指南介绍了与 Ultralytics YOLO 模型兼容的各种数据集格式，并提供了关于其结构、使用方法以及如何在不同格式之间转换的见解。

## 支持的数据集格式

### Ultralytics YOLO 格式

Ultralytics YOLO 格式是一种数据集配置格式，允许您定义数据集根目录、训练/验证/测试图像目录的相对路径或包含图像路径的 `*.txt` 文件，以及类名字典。以下是一个示例：

!!! example "ultralytics/cfg/datasets/coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

此格式的标签应导出为 YOLO 格式，每张图像一个 `*.txt` 文件。如果图像中没有目标，则不需要 `*.txt` 文件。`*.txt` 文件应按 `class x_center y_center width height` 格式每行一个目标进行格式化。边界框坐标必须采用**归一化的 xywh** 格式（从 0 到 1）。如果您的边界框以像素为单位，则应将 `x_center` 和 `width` 除以图像宽度，将 `y_center` 和 `height` 除以图像高度。类别编号应从零开始索引（从 0 开始）。

<p align="center"><img width="750" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie.avif" alt="标注图像示例"></p>

上图对应的标签文件包含 2 个人（类别 `0`）和一条领带（类别 `27`）：

<p align="center"><img width="428" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie-1.avif" alt="标签文件示例"></p>

使用 Ultralytics YOLO 格式时，请按照下面 [COCO8 数据集](coco8.md)示例所示组织您的训练和验证图像及标签。

<p align="center"><img width="800" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie-2.avif" alt="数据集目录结构示例"></p>

#### 使用示例

以下是如何使用 YOLO 格式数据集训练模型：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

### Ultralytics NDJSON 格式

NDJSON（换行符分隔的 JSON）格式提供了一种为 Ultralytics YOLO11 模型定义数据集的替代方式。此格式将数据集元数据和标注存储在单个文件中，其中每行包含一个单独的 JSON 对象。

NDJSON 数据集文件包含：

1. **数据集记录**（第一行）：包含数据集元数据，包括任务类型、类名和一般信息
2. **图像记录**（后续行）：包含单个图像数据，包括尺寸、标注和文件路径

!!! example "NDJSON 示例"

    === "数据集记录（第 1 行）"

        ```json
        {
            "type": "dataset",
            "task": "detect",
            "name": "Example",
            "description": "COCO NDJSON 示例数据集",
            "url": "https://app.ultralytics.com/user/datasets/example",
            "class_names": { "0": "person", "1": "bicycle", "2": "car" },
            "bytes": 426342,
            "version": 0,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z"
        }
        ```

    === "图像记录（第 2 行及以后）"

        ```json
        {
            "type": "image",
            "file": "image1.jpg",
            "url": "https://www.url.com/path/to/image1.jpg",
            "width": 640,
            "height": 480,
            "split": "train",
            "annotations": {
                "boxes": [
                    [0, 0.52481, 0.37629, 0.28394, 0.41832],
                    [1, 0.73526, 0.29847, 0.19275, 0.33691]
                ]
            }
        }
        ```

**按任务划分的标注格式：**

- **检测：** `"annotations": {"boxes": [[class_id, x_center, y_center, width, height], ...]}`
- **分割：** `"annotations": {"segments": [[class_id, x1, y1, x2, y2, ...], ...]}`
- **姿态：** `"annotations": {"pose": [[class_id, x1, y1, v1, x2, y2, v2, ...], ...]}`
- **OBB：** `"annotations": {"obb": [[class_id, x_center, y_center, width, height, angle], ...]}`
- **分类：** `"annotations": {"classification": [class_id]}`

#### 使用示例

要将 NDJSON 数据集与 YOLO11 一起使用，只需指定 `.ndjson` 文件的路径：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")

        # 使用 NDJSON 数据集训练
        results = model.train(data="path/to/dataset.ndjson", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 使用 NDJSON 数据集开始训练
        yolo detect train data=path/to/dataset.ndjson model=yolo11n.pt epochs=100 imgsz=640
        ```

#### NDJSON 格式的优势

- **单文件**：所有数据集信息包含在一个文件中
- **流式处理**：可以逐行处理大型数据集，无需将所有内容加载到内存中
- **云集成**：支持远程图像 URL，用于基于云的训练
- **可扩展**：易于添加自定义元数据字段
- **版本控制**：单文件格式与 git 和版本控制系统配合良好

## 支持的数据集

以下是支持的数据集列表及每个数据集的简要描述：

- [African-wildlife](african-wildlife.md)：包含非洲野生动物图像的数据集，包括水牛、大象、犀牛和斑马。
- [Argoverse](argoverse.md)：包含来自城市环境的 3D 跟踪和运动预测数据的数据集，具有丰富的标注。
- [Brain-tumor](brain-tumor.md)：用于检测脑肿瘤的数据集，包含 MRI 或 CT 扫描图像，详细说明肿瘤的存在、位置和特征。
- [COCO](coco.md)：Common Objects in Context（COCO）是一个大规模[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和图像描述数据集，包含 80 个目标类别。
- [COCO8](coco8.md)：COCO 训练集和验证集前 4 张图像的较小子集，适合快速测试。
- [COCO8-Grayscale](coco8-grayscale.md)：通过将 RGB 转换为灰度创建的 COCO8 灰度版本，适用于单通道模型评估。
- [COCO8-Multispectral](coco8-multispectral.md)：通过插值 RGB 波长创建的 COCO8 10 通道多光谱版本，适用于光谱感知模型评估。
- [COCO128](coco128.md)：COCO 训练集和验证集前 128 张图像的较小子集，适合测试。
- [Construction-PPE](construction-ppe.md)：包含建筑工地工人的数据集，标注了安全帽、背心、手套、靴子和护目镜等安全装备，包括缺失装备标注如 no_helmet、no_googles，用于真实世界的合规监控。
- [Global Wheat 2020](globalwheat2020.md)：包含 2020 年全球小麦挑战赛小麦穗图像的数据集。
- [HomeObjects-3K](homeobjects-3k.md)：包含床、椅子、电视等室内家居物品的数据集，非常适合智能家居自动化、机器人、增强现实和房间布局分析等应用。
- [KITTI](kitti.md)：包含真实世界驾驶场景的数据集，具有立体、LiDAR 和 GPS/IMU 数据，此处用于**2D 目标检测**任务，如在城市、农村和高速公路环境中识别汽车、行人和骑自行车的人。
- [LVIS](lvis.md)：包含 1203 个目标类别的大规模目标检测、分割和图像描述数据集。
- [Medical-pills](medical-pills.md)：包含药丸图像的数据集，标注用于制药质量保证、药丸分类和法规合规等应用。
- [Objects365](objects365.md)：包含 365 个目标类别和超过 60 万张标注图像的高质量大规模目标检测数据集。
- [OpenImagesV7](open-images-v7.md)：Google 提供的综合数据集，包含 170 万张训练图像和 4.2 万张验证图像。
- [Roboflow 100](roboflow-100.md)：包含 100 个数据集的多样化目标检测基准，涵盖七个图像领域，用于全面的模型评估。
- [Signature](signature.md)：包含各种文档图像的数据集，标注了签名，支持文档验证和欺诈检测研究。
- [SKU-110K](sku-110k.md)：包含零售环境中密集目标检测的数据集，超过 1.1 万张图像和 170 万个[边界框](https://www.ultralytics.com/glossary/bounding-box)。
- [VisDrone](visdrone.md)：包含无人机拍摄图像的目标检测和多目标跟踪数据的数据集，超过 1 万张图像和视频序列。
- [VOC](voc.md)：Pascal Visual Object Classes（VOC）目标检测和分割数据集，包含 20 个目标类别和超过 1.1 万张图像。
- [xView](xview.md)：用于航拍图像目标检测的数据集，包含 60 个目标类别和超过 100 万个标注目标。

### 添加您自己的数据集

如果您有自己的数据集并希望使用它来训练 Ultralytics YOLO 格式的检测模型，请确保它遵循上面"Ultralytics YOLO 格式"中指定的格式。将您的标注转换为所需格式，并在 YAML 配置文件中指定路径、类别数量和类名。


## 转换标签格式

### COCO 数据集格式转 YOLO 格式

您可以使用以下代码片段轻松地将流行的 [COCO 数据集](coco.md)格式的标签转换为 YOLO 格式：

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/")
        ```

此转换工具可用于将 COCO 数据集或任何 COCO 格式的数据集转换为 Ultralytics YOLO 格式。该过程将基于 JSON 的 COCO 标注转换为更简单的基于文本的 YOLO 格式，使其与 [Ultralytics YOLO 模型](../../models/yolo11.md)兼容。

请务必仔细检查您要使用的数据集是否与您的模型兼容，并遵循必要的格式约定。正确格式化的数据集对于训练成功的目标检测模型至关重要。

## 常见问题

### Ultralytics YOLO 数据集格式是什么，如何构建？

Ultralytics YOLO 格式是一种用于在训练项目中定义数据集的结构化配置。它涉及设置训练、验证和测试图像及相应标签的路径。例如：

```yaml
--8<-- "ultralytics/cfg/datasets/coco8.yaml"
```

标签保存在 `*.txt` 文件中，每张图像一个文件，格式为 `class x_center y_center width height`，使用归一化坐标。详细指南请参阅 [COCO8 数据集示例](coco8.md)。

### 如何将 COCO 数据集转换为 YOLO 格式？

您可以使用 [Ultralytics 转换工具](../../reference/data/converter.md)将 COCO 数据集转换为 YOLO 格式。以下是一个快速方法：

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/coco/annotations/")
```

此代码将把您的 COCO 标注转换为 YOLO 格式，实现与 Ultralytics YOLO 模型的无缝集成。更多详情请访问[转换标签格式](#转换标签格式)部分。

### Ultralytics YOLO 支持哪些目标检测数据集？

Ultralytics YOLO 支持广泛的数据集，包括：

- [Argoverse](argoverse.md)
- [COCO](coco.md)
- [LVIS](lvis.md)
- [COCO8](coco8.md)
- [Global Wheat 2020](globalwheat2020.md)
- [Objects365](objects365.md)
- [OpenImagesV7](open-images-v7.md)

每个数据集页面都提供了针对高效 YOLO11 训练的结构和使用详细信息。在[支持的数据集](#支持的数据集)部分探索完整列表。

### 如何使用我的数据集开始训练 YOLO11 模型？

要开始训练 YOLO11 模型，请确保您的数据集格式正确，并在 YAML 文件中定义路径。使用以下脚本开始训练：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # 加载预训练模型
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=path/to/your_dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关使用不同模式（包括 CLI 命令）的更多详情，请参阅[使用示例](#使用示例)部分。

### 在哪里可以找到使用 Ultralytics YOLO 进行目标检测的实际示例？

Ultralytics 提供了大量使用 YOLO11 进行各种应用的示例和实用指南。如需全面概述，请访问 [Ultralytics 博客](https://www.ultralytics.com/blog)，您可以在那里找到案例研究、详细教程和社区故事，展示使用 YOLO11 进行目标检测、分割等。有关具体示例，请查看文档中的[使用](../../modes/predict.md)部分。
