---
comments: true
description: 探索 Google 提供的综合 Open Images V7 数据集。了解其标注、应用，并使用 YOLO11 预训练模型进行计算机视觉任务。
keywords: Open Images V7, Google 数据集, 计算机视觉, YOLO11 模型, 目标检测, 图像分割, 视觉关系, AI 研究, Ultralytics
---

# Open Images V7 数据集

[Open Images V7](https://storage.googleapis.com/openimages/web/index.html) 是 Google 推出的一个多功能且广泛的数据集。旨在推动[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域的研究，它拥有大量标注了丰富数据的图像集合，包括图像级标签、目标边界框、目标分割掩码、视觉关系和局部叙述。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/u3pLlgzUeV8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 OpenImagesV7 预训练模型进行<a href="https://www.ultralytics.com/glossary/object-detection">目标检测</a>
</p>

## Open Images V7 预训练模型

| 模型 | 尺寸<br><sup>(像素)</sup> | mAP<sup>val<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| --- | --- | --- | --- | --- | --- | --- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-oiv7.pt) | 640 | 18.4 | 142.4 | 1.21 | 3.5 | 10.5 |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-oiv7.pt) | 640 | 27.7 | 183.1 | 1.40 | 11.4 | 29.7 |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-oiv7.pt) | 640 | 33.6 | 408.5 | 2.26 | 26.2 | 80.6 |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-oiv7.pt) | 640 | 34.9 | 596.9 | 2.43 | 44.1 | 167.4 |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-oiv7.pt) | 640 | 36.3 | 860.6 | 3.56 | 68.7 | 260.6 |

您可以按以下方式使用这些预训练模型进行推理或微调。

!!! example "预训练模型使用示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 Open Images Dataset V7 预训练的 YOLOv8n 模型
        model = YOLO("yolov8n-oiv7.pt")

        # 运行预测
        results = model.predict(source="image.jpg")

        # 从预训练检查点开始训练
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 使用 Open Images Dataset V7 预训练模型进行预测
        yolo detect predict source=image.jpg model=yolov8n-oiv7.pt

        # 从 Open Images Dataset V7 预训练检查点开始训练
        yolo detect train data=coco8.yaml model=yolov8n-oiv7.pt epochs=100 imgsz=640
        ```

![Open Images V7 类别可视化](https://github.com/ultralytics/docs/releases/download/0/open-images-v7-classes-visual.avif)


## 主要特点

- 包含约 900 万张图像，以各种方式标注以适应多种计算机视觉任务。
- 拥有惊人的 1600 万个边界框，涵盖 190 万张图像中的 600 个目标类别。这些边界框主要由专家手工绘制，确保高[精确率](https://www.ultralytics.com/glossary/precision)。
- 提供 330 万个视觉关系标注，详细说明 1,466 个独特的关系三元组、目标属性和人类活动。
- V5 引入了 350 个类别中 280 万个目标的分割掩码。
- V6 引入了 67.5 万个局部叙述，结合语音、文本和鼠标轨迹来突出描述的目标。
- V7 引入了 140 万张图像上的 6640 万个点级标签，涵盖 5,827 个类别。
- 包含 20,638 个类别中的 6140 万个图像级标签。
- 为[图像分类](https://www.ultralytics.com/glossary/image-classification)、目标检测、关系检测、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)和多模态图像描述提供统一平台。

## 数据集结构

Open Images V7 由多个组件构成，以满足各种计算机视觉挑战：

- **图像**：约 900 万张图像，通常展示复杂场景，平均每张图像有 8.3 个目标。
- **边界框**：超过 1600 万个边界框，划分 600 个类别的目标。
- **分割掩码**：详细说明 350 个类别中 280 万个目标的精确边界。
- **视觉关系**：330 万个标注，指示目标关系、属性和动作。
- **局部叙述**：67.5 万个描述，结合语音、文本和鼠标轨迹。
- **点级标签**：140 万张图像上的 6640 万个标签，适用于零样本/少样本[语义分割](https://www.ultralytics.com/glossary/semantic-segmentation)。

## 应用

Open Images V7 是训练和评估各种计算机视觉任务中最先进模型的基石。该数据集的广泛范围和高质量标注使其对专注于[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)的研究人员和开发者不可或缺。

一些关键应用包括：

- **高级目标检测**：训练模型以高准确率识别和定位复杂场景中的多个目标。
- **语义理解**：开发理解目标之间视觉关系的系统。
- **图像分割**：为目标创建精确的像素级掩码，实现详细的场景分析。
- **多模态学习**：将视觉数据与文本描述结合，实现更丰富的 AI 理解。
- **零样本学习**：利用广泛的类别覆盖来识别训练期间未见过的目标。

## 数据集 YAML

Ultralytics 维护一个 `open-images-v7.yaml` 文件，指定训练所需的数据集路径、类名和其他配置详情。

!!! example "OpenImagesV7.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/open-images-v7.yaml"
    ```

## 使用方法

要在 Open Images V7 数据集上训练 YOLO11n 模型 100 个[训练周期](https://www.ultralytics.com/glossary/epoch)，图像尺寸为 640，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! warning

    完整的 Open Images V7 数据集包含 1,743,042 张训练图像和 41,620 张验证图像，下载后需要约 **561 GB 的存储空间**。

    执行以下命令将在本地不存在时自动触发完整数据集的下载。在运行以下示例之前，请务必：

    - 验证您的设备有足够的存储容量。
    - 确保有稳定且快速的互联网连接。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 在 Open Images V7 数据集上训练模型
        results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 在 Open Images V7 数据集上训练 COCO 预训练的 YOLO11n 模型
        yolo detect train data=open-images-v7.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例数据和标注

数据集的图示有助于深入了解其丰富性：

![数据集示例图像](https://github.com/ultralytics/docs/releases/download/0/oidv7-all-in-one-example-ab.avif)

- **Open Images V7**：此图像展示了可用标注的深度和细节，包括边界框、关系和分割掩码。

研究人员可以从数据集解决的各种计算机视觉挑战中获得宝贵见解，从基本目标检测到复杂的关系识别。[标注的多样性](https://docs.ultralytics.com/datasets/explorer/)使 Open Images V7 对于开发能够理解复杂视觉场景的模型特别有价值。

## 引用和致谢

对于在工作中使用 Open Images V7 的人，请引用相关论文并致谢创建者：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{OpenImages,
          author = {Alina Kuznetsova and Hassan Rom and Neil Alldrin and Jasper Uijlings and Ivan Krasin and Jordi Pont-Tuset and Shahab Kamali and Stefan Popov and Matteo Malloci and Alexander Kolesnikov and Tom Duerig and Vittorio Ferrari},
          title = {The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale},
          year = {2020},
          journal = {IJCV}
        }
        ```

衷心感谢 Google AI 团队创建和维护 Open Images V7 数据集。要深入了解数据集及其内容，请访问[官方 Open Images V7 网站](https://storage.googleapis.com/openimages/web/index.html)。

## 常见问题

### 什么是 Open Images V7 数据集？

Open Images V7 是 Google 创建的一个广泛且多功能的数据集，旨在推进计算机视觉研究。它包括图像级标签、目标边界框、目标分割掩码、视觉关系和局部叙述，使其非常适合各种计算机视觉任务，如目标检测、分割和关系检测。

### 如何在 Open Images V7 数据集上训练 YOLO11 模型？

要在 Open Images V7 数据集上训练 YOLO11 模型，您可以使用 Python 和 CLI 命令。以下是训练 YOLO11n 模型 100 个训练周期、图像尺寸为 640 的示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 在 Open Images V7 数据集上训练模型
        results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 在 Open Images V7 数据集上训练 COCO 预训练的 YOLO11n 模型
        yolo detect train data=open-images-v7.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

有关参数和设置的详情，请参阅[训练](../../modes/train.md)页面。

### Open Images V7 数据集有哪些主要特点？

Open Images V7 数据集包含约 900 万张图像，具有各种标注：

- **边界框**：600 个目标类别中的 1600 万个边界框。
- **分割掩码**：350 个类别中 280 万个目标的掩码。
- **视觉关系**：330 万个标注，指示关系、属性和动作。
- **局部叙述**：67.5 万个描述，结合语音、文本和鼠标轨迹。
- **点级标签**：140 万张图像上的 6640 万个标签。
- **图像级标签**：20,638 个类别中的 6140 万个标签。

### Open Images V7 数据集有哪些可用的预训练模型？

Ultralytics 为 Open Images V7 数据集提供了几个 YOLOv8 预训练模型，每个模型具有不同的大小和性能指标：

| 模型 | 尺寸<br><sup>(像素)</sup> | mAP<sup>val<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| --- | --- | --- | --- | --- | --- | --- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-oiv7.pt) | 640 | 18.4 | 142.4 | 1.21 | 3.5 | 10.5 |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-oiv7.pt) | 640 | 27.7 | 183.1 | 1.40 | 11.4 | 29.7 |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-oiv7.pt) | 640 | 33.6 | 408.5 | 2.26 | 26.2 | 80.6 |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-oiv7.pt) | 640 | 34.9 | 596.9 | 2.43 | 44.1 | 167.4 |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-oiv7.pt) | 640 | 36.3 | 860.6 | 3.56 | 68.7 | 260.6 |

### Open Images V7 数据集可以用于哪些应用？

Open Images V7 数据集支持各种计算机视觉任务，包括：

- **[图像分类](https://www.ultralytics.com/glossary/image-classification)**
- **目标检测**
- **实例分割**
- **视觉关系检测**
- **多模态图像描述**

其全面的标注和广泛的范围使其适合训练和评估高级[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型，如我们[应用](#应用)部分中详细介绍的实际用例所示。
