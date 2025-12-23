---
comments: true
description: 探索 Ultralytics YOLOv8，这是实时目标检测的进步，通过一系列预训练模型优化性能以适应各种任务。
keywords: YOLOv8, 实时目标检测, YOLO 系列, Ultralytics, 计算机视觉, 高级目标检测, AI, 机器学习, 深度学习
---

# 探索 Ultralytics YOLOv8

## 概述

YOLOv8 由 Ultralytics 于 2023 年 1 月 10 日发布，在精度和速度方面提供了尖端性能。基于之前 YOLO 版本的进步，YOLOv8 引入了新功能和优化，使其成为各种[目标检测](https://www.ultralytics.com/blog/a-guide-to-deep-dive-into-object-detection-in-2025)任务在广泛应用中的理想选择。

![Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/yolov8-comparison-plots.avif)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Na0HvJ4hkk0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics YOLOv8 模型概述
</p>

## YOLOv8 的关键特性

- **先进的主干网络和颈部架构：** YOLOv8 采用最先进的主干网络和颈部架构，从而改进了[特征提取](https://www.ultralytics.com/glossary/feature-extraction)和[目标检测](https://www.ultralytics.com/glossary/object-detection)性能。
- **无锚点分离式 Ultralytics 头：** YOLOv8 采用无锚点分离式 Ultralytics 头，与基于锚点的方法相比，有助于提高精度和更高效的检测过程。
- **优化的精度-速度权衡：** YOLOv8 专注于保持精度和速度之间的最佳平衡，适用于各种应用领域的实时目标检测任务。
- **多种预训练模型：** YOLOv8 提供一系列预训练模型以满足各种任务和性能要求，使您更容易找到适合特定用例的正确模型。

## 支持的任务和模式

YOLOv8 系列提供多种模型，每种模型都专门用于计算机视觉中的特定任务。这些模型旨在满足各种需求，从目标检测到更复杂的任务，如[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)、姿态/关键点检测、旋转目标检测和分类。

YOLOv8 系列的每个变体都针对其各自的任务进行了优化，确保高性能和精度。此外，这些模型与各种操作模式兼容，包括[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)，便于在部署和开发的不同阶段使用。

| 模型        | 文件名                                                                                                         | 任务                                         | 推理 | 验证 | 训练 | 导出 |
| ----------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ---- | ---- | ---- | ---- |
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [检测](../tasks/detect.md)                   | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [实例分割](../tasks/segment.md)              | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [姿态/关键点](../tasks/pose.md)              | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-obb  | `yolov8n-obb.pt` `yolov8s-obb.pt` `yolov8m-obb.pt` `yolov8l-obb.pt` `yolov8x-obb.pt`                           | [旋转检测](../tasks/obb.md)                  | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [分类](../tasks/classify.md)                 | ✅   | ✅   | ✅   | ✅   |

此表概述了 YOLOv8 模型变体，突出了它们在特定任务中的适用性以及与各种操作模式（如推理、验证、训练和导出）的兼容性。它展示了 YOLOv8 系列的多功能性和稳健性，使其适用于[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)中的各种应用。

## 性能指标

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv8"]'></canvas>

!!! tip "性能"

    === "检测 (COCO)"

        有关在 [COCO](../datasets/detect/coco.md) 上训练的这些模型的使用示例，请参阅[检测文档](../tasks/detect.md)，其中包含 80 个预训练类别。

        | 模型                                                                                 | 尺寸<br><sup>(像素)</sup> | mAP<sup>val<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "检测 (Open Images V7)"

        有关在 [Open Image V7](../datasets/detect/open-images-v7.md) 上训练的这些模型的使用示例，请参阅[检测文档](../tasks/detect.md)，其中包含 600 个预训练类别。

        | 模型                                                                                     | 尺寸<br><sup>(像素)</sup> | mAP<sup>val<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "分割 (COCO)"

        有关在 [COCO](../datasets/segment/coco.md) 上训练的这些模型的使用示例，请参阅[分割文档](../tasks/segment.md)，其中包含 80 个预训练类别。

        | 模型                                                                                        | 尺寸<br><sup>(像素)</sup> | mAP<sup>box<br>50-95</sup> | mAP<sup>mask<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "分类 (ImageNet)"

        有关在 [ImageNet](../datasets/classify/imagenet.md) 上训练的这些模型的使用示例，请参阅[分类文档](../tasks/classify.md)，其中包含 1000 个预训练类别。

        | 模型                                                                                        | 尺寸<br><sup>(像素)</sup> | acc<br><sup>top1</sup> | acc<br><sup>top5</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(B) at 224</sup> |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt) | 224                   | 69.0             | 88.3             | 12.9                           | 0.31                                | 2.7                | 0.5                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-cls.pt) | 224                   | 73.8             | 91.7             | 23.4                           | 0.35                                | 6.4                | 1.7                      |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-cls.pt) | 224                   | 76.8             | 93.5             | 85.4                           | 0.62                                | 17.0               | 5.3                      |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-cls.pt) | 224                   | 76.8             | 93.5             | 163.0                          | 0.87                                | 37.5               | 12.3                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-cls.pt) | 224                   | 79.0             | 94.6             | 232.0                          | 1.01                                | 57.4               | 19.0                     |

    === "姿态 (COCO)"

        有关在 [COCO](../datasets/pose/coco.md) 上训练的这些模型的使用示例，请参阅[姿态估计文档](../tasks/pose.md)，其中包含 1 个预训练类别"person"。

        | 模型                                                                                                | 尺寸<br><sup>(像素)</sup> | mAP<sup>pose<br>50-95</sup> | mAP<sup>pose<br>50</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数量<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

    === "OBB (DOTAv1)"

        有关在 [DOTAv1](../datasets/obb/dota-v2.md#dota-v10) 上训练的这些模型的使用示例，请参阅[旋转检测文档](../tasks/obb.md)，其中包含 15 个预训练类别。

        | 模型                                                                                        | 尺寸<br><sup>(像素)</sup> | mAP<sup>test<br>50</sup>   | 速度<br><sup>CPU ONNX<br>(ms)</sup>   | 速度<br><sup>A100 TensorRT<br>(ms)</sup>   | 参数量<br><sup>(M)</sup>   | FLOPs<br><sup>(B)</sup> |
        |----------------------------------------------------------------------------------------------|-----------------------| -------------------- | -------------------------------- | ------------------------------------- | -------------------- | ----------------- |
        | [YOLOv8n-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt) | 1024                  | 78.0                 | 204.77                           | 3.57                                  | 3.1                  | 23.3              |
        | [YOLOv8s-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-obb.pt) | 1024                  | 79.5                 | 424.88                           | 4.07                                  | 11.4                 | 76.3              |
        | [YOLOv8m-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-obb.pt) | 1024                  | 80.5                 | 763.48                           | 7.61                                  | 26.4                 | 208.6             |
        | [YOLOv8l-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-obb.pt) | 1024                  | 80.7                 | 1278.42                          | 11.83                                 | 44.5                 | 433.8             |
        | [YOLOv8x-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-obb.pt) | 1024                  | 81.36                | 1759.10                          | 13.23                                 | 69.5                 | 676.7             |

## YOLOv8 使用示例

此示例提供简单的 YOLOv8 训练和推理示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参阅[预测](../modes/predict.md)、[训练](../modes/train.md)、[验证](../modes/val.md)和[导出](../modes/export.md)文档页面。

请注意，以下示例适用于用于目标检测的 YOLOv8 [检测](../tasks/detect.md)模型。有关其他支持的任务，请参阅[分割](../tasks/segment.md)、[分类](../tasks/classify.md)、[OBB](../tasks/obb.md)文档和[姿态](../tasks/pose.md)文档。

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) 预训练的 `*.pt` 模型以及配置 `*.yaml` 文件可以传递给 `YOLO()` 类以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLOv8n 模型
        model = YOLO("yolov8n.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个轮次
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLOv8n 模型对 'bus.jpg' 图像运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI 命令可直接运行模型：

        ```bash
        # 加载 COCO 预训练的 YOLOv8n 模型并在 COCO8 示例数据集上训练 100 个轮次
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载 COCO 预训练的 YOLOv8n 模型并对 'bus.jpg' 图像运行推理
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 引用和致谢

!!! tip "Ultralytics YOLOv8 出版物"

    由于模型快速发展的特性，Ultralytics 尚未发布 YOLOv8 的正式研究论文。我们专注于推进技术并使其更易于使用，而不是制作静态文档。有关 YOLO 架构、功能和使用的最新信息，请参阅我们的 [GitHub 仓库](https://github.com/ultralytics/ultralytics)和[文档](https://docs.ultralytics.com/)。

如果您在工作中使用 YOLOv8 模型或此仓库中的任何其他软件，请使用以下格式引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

请注意，DOI 正在等待中，一旦可用将添加到引用中。YOLOv8 模型在 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 和[企业](https://www.ultralytics.com/license)许可证下提供。

## 常见问题

### 什么是 YOLOv8，它与之前的 YOLO 版本有何不同？

YOLOv8 旨在提高实时目标检测性能，具有先进的功能。与早期版本不同，YOLOv8 采用了**无锚点分离式 Ultralytics 头**、最先进的[主干网络](https://www.ultralytics.com/glossary/backbone)和颈部架构，并提供优化的[精度](https://www.ultralytics.com/glossary/accuracy)-速度权衡，使其成为各种应用的理想选择。有关更多详细信息，请查看[概述](#概述)和 [YOLOv8 的关键特性](#yolov8-的关键特性)部分。

### 如何将 YOLOv8 用于不同的计算机视觉任务？

YOLOv8 支持广泛的计算机视觉任务，包括目标检测、实例分割、姿态/关键点检测、旋转目标检测和分类。每个模型变体都针对其特定任务进行了优化，并与各种操作模式兼容，如[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)。有关更多信息，请参阅[支持的任务和模式](#支持的任务和模式)部分。

### YOLOv8 模型的性能指标是什么？

YOLOv8 模型在各种基准数据集上实现了最先进的性能。例如，YOLOv8n 模型在 COCO 数据集上实现了 37.3 的 mAP（平均精度均值），在 A100 TensorRT 上的速度为 0.99 ms。每个模型变体在不同任务和数据集上的详细性能指标可在[性能指标](#性能指标)部分找到。

### 如何训练 YOLOv8 模型？

可以使用 Python 或 CLI 训练 YOLOv8 模型。以下是使用 COCO 预训练的 YOLOv8 模型在 COCO8 数据集上训练 100 个[轮次](https://www.ultralytics.com/glossary/epoch)的示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLOv8n 模型
        model = YOLO("yolov8n.pt")

        # 在 COCO8 示例数据集上训练模型 100 个轮次
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640
        ```

有关更多详细信息，请访问[训练](../modes/train.md)文档。

### 我可以对 YOLOv8 模型进行性能基准测试吗？

是的，YOLOv8 模型可以在各种导出格式中进行速度和精度的基准测试。您可以使用 PyTorch、ONNX、TensorRT 等进行基准测试。以下是使用 Python 和 CLI 进行基准测试的示例命令：

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在 GPU 上进行基准测试
        benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

有关更多信息，请查看[性能指标](#性能指标)部分。
