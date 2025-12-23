---
comments: true
description: 探索 YOLOv5u，一种具有优化精度-速度权衡的先进目标检测模型，采用无锚点 Ultralytics 头部和各种预训练模型。
keywords: YOLOv5, YOLOv5u, 目标检测, Ultralytics, 无锚点, 预训练模型, 精度, 速度, 实时检测
---

# Ultralytics YOLOv5

## 概述

YOLOv5u 代表了[目标检测](https://www.ultralytics.com/glossary/object-detection)方法的进步。源自 Ultralytics 开发的 [YOLOv5](https://github.com/ultralytics/yolov5) 模型的基础架构，YOLOv5u 集成了无锚点、无目标性分离头部，这是之前在 [YOLOv8](yolov8.md) 模型中引入的特性。这种适应改进了模型的架构，在目标检测任务中实现了更好的精度-速度权衡。鉴于实证结果及其衍生特性，YOLOv5u 为那些在研究和实际应用中寻求稳健解决方案的人提供了高效的替代方案。

![Ultralytics YOLOv5](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov5-splash.avif)

!!! warning "由 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 训练的 YOLOv5 模型与 [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) 库不兼容"

    Ultralytics 提供了 [YOLOv5 模型的无锚点变体](#性能指标)。使用[原始 YOLOv5 仓库](https://github.com/ultralytics/yolov5)训练的模型无法与 Ultralytics 库一起使用。

## 关键特性

- **无锚点分离 Ultralytics 头部：** 传统目标检测模型依赖预定义的锚框来预测对象位置。然而，YOLOv5u 现代化了这种方法。通过采用无锚点分离 Ultralytics 头部，它确保了更灵活和自适应的检测机制，从而在各种场景中提高了性能。

- **优化的精度-速度权衡：** 速度和精度通常朝相反方向拉动。但 YOLOv5u 挑战了这种权衡。它提供了校准的平衡，确保实时检测而不影响精度。这一特性对于需要快速响应的应用（如[自动驾驶车辆](https://www.ultralytics.com/glossary/autonomous-vehicles)、[机器人](https://www.ultralytics.com/glossary/robotics)和实时视频分析）特别宝贵。

- **多种预训练模型：** 理解不同任务需要不同的工具集，YOLOv5u 提供了大量预训练模型。无论您专注于推理、验证还是训练，都有一个为您量身定制的模型等待着您。这种多样性确保您不只是使用一刀切的解决方案，而是专门针对您独特挑战进行微调的模型。

## 支持的任务和模式

YOLOv5u 模型具有各种预训练权重，在[目标检测](../tasks/detect.md)任务中表现出色。它们支持全面的模式范围，使其适用于从开发到部署的各种应用。

| 模型类型 | 预训练权重                                                                                                                  | 任务                                   | 推理 | 验证 | 训练 | 导出 |
| -------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ---- | ---- | ---- | ---- |
| YOLOv5u  | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [目标检测](../tasks/detect.md)         | ✅   | ✅   | ✅   | ✅   |

此表提供了 YOLOv5u 模型变体的详细概述，突显了它们在目标检测任务中的适用性以及对各种操作模式（如[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)）的支持。这种全面的支持确保用户可以在广泛的目标检测场景中充分利用 YOLOv5u 模型的能力。

## 性能指标

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5"]'></canvas>

!!! tip "性能"

    === "检测 (COCO)"

    查看[检测文档](../tasks/detect.md)了解在 [COCO](../datasets/detect/coco.md) 上训练的这些模型的使用示例，其中包含 80 个预训练类别。

    | 模型                                                                                        | YAML                                                                                                           | 尺寸<br><sup>(像素)</sup> | mAP<sup>val<br>50-95</sup> | 速度<br><sup>CPU ONNX<br>(ms)</sup> | 速度<br><sup>A100 TensorRT<br>(ms)</sup> | 参数<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    |---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|---------------------------|----------------------------|-------------------------------------|------------------------------------------|------------------------|-------------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5nu.pt)   | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                       | 34.3                       | 73.6                                | 1.06                                     | 2.6                    | 7.7                     |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5su.pt)   | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                       | 43.0                       | 120.7                               | 1.27                                     | 9.1                    | 24.0                    |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5mu.pt)   | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                       | 49.0                       | 233.9                               | 1.86                                     | 25.1                   | 64.2                    |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5lu.pt)   | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                       | 52.2                       | 408.4                               | 2.50                                     | 53.2                   | 135.0                   |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5xu.pt)   | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                       | 53.2                       | 763.2                               | 3.81                                     | 97.2                   | 246.4                   |
    |                                                                                             |                                                                                                                |                           |                            |                                     |                                          |                        |                         |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                      | 42.1                       | 211.0                               | 1.83                                     | 4.3                    | 7.8                     |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                      | 48.6                       | 422.6                               | 2.34                                     | 15.3                   | 24.6                    |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                      | 53.6                       | 810.9                               | 4.36                                     | 41.2                   | 65.7                    |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                      | 55.7                       | 1470.9                              | 5.47                                     | 86.1                   | 137.4                   |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                      | 56.8                       | 2436.5                              | 8.98                                     | 155.4                  | 250.7                   |

## 使用示例

本示例提供简单的 YOLOv5 训练和推理示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参阅[预测](../modes/predict.md)、[训练](../modes/train.md)、[验证](../modes/val.md)和[导出](../modes/export.md)文档页面。

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) 预训练的 `*.pt` 模型以及配置 `*.yaml` 文件可以传递给 `YOLO()` 类以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLOv5n 模型
        model = YOLO("yolov5n.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个 epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLOv5n 模型在 'bus.jpg' 图像上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI 命令可直接运行模型：

        ```bash
        # 加载 COCO 预训练的 YOLOv5n 模型并在 COCO8 示例数据集上训练 100 个 epoch
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载 COCO 预训练的 YOLOv5n 模型并在 'bus.jpg' 图像上运行推理
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## 引用和致谢

!!! tip "Ultralytics YOLOv5 出版物"

    由于模型快速发展的特性，Ultralytics 尚未发布 YOLOv5 的正式研究论文。我们专注于推进技术并使其更易于使用，而不是制作静态文档。有关 YOLO 架构、特性和使用的最新信息，请参阅我们的 [GitHub 仓库](https://github.com/ultralytics/ultralytics)和[文档](https://docs.ultralytics.com/)。

如果您在研究中使用 YOLOv5 或 YOLOv5u，请按以下方式引用 Ultralytics YOLOv5 仓库：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

请注意，YOLOv5 模型根据 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 和[企业](https://www.ultralytics.com/license)许可证提供。

## 常见问题

### 什么是 Ultralytics YOLOv5u，它与 YOLOv5 有何不同？

Ultralytics YOLOv5u 是 YOLOv5 的高级版本，集成了无锚点、无目标性分离头部，增强了实时目标检测任务的[精度](https://www.ultralytics.com/glossary/accuracy)-速度权衡。与传统 YOLOv5 不同，YOLOv5u 采用无锚点检测机制，使其在各种场景中更加灵活和自适应。

### YOLOv5u 中的无锚点 Ultralytics 头部如何改进目标检测性能？

YOLOv5u 中的无锚点 Ultralytics 头部通过消除对预定义锚框的依赖来改进目标检测性能。这导致了更灵活和自适应的检测机制，可以更高效地处理各种对象大小和形状。这种增强直接有助于精度和速度之间的平衡权衡，使 YOLOv5u 适合实时应用。

### 我可以将预训练的 YOLOv5u 模型用于不同的任务和模式吗？

是的，您可以将预训练的 YOLOv5u 模型用于各种任务，如[目标检测](../tasks/detect.md)。这些模型支持多种模式，包括[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)。这种灵活性允许用户在不同的操作需求中利用 YOLOv5u 模型的能力。

### 如何使用 Ultralytics Python API 训练 YOLOv5u 模型？

您可以通过加载预训练模型并使用数据集运行训练命令来训练 YOLOv5u 模型。以下是一个快速示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLOv5n 模型
        model = YOLO("yolov5n.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个 epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 加载 COCO 预训练的 YOLOv5n 模型并在 COCO8 示例数据集上训练 100 个 epoch
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640
        ```

有关更详细的说明，请访问[使用示例](#使用示例)部分。
