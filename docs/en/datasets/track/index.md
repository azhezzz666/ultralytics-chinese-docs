---
comments: true
description: 学习如何使用 YOLO 进行多目标跟踪。探索数据集格式、跟踪算法，以及使用 Python 或 CLI 进行实时目标跟踪的实现示例。
keywords: YOLO, 多目标跟踪, 跟踪数据集, Python 跟踪示例, CLI 跟踪示例, 目标检测, Ultralytics, AI, 机器学习, BoT-SORT, ByteTrack
---

# 多目标跟踪数据集概览

多目标跟踪是视频分析中的关键组件，用于识别目标并在视频帧之间为每个检测到的目标维护唯一 ID。Ultralytics YOLO 提供强大的跟踪功能，可应用于监控、体育分析和交通监控等各种领域。

## 数据集格式（即将推出）

Ultralytics 跟踪目前复用检测、分割或姿态模型，无需特定于跟踪器的训练。原生跟踪器训练支持正在积极开发中。

## 可用跟踪器

Ultralytics YOLO 支持以下跟踪算法：

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 使用 `botsort.yaml` 启用此跟踪器（默认）
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) - 使用 `bytetrack.yaml` 启用此跟踪器

## 使用方法

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3 iou=0.5 show=True
        ```

## 跨帧持久化跟踪

对于视频帧之间的连续跟踪，可以使用 `persist=True` 参数：

!!! example

    === "Python"

        ```python
        import cv2

        from ultralytics import YOLO

        # 加载 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 打开视频文件
        cap = cv2.VideoCapture("path/to/video.mp4")

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # 运行跟踪并在帧之间保持持久性
                results = model.track(frame, persist=True)

                # 可视化结果
                annotated_frame = results[0].plot()
                cv2.imshow("Tracking", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        ```

## 常见问题

### 如何使用 Ultralytics YOLO 进行多目标跟踪？

要使用 Ultralytics YOLO 进行多目标跟踪，可以使用提供的 Python 或 CLI 示例开始。以下是入门方法：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # 加载 YOLO11 模型
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3 iou=0.5 show=True
        ```

这些命令加载 YOLO11 模型，并使用它以指定的置信度（`conf`）和[交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou)（`iou`）阈值跟踪给定视频源中的目标。有关更多详细信息，请参阅[跟踪模式文档](../../modes/track.md)。

### Ultralytics 跟踪器训练有哪些即将推出的功能？

Ultralytics 正在持续增强其 AI 模型。即将推出的功能将支持独立跟踪器的训练。在此之前，多目标检测器利用预训练的检测、分割或姿态模型进行跟踪，无需独立训练。关注我们的[博客](https://www.ultralytics.com/blog)或查看[即将推出的功能](../../reference/trackers/track.md)以获取更新。

### 为什么应该使用 Ultralytics YOLO 进行多目标跟踪？

Ultralytics YOLO 是最先进的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型，以其实时性能和高[准确性](https://www.ultralytics.com/glossary/accuracy)而闻名。使用 YOLO 进行多目标跟踪有以下几个优势：

- **实时跟踪**：实现高效、高速的跟踪，非常适合动态环境。
- **预训练模型的灵活性**：无需从头训练；只需使用预训练的检测、分割或姿态模型。
- **易于使用**：简单的 API 集成，支持 Python 和 CLI，使设置跟踪流程变得简单。
- **广泛的文档和社区支持**：Ultralytics 提供全面的文档和活跃的社区论坛，帮助解决问题并增强您的跟踪模型。

有关设置和使用 YOLO 进行跟踪的更多详细信息，请访问我们的[跟踪使用指南](../../modes/track.md)。

### 我可以使用自定义数据集进行 Ultralytics YOLO 多目标跟踪吗？

是的，您可以使用自定义数据集进行 Ultralytics YOLO 多目标跟踪。虽然独立跟踪器训练支持是即将推出的功能，但您已经可以在自定义数据集上使用预训练模型。以与 YOLO 兼容的适当格式准备数据集，并按照文档进行集成。

### 如何解释 Ultralytics YOLO 跟踪模型的结果？

运行 Ultralytics YOLO 跟踪任务后，结果包括各种数据点，如跟踪目标 ID、边界框和置信度分数。以下是如何解释这些结果的简要概述：

- **跟踪 ID**：每个目标被分配一个唯一 ID，有助于跨帧跟踪它。
- **边界框**：这些指示跟踪目标在帧中的位置。
- **置信度分数**：这些反映模型检测跟踪目标的置信度。

有关解释和可视化这些结果的详细指导，请参阅[结果处理指南](../../reference/engine/results.md)。

### 如何自定义跟踪器配置？

您可以通过创建跟踪器配置文件的修改版本来自定义跟踪器。从 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 复制现有的跟踪器配置文件，根据需要修改参数，并在运行跟踪器时指定此文件：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.track(source="video.mp4", tracker="custom_tracker.yaml")
```
