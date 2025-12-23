---
comments: true
description: 探索使用 Ultralytics YOLO 进行高效、灵活和可定制的多目标跟踪。学习轻松跟踪实时视频流。
keywords: 多目标跟踪, Ultralytics YOLO, 视频分析, 实时跟踪, 目标检测, AI, 机器学习
---

# 使用 Ultralytics YOLO 进行多目标跟踪

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/multi-object-tracking-examples.avif" alt="多目标跟踪示例">

视频分析领域中的目标跟踪是一项关键任务，它不仅识别帧内目标的位置和类别，还在视频进行过程中为每个检测到的目标维护唯一 ID。应用范围无限——从监控和安全到实时体育分析。

## 为什么选择 Ultralytics YOLO 进行目标跟踪？

Ultralytics 跟踪器的输出与标准[目标检测](https://www.ultralytics.com/glossary/object-detection)一致，但增加了目标 ID 的价值。这使得在视频流中跟踪目标并执行后续分析变得容易。以下是您应该考虑使用 Ultralytics YOLO 满足目标跟踪需求的原因：

- **效率：** 实时处理视频流而不牺牲[精度](https://www.ultralytics.com/glossary/accuracy)。
- **灵活性：** 支持多种跟踪算法和配置。
- **易用性：** 简单的 Python API 和 CLI 选项，便于快速集成和部署。
- **可定制性：** 易于与自定义训练的 YOLO 模型一起使用，允许集成到特定领域的应用中。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 使用 Ultralytics YOLO 进行目标检测和跟踪。
</p>

## 实际应用

|           交通运输           |              零售              |         水产养殖          |
| :--------------------------: | :----------------------------: | :-----------------------: |
| ![车辆跟踪][vehicle track]   | ![人员跟踪][people track]      | ![鱼类跟踪][fish track]   |
|          车辆跟踪            |         人员跟踪               |        鱼类跟踪           |

## 功能概览

Ultralytics YOLO 扩展了其目标检测功能，提供稳健且多功能的目标跟踪：

- **实时跟踪：** 在高帧率视频中无缝跟踪目标。
- **多跟踪器支持：** 从多种成熟的跟踪算法中选择。
- **可定制的跟踪器配置：** 通过调整各种参数来定制跟踪算法以满足特定要求。

## 可用的跟踪器

Ultralytics YOLO 支持以下跟踪算法。可以通过传递相关的 YAML 配置文件（如 `tracker=tracker_type.yaml`）来启用它们：

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 使用 `botsort.yaml` 启用此跟踪器。
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) - 使用 `bytetrack.yaml` 启用此跟踪器。

默认跟踪器是 BoT-SORT。

## 跟踪

要在视频流上运行跟踪器，请使用训练好的检测、分割或姿态模型，如 YOLO11n、YOLO11n-seg 或 YOLO11n-pose。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载官方或自定义模型
        model = YOLO("yolo11n.pt")  # 加载官方检测模型
        model = YOLO("yolo11n-seg.pt")  # 加载官方分割模型
        model = YOLO("yolo11n-pose.pt")  # 加载官方姿态模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 使用模型执行跟踪
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # 使用默认跟踪器跟踪
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # 使用 ByteTrack
        ```

    === "CLI"

        ```bash
        # 使用命令行界面使用各种模型执行跟踪
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4"      # 官方检测模型
        yolo track model=yolo11n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # 官方分割模型
        yolo track model=yolo11n-pose.pt source="https://youtu.be/LNwODJXcvt4" # 官方姿态模型
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" # 自定义训练的模型

        # 使用 ByteTrack 跟踪器跟踪
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" tracker="bytetrack.yaml"
        ```

如上所示，跟踪可用于在视频或流源上运行的所有检测、分割和姿态模型。

## 配置

### 跟踪参数

跟踪配置与预测模式共享属性，如 `conf`、`iou` 和 `show`。有关更多配置，请参阅[预测](../modes/predict.md#推理参数)模型页面。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 配置跟踪参数并运行跟踪器
        model = YOLO("yolo11n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # 使用命令行界面配置跟踪参数并运行跟踪器
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### 跟踪器选择

Ultralytics 还允许您使用修改后的跟踪器配置文件。为此，只需从 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 复制跟踪器配置文件（例如 `custom_tracker.yaml`），并根据需要修改任何配置（除了 `tracker_type`）。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型并使用自定义配置文件运行跟踪器
        model = YOLO("yolo11n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        # 使用命令行界面加载模型并使用自定义配置文件运行跟踪器
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

### 跟踪器参数

通过编辑每个跟踪算法特定的 YAML 配置文件，可以微调某些跟踪行为。这些文件定义了阈值、缓冲区和匹配逻辑等参数：

| **参数**            | **有效值或范围**                              | **描述**                                                                                                                       |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `tracker_type`      | `botsort`, `bytetrack`                        | 指定跟踪器类型。选项为 `botsort` 或 `bytetrack`。                                                                              |
| `track_high_thresh` | `0.0-1.0`                                     | 跟踪期间第一次关联使用的阈值。影响检测与现有轨迹匹配的置信度。                                                                 |
| `track_low_thresh`  | `0.0-1.0`                                     | 跟踪期间第二次关联的阈值。当第一次关联失败时使用，标准更宽松。                                                                 |
| `new_track_thresh`  | `0.0-1.0`                                     | 如果检测不匹配任何现有轨迹，则初始化新轨迹的阈值。控制何时认为新目标出现。                                                     |
| `track_buffer`      | `>=0`                                         | 用于指示丢失轨迹在被移除之前应保持活动的帧数的缓冲区。值越高，对遮挡的容忍度越高。                                             |
| `match_thresh`      | `0.0-1.0`                                     | 匹配轨迹的阈值。值越高，匹配越宽松。                                                                                           |
| `fuse_score`        | `True`, `False`                               | 确定是否在匹配前将置信度分数与 IoU 距离融合。有助于在关联时平衡空间和置信度信息。                                              |
| `gmc_method`        | `orb`, `sift`, `ecc`, `sparseOptFlow`, `None` | 用于全局运动补偿的方法。有助于考虑相机移动以改善跟踪。                                                                         |
| `proximity_thresh`  | `0.0-1.0`                                     | 与 ReID（重识别）有效匹配所需的最小 IoU。确保在使用外观线索之前的空间接近性。                                                  |
| `appearance_thresh` | `0.0-1.0`                                     | ReID 所需的最小外观相似度。设置两个检测必须在视觉上多相似才能被链接。                                                          |
| `with_reid`         | `True`, `False`                               | 指示是否使用 ReID。启用基于外观的匹配，以便在遮挡时更好地跟踪。仅 BoTSORT 支持。                                               |

## Python 示例

### 持久化轨迹循环

这是一个使用 [OpenCV](https://www.ultralytics.com/glossary/opencv) (`cv2`) 和 YOLO11 在视频帧上运行目标跟踪的 Python 脚本。此脚本假设已安装必要的包（`opencv-python` 和 `ultralytics`）。`persist=True` 参数告诉跟踪器当前图像或帧是序列中的下一个，并期望从当前图像中的前一个图像获取轨迹。

!!! example "带跟踪的流式 for 循环"

    ```python
    import cv2

    from ultralytics import YOLO

    # 加载 YOLO11 模型
    model = YOLO("yolo11n.pt")

    # 打开视频文件
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频读取一帧
        success, frame = cap.read()

        if success:
            # 在帧上运行 YOLO11 跟踪，在帧之间持久化轨迹
            results = model.track(frame, persist=True)

            # 在帧上可视化结果
            annotated_frame = results[0].plot()

            # 显示标注的帧
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # 如果按下 'q' 则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果到达视频末尾则退出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

请注意从 `model(frame)` 到 `model.track(frame)` 的更改，这启用了目标跟踪而不是简单的检测。

### 随时间绘制轨迹

在连续帧上可视化目标轨迹可以提供有关视频中检测到的目标的移动模式和行为的宝贵见解。使用 Ultralytics YOLO11，绘制这些轨迹是一个无缝且高效的过程。

!!! example "在多个视频帧上绘制轨迹"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    # 加载 YOLO11 模型
    model = YOLO("yolo11n.pt")

    # 打开视频文件
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 存储轨迹历史
    track_history = defaultdict(lambda: [])

    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频读取一帧
        success, frame = cap.read()

        if success:
            # 在帧上运行 YOLO11 跟踪，在帧之间持久化轨迹
            result = model.track(frame, persist=True)[0]

            # 获取边界框和轨迹 ID
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # 在帧上可视化结果
                frame = result.plot()

                # 绘制轨迹
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y 中心点
                    if len(track) > 30:  # 保留 30 帧的 30 个轨迹
                        track.pop(0)

                    # 绘制跟踪线
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # 显示标注的帧
            cv2.imshow("YOLO11 Tracking", frame)

            # 如果按下 'q' 则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果到达视频末尾则退出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

[fish track]: https://github.com/ultralytics/docs/releases/download/0/fish-tracking.avif
[people track]: https://github.com/ultralytics/docs/releases/download/0/people-tracking.avif
[vehicle track]: https://github.com/ultralytics/docs/releases/download/0/vehicle-tracking.avif

## 常见问题

### 什么是多目标跟踪，Ultralytics YOLO 如何支持它？

视频分析中的多目标跟踪涉及识别目标并在视频帧中为每个检测到的目标维护唯一 ID。Ultralytics YOLO 通过提供实时跟踪和目标 ID 来支持此功能，便于执行监控和安全、实时体育分析等任务。系统使用 [BoT-SORT](https://github.com/NirAharon/BoT-SORT) 和 [ByteTrack](https://github.com/FoundationVision/ByteTrack) 等跟踪器，可通过 YAML 文件配置。

### 如何为 Ultralytics YOLO 配置自定义跟踪器？

您可以通过从 [Ultralytics 跟踪器配置目录](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)复制现有跟踪器配置文件（例如 `custom_tracker.yaml`）并根据需要修改参数（除了 `tracker_type`）来配置自定义跟踪器。在跟踪模型中使用此文件：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

### 使用 Ultralytics YOLO 进行多目标跟踪有哪些实际应用？

使用 Ultralytics YOLO 进行多目标跟踪有许多应用，包括：

- **交通运输：** 用于交通管理和[自动驾驶](https://www.ultralytics.com/blog/ai-in-self-driving-cars)的车辆跟踪。
- **零售：** 用于店内分析和安全的人员跟踪。
- **水产养殖：** 用于监测水生环境的鱼类跟踪。
- **体育分析：** 跟踪运动员和设备进行性能分析。
- **安全系统：** [监控可疑活动](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8)和创建[安全警报](https://docs.ultralytics.com/guides/security-alarm-system/)。

这些应用受益于 Ultralytics YOLO 以卓越精度实时处理高帧率视频的能力。
