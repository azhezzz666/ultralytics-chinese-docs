---
comments: true
description: 学习如何使用 Ultralytics YOLO11 进行速度估计，适用于交通控制、自动驾驶导航和监控等应用。
keywords: Ultralytics YOLO11, 速度估计, 对象跟踪, 计算机视觉, 交通控制, 自动驾驶导航, 监控, 安防
---

# 使用 Ultralytics YOLO11 进行速度估计 🚀

## 什么是速度估计？

[速度估计](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects)是在给定上下文中计算对象移动速率的过程，通常用于[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用。使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/)，您现在可以结合[对象跟踪](../modes/track.md)以及距离和时间数据来计算对象的速度，这对于交通监控和监控等任务至关重要。速度估计的准确性直接影响各种应用的效率和可靠性，使其成为智能系统和实时决策过程发展的关键组成部分。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rCggzXRRSRo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO11 进行速度估计
</p>

!!! tip "查看我们的博客"

    如需深入了解速度估计，请查看我们的博客文章：[Ultralytics YOLO11 在计算机视觉项目中的速度估计](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects)

## 速度估计的优势

- **高效交通控制：** 准确的速度估计有助于管理交通流量，提高安全性并减少道路拥堵。
- **精确自动驾驶导航：** 在自动驾驶系统（如[自动驾驶汽车](https://www.ultralytics.com/solutions/ai-in-automotive)）中，可靠的速度估计确保安全准确的车辆导航。
- **增强监控安全：** 监控分析中的速度估计有助于识别异常行为或潜在威胁，提高安全措施的有效性。

## 实际应用

|                                                                            交通运输                                                                            |                                                                              交通运输                                                                              |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO11 在道路上进行速度估计](https://github.com/ultralytics/docs/releases/download/0/speed-estimation-on-road-using-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO11 在桥梁上进行速度估计](https://github.com/ultralytics/docs/releases/download/0/speed-estimation-on-bridge-using-ultralytics-yolov8.avif) |
|                                                          使用 Ultralytics YOLO11 在道路上进行速度估计                                                           |                                                           使用 Ultralytics YOLO11 在桥梁上进行速度估计                                                            |

???+ warning "速度是估计值"

    速度将是估计值，可能不完全准确。此外，估计值可能因摄像头规格和相关因素而有所不同。

!!! example "使用 Ultralytics YOLO 进行速度估计"

    === "命令行"

        ```bash
        # 运行速度估计示例
        yolo solutions speed show=True

        # 传入源视频
        yolo solutions speed source="path/to/video.mp4"

        # 根据摄像头配置调整每像素米数值
        yolo solutions speed meter_per_pixel=0.05
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件时出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化速度估计对象
        speedestimator = solutions.SpeedEstimator(
            show=True,  # 显示输出
            model="yolo11n.pt",  # YOLO11 模型文件路径
            fps=fps,  # 根据每秒帧数调整速度
            # max_speed=120,  # 将速度限制在最大值（km/h）以避免异常值
            # max_hist=5,  # 计算速度前对象被跟踪的最小帧数
            # meter_per_pixel=0.05,  # 高度依赖于摄像头配置
            # classes=[0, 2],  # 估计特定类别的速度
            # line_width=2,  # 调整边界框的线宽
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或处理已完成。")
                break

            results = speedestimator(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `SpeedEstimator` 参数

以下是 `SpeedEstimator` 参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "fps", "max_hist", "meter_per_pixel", "max_speed"]) }}

`SpeedEstimator` 解决方案允许使用 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化选项：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 常见问题

### 如何使用 Ultralytics YOLO11 估计对象速度？

使用 Ultralytics YOLO11 估计对象速度涉及结合[目标检测](https://www.ultralytics.com/glossary/object-detection)和跟踪技术。首先，您需要使用 YOLO11 模型在每一帧中检测对象。然后，跨帧跟踪这些对象以计算它们随时间的移动。最后，使用对象在帧之间移动的距离和帧率来估计其速度。

**示例**：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 初始化 SpeedEstimator
speedestimator = solutions.SpeedEstimator(
    model="yolo11n.pt",
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = speedestimator(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

更多详情，请参阅我们的[官方博客文章](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects)。

### 在交通管理中使用 Ultralytics YOLO11 进行速度估计有什么好处？

使用 Ultralytics YOLO11 进行速度估计在交通管理中具有显著优势：

- **增强安全性：** 准确估计车辆速度以检测超速并提高道路安全。
- **实时监控：** 受益于 YOLO11 的实时目标检测能力，有效监控交通流量和拥堵情况。
- **可扩展性：** 在各种硬件设置上部署模型，从[边缘设备](https://docs.ultralytics.com/guides/nvidia-jetson/)到服务器，确保大规模实施的灵活和可扩展解决方案。

更多应用，请参阅[速度估计的优势](#速度估计的优势)。

### YOLO11 可以与其他 AI 框架（如 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 或 [PyTorch](https://www.ultralytics.com/glossary/pytorch)）集成吗？

是的，YOLO11 可以与其他 AI 框架（如 TensorFlow 和 PyTorch）集成。Ultralytics 支持将 YOLO11 模型导出为各种格式，如 [ONNX](../integrations/onnx.md)、[TensorRT](../integrations/tensorrt.md) 和 [CoreML](../integrations/coreml.md)，确保与其他机器学习框架的顺畅互操作性。

将 YOLO11 模型导出为 ONNX 格式：

```bash
yolo export model=yolo11n.pt format=onnx
```

在我们的[导出指南](../modes/export.md)中了解更多关于导出模型的信息。

### 使用 Ultralytics YOLO11 进行速度估计的准确性如何？

使用 Ultralytics YOLO11 进行速度估计的[准确性](https://www.ultralytics.com/glossary/accuracy)取决于多个因素，包括对象跟踪的质量、视频的分辨率和帧率以及环境变量。虽然速度估计器提供可靠的估计，但由于帧处理速度和对象遮挡的差异，可能无法达到 100% 准确。

**注意**：始终考虑误差范围，并在可能的情况下使用真实数据验证估计值。

有关进一步提高准确性的技巧，请查看 [`SpeedEstimator` 参数部分](#speedestimator-参数)。
