---
comments: true
description: 使用 Ultralytics YOLO11 实时监控健身训练，优化您的健身计划。跟踪并改善您的运动姿势和表现。
keywords: 健身监控, Ultralytics YOLO11, 姿态估计, 健身跟踪, 运动评估, 实时反馈, 运动姿势, 性能指标
---

# 使用 Ultralytics YOLO11 进行健身训练监控

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-monitor-workouts-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开健身监控"></a>

通过 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) 的姿态估计监控健身训练，可以通过实时准确跟踪关键身体标志点和关节来增强运动评估。这项技术提供即时的运动姿势反馈、跟踪训练计划并测量性能指标，为用户和教练优化训练课程。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Ck7DW96dNok"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO 监控健身运动 | 深蹲、腿部伸展、俯卧撑等
</p>

## 健身训练监控的优势

- **优化表现：** 根据监控数据调整训练以获得更好的效果。
- **目标达成：** 跟踪和调整健身目标以实现可衡量的进步。
- **个性化：** 基于个人数据定制训练计划以提高效果。
- **健康意识：** 早期检测表明健康问题或过度训练的模式。
- **明智决策：** 数据驱动的决策用于调整计划和设定现实目标。

## 实际应用

|                                        健身训练监控                                         |                                        健身训练监控                                         |
| :------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| ![俯卧撑计数](https://github.com/ultralytics/docs/releases/download/0/pushups-counting.avif) | ![引体向上计数](https://github.com/ultralytics/docs/releases/download/0/pullups-counting.avif) |
|                                          俯卧撑计数                                          |                                          引体向上计数                                          |

!!! example "使用 Ultralytics YOLO 进行健身训练监控"

    === "CLI"

        ```bash
        # 运行健身示例
        yolo solutions workout show=True

        # 传入源视频
        yolo solutions workout source="path/to/video.mp4"

        # 使用俯卧撑的关键点
        yolo solutions workout kpts="[6, 8, 10]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件时出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("workouts_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化 AIGym
        gym = solutions.AIGym(
            show=True,  # 显示帧
            kpts=[6, 8, 10],  # 用于监控特定运动的关键点，默认为俯卧撑
            model="yolo11n-pose.pt",  # YOLO11 姿态估计模型文件路径
            # line_width=2,  # 调整边界框和文本显示的线宽
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或处理已完成。")
                break

            results = gym(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### 关键点映射图

![Ultralytics YOLO11 姿态关键点顺序](https://github.com/ultralytics/docs/releases/download/0/keypoints-order-ultralytics-yolov8-pose.avif)

### `AIGym` 参数

以下是 `AIGym` 参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "up_angle", "down_angle", "kpts"]) }}

`AIGym` 解决方案还支持一系列对象跟踪参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还可以应用以下可视化设置：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 常见问题

### 如何使用 Ultralytics YOLO11 监控我的健身训练？

要使用 Ultralytics YOLO11 监控您的健身训练，您可以利用[姿态估计功能](https://docs.ultralytics.com/tasks/pose/)实时跟踪和分析关键身体标志点和关节。这使您能够获得即时的运动姿势反馈、计算重复次数并测量性能指标。您可以使用提供的俯卧撑、引体向上或腹部训练示例代码开始：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "读取视频文件时出错"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("视频帧为空或处理已完成。")
        break
    results = gym(im0)

cv2.destroyAllWindows()
```

有关更多自定义和设置，您可以参考文档中的 [AIGym](#aigym-参数) 部分。

### 使用 Ultralytics YOLO11 进行健身监控有哪些好处？

使用 Ultralytics YOLO11 进行健身监控提供了几个关键好处：

- **优化表现：** 通过根据监控数据调整训练，您可以获得更好的效果。
- **目标达成：** 轻松跟踪和调整健身目标以实现可衡量的进步。
- **个性化：** 基于您的个人数据获得定制的训练计划以获得最佳效果。
- **健康意识：** 早期检测表明潜在健康问题或过度训练的模式。
- **明智决策：** 做出数据驱动的决策来调整计划和设定现实目标。

您可以观看 [YouTube 视频演示](https://www.youtube.com/watch?v=LGGxqLZtvuw)来了解这些好处的实际效果。

### Ultralytics YOLO11 在检测和跟踪运动方面有多准确？

由于其最先进的[姿态估计](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-pose-estimation)功能，Ultralytics YOLO11 在检测和跟踪运动方面非常准确。它可以准确跟踪关键身体标志点和关节，提供实时的运动姿势和性能指标反馈。该模型的预训练权重和强大架构确保了高[精度](https://www.ultralytics.com/glossary/precision)和可靠性。有关实际示例，请查看文档中的[实际应用](#实际应用)部分，其中展示了俯卧撑和引体向上计数。

### 我可以使用 Ultralytics YOLO11 进行自定义健身计划吗？

是的，Ultralytics YOLO11 可以适应自定义健身计划。`AIGym` 类支持不同的姿势类型，如 `pushup`、`pullup` 和 `abworkout`。您可以指定关键点和角度来检测特定运动。以下是一个示例设置：

```python
from ultralytics import solutions

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],  # 用于俯卧撑 - 可以为其他运动自定义
)
```

有关设置参数的更多详细信息，请参考 [AIGym 参数](#aigym-参数)部分。这种灵活性允许您监控各种运动并根据您的[健身目标](https://www.ultralytics.com/blog/ai-in-our-day-to-day-health-and-fitness)自定义计划。

### 如何使用 Ultralytics YOLO11 保存健身监控输出？

要保存健身监控输出，您可以修改代码以包含一个视频写入器来保存处理后的帧。以下是一个示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "读取视频文件时出错"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("workouts.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("视频帧为空或处理已完成。")
        break
    results = gym(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

此设置将监控的视频写入输出文件，允许您稍后查看健身表现或与教练分享以获得额外反馈。
