---
comments: true
description: 探索使用 YOLO11 的 Ultralytics 解决方案，包括目标计数、模糊处理、安全系统等。通过尖端 AI 提升效率并解决实际问题。
keywords: Ultralytics, YOLO11, 目标计数, 目标模糊, 安全系统, AI 解决方案, 实时分析, 计算机视觉应用
---

# Ultralytics 解决方案：利用 YOLO11 解决实际问题

Ultralytics 解决方案提供 YOLO 模型的尖端应用，提供目标计数、模糊处理和安全系统等实际解决方案，在各行各业提升效率和[准确性](https://www.ultralytics.com/glossary/accuracy)。探索 YOLO11 在实际应用中的强大功能。

![Ultralytics 解决方案缩略图](https://github.com/ultralytics/docs/releases/download/0/ultralytics-solutions-thumbnail.avif)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/bjkt5OE_ANA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何从命令行（CLI）运行 Ultralytics 解决方案 | Ultralytics YOLO11 🚀
</p>

## 解决方案

以下是我们精选的 Ultralytics 解决方案列表，可用于创建出色的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目。

- [分析](../guides/analytics.md)：进行全面的数据分析以发现模式并做出明智决策，利用 YOLO11 进行描述性、预测性和规范性分析。
- [距离计算](../guides/distance-calculation.md)：使用 YOLO11 中的[边界框](https://www.ultralytics.com/glossary/bounding-box)质心计算目标之间的距离，对空间分析至关重要。
- [热力图](../guides/heatmaps.md)：利用检测热力图在矩阵中可视化数据强度，在计算机视觉任务中提供清晰的洞察。
- [实例分割与目标跟踪](../guides/instance-segmentation-and-tracking.md)：使用 YOLO11 实现[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)和目标跟踪，实现精确的目标边界和持续监控。
- [Streamlit 实时推理](../guides/streamlit-live-inference.md)：利用 YOLO11 的强大功能，通过用户友好的 Streamlit 界面直接在网页浏览器中进行实时[目标检测](https://www.ultralytics.com/glossary/object-detection)。
- [目标模糊](../guides/object-blurring.md)：使用 YOLO11 应用目标模糊，在图像和视频处理中保护隐私。
- [目标计数](../guides/object-counting.md)：学习使用 YOLO11 进行实时目标计数。掌握在实时视频流中准确计数目标的专业知识。
- [区域目标计数](../guides/region-counting.md)：使用 YOLO11 在特定区域内计数目标，实现不同区域的准确检测。
- [目标裁剪](../guides/object-cropping.md)：掌握使用 YOLO11 进行目标裁剪，从图像和视频中精确提取目标。
- [停车管理](../guides/parking-management.md)：使用 YOLO11 组织和引导停车区域的车辆流动，优化空间利用和用户体验。
- [队列管理](../guides/queue-management.md)：使用 YOLO11 实现高效的队列管理系统，最小化等待时间并提高生产力。
- [安全报警系统](../guides/security-alarm-system.md)：使用 YOLO11 创建安全报警系统，在检测到新目标时触发警报。自定义系统以满足您的特定需求。
- [相似性搜索](../guides/similarity-search.md)：通过结合 [OpenAI CLIP](https://cookbook.openai.com/examples/custom_image_embedding_search) 嵌入和 [Meta FAISS](https://ai.meta.com/tools/faiss/) 实现智能图像检索，支持"拿着包的人"或"行驶中的车辆"等自然语言查询。
- [速度估计](../guides/speed-estimation.md)：使用 YOLO11 和目标跟踪技术估计目标速度，对自动驾驶车辆和交通监控等应用至关重要。
- [区域目标跟踪](../guides/trackzone.md)：学习如何使用 YOLO11 在视频帧的特定区域内跟踪目标，实现精确高效的监控。
- [VisionEye 视觉目标映射](../guides/vision-eye.md)：开发模拟人眼聚焦特定目标的系统，增强计算机识别和优先处理细节的能力。
- [健身监控](../guides/workouts-monitoring.md)：了解如何使用 YOLO11 监控健身活动。学习实时跟踪和分析各种健身动作。

### 解决方案参数

{% from "macros/solutions-args.md" import param_table %}
{{ param_table() }}

!!! note "跟踪参数"

     解决方案还支持 `track` 的一些参数，包括 `conf`、`line_width`、`tracker`、`model`、`show`、`verbose` 和 `classes` 等参数。

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

!!! note "可视化参数"

    您可以使用 `show_conf`、`show_labels` 和其他上述参数来自定义可视化效果。

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

### SolutionAnnotator 的使用

所有 Ultralytics 解决方案都使用独立的 [`SolutionAnnotator`](https://docs.ultralytics.com/reference/solutions/solutions/#ultralytics.solutions.solutions.SolutionAnnotator) 类，该类扩展了主 [`Annotator`](https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Annotator) 类，并具有以下方法：

| 方法                               | 返回类型    | 描述                                                                 |
| ---------------------------------- | ----------- | -------------------------------------------------------------------- |
| `draw_region()`                    | `None`      | 使用指定的点、颜色和厚度绘制区域。                                   |
| `queue_counts_display()`           | `None`      | 在指定区域显示队列计数。                                             |
| `display_analytics()`              | `None`      | 显示停车场管理的整体统计信息。                                       |
| `estimate_pose_angle()`            | `float`     | 计算目标姿态中三个点之间的角度。                                     |
| `draw_specific_points()`           | `None`      | 在图像上绘制特定关键点。                                             |
| `plot_workout_information()`       | `None`      | 在图像上绘制带标签的文本框。                                         |
| `plot_angle_and_count_and_stage()` | `None`      | 可视化健身监控的角度、步数和阶段。                                   |
| `plot_distance_and_line()`         | `None`      | 显示质心之间的距离并用线连接它们。                                   |
| `display_objects_labels()`         | `None`      | 用目标类别标签标注边界框。                                           |
| `sweep_annotator()`                | `None`      | 可视化垂直扫描线和可选标签。                                         |
| `visioneye()`                      | `None`      | 将目标质心映射并连接到视觉"眼睛"点。                                 |
| `adaptive_label()`                 | `None`      | 在边界框中心绘制圆形或矩形背景形状标签。                             |

### 使用 SolutionResults

除了 [`相似性搜索`](../guides/similarity-search.md) 外，每个解决方案调用都返回一个 `SolutionResults` 对象列表。

- 对于目标计数，结果包括 `in_count`、`out_count` 和 `classwise_count`。

!!! example "SolutionResults"

    ```python
    import cv2

    from ultralytics import solutions

    im0 = cv2.imread("path/to/img")

    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

    counter = solutions.ObjectCounter(
        show=True,  # 显示输出
        region=region_points,  # 传递区域点
        model="yolo11n.pt",  # model="yolo11n-obb.pt" 用于 OBB 模型的目标计数
        # classes=[0, 2],  # 使用 COCO 预训练模型计数特定类别，如人和汽车
        # tracker="botsort.yaml"  # 选择跟踪器，如 "bytetrack.yaml"
    )
    results = counter(im0)
    print(results.in_count)  # 显示进入计数
    print(results.out_count)  # 显示离开计数
    print(results.classwise_count)  # 显示按类别计数
    ```

`SolutionResults` 对象具有以下属性：

| 属性                 | 类型               | 描述                                                                                                   |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------ |
| `plot_im`            | `np.ndarray`       | 带有视觉叠加层的图像，如计数、模糊效果或解决方案特定增强。                                             |
| `in_count`           | `int`              | 视频流中检测到进入定义区域的目标总数。                                                                 |
| `out_count`          | `int`              | 视频流中检测到离开定义区域的目标总数。                                                                 |
| `classwise_count`    | `Dict[str, int]`   | 记录按类别进出目标计数的字典，用于高级分析。                                                           |
| `queue_count`        | `int`              | 当前在预定义队列或等待区域内的目标数量（适用于队列管理）。                                             |
| `workout_count`      | `int`              | 运动跟踪期间完成的健身重复次数总数。                                                                   |
| `workout_angle`      | `float`            | 健身期间计算的关节或姿态角度，用于姿势评估。                                                           |
| `workout_stage`      | `str`              | 当前健身阶段或运动阶段（如 'up'、'down'）。                                                            |
| `pixels_distance`    | `float`            | 两个目标或点（如边界框）之间的像素距离（适用于距离计算）。                                             |
| `available_slots`    | `int`              | 监控区域中未占用的停车位数量（适用于停车管理）。                                                       |
| `filled_slots`       | `int`              | 监控区域中已占用的停车位数量（适用于停车管理）。                                                       |
| `email_sent`         | `bool`             | 指示通知或警报邮件是否已成功发送（适用于安全报警）。                                                   |
| `total_tracks`       | `int`              | 视频分析期间观察到的唯一目标轨迹总数。                                                                 |
| `region_counts`      | `Dict[str, int]`   | 用户定义区域或区域内的目标计数。                                                                       |
| `speed_dict`         | `Dict[str, float]` | 按轨迹计算的目标速度字典，用于速度分析。                                                               |
| `total_crop_objects` | `int`              | ObjectCropper 解决方案生成的裁剪目标图像总数。                                                         |
| `speed`              | `Dict[str, float]` | 包含跟踪和解决方案处理性能指标的字典。                                                                 |

更多详情，请参阅 [`SolutionResults` 类文档](https://docs.ultralytics.com/reference/solutions/solutions/#ultralytics.solutions.solutions.SolutionAnnotator)。

### 通过 CLI 使用解决方案

!!! tip "命令信息"

    大多数解决方案可以直接通过命令行界面使用，包括：

    `Count`、`Crop`、`Blur`、`Workout`、`Heatmap`、`Isegment`、`Visioneye`、`Speed`、`Queue`、`Analytics`、`Inference`

    **语法**

        yolo SOLUTIONS SOLUTION_NAME ARGS

    - **SOLUTIONS** 是必需的关键字。
    - **SOLUTION_NAME** 是以下之一：`['count', 'crop', 'blur', 'workout', 'heatmap', 'isegment', 'queue', 'speed', 'analytics', 'trackzone', 'inference', 'visioneye']`。
    - **ARGS**（可选）是自定义的 `arg=value` 对，如 `show_in=True`，用于覆盖默认设置。

```bash
yolo solutions count show=True # 用于目标计数

yolo solutions count source="path/to/video.mp4" # 指定视频文件路径
```

### 为我们的解决方案做贡献

我们欢迎社区的贡献！如果您已经掌握了 Ultralytics YOLO 的某个特定方面，而我们的解决方案尚未涵盖，我们鼓励您分享您的专业知识。编写指南是回馈社区的好方法，有助于使我们的文档更加全面和用户友好。

要开始，请阅读我们的[贡献指南](../help/contributing.md)，了解如何提交 Pull Request (PR) 🛠️。我们期待您的贡献！

让我们共同努力，使 Ultralytics YOLO 生态系统更加强大和多功能 🙏！

## 常见问题

### 如何使用 Ultralytics YOLO 进行实时目标计数？

Ultralytics YOLO11 可以利用其先进的目标检测功能进行实时目标计数。您可以按照我们的[目标计数](../guides/object-counting.md)详细指南设置 YOLO11 进行实时视频流分析。只需安装 YOLO11，加载模型，然后处理视频帧即可动态计数目标。

### 使用 Ultralytics YOLO 用于安全系统有什么好处？

Ultralytics YOLO11 通过提供实时目标检测和警报机制来增强安全系统。通过使用 YOLO11，您可以创建一个安全报警系统，在监控区域检测到新目标时触发警报。了解如何使用 YOLO11 设置[安全报警系统](../guides/security-alarm-system.md)以实现强大的安全监控。

### Ultralytics YOLO 如何改进队列管理系统？

Ultralytics YOLO11 可以通过准确计数和跟踪队列中的人员来显著改进队列管理系统，从而帮助减少等待时间并优化服务效率。按照我们的[队列管理](../guides/queue-management.md)详细指南了解如何实现 YOLO11 进行有效的队列监控和分析。

### Ultralytics YOLO 可以用于健身监控吗？

是的，Ultralytics YOLO11 可以有效地用于实时跟踪和分析健身动作来监控健身活动。这允许对运动姿势和表现进行精确评估。探索我们的[健身监控](../guides/workouts-monitoring.md)指南，了解如何使用 YOLO11 设置 AI 驱动的健身监控系统。

### Ultralytics YOLO 如何帮助创建用于[数据可视化](https://www.ultralytics.com/glossary/data-visualization)的热力图？

Ultralytics YOLO11 可以生成热力图来可视化给定区域的数据强度，突出显示高活动或感兴趣的区域。此功能在理解各种计算机视觉任务中的模式和趋势方面特别有用。了解更多关于使用 YOLO11 创建和使用[热力图](../guides/heatmaps.md)进行全面数据分析和可视化的信息。
