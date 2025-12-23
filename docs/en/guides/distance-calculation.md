---
comments: true
description: 学习如何使用 Ultralytics YOLO11 计算物体之间的距离，实现精确的空间定位和场景理解。
keywords: Ultralytics, YOLO11, 距离计算, 计算机视觉, 目标跟踪, 空间定位
---

# 使用 Ultralytics YOLO11 进行距离计算

## 什么是距离计算？

在指定空间内测量两个物体之间的间隙称为距离计算。在 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 中，使用[边界框](https://www.ultralytics.com/glossary/bounding-box)质心来计算用户高亮显示的边界框之间的距离。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Oe0vmsvnY74"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO 以像素为单位估计检测到的物体之间的距离 🚀
</p>

## 可视化效果

|                                         使用 Ultralytics YOLO11 进行距离计算                                         |
| :---------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLO11 距离计算](https://github.com/ultralytics/docs/releases/download/0/distance-calculation.avif) |

## 距离计算的优势

- **定位[精度](https://www.ultralytics.com/glossary/precision)**：增强[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务中的精确空间定位。
- **尺寸估计**：允许估计物体大小，以便更好地理解上下文。
- **场景理解**：改善 3D 场景理解，以便在[自动驾驶车辆](https://www.ultralytics.com/glossary/autonomous-vehicles)和监控系统等应用中做出更好的决策。
- **碰撞避免**：使系统能够通过监控移动物体之间的距离来检测潜在碰撞。
- **空间分析**：便于分析监控环境中物体的关系和交互。

???+ tip "距离计算"

    - 用鼠标左键点击任意两个边界框来计算距离。
    - 使用鼠标右键删除所有绘制的点。
    - 在帧中任意位置左键点击以添加新点。

???+ warning "距离是估计值"

    距离是估计值，可能不完全准确，因为它是使用 2D 数据计算的，
    缺乏深度信息。

!!! example "使用 Ultralytics YOLO 进行距离计算"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化距离计算对象
        distancecalculator = solutions.DistanceCalculation(
            model="yolo11n.pt",  # YOLO11 模型文件路径
            show=True,  # 显示输出
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或处理完成。")
                break

            results = distancecalculator(im0)

            print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `DistanceCalculation()` 参数

下表列出了 `DistanceCalculation` 的参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model"]) }}

您还可以在 `DistanceCalculation` 解决方案中使用各种 `track` 参数。

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还提供以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 实现细节

`DistanceCalculation` 类通过跨视频帧跟踪物体并计算所选边界框质心之间的欧几里得距离来工作。当您点击两个物体时，该解决方案：

1. 提取所选边界框的质心（中心点）
2. 计算这些质心之间的欧几里得距离（以像素为单位）
3. 在帧上显示距离，并在物体之间绘制连接线

该实现使用 `mouse_event_for_distance` 方法来处理鼠标交互，允许用户根据需要选择物体和清除选择。`process` 方法处理逐帧处理、跟踪物体和计算距离。

## 应用场景

使用 YOLO11 进行距离计算有许多实际应用：

- **零售分析**：测量顾客与产品的接近程度，分析商店布局效果
- **工业安全**：监控工人与机械之间的安全距离
- **交通管理**：分析车辆间距，检测跟车过近
- **体育分析**：计算球员、球和关键场地位置之间的距离
- **医疗保健**：确保等候区的适当距离，监控患者移动
- **机器人技术**：使机器人能够与障碍物和人保持适当距离

## 常见问题

### 如何使用 Ultralytics YOLO11 计算物体之间的距离？

要使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 计算物体之间的距离，您需要识别检测到的物体的边界框质心。此过程涉及从 Ultralytics 的 `solutions` 模块初始化 `DistanceCalculation` 类，并使用模型的跟踪输出来计算距离。

### 使用 Ultralytics YOLO11 进行距离计算有什么优势？

使用 Ultralytics YOLO11 进行距离计算有以下几个优势：

- **定位精度**：为物体提供精确的空间定位。
- **尺寸估计**：帮助估计物理尺寸，有助于更好地理解上下文。
- **场景理解**：增强 3D 场景理解，有助于在自动驾驶和监控等应用中做出更好的决策。
- **实时处理**：即时执行计算，适合实时视频分析。
- **集成能力**：与其他 YOLO11 解决方案（如[目标跟踪](../modes/track.md)和[速度估计](speed-estimation.md)）无缝协作。

### 我可以使用 Ultralytics YOLO11 在实时视频流中进行距离计算吗？

是的，您可以使用 Ultralytics YOLO11 在实时视频流中进行距离计算。该过程涉及使用 [OpenCV](https://www.ultralytics.com/glossary/opencv) 捕获视频帧，运行 YOLO11 [目标检测](https://www.ultralytics.com/glossary/object-detection)，并使用 `DistanceCalculation` 类计算连续帧中物体之间的距离。有关详细实现，请参阅[视频流示例](#使用-ultralytics-yolo11-进行距离计算)。

### 如何删除使用 Ultralytics YOLO11 进行距离计算时绘制的点？

要删除使用 Ultralytics YOLO11 进行距离计算时绘制的点，您可以使用鼠标右键点击。此操作将清除您绘制的所有点。有关更多详细信息，请参阅[距离计算示例](#使用-ultralytics-yolo11-进行距离计算)下的注释部分。

### 在 Ultralytics YOLO11 中初始化 DistanceCalculation 类的关键参数是什么？

在 Ultralytics YOLO11 中初始化 `DistanceCalculation` 类的关键参数包括：

- `model`：YOLO11 模型文件的路径。
- `tracker`：要使用的跟踪算法（默认为 'botsort.yaml'）。
- `conf`：检测的置信度阈值。
- `show`：显示输出的标志。

有关详尽的列表和默认值，请参阅 [DistanceCalculation 的参数](#distancecalculation-参数)。
