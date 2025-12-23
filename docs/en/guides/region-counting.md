---
comments: true
description: 学习如何使用 Ultralytics YOLO11 在指定区域内进行精确目标计数，提高各种应用的效率。
keywords: 目标计数, 区域, YOLO11, 计算机视觉, Ultralytics, 效率, 准确率, 自动化, 实时, 应用, 监控, 监测
---

# 使用 Ultralytics YOLO 在不同区域进行目标计数 🚀

## 什么是区域目标计数？

使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) 在区域内进行[目标计数](../guides/object-counting.md)涉及使用先进的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)精确确定指定区域内的目标数量。这种方法对于优化流程、增强安全性和提高各种应用的效率非常有价值。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/mzLfC13ISF4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO11 在不同区域进行目标计数 | Ultralytics 解决方案 🚀
</p>

## 区域目标计数的优势

- **[精度](https://www.ultralytics.com/glossary/precision)和准确率**：使用先进计算机视觉在区域内进行目标计数可确保精确准确的计数，最大限度地减少通常与手动计数相关的错误。
- **效率提升**：自动化目标计数提高运营效率，提供实时结果并简化不同应用的流程。
- **多功能性和应用**：区域目标计数的多功能性使其适用于从制造和监控到交通监测的各种领域，有助于其广泛的实用性和有效性。

## 实际应用

|                                                                                      零售                                                                                       |                                                                                 市场街道                                                                                  |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO11 在不同区域进行人员计数](https://github.com/ultralytics/docs/releases/download/0/people-counting-different-region-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO11 在不同区域进行人群计数](https://github.com/ultralytics/docs/releases/download/0/crowd-counting-different-region-ultralytics-yolov8.avif) |
|                                                           使用 Ultralytics YOLO11 在不同区域进行人员计数                                                            |                                                           使用 Ultralytics YOLO11 在不同区域进行人群计数                                                           |

## 使用示例

!!! example "使用 Ultralytics YOLO 进行区域计数"

    === "Python"

         ```python
         import cv2

         from ultralytics import solutions

         cap = cv2.VideoCapture("path/to/video.mp4")
         assert cap.isOpened(), "读取视频文件出错"

         # 以列表形式传入区域
         # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

         # 以字典形式传入区域
         region_points = {
             "region-01": [(50, 50), (250, 50), (250, 250), (50, 250)],
             "region-02": [(640, 640), (780, 640), (780, 720), (640, 720)],
         }

         # 视频写入器
         w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
         video_writer = cv2.VideoWriter("region_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

         # 初始化区域计数器对象
         regioncounter = solutions.RegionCounter(
             show=True,  # 显示帧
             region=region_points,  # 传入区域点
             model="yolo11n.pt",  # 用于区域计数的模型，例如 yolo11s.pt
         )

         # 处理视频
         while cap.isOpened():
             success, im0 = cap.read()

             if not success:
                 print("视频帧为空或处理完成。")
                 break

             results = regioncounter(im0)

             # print(results)  # 访问输出

             video_writer.write(results.plot_im)

         cap.release()
         video_writer.release()
         cv2.destroyAllWindows()  # 销毁所有打开的窗口
         ```

!!! tip "Ultralytics 示例代码"

      Ultralytics 区域计数模块可在我们的[示例部分](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Region-Counter/yolov8_region_counter.py)中找到。您可以探索此示例进行代码自定义并修改以适应您的特定用例。

### `RegionCounter` 参数

下表列出了 `RegionCounter` 的参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

`RegionCounter` 解决方案支持使用目标跟踪参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化设置：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 常见问题

### 什么是使用 Ultralytics YOLO11 在指定区域内进行目标计数？

使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 在指定区域内进行目标计数涉及使用先进的计算机视觉检测和统计定义区域内的目标数量。这种精确的方法提高了制造、监控和交通监测等各种应用的效率和[准确率](https://www.ultralytics.com/glossary/accuracy)。

### 如何使用 Ultralytics YOLO11 运行基于区域的目标计数脚本？

按照以下步骤在 Ultralytics YOLO11 中运行目标计数：

1. 克隆 Ultralytics 仓库并导航到目录：

    ```bash
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics/examples/YOLOv8-Region-Counter
    ```

2. 执行区域计数脚本：
    ```bash
    python yolov8_region_counter.py --source "path/to/video.mp4" --save-img
    ```

有关更多选项，请访问[使用示例](#使用示例)部分。

### 为什么我应该使用 Ultralytics YOLO11 进行区域目标计数？

使用 Ultralytics YOLO11 进行区域目标计数有多项优势：

1. **实时处理**：YOLO11 的架构支持快速推理，使其非常适合需要即时计数结果的应用。
2. **灵活的区域定义**：该解决方案允许您将多个自定义区域定义为多边形、矩形或线，以满足您的特定监控需求。
3. **多类别支持**：在同一区域内同时计数不同的目标类型，提供全面的分析。
4. **集成能力**：通过 Ultralytics Python API 或命令行界面轻松与现有系统集成。

在[优势](#区域目标计数的优势)部分探索更深入的好处。

### 区域目标计数有哪些实际应用？

使用 Ultralytics YOLO11 进行目标计数可应用于众多实际场景：

- **零售分析**：计数商店不同区域的顾客以优化布局和人员配置。
- **交通管理**：监控特定路段或交叉口的车辆流量。
- **制造**：跟踪通过不同生产区域的产品。
- **仓库运营**：计数指定存储区域的库存物品。
- **公共安全**：在活动期间监控特定区域的人群密度。

在[实际应用](#实际应用)部分和 [TrackZone](../guides/trackzone.md) 解决方案中探索更多示例，了解额外的基于区域的监控功能。
