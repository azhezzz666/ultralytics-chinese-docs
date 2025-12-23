---
comments: true
description: 使用 Ultralytics YOLO11 优化停车位并增强安全性。探索实时车辆检测和智能停车解决方案。
keywords: 停车管理, YOLO11, Ultralytics, 车辆检测, 实时跟踪, 停车场优化, 智能停车
---

# 使用 Ultralytics YOLO11 进行停车管理 🚀

## 什么是停车管理系统？

使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) 进行停车管理通过组织车位和监控可用性来确保高效安全的停车。YOLO11 可以通过实时车辆检测和停车占用洞察来改善停车场管理。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/WwXnljc7ZUM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO 实现停车管理 🚀
</p>

## 停车管理系统的优势

- **效率**：停车场管理优化停车位的使用并减少拥堵。
- **安全和保障**：使用 YOLO11 的停车管理通过监控和安全措施提高人员和车辆的安全性。
- **减少排放**：使用 YOLO11 的停车管理通过管理交通流量来最小化停车场中的怠速时间和排放。

## 实际应用

|                                                                     停车管理系统                                                                      |                                                                      停车管理系统                                                                       |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO11 进行停车场分析](https://github.com/ultralytics/docs/releases/download/0/parking-management-aerial-view-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO11 进行停车管理俯视图](https://github.com/ultralytics/docs/releases/download/0/parking-management-top-view-ultralytics-yolov8.avif) |
|                                                      使用 Ultralytics YOLO11 进行停车管理航拍视图                                                       |                                                         使用 Ultralytics YOLO11 进行停车管理俯视图                                                         |

## 停车管理系统代码工作流程

??? note "选择点现在变得简单"

    在停车管理系统中选择停车点是一项关键且复杂的任务。Ultralytics 通过提供"停车位标注器"工具简化了这一过程，让您可以定义停车区域，这些区域可以在后续处理中使用。

**步骤 1：** 从您想要管理停车场的视频或摄像头流中捕获一帧。

**步骤 2：** 使用提供的代码启动图形界面，您可以在其中选择图像并通过鼠标点击开始勾勒停车区域以创建多边形。

!!! example "Ultralytics YOLO 停车位标注器"

    ??? note "安装 `tkinter` 的额外步骤"

        通常，`tkinter` 随 Python 预装。但是，如果没有，您可以使用以下步骤安装：

        - **Linux**: (Debian/Ubuntu): `sudo apt install python3-tk`
        - **Fedora**: `sudo dnf install python3-tkinter`
        - **Arch**: `sudo pacman -S tk`
        - **Windows**: 重新安装 Python 并在安装过程中的**可选功能**中启用 `tcl/tk and IDLE` 复选框
        - **MacOS**: 从 [https://www.python.org/downloads/macos/](https://www.python.org/downloads/macos/) 重新安装 Python 或 `brew install python-tk`

    === "Python"

        ```python
        from ultralytics import solutions

        solutions.ParkingPtsSelection()
        ```

**步骤 3：** 使用多边形定义停车区域后，点击 `save` 将包含数据的 JSON 文件存储在您的工作目录中。

![Ultralytics YOLO11 点选择演示](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-points-selection-demo.avif)

**步骤 4：** 现在您可以使用提供的代码通过 Ultralytics YOLO 进行停车管理。

!!! example "使用 Ultralytics YOLO 进行停车管理"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        # 视频捕获
        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("parking management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化停车管理对象
        parkingmanager = solutions.ParkingManagement(
            model="yolo11n.pt",  # 模型文件路径
            json_file="bounding_boxes.json",  # 停车标注文件路径
        )

        while cap.isOpened():
            ret, im0 = cap.read()
            if not ret:
                break

            results = parkingmanager(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `ParkingManagement` 参数

下表列出了 `ParkingManagement` 的参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "json_file"]) }}

`ParkingManagement` 解决方案允许使用多个 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化选项：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## 常见问题

### Ultralytics YOLO11 如何增强停车管理系统？

Ultralytics YOLO11 通过提供**实时车辆检测**和监控大大增强了停车管理系统。这导致停车位使用优化、拥堵减少，并通过持续监控提高安全性。[停车管理系统](https://github.com/ultralytics/ultralytics)实现高效的交通流量，最小化怠速时间和排放，从而有助于环境可持续性。有关更多详细信息，请参阅[停车管理代码工作流程](#停车管理系统代码工作流程)。

### 使用 Ultralytics YOLO11 进行智能停车有什么好处？

使用 Ultralytics YOLO11 进行智能停车有众多好处：

- **效率**：优化停车位的使用并减少拥堵。
- **安全和保障**：增强监控并确保车辆和行人的安全。
- **环境影响**：通过最小化车辆怠速时间帮助减少排放。在[停车管理系统的优势部分](#停车管理系统的优势)探索更多好处。

### 如何使用 Ultralytics YOLO11 定义停车位？

使用 Ultralytics YOLO11 定义停车位非常简单：

1. 从视频或摄像头流中捕获一帧。
2. 使用提供的代码启动 GUI 以选择图像并绘制多边形来定义停车位。
3. 将标注数据保存为 JSON 格式以进行进一步处理。有关全面说明，请查看上面的点选择部分。

### 我可以为特定的停车管理需求自定义 YOLO11 模型吗？

是的，Ultralytics YOLO11 允许为特定的停车管理需求进行自定义。您可以调整参数，如**占用和可用区域颜色**、文本显示边距等。利用 `ParkingManagement` 类的[参数](#parkingmanagement-参数)，您可以定制模型以满足您的特定需求，确保最大效率和有效性。

### Ultralytics YOLO11 在停车场管理中有哪些实际应用？

Ultralytics YOLO11 在停车场管理中有多种实际应用，包括：

- **停车位检测**：准确识别可用和占用的车位。
- **监控**：通过实时监控增强安全性。
- **交通流量管理**：通过高效的交通处理减少怠速时间和拥堵。展示这些应用的图像可以在[实际应用](#实际应用)中找到。
