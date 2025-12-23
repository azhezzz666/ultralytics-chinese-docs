---
comments: true
description: 学习如何使用 Ultralytics YOLO11 管理和优化队列，以减少等待时间并提高各种实际应用的效率。
keywords: 队列管理, YOLO11, Ultralytics, 减少等待时间, 效率, 客户满意度, 零售, 机场, 医疗, 银行
---

# 使用 Ultralytics YOLO11 进行队列管理 🚀

## 什么是队列管理？

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-monitor-objects-in-queue-using-queue-management-solution.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开队列管理"></a>

使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) 进行队列管理涉及组织和控制人员或车辆排队，以减少等待时间并提高效率。它是关于优化队列以提高零售、银行、机场和医疗设施等各种环境中的客户满意度和系统性能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gxr9SpYPLh0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO 构建队列管理系统 | 零售、银行和人群用例 🚀
</p>

## 队列管理的优势

- **减少等待时间**：队列管理系统高效组织队列，最大限度地减少客户等待时间。这提高了满意度，因为客户花更少的时间等待，更多的时间与产品或服务互动。
- **提高效率**：实施队列管理使企业能够更有效地分配资源。通过分析队列数据和优化人员部署，企业可以简化运营、降低成本并提高整体生产力。
- **实时洞察**：YOLO11 驱动的队列管理提供队列长度和等待时间的即时数据，使管理人员能够快速做出明智的决策。
- **增强客户体验**：通过减少与长时间等待相关的挫败感，企业可以显著提高客户满意度和忠诚度。

## 实际应用

|                                                                                            物流                                                                                            |                                                                            零售                                                                             |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO11 在机场售票柜台进行队列管理](https://github.com/ultralytics/docs/releases/download/0/queue-management-airport-ticket-counter-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO11 在人群中进行队列监控](https://github.com/ultralytics/docs/releases/download/0/queue-monitoring-crowd-ultralytics-yolov8.avif) |
|                                                               使用 Ultralytics YOLO11 在机场售票柜台进行队列管理                                                               |                                                         使用 Ultralytics YOLO11 在人群中进行队列监控                                                          |

!!! example "使用 Ultralytics YOLO 进行队列管理"

    === "命令行"

        ```bash
        # 运行队列示例
        yolo solutions queue show=True

        # 传入视频源
        yolo solutions queue source="path/to/video.mp4"

        # 传入队列坐标
        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 定义队列点
        queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # 区域点
        # queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]    # 多边形点

        # 初始化队列管理器对象
        queuemanager = solutions.QueueManager(
            show=True,  # 显示输出
            model="yolo11n.pt",  # YOLO11 模型文件路径
            region=queue_region,  # 传入队列区域点
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("视频帧为空或处理完成。")
                break
            results = queuemanager(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `QueueManager` 参数

下表列出了 `QueueManager` 的参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

`QueueManagement` 解决方案还支持一些 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还可以使用以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 实施策略

在使用 YOLO11 实施队列管理时，请考虑以下最佳实践：

1. **战略性摄像头放置**：将摄像头放置在能够捕获整个队列区域且无遮挡的位置。
2. **定义适当的队列区域**：根据您空间的物理布局仔细设置队列边界。
3. **调整检测置信度**：根据照明条件和人群密度微调置信度阈值。
4. **与现有系统集成**：将您的队列管理解决方案与数字标牌或员工通知系统连接，以实现自动响应。

## 常见问题

### 如何使用 Ultralytics YOLO11 进行实时队列管理？

要使用 Ultralytics YOLO11 进行实时队列管理，您可以按照以下步骤操作：

1. 使用 `YOLO("yolo11n.pt")` 加载 YOLO11 模型。
2. 使用 `cv2.VideoCapture` 捕获视频流。
3. 定义队列管理的感兴趣区域 (ROI)。
4. 处理帧以检测目标并管理队列。

以下是一个最小示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

queuemanager = solutions.QueueManager(
    model="yolo11n.pt",
    region=queue_region,
    line_width=3,
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        results = queuemanager(im0)

cap.release()
cv2.destroyAllWindows()
```

利用 Ultralytics [HUB](https://docs.ultralytics.com/hub/) 可以通过提供用户友好的平台来部署和管理您的队列管理解决方案，从而简化此过程。

### 使用 Ultralytics YOLO11 进行队列管理有哪些主要优势？

使用 Ultralytics YOLO11 进行队列管理有多项好处：

- **大幅减少等待时间**：高效组织队列，减少客户等待时间并提高满意度。
- **提高效率**：分析队列数据以优化人员部署和运营，从而降低成本。
- **实时警报**：为长队列提供实时通知，实现快速干预。
- **可扩展性**：易于在零售、机场和医疗等不同环境中扩展。

有关更多详细信息，请探索我们的[队列管理](https://docs.ultralytics.com/reference/solutions/queue_management/)解决方案。

### 为什么我应该选择 Ultralytics YOLO11 而不是 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 或 Detectron2 等竞争对手进行队列管理？

Ultralytics YOLO11 相比 TensorFlow 和 Detectron2 在队列管理方面有多项优势：

- **实时性能**：YOLO11 以其实时检测能力著称，提供更快的处理速度。
- **易用性**：Ultralytics 通过 [Ultralytics HUB](https://docs.ultralytics.com/hub/) 提供从训练到部署的用户友好体验。
- **预训练模型**：访问一系列预训练模型，最大限度地减少设置所需的时间。
- **社区支持**：广泛的文档和活跃的社区支持使问题解决更加容易。

了解如何开始使用 [Ultralytics YOLO](https://docs.ultralytics.com/quickstart/)。

### Ultralytics YOLO11 能否处理多种类型的队列，例如机场和零售环境中的队列？

是的，Ultralytics YOLO11 可以管理各种类型的队列，包括机场和零售环境中的队列。通过使用特定区域和设置配置 QueueManager，YOLO11 可以适应不同的队列布局和密度。

机场示例：

```python
queue_region_airport = [(50, 600), (1200, 600), (1200, 550), (50, 550)]
queue_airport = solutions.QueueManager(
    model="yolo11n.pt",
    region=queue_region_airport,
    line_width=3,
)
```

有关多样化应用的更多信息，请查看我们的[实际应用](#实际应用)部分。

### Ultralytics YOLO11 在队列管理中有哪些实际应用？

Ultralytics YOLO11 在队列管理中有多种实际应用：

- **零售**：监控结账队列以减少等待时间并提高客户满意度。
- **机场**：管理售票柜台和安检站的队列，以获得更顺畅的乘客体验。
- **医疗**：优化诊所和医院的患者流量。
- **银行**：通过高效管理队列来增强银行的客户服务。

查看我们关于[实时队列监控的博客](https://www.ultralytics.com/blog/a-look-at-real-time-queue-monitoring-enabled-by-computer-vision)，了解更多关于计算机视觉如何改变各行业队列监控的信息。
