---
comments: true
description: 学习使用 Ultralytics YOLO11 准确识别和实时计数目标，适用于人群分析和监控等应用。
keywords: 目标计数, YOLO11, Ultralytics, 实时目标检测, 人工智能, 深度学习, 目标跟踪, 人群分析, 监控, 资源优化
---

# 使用 Ultralytics YOLO11 进行目标计数

## 什么是目标计数？

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-count-the-objects-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开目标计数"></a>

使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) 进行目标计数涉及在视频和摄像头流中准确识别和计数特定目标。YOLO11 凭借其最先进的算法和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)能力，在人群分析和监控等各种场景中表现出色，提供高效精确的实时目标计数。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/vKcD44GkSF8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO11 进行实时目标计数 🍏
</p>

## 目标计数的优势

- **资源优化**：目标计数通过提供准确的计数来促进高效的资源管理，优化[库存管理](https://docs.ultralytics.com/guides/analytics/)等应用中的资源分配。
- **增强安全性**：目标计数通过准确跟踪和计数实体来增强安全和监控，有助于主动[威胁检测](https://docs.ultralytics.com/guides/security-alarm-system/)。
- **明智决策**：目标计数为决策提供有价值的洞察，优化零售、[交通管理](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination)和其他各种领域的流程。

## 实际应用

|                                                                        物流                                                                        |                                                                         水产养殖                                                                          |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO11 进行传送带包裹计数](https://github.com/ultralytics/docs/releases/download/0/conveyor-belt-packets-counting.avif) | ![使用 Ultralytics YOLO11 进行海洋鱼类计数](https://github.com/ultralytics/docs/releases/download/0/fish-counting-in-sea-using-ultralytics-yolov8.avif) |
|                                                 使用 Ultralytics YOLO11 进行传送带包裹计数                                                 |                                                        使用 Ultralytics YOLO11 进行海洋鱼类计数                                                         |

!!! example "使用 Ultralytics YOLO 进行目标计数"

    === "命令行"

        ```bash
        # 运行计数示例
        yolo solutions count show=True

        # 传入视频源
        yolo solutions count source="path/to/video.mp4"

        # 传入区域坐标
        yolo solutions count region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

        `region` 参数接受两个点（用于线）或三个或更多点的多边形。按照它们应该连接的顺序定义坐标，以便计数器准确知道进入和退出发生的位置。

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # region_points = [(20, 400), (1080, 400)]                                      # 线计数
        region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # 矩形区域
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # 多边形区域

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化目标计数器对象
        counter = solutions.ObjectCounter(
            show=True,  # 显示输出
            region=region_points,  # 传入区域点
            model="yolo11n.pt",  # model="yolo11n-obb.pt" 用于使用 OBB 模型进行目标计数
            # classes=[0, 2],  # 计数特定类别，例如使用 COCO 预训练模型计数人和汽车
            # tracker="botsort.yaml",  # 选择跟踪器，例如 "bytetrack.yaml"
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或处理完成。")
                break

            results = counter(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `ObjectCounter` 参数

下表列出了 `ObjectCounter` 的参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "show_in", "show_out", "region"]) }}

`ObjectCounter` 解决方案允许使用多个 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 常见问题

### 如何使用 Ultralytics YOLO11 在视频中计数目标？

要使用 Ultralytics YOLO11 在视频中计数目标，您可以按照以下步骤操作：

1. 导入必要的库（`cv2`、`ultralytics`）。
2. 定义计数区域（例如，多边形、线等）。
3. 设置视频捕获并初始化目标计数器。
4. 处理每一帧以跟踪目标并在定义的区域内计数。

以下是在区域内计数的简单示例：

```python
import cv2

from ultralytics import solutions


def count_objects_in_region(video_path, output_video_path, model_path):
    """在视频中的特定区域内计数目标。"""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "读取视频文件出错"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("视频帧为空或处理完成。")
            break
        results = counter(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region("path/to/video.mp4", "output_video.avi", "yolo11n.pt")
```

有关更高级的配置和选项，请查看 [RegionCounter 解决方案](https://docs.ultralytics.com/guides/region-counting/)以同时在多个区域中计数目标。

### 使用 Ultralytics YOLO11 进行目标计数有什么优势？

使用 Ultralytics YOLO11 进行目标计数有多项优势：

1. **资源优化**：它通过提供准确的计数来促进高效的资源管理，帮助优化[库存管理](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management)等行业的资源分配。
2. **增强安全性**：它通过准确跟踪和计数实体来增强安全和监控，有助于主动威胁检测和[安全系统](https://docs.ultralytics.com/guides/security-alarm-system/)。
3. **明智决策**：它为决策提供有价值的洞察，优化零售、交通管理等领域的流程。
4. **实时处理**：YOLO11 的架构支持[实时推理](https://www.ultralytics.com/glossary/real-time-inference)，使其适用于实时视频流和时间敏感的应用。

有关实现示例和实际应用，请探索 [TrackZone 解决方案](https://docs.ultralytics.com/guides/trackzone/)以在特定区域中跟踪目标。

### 如何使用 Ultralytics YOLO11 计数特定类别的目标？

要使用 Ultralytics YOLO11 计数特定类别的目标，您需要在跟踪阶段指定您感兴趣的类别。以下是 Python 示例：

```python
import cv2

from ultralytics import solutions


def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """在视频中计数特定类别的目标。"""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "读取视频文件出错"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_points = [(20, 400), (1080, 400)]
    counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, classes=classes_to_count)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("视频帧为空或处理完成。")
            break
        results = counter(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_specific_classes("path/to/video.mp4", "output_specific_classes.avi", "yolo11n.pt", [0, 2])
```

在此示例中，`classes_to_count=[0, 2]` 表示它计数类别 `0` 和 `2` 的目标（例如，COCO 数据集中的人和汽车）。您可以在 [COCO 数据集文档](https://docs.ultralytics.com/datasets/detect/coco/)中找到有关类别索引的更多信息。

### 为什么我应该在实时应用中使用 YOLO11 而不是其他[目标检测](https://www.ultralytics.com/glossary/object-detection)模型？

Ultralytics YOLO11 相比 [Faster R-CNN](https://docs.ultralytics.com/compare/yolo11-vs-efficientdet/)、SSD 和以前的 YOLO 版本等其他目标检测模型提供了多项优势：

1. **速度和效率**：YOLO11 提供实时处理能力，使其非常适合需要高速推理的应用，如监控和[自动驾驶](https://www.ultralytics.com/blog/ai-in-self-driving-cars)。
2. **[准确率](https://www.ultralytics.com/glossary/accuracy)**：它为目标检测和跟踪任务提供最先进的准确率，减少误报数量并提高整体系统可靠性。
3. **易于集成**：YOLO11 提供与各种平台和设备的无缝集成，包括移动和[边缘设备](https://docs.ultralytics.com/guides/nvidia-jetson/)，这对于现代 AI 应用至关重要。
4. **灵活性**：支持目标检测、[分割](https://docs.ultralytics.com/tasks/segment/)和跟踪等各种任务，具有可配置的模型以满足特定用例需求。

查看 Ultralytics [YOLO11 文档](https://docs.ultralytics.com/models/yolo11/)以深入了解其功能和性能比较。

### YOLO11 可以用于人群分析和交通管理等高级应用吗？

是的，Ultralytics YOLO11 非常适合人群分析和交通管理等高级应用，因为它具有实时检测能力、可扩展性和集成灵活性。其高级功能允许在动态环境中进行高精度的目标跟踪、计数和分类。示例用例包括：

- **人群分析**：监控和管理大型聚会，确保安全并通过[基于区域的计数](https://docs.ultralytics.com/guides/region-counting/)优化人流。
- **交通管理**：跟踪和计数车辆，分析交通模式，并通过[速度估计](https://docs.ultralytics.com/guides/speed-estimation/)功能实时管理拥堵。
- **零售分析**：分析客户移动模式和产品互动，以优化商店布局并改善客户体验。
- **工业自动化**：计数传送带上的产品并监控生产线以进行质量控制和效率改进。

有关更专业的应用，请探索 [Ultralytics 解决方案](https://docs.ultralytics.com/solutions/)以获取为现实世界计算机视觉挑战设计的全面工具集。
