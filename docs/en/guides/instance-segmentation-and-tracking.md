---
comments: true
description: 使用 Ultralytics YOLO11 掌握实例分割和跟踪。学习精确目标识别和跟踪的技术。
keywords: 实例分割, 跟踪, YOLO11, Ultralytics, 目标检测, 机器学习, 计算机视觉, python
---

# 使用 Ultralytics YOLO11 进行实例分割和跟踪 🚀

## 什么是实例分割？

[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)是一项计算机视觉任务，涉及在像素级别识别和勾勒图像中的各个物体。与仅按类别对像素进行分类的[语义分割](https://www.ultralytics.com/glossary/semantic-segmentation)不同，实例分割会唯一标记并精确描绘每个物体实例，这对于需要详细空间理解的应用（如医学成像、自动驾驶和工业自动化）至关重要。

[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) 提供强大的实例分割功能，能够实现精确的物体边界检测，同时保持 YOLO 模型闻名的速度和效率。

Ultralytics 包中有两种类型的实例分割跟踪可用：

- **带类别物体的实例分割**：每个类别物体被分配一个唯一的颜色，以便清晰地进行视觉分离。

- **带物体轨迹的实例分割**：每个轨迹用不同的颜色表示，便于在视频帧之间轻松识别和跟踪。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/75G_S1Ngji8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO11 进行带物体跟踪的实例分割
</p>

## 示例

|                                                        实例分割                                                         |                                                                  实例分割 + 物体跟踪                                                                  |
| :----------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics 实例分割](https://github.com/ultralytics/docs/releases/download/0/ultralytics-instance-segmentation.avif) | ![Ultralytics 带物体跟踪的实例分割](https://github.com/ultralytics/docs/releases/download/0/ultralytics-instance-segmentation-object-tracking.avif) |
|                                                 Ultralytics 实例分割 😍                                                 |                                                         Ultralytics 带物体跟踪的实例分割 🔥                                                         |

!!! example "使用 Ultralytics YOLO 进行实例分割"

    === "命令行"

        ```bash
        # 使用 Ultralytics YOLO11 进行实例分割
        yolo solutions isegment show=True

        # 传入视频源
        yolo solutions isegment source="path/to/video.mp4"

        # 监控特定类别
        yolo solutions isegment classes="[0, 5]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("isegment_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化实例分割对象
        isegment = solutions.InstanceSegmentation(
            show=True,  # 显示输出
            model="yolo11n-seg.pt",  # model="yolo11n-seg.pt" 用于使用 YOLO11 进行物体分割
            # classes=[0, 2],  # 分割特定类别，例如使用预训练模型分割人和汽车
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或视频处理已成功完成。")
                break

            results = isegment(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `InstanceSegmentation` 参数

下表列出了 `InstanceSegmentation` 的参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

您还可以在 `InstanceSegmentation` 解决方案中利用 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还提供以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 实例分割的应用

使用 YOLO11 进行实例分割在各个行业有众多实际应用：

### 废物管理和回收

YOLO11 可用于[废物管理设施](https://www.ultralytics.com/blog/simplifying-e-waste-management-with-ai-innovations)中识别和分类不同类型的材料。该模型可以高精度地分割塑料废物、纸板、金属和其他可回收物，使自动分拣系统能够更高效地处理废物。考虑到全球产生的 70 亿吨塑料废物中只有约 10% 被回收，这一点尤为重要。

### 自动驾驶车辆

在[自动驾驶汽车](https://www.ultralytics.com/solutions/ai-in-automotive)中，实例分割有助于在像素级别识别和跟踪行人、车辆、交通标志和其他道路元素。这种对环境的精确理解对于导航和安全决策至关重要。YOLO11 的实时性能使其非常适合这些时间敏感的应用。

### 医学成像

实例分割可以在医学扫描中识别和勾勒肿瘤、器官或细胞结构。YOLO11 精确描绘物体边界的能力使其在[医学诊断](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency)和治疗规划中非常有价值。

### 建筑工地监控

在建筑工地，实例分割可以跟踪重型机械、工人和材料。这有助于通过监控设备位置和检测工人进入危险区域来确保安全，同时还可以优化工作流程和资源分配。

## 注意

如有任何疑问，请随时在 [Ultralytics Issues 部分](https://github.com/ultralytics/ultralytics/issues/new/choose)或下面提到的讨论部分发布您的问题。

## 常见问题

### 如何使用 Ultralytics YOLO11 执行实例分割？

要使用 Ultralytics YOLO11 执行实例分割，请使用 YOLO11 的分割版本初始化 YOLO 模型，并通过它处理视频帧。这是一个简化的代码示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "读取视频文件出错"

# 视频写入器
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 初始化 InstanceSegmentation
isegment = solutions.InstanceSegmentation(
    show=True,  # 显示输出
    model="yolo11n-seg.pt",  # model="yolo11n-seg.pt" 用于使用 YOLO11 进行物体分割
)

# 处理视频
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("视频帧为空或处理完成。")
        break
    results = isegment(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

在 [Ultralytics YOLO11 指南](https://docs.ultralytics.com/tasks/segment/)中了解更多关于实例分割的信息。

### Ultralytics YOLO11 中实例分割和物体跟踪有什么区别？

实例分割识别并勾勒图像中的各个物体，为每个物体提供唯一的标签和掩码。物体跟踪通过在视频帧之间为物体分配一致的 ID 来扩展此功能，便于随时间持续跟踪相同的物体。当两者结合时，如 YOLO11 的实现，您可以获得强大的功能来分析视频中的物体移动和行为，同时保持精确的边界信息。

### 为什么我应该使用 Ultralytics YOLO11 而不是 Mask R-CNN 或 Faster R-CNN 等其他模型进行实例分割和跟踪？

与 Mask R-CNN 或 Faster R-CNN 等其他模型相比，Ultralytics YOLO11 提供实时性能、卓越的[准确率](https://www.ultralytics.com/glossary/accuracy)和易用性。YOLO11 在单次传递中处理图像（单阶段检测），使其在保持高精度的同时显著更快。它还提供与 [Ultralytics HUB](https://www.ultralytics.com/hub) 的无缝集成，允许用户高效地管理模型、数据集和训练管道。对于需要速度和准确率的应用，YOLO11 提供了最佳平衡。

### Ultralytics 是否提供适合训练 YOLO11 模型进行实例分割和跟踪的数据集？

是的，Ultralytics 提供了几个适合训练 YOLO11 模型进行实例分割的数据集，包括 [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/)、[COCO8-Seg](https://docs.ultralytics.com/datasets/segment/coco8-seg/)（用于快速测试的较小子集）、[Package-Seg](https://docs.ultralytics.com/datasets/segment/package-seg/) 和 [Crack-Seg](https://docs.ultralytics.com/datasets/segment/crack-seg/)。这些数据集带有实例分割任务所需的像素级标注。对于更专业的应用，您还可以按照 Ultralytics 格式创建自定义数据集。完整的数据集信息和使用说明可以在 [Ultralytics 数据集文档](https://docs.ultralytics.com/datasets/)中找到。
