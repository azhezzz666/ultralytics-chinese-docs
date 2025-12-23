---
comments: true
description: 利用 Ultralytics YOLO11 的强大功能在各种数据源上进行实时高速推理。了解预测模式、关键特性和实际应用。
keywords: Ultralytics, YOLO11, 模型预测, 推理, 预测模式, 实时推理, 计算机视觉, 机器学习, 流式处理, 高性能
---

# 使用 Ultralytics YOLO 进行模型预测

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统和集成">

## 简介

在[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域，理解视觉数据的过程通常称为推理或预测。Ultralytics YOLO11 提供了一个称为**预测模式**的强大功能，专为在各种数据源上进行高性能实时推理而设计。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/YKbBXWBJloY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何从 Ultralytics YOLO11 任务中提取结果用于自定义项目 🚀
</p>

## 实际应用

|                   制造业                    |                        体育                        |                   安全                    |
| :-----------------------------------------: | :------------------------------------------------: | :---------------------------------------: |
| ![车辆零部件检测][car spare parts]          | ![足球运动员检测][football player detect]          | ![人员跌倒检测][human fall detect]        |
|           车辆零部件检测                    |              足球运动员检测                        |            人员跌倒检测                   |

## 为什么使用 Ultralytics YOLO 进行推理？

以下是您应该考虑使用 YOLO11 预测模式满足各种推理需求的原因：

- **多功能性：** 能够对图像、视频甚至实时流进行推理。
- **性能：** 专为实时高速处理而设计，不牺牲[精度](https://www.ultralytics.com/glossary/accuracy)。
- **易用性：** 直观的 Python 和 CLI 接口，便于快速部署和测试。
- **高度可定制：** 各种设置和参数，可根据您的特定要求调整模型的推理行为。

### 预测模式的关键特性

YOLO11 的预测模式设计为稳健且多功能，具有以下特点：

- **多数据源兼容性：** 无论您的数据是单个图像、图像集合、视频文件还是实时视频流，预测模式都能处理。
- **流式模式：** 使用流式功能生成内存高效的 `Results` 对象生成器。通过在预测器的调用方法中设置 `stream=True` 来启用。
- **批处理：** 在单个批次中处理多个图像或视频帧，进一步减少总推理时间。
- **集成友好：** 由于其灵活的 API，可轻松与现有数据管道和其他软件组件集成。

Ultralytics YOLO 模型在推理期间传递 `stream=True` 时返回 Python `Results` 对象列表或内存高效的 `Results` 对象生成器：

!!! example "预测"

    === "使用 `stream=False` 返回列表"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 预训练的 YOLO11n 模型

        # 对图像列表运行批量推理
        results = model(["image1.jpg", "image2.jpg"])  # 返回 Results 对象列表

        # 处理结果列表
        for result in results:
            boxes = result.boxes  # 边界框输出的 Boxes 对象
            masks = result.masks  # 分割掩码输出的 Masks 对象
            keypoints = result.keypoints  # 姿态输出的 Keypoints 对象
            probs = result.probs  # 分类输出的 Probs 对象
            obb = result.obb  # OBB 输出的旋转框对象
            result.show()  # 显示到屏幕
            result.save(filename="result.jpg")  # 保存到磁盘
        ```

    === "使用 `stream=True` 返回生成器"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 预训练的 YOLO11n 模型

        # 对图像列表运行批量推理
        results = model(["image1.jpg", "image2.jpg"], stream=True)  # 返回 Results 对象生成器

        # 处理结果生成器
        for result in results:
            boxes = result.boxes  # 边界框输出的 Boxes 对象
            masks = result.masks  # 分割掩码输出的 Masks 对象
            keypoints = result.keypoints  # 姿态输出的 Keypoints 对象
            probs = result.probs  # 分类输出的 Probs 对象
            obb = result.obb  # OBB 输出的旋转框对象
            result.show()  # 显示到屏幕
            result.save(filename="result.jpg")  # 保存到磁盘
        ```

## 推理源

YOLO11 可以处理不同类型的输入源进行推理，如下表所示。源包括静态图像、视频流和各种数据格式。该表还指示每个源是否可以使用参数 `stream=True` ✅ 在流式模式下使用。流式模式有利于处理视频或实时流，因为它创建结果生成器而不是将所有帧加载到内存中。

!!! tip

    对于处理长视频或大型数据集，使用 `stream=True` 可以有效管理内存。当 `stream=False` 时，所有帧或数据点的结果都存储在内存中，这对于大型输入可能会快速累积并导致内存不足错误。相比之下，`stream=True` 使用生成器，仅将当前帧或数据点的结果保留在内存中，显著减少内存消耗并防止内存不足问题。

| 源         | 示例                                       | 类型            | 备注                                                                                         |
| ---------- | ------------------------------------------ | --------------- | -------------------------------------------------------------------------------------------- |
| 图像       | `'image.jpg'`                              | `str` 或 `Path` | 单个图像文件。                                                                               |
| URL        | `'https://ultralytics.com/images/bus.jpg'` | `str`           | 图像的 URL。                                                                                 |
| 截图       | `'screen'`                                 | `str`           | 捕获屏幕截图。                                                                               |
| PIL        | `Image.open('image.jpg')`                  | `PIL.Image`     | HWC 格式，RGB 通道。                                                                         |
| [OpenCV](https://www.ultralytics.com/glossary/opencv) | `cv2.imread('image.jpg')`                  | `np.ndarray`    | HWC 格式，BGR 通道 `uint8 (0-255)`。                                                         |
| numpy      | `np.zeros((640,1280,3))`                   | `np.ndarray`    | HWC 格式，BGR 通道 `uint8 (0-255)`。                                                         |
| torch      | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | BCHW 格式，RGB 通道 `float32 (0.0-1.0)`。                                                    |
| CSV        | `'sources.csv'`                            | `str` 或 `Path` | 包含图像、视频或目录路径的 CSV 文件。                                                        |
| 视频 ✅    | `'video.mp4'`                              | `str` 或 `Path` | MP4、AVI 等格式的视频文件。                                                                  |
| 目录 ✅    | `'path/'`                                  | `str` 或 `Path` | 包含图像或视频的目录路径。                                                                   |
| glob ✅    | `'path/*.jpg'`                             | `str`           | 匹配多个文件的 glob 模式。使用 `*` 字符作为通配符。                                          |
| YouTube ✅ | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | YouTube 视频的 URL。                                                                         |
| 流 ✅      | `'rtsp://example.com/media.mp4'`           | `str`           | RTSP、RTMP、TCP 等流协议的 URL，或 IP 地址。                                                 |
| 多流 ✅    | `'list.streams'`                           | `str` 或 `Path` | `*.streams` 文本文件，每行一个流 URL，即 8 个流将以批次大小 8 运行。                         |
| 网络摄像头 ✅ | `0`                                        | `int`           | 要运行推理的已连接摄像头设备的索引。                                                         |

## 推理参数

`model.predict()` 接受多个可在推理时传递的参数以覆盖默认值：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 使用参数对 'bus.jpg' 运行推理
        model.predict("https://ultralytics.com/images/bus.jpg", save=True, imgsz=320, conf=0.5)
        ```

    === "CLI"

        ```bash
        # 对 'bus.jpg' 运行推理
        yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
        ```

推理参数：

{% include "macros/predict-args.md" %}

## 处理结果

所有 Ultralytics `predict()` 调用将返回 `Results` 对象列表：

!!! example "Results"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO11n 模型
    model = YOLO("yolo11n.pt")

    # 对图像运行推理
    results = model("https://ultralytics.com/images/bus.jpg")
    results = model(
        [
            "https://ultralytics.com/images/bus.jpg",
            "https://ultralytics.com/images/zidane.jpg",
        ]
    )  # 批量推理
    ```

`Results` 对象具有以下属性：

| 属性         | 类型                  | 描述                                                                             |
| ------------ | --------------------- | -------------------------------------------------------------------------------- |
| `orig_img`   | `np.ndarray`          | 原始图像的 numpy 数组。                                                          |
| `orig_shape` | `tuple`               | 原始图像形状，格式为 (height, width)。                                           |
| `boxes`      | `Boxes, optional`     | 包含检测边界框的 Boxes 对象。                                                    |
| `masks`      | `Masks, optional`     | 包含检测掩码的 Masks 对象。                                                      |
| `probs`      | `Probs, optional`     | 包含分类任务每个类别概率的 Probs 对象。                                          |
| `keypoints`  | `Keypoints, optional` | 包含每个对象检测到的关键点的 Keypoints 对象。                                    |
| `obb`        | `OBB, optional`       | 包含旋转边界框的 OBB 对象。                                                      |
| `speed`      | `dict`                | 每张图像的预处理、推理和后处理速度字典（毫秒）。                                 |
| `names`      | `dict`                | 类别索引到类别名称的映射字典。                                                   |
| `path`       | `str`                 | 图像文件的路径。                                                                 |

`Results` 对象具有以下方法：

| 方法          | 返回类型               | 描述                                                                              |
| ------------- | ---------------------- | --------------------------------------------------------------------------------- |
| `update()`    | `None`                 | 使用新的检测数据（boxes、masks、probs、obb、keypoints）更新 Results 对象。        |
| `cpu()`       | `Results`              | 返回所有张量移动到 CPU 内存的 Results 对象副本。                                  |
| `numpy()`     | `Results`              | 返回所有张量转换为 numpy 数组的 Results 对象副本。                                |
| `cuda()`      | `Results`              | 返回所有张量移动到 GPU 内存的 Results 对象副本。                                  |
| `to()`        | `Results`              | 返回张量移动到指定设备和 dtype 的 Results 对象副本。                              |
| `new()`       | `Results`              | 创建具有相同图像、路径、名称和速度属性的新 Results 对象。                         |
| `plot()`      | `np.ndarray`           | 在输入 RGB 图像上绘制检测结果并返回标注图像。                                     |
| `show()`      | `None`                 | 显示带有标注推理结果的图像。                                                      |
| `save()`      | `str`                  | 将标注的推理结果图像保存到文件并返回文件名。                                      |
| `verbose()`   | `str`                  | 返回每个任务的日志字符串，详细说明检测和分类结果。                                |
| `save_txt()`  | `str`                  | 将检测结果保存到文本文件并返回保存文件的路径。                                    |
| `save_crop()` | `None`                 | 将裁剪的检测图像保存到指定目录。                                                  |
| `summary()`   | `List[Dict[str, Any]]` | 将推理结果转换为带有可选归一化的摘要字典。                                        |
| `to_df()`     | `DataFrame`            | 将检测结果转换为 Polars DataFrame。                                               |
| `to_csv()`    | `str`                  | 将检测结果转换为 CSV 格式。                                                       |
| `to_json()`   | `str`                  | 将检测结果转换为 JSON 格式。                                                      |

[car spare parts]: https://github.com/ultralytics/docs/releases/download/0/vehicle-spare-parts-detection.avif
[football player detect]: https://github.com/ultralytics/docs/releases/download/0/football-player-detection.avif
[human fall detect]: https://github.com/ultralytics/docs/releases/download/0/people-fall-detection.avif

## 常见问题

### 如何使用 Ultralytics YOLO11 对图像运行推理？

要使用 Ultralytics YOLO11 对图像运行推理，您可以使用 Python API 或 CLI：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 定义图像文件路径
        source = "path/to/image.jpg"

        # 对源运行推理
        results = model(source)  # Results 对象列表
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo11n.pt source=path/to/image.jpg
        ```

### YOLO11 预测模式支持哪些数据源？

YOLO11 预测模式支持多种数据源，包括单个图像、目录、视频文件、URL、流和网络摄像头。您可以在推理源表中查看支持的源及其对应的示例。

### 如何在 YOLO11 推理中启用流式模式？

要在 YOLO11 推理中启用流式模式，请在调用预测方法时设置 `stream=True`。这将创建一个内存高效的 `Results` 对象生成器，而不是将所有结果加载到内存中：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("path/to/video.mp4", stream=True)

for result in results:
    # 处理每一帧
    boxes = result.boxes
    result.show()
```

这对于处理长视频或大型数据集特别有用。
