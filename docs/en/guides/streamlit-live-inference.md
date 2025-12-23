---
comments: true
description: 学习如何使用 Streamlit 和 Ultralytics YOLO11 设置实时目标检测应用程序。按照本分步指南实现基于网络摄像头的目标检测。
keywords: Streamlit, YOLO11, 实时目标检测, Streamlit 应用程序, YOLO11 Streamlit 教程, 网络摄像头目标检测
---

# 使用 Ultralytics YOLO11 的 Streamlit 应用程序进行实时推理

## 简介

Streamlit 使构建和部署交互式 Web 应用程序变得简单。将其与 Ultralytics YOLO11 结合，可以直接在浏览器中进行实时[目标检测](https://www.ultralytics.com/glossary/object-detection)和分析。YOLO11 的高准确性和速度确保了实时视频流的无缝性能，使其非常适合安防、零售等领域的应用。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/1H9ktpHUUB8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在浏览器中使用 Streamlit 与 Ultralytics 进行实时<a href="https://www.ultralytics.com/glossary/computer-vision-cv">计算机视觉</a>
</p>

|                                                                水产养殖                                                                 |                                                           畜牧业                                                           |
| :----------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO11 进行鱼类检测](https://github.com/ultralytics/docs/releases/download/0/fish-detection-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO11 进行动物检测](https://github.com/ultralytics/docs/releases/download/0/animals-detection-yolov8.avif) |
|                                                  使用 Ultralytics YOLO11 进行鱼类检测                                                   |                                              使用 Ultralytics YOLO11 进行动物检测                                              |

## 实时推理的优势

- **无缝实时目标检测**：Streamlit 与 YOLO11 结合，可直接从网络摄像头画面进行实时目标检测。这允许即时分析和洞察，非常适合[需要即时反馈的应用](https://docs.ultralytics.com/modes/predict/)。
- **用户友好的部署**：Streamlit 的交互式界面使部署和使用应用程序变得容易，无需广泛的技术知识。用户只需简单点击即可开始实时推理，增强了可访问性和可用性。
- **高效的资源利用**：YOLO11 的优化算法确保以最少的计算资源进行高速处理。这种效率允许即使在标准硬件上也能进行流畅可靠的网络摄像头推理，使先进的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)对更广泛的受众可用。

## Streamlit 应用程序代码

!!! tip "Ultralytics 安装"

    在开始构建应用程序之前，请确保已安装 Ultralytics Python 包。

    ```bash
    pip install ultralytics
    ```

!!! example "使用 Streamlit 与 Ultralytics YOLO 进行推理"

    === "命令行"

        ```bash
        yolo solutions inference

        yolo solutions inference model="path/to/model.pt"
        ```

        这些命令启动 Ultralytics 附带的默认 Streamlit 界面。使用 `yolo solutions inference --help` 查看其他标志，如 `source`、`conf` 或 `persist`，如果您想在不编辑 Python 代码的情况下自定义体验。

    === "Python"

        ```python
        from ultralytics import solutions

        inf = solutions.Inference(
            model="yolo11n.pt",  # 您可以使用 Ultralytics 支持的任何模型，例如 YOLO11 或自定义训练的模型
        )

        inf.inference()

        # 确保使用命令 `streamlit run path/to/file.py` 运行文件
        ```

这将在默认 Web 浏览器中启动 Streamlit 应用程序。您将看到主标题、副标题和带有配置选项的侧边栏。选择所需的 YOLO11 模型，设置置信度和 [NMS 阈值](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)，然后点击"开始"按钮开始实时目标检测。

## 工作原理

在底层，Streamlit 应用程序使用 [Ultralytics solutions 模块](https://docs.ultralytics.com/reference/solutions/streamlit_inference/)创建交互式界面。当您开始推理时，应用程序：

1. 从网络摄像头或上传的视频文件捕获视频
2. 通过 YOLO11 模型处理每一帧
3. 使用您指定的置信度和 IoU 阈值应用目标检测
4. 实时显示原始帧和标注帧
5. 如果选择，可选择启用对象跟踪

该应用程序提供了一个干净、用户友好的界面，带有调整模型参数和随时启动/停止推理的控件。

## 结论

通过遵循本指南，您已成功使用 Streamlit 和 Ultralytics YOLO11 创建了一个实时目标检测应用程序。此应用程序允许您通过网络摄像头体验 YOLO11 检测对象的强大功能，具有用户友好的界面和随时停止视频流的能力。

如需进一步增强，您可以探索添加更多功能，如录制视频流、保存标注帧或与其他[计算机视觉库](https://www.ultralytics.com/blog/exploring-vision-ai-frameworks-tensorflow-pytorch-and-opencv)集成。

## 与社区分享您的想法

与社区互动以了解更多、排除问题并分享您的项目：

### 在哪里寻求帮助和支持

- **GitHub Issues：** 访问 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics/issues)提出问题、报告错误和建议功能。
- **Ultralytics Discord 服务器：** 加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)与其他用户和开发者联系，获得支持，分享知识，并集思广益。

### 官方文档

- **Ultralytics YOLO11 文档：** 参阅[官方 YOLO11 文档](https://docs.ultralytics.com/)获取关于各种计算机视觉任务和项目的全面指南和见解。

## 常见问题

### 如何使用 Streamlit 和 Ultralytics YOLO11 设置实时目标检测应用程序？

使用 Streamlit 和 Ultralytics YOLO11 设置实时目标检测应用程序非常简单。首先，确保使用以下命令安装 Ultralytics Python 包：

```bash
pip install ultralytics
```

然后，您可以创建一个基本的 Streamlit 应用程序来运行实时推理：

!!! example "Streamlit 应用程序"

    === "Python"

        ```python
        from ultralytics import solutions

        inf = solutions.Inference(
            model="yolo11n.pt",  # 您可以使用 Ultralytics 支持的任何模型，例如 YOLO11、YOLOv10
        )

        inf.inference()

        # 确保使用命令 `streamlit run path/to/file.py` 运行文件
        ```

    === "命令行"

        ```bash
        yolo solutions inference
        ```

有关实际设置的更多详细信息，请参阅文档的 [Streamlit 应用程序代码部分](#streamlit-应用程序代码)。

### 使用 Ultralytics YOLO11 与 Streamlit 进行实时目标检测的主要优势是什么？

使用 Ultralytics YOLO11 与 Streamlit 进行实时目标检测有几个优势：

- **无缝实时检测**：直接从网络摄像头画面实现高[准确性](https://www.ultralytics.com/glossary/accuracy)的实时目标检测。
- **用户友好的界面**：Streamlit 的直观界面允许轻松使用和部署，无需广泛的技术知识。
- **资源效率**：YOLO11 的优化算法确保以最少的计算资源进行高速处理。

在[实时推理的优势部分](#实时推理的优势)了解更多关于这些优势的信息。

### 如何在 Web 浏览器中部署 Streamlit 目标检测应用程序？

在编写集成 Ultralytics YOLO11 的 Streamlit 应用程序后，您可以通过运行以下命令部署它：

```bash
streamlit run path/to/file.py
```

此命令将在默认 Web 浏览器中启动应用程序，使您能够选择 YOLO11 模型、设置置信度和 NMS 阈值，并通过简单点击开始实时目标检测。有关详细指南，请参阅 [Streamlit 应用程序代码](#streamlit-应用程序代码)部分。

### 使用 Streamlit 和 Ultralytics YOLO11 进行实时目标检测有哪些用例？

使用 Streamlit 和 Ultralytics YOLO11 进行实时目标检测可以应用于各个领域：

- **安防**：实时监控未经授权的访问和[安全警报系统](https://docs.ultralytics.com/guides/security-alarm-system/)。
- **零售**：客户计数、货架管理和[库存跟踪](https://www.ultralytics.com/blog/from-shelves-to-sales-exploring-yolov8s-impact-on-inventory-management)。
- **野生动物和农业**：监控动物和作物状况以进行[保护工作](https://www.ultralytics.com/blog/ai-in-wildlife-conservation)。

有关更深入的用例和示例，请探索 [Ultralytics Solutions](https://docs.ultralytics.com/solutions/)。

### Ultralytics YOLO11 与 YOLOv5 和 RCNN 等其他目标检测模型相比如何？

Ultralytics YOLO11 相比 YOLOv5 和 RCNN 等先前模型提供了几项增强：

- **更高的速度和准确性**：改进了实时应用的性能。
- **易于使用**：简化的界面和部署。
- **资源效率**：优化以更少的计算要求获得更好的速度。

有关全面比较，请查看 [Ultralytics YOLO11 文档](https://docs.ultralytics.com/models/yolo11/)和讨论模型性能的相关博客文章。
