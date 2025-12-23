---
comments: true
description: 学习如何使用 Ultralytics YOLO11 裁剪和提取目标，以进行聚焦分析、减少数据量和提高精度。
keywords: Ultralytics, YOLO11, 目标裁剪, 目标检测, 图像处理, 视频分析, 人工智能, 机器学习
---

# 使用 Ultralytics YOLO11 进行目标裁剪

## 什么是目标裁剪？

使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) 进行目标裁剪涉及从图像或视频中隔离和提取特定检测到的目标。YOLO11 模型的能力被用于准确识别和描绘目标，从而实现精确裁剪以进行进一步分析或处理。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/J1BaCqytBmA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO 进行目标裁剪
</p>

## 目标裁剪的优势

- **聚焦分析**：YOLO11 促进有针对性的目标裁剪，允许对场景中的单个项目进行深入检查或处理。
- **减少数据量**：通过仅提取相关目标，目标裁剪有助于最小化数据大小，使其在存储、传输或后续计算任务中更加高效。
- **增强精度**：YOLO11 的[目标检测](https://www.ultralytics.com/glossary/object-detection)[准确率](https://www.ultralytics.com/glossary/accuracy)确保裁剪的目标保持其空间关系，保留视觉信息的完整性以进行详细分析。

## 可视化

|                                                                                机场行李                                                                                 |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO11 在机场传送带上裁剪行李箱](https://github.com/ultralytics/docs/releases/download/0/suitcases-cropping-airport-conveyor-belt.avif) |
|                                                      使用 Ultralytics YOLO11 在机场传送带上裁剪行李箱                                                      |

!!! example "使用 Ultralytics YOLO 进行目标裁剪"

    === "命令行"

        ```bash
        # 裁剪目标
        yolo solutions crop show=True

        # 传入视频源
        yolo solutions crop source="path/to/video.mp4"

        # 裁剪特定类别
        yolo solutions crop classes="[0, 2]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # 初始化目标裁剪器
        cropper = solutions.ObjectCropper(
            show=True,  # 显示输出
            model="yolo11n.pt",  # 用于目标裁剪的模型，例如 yolo11x.pt
            classes=[0, 2],  # 裁剪特定类别，如使用 COCO 预训练模型裁剪人和汽车
            # conf=0.5,  # 调整目标的置信度阈值
            # crop_dir="cropped-detections",  # 设置裁剪检测的目录名称
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或处理完成。")
                break

            results = cropper(im0)

            # print(results)  # 访问输出

        cap.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

        当您提供可选的 `crop_dir` 参数时，每个裁剪的目标都会写入该文件夹，文件名包含源图像名称和类别。这使得检查检测结果或构建下游数据集变得容易，无需编写额外代码。

### `ObjectCropper` 参数

下表列出了 `ObjectCropper` 的参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "crop_dir"]) }}

此外，还可以使用以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## 常见问题

### Ultralytics YOLO11 中的目标裁剪是什么，它是如何工作的？

使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 进行目标裁剪涉及基于 YOLO11 的检测能力从图像或视频中隔离和提取特定目标。这个过程允许聚焦分析、减少数据量和增强[精度](https://www.ultralytics.com/glossary/precision)，通过利用 YOLO11 高精度地识别目标并相应地裁剪它们。有关深入教程，请参阅[目标裁剪示例](#使用-ultralytics-yolo11-进行目标裁剪)。

### 为什么我应该使用 Ultralytics YOLO11 而不是其他解决方案进行目标裁剪？

Ultralytics YOLO11 因其精度、速度和易用性而脱颖而出。它允许详细准确的目标检测和裁剪，这对于[聚焦分析](#目标裁剪的优势)和需要高数据完整性的应用至关重要。此外，YOLO11 与 [OpenVINO](../integrations/openvino.md) 和 [TensorRT](../integrations/tensorrt.md) 等工具无缝集成，用于需要实时能力和在各种硬件上优化的部署。在[模型导出指南](../modes/export.md)中探索这些优势。

### 如何使用目标裁剪减少数据集的数据量？

通过使用 Ultralytics YOLO11 仅从图像或视频中裁剪相关目标，您可以显著减少数据大小，使其在存储和处理方面更加高效。这个过程涉及训练模型以检测特定目标，然后使用结果仅裁剪和保存这些部分。有关利用 Ultralytics YOLO11 功能的更多信息，请访问我们的[快速入门指南](../quickstart.md)。

### 我可以使用 Ultralytics YOLO11 进行实时视频分析和目标裁剪吗？

是的，Ultralytics YOLO11 可以处理实时视频流以动态检测和裁剪目标。该模型的高速推理能力使其非常适合[监控](security-alarm-system.md)、体育分析和自动检测系统等实时应用。查看[跟踪](../modes/track.md)和[预测模式](../modes/predict.md)以了解如何实现实时处理。

### 高效运行 YOLO11 进行目标裁剪的硬件要求是什么？

Ultralytics YOLO11 针对 CPU 和 GPU 环境进行了优化，但为了实现最佳性能，特别是对于实时或大批量推理，建议使用专用 GPU（例如 NVIDIA Tesla、RTX 系列）。对于在轻量级设备上部署，考虑使用 [CoreML](../integrations/coreml.md) 用于 iOS 或 [TFLite](../integrations/tflite.md) 用于 Android。有关支持的设备和格式的更多详细信息，请参阅我们的[模型部署选项](../guides/model-deployment-options.md)。
