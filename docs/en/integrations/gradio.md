---
comments: true
description: 探索使用 Gradio 和 Ultralytics YOLO11 执行目标检测的交互式方式。上传图像并调整设置以获得实时结果。
keywords: Ultralytics, YOLO11, Gradio, 目标检测, 交互式, 实时, 图像处理, AI
---

# 交互式目标检测：Gradio 和 Ultralytics YOLO11 🚀

## 交互式目标检测简介

这个 Gradio 界面提供了一种简单且交互式的方式，使用 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) 模型执行[目标检测](https://www.ultralytics.com/glossary/object-detection)。用户可以上传图像并调整置信度阈值和交并比（IoU）阈值等参数，以获得实时检测结果。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/pWYiene9lYw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Gradio 与 Ultralytics YOLO11 集成
</p>

## 为什么使用 Gradio 进行目标检测？

- **用户友好的界面**：Gradio 为用户提供了一个简单的平台来上传图像和可视化检测结果，无需任何编码要求。
- **实时调整**：可以即时调整置信度和 IoU 阈值等参数，允许立即反馈和优化检测结果。
- **广泛的可访问性**：Gradio Web 界面可供任何人访问，使其成为演示、教育目的和快速实验的绝佳工具。

<p align="center">
   <img width="800" alt="Gradio 示例截图" src="https://github.com/ultralytics/docs/releases/download/0/gradio-example-screenshot.avif">
</p>

## 如何安装 Gradio

```bash
pip install gradio
```

## 如何使用界面

1. **上传图像**：点击"上传图像"选择要进行目标检测的图像文件。
2. **调整参数**：
    - **置信度阈值**：滑块用于设置检测对象的最小置信度级别。
    - **IoU 阈值**：滑块用于设置区分不同对象的 IoU 阈值。
3. **查看结果**：将显示带有检测到的对象及其标签的处理后图像。

## 示例用例

- **示例图像 1**：使用默认阈值进行公交车检测。
- **示例图像 2**：使用默认阈值对体育图像进行检测。

## 使用示例

本节提供用于使用 Ultralytics YOLO11 模型创建 Gradio 界面的 Python 代码。该代码支持分类任务、检测任务、分割任务和关键点任务。

```python
import gradio as gr
import PIL.Image as Image

from ultralytics import ASSETS, YOLO

model = YOLO("yolo11n.pt")


def predict_image(img, conf_threshold, iou_threshold):
    """使用可调整的置信度和 IoU 阈值的 YOLO11 模型预测图像中的对象。"""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="上传图像"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="置信度阈值"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU 阈值"),
    ],
    outputs=gr.Image(type="pil", label="结果"),
    title="Ultralytics Gradio",
    description="上传图像进行推理。默认使用 Ultralytics YOLO11n 模型。",
    examples=[
        [ASSETS / "bus.jpg", 0.25, 0.45],
        [ASSETS / "zidane.jpg", 0.25, 0.45],
    ],
)

if __name__ == "__main__":
    iface.launch()
```

## 参数说明

| 参数名称         | 类型    | 描述                                     |
| ---------------- | ------- | ---------------------------------------- |
| `img`            | `Image` | 将在其上执行目标检测的图像。             |
| `conf_threshold` | `float` | 检测对象的置信度阈值。                   |
| `iou_threshold`  | `float` | 用于对象分离的交并比阈值。               |

### Gradio 界面组件

| 组件         | 描述                           |
| ------------ | ------------------------------ |
| 图像输入     | 用于上传要检测的图像。         |
| 滑块         | 用于调整置信度和 IoU 阈值。    |
| 图像输出     | 用于显示检测结果。             |

## 常见问题

### 如何将 Gradio 与 Ultralytics YOLO11 一起用于目标检测？

要将 Gradio 与 Ultralytics YOLO11 一起用于目标检测，您可以按照以下步骤操作：

1. **安装 Gradio**：使用命令 `pip install gradio`。
2. **创建界面**：编写 Python 脚本来初始化 Gradio 界面。您可以参考文档中提供的[使用示例](#使用示例)代码了解详细信息。
3. **上传和调整**：上传您的图像并在 Gradio 界面上调整置信度和 IoU 阈值，以获得实时目标检测结果。

以下是供参考的最小代码片段：

```python
import gradio as gr

from ultralytics import YOLO

model = YOLO("yolo11n.pt")


def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
    )
    return results[0].plot() if results else None


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="上传图像"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="置信度阈值"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU 阈值"),
    ],
    outputs=gr.Image(type="pil", label="结果"),
    title="Ultralytics Gradio YOLO11",
    description="上传图像进行 YOLO11 目标检测。",
)
iface.launch()
```

### 将 Gradio 用于 Ultralytics YOLO11 目标检测有什么好处？

将 Gradio 用于 Ultralytics YOLO11 目标检测提供了几个好处：

- **用户友好的界面**：Gradio 为用户提供了一个直观的界面来上传图像和可视化检测结果，无需编码工作。
- **实时调整**：您可以动态调整置信度和 IoU 阈值等检测参数，并立即看到效果。
- **可访问性**：Web 界面对任何人都可访问，使其对快速实验、教育目的和演示非常有用。

有关更多详细信息，您可以阅读这篇关于[放射学中的 AI](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency) 的博客文章，其中展示了类似的交互式可视化技术。

### Gradio 和 Ultralytics YOLO11 可以一起用于教育目的吗？

是的，Gradio 和 Ultralytics YOLO11 可以有效地一起用于教育目的。Gradio 直观的 Web 界面使学生和教育工作者可以轻松地与 Ultralytics YOLO11 等最先进的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型进行交互，而无需高级编程技能。这种设置非常适合演示目标检测和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)中的关键概念，因为 Gradio 提供即时的视觉反馈，有助于理解不同参数对检测性能的影响。

### 如何在 YOLO11 的 Gradio 界面中调整置信度和 IoU 阈值？

在 YOLO11 的 Gradio 界面中，您可以使用提供的滑块调整置信度和 IoU 阈值。这些阈值有助于控制预测[准确率](https://www.ultralytics.com/glossary/accuracy)和对象分离：

- **置信度阈值**：确定检测对象的最小置信度级别。滑动以增加或减少所需的置信度。
- **IoU 阈值**：设置区分重叠对象的交并比阈值。调整此值以优化对象分离。

有关这些参数的更多信息，请访问[参数说明部分](#参数说明)。

### 将 Ultralytics YOLO11 与 Gradio 一起使用有哪些实际应用？

将 Ultralytics YOLO11 与 Gradio 结合使用的实际应用包括：

- **实时目标检测演示**：非常适合展示目标检测如何实时工作。
- **教育工具**：在学术环境中教授目标检测和计算机视觉概念非常有用。
- **原型开发**：高效地快速开发和测试原型目标检测应用程序。
- **社区和协作**：使与社区分享模型以获取反馈和协作变得容易。

有关类似用例的示例，请查看 [Ultralytics 关于动物行为监测的博客](https://www.ultralytics.com/blog/monitoring-animal-behavior-using-ultralytics-yolov8)，该博客展示了交互式可视化如何增强野生动物保护工作。
