# 使用 Ultralytics YOLOv8 的区域计数（视频推理）

> **注意：** 区域计数器现已成为 **[Ultralytics 解决方案](https://docs.ultralytics.com/solutions/)** 的一部分，提供增强功能和持续更新。
>
> 🔗 **探索官方[区域计数指南](https://docs.ultralytics.com/guides/region-counting/)获取最新实现。**

> 🔔 **通知：**
>
> 此 GitHub 示例（`ultralytics/examples/YOLOv8-Region-Counter/`）将保持可用，但**不再积极维护**。如需最新功能、更新和支持，请参阅 Ultralytics 文档中的官方[区域计数指南](https://docs.ultralytics.com/guides/region-counting/)。谢谢！

区域计数是一种用于统计视频画面中预定义区域或区块内目标数量的技术。这允许进行更详细的分析，特别是在同时监控多个不同区域时。用户可以通过左键点击和拖动来交互式调整这些区域，实现针对特定需求和布局的实时计数。此方法在各种[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用中很有价值，从交通分析到零售分析。

<div>
<p align="center">
  <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/5ab3bbd7-fd12-4849-928e-5f294d6c3fcf" width="45%" alt="YOLOv8 区域计数可视化 1">
  <img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/e7c1aea7-474d-4d78-8d48-b50854ffe1ca" width="45%" alt="YOLOv8 区域计数可视化 2">
</p>
</div>

## 📚 目录

- [步骤 1：安装所需库](#-步骤-1安装所需库)
- [步骤 2：使用 Ultralytics YOLOv8 运行区域计数](#-步骤-2使用-ultralytics-yolov8-运行区域计数)
- [使用选项](#-使用选项)
- [常见问题（FAQ）](#-常见问题faq)
- [贡献](#-贡献)

## ⚙️ 步骤 1：安装所需库

首先，克隆 Ultralytics 仓库并进入示例目录。确保已安装 Python 以及 [PyTorch](https://pytorch.org/) 和 [OpenCV](https://opencv.org/) 等必要依赖。

```bash
# 克隆 ultralytics 仓库
git clone https://github.com/ultralytics/ultralytics

# 进入示例目录
cd ultralytics/examples/YOLOv8-Region-Counter

# 安装所需包（如果尚未安装）
pip install ultralytics shapely
```

## ▶️ 步骤 2：使用 Ultralytics YOLOv8 运行区域计数

使用以下命令执行脚本。你可以自定义源、模型、设备和其他参数。

### 注意

视频开始播放后，你可以通过左键点击并拖动来动态重新定位视频帧内的计数区域。

```bash
# 在视频源上运行推理，保存结果并查看输出
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --view-img

# 使用 CPU 运行推理
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --view-img --device cpu

# 使用特定的 Ultralytics YOLOv8 模型文件
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --weights "path/to/yolov8n.pt"

# 仅检测特定类别（如类别 0 和类别 2）
python yolov8_region_counter.py --source "path/to/video.mp4" --classes 0 2 --weights "path/to/yolov8m.pt"

# 运行推理但不保存输出视频/图像
python yolov8_region_counter.py --source "path/to/video.mp4" --view-img
```

在 Ultralytics [预测模式文档](https://docs.ultralytics.com/modes/predict/)中了解更多关于推理参数的信息。


## 🛠️ 使用选项

脚本接受多个命令行参数进行自定义：

- `--source`：输入视频文件路径。
- `--device`：计算设备（`cpu` 或 GPU ID 如 `0`）。
- `--save-img`：布尔标志，保存带有检测结果的输出帧。
- `--view-img`：布尔标志，实时显示输出视频流。
- `--weights`：[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 模型文件路径（`.pt`）。默认通常使用标准模型如 `yolov8n.pt`。
- `--classes`：按特定类别 ID 过滤检测（如 `--classes 0 2 3` 检测类别 0、2 和 3）。
- `--line-thickness`：[边界框](https://www.ultralytics.com/glossary/bounding-box)线条粗细。
- `--region-thickness`：定义计数区域的线条粗细。
- `--track-thickness`：目标跟踪线条粗细。

在 [Ultralytics 文档](https://docs.ultralytics.com/)中探索不同的模型和训练选项。

## ❓ 常见问题（FAQ）

### 什么是区域计数？

区域计数是[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)中用于确定图像或视频帧预定义区域内目标数量的过程。它通常应用于[图像处理](https://en.wikipedia.org/wiki/Image_processing)和[模式识别](https://en.wikipedia.org/wiki/Pattern_recognition)等领域，用于分析空间分布和基于位置分割目标。

### 区域计数器是否支持自定义区域形状？

是的，区域计数器允许使用多边形（包括矩形）定义区域。你可以直接在脚本中自定义区域属性，如坐标、名称和颜色。`shapely` 库用于多边形定义。查看 [Shapely 用户手册](https://shapely.readthedocs.io/en/stable/manual.html#polygons)了解更多关于多边形创建的详情。

```python
from shapely.geometry import Polygon

# 计数区域定义示例
counting_regions = [
    {
        "name": "区域 1（五边形）",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # 5 点多边形
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # 区域 BGR 颜色
        "text_color": (255, 255, 255),  # 文本 BGR 颜色
    },
    {
        "name": "区域 2（矩形）",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # 4 点多边形（矩形）
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # 区域 BGR 颜色
        "text_color": (0, 0, 0),  # 文本 BGR 颜色
    },
]
```

### 为什么将区域计数与 YOLOv8 结合？

[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 擅长在视频流中进行[目标检测](https://www.ultralytics.com/glossary/object-detection)和[跟踪](https://www.ultralytics.com/glossary/object-tracking)。集成区域计数增强了其能力，能够在特定区域内量化目标，使其适用于人群监控、交通流量分析和零售客流量统计等应用。查看我们关于[使用 Ultralytics YOLOv8 进行目标检测和跟踪](https://www.ultralytics.com/blog/object-detection-and-tracking-with-ultralytics-yolov8)的博客文章。

### 如何排除故障？

对于调试，你可以启用更详细的输出。虽然此特定脚本没有专用的 `--debug` 标志，但你可以在代码中添加 print 语句来检查变量或使用标准 Python 调试工具。确保视频路径和模型权重路径正确。有关常见问题，请参阅 [Ultralytics FAQ](https://docs.ultralytics.com/help/FAQ/)。

### 我可以使用其他 YOLO 版本或自定义模型吗？

是的，你可以通过使用 `--weights` 参数指定 `.pt` 文件路径来使用不同的 Ultralytics YOLO 模型版本（如 YOLOv5、YOLOv9、YOLOv10、YOLO11）或你自己的自定义训练模型。确保模型与 Ultralytics 框架兼容。在[模型训练指南](https://docs.ultralytics.com/modes/train/)中了解更多关于训练自定义模型的信息。

## 🤝 贡献

欢迎贡献以改进此示例或添加新功能！请随时在 [Ultralytics 主仓库](https://github.com/ultralytics/ultralytics)上提交 Pull Request 或提交 Issue。请记得查看官方[区域计数指南](https://docs.ultralytics.com/guides/region-counting/)获取最新维护版本。
