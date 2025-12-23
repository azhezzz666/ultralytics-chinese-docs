# YOLO11 与 SAHI 视频推理

[切片辅助超推理（SAHI）](https://github.com/obss/sahi)是一种强大的技术，旨在优化[目标检测](https://en.wikipedia.org/wiki/Object_detection)算法，特别适用于大规模和[高分辨率图像](https://en.wikipedia.org/wiki/Image_resolution)。它通过将图像或视频帧分割成可管理的切片，使用 [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) 等模型对每个切片执行检测，然后智能合并结果。这种方法显著提高了小目标的检测精度，并在高分辨率输入上保持性能。

本教程指导你使用 SAHI 库在视频文件上运行 Ultralytics YOLO11 推理以增强检测能力。有关使用 SAHI 与 Ultralytics 模型的详细指南，请参阅 [SAHI 切片推理指南](https://docs.ultralytics.com/guides/sahi-tiled-inference/)。

## 📋 目录

- [步骤 1：安装所需库](#-步骤-1安装所需库)
- [步骤 2：使用 SAHI 和 Ultralytics YOLO11 运行推理](#-步骤-2使用-sahi-和-ultralytics-yolo11-运行推理)
- [使用选项](#-使用选项)
- [贡献](#-贡献)

## ⚙️ 步骤 1：安装所需库

首先，克隆 [Ultralytics 仓库](https://github.com/ultralytics/ultralytics)以访问示例脚本。然后，使用 [pip](https://pip.pypa.io/en/stable/) 安装必要的 [Python](https://www.python.org/) 包，包括 `sahi` 和 `ultralytics`。最后，进入示例目录。

```bash
# 克隆 ultralytics 仓库
git clone https://github.com/ultralytics/ultralytics

# 安装依赖
# 确保已安装 Python 3.8 或更高版本
pip install -U sahi ultralytics opencv-python

# 进入示例文件夹
cd ultralytics/examples/YOLOv8-SAHI-Inference-Video
```

## 🚀 步骤 2：使用 SAHI 和 Ultralytics YOLO11 运行推理

设置完成后，你可以在视频文件上运行推理。提供的脚本 `yolov8_sahi.py` 利用 SAHI 与指定的 YOLO11 模型进行切片推理。

使用命令行执行脚本，指定视频文件路径。你还可以选择不同的 YOLO11 模型权重。

```bash
# 运行推理并保存带有边界框的输出视频
python yolov8_sahi.py --source "path/to/your/video.mp4" --save-img

# 使用特定 YOLO11 模型（如 yolo11n.pt）运行推理并保存结果
python yolov8_sahi.py --source "path/to/your/video.mp4" --save-img --weights "yolo11n.pt"

# 使用更小的切片运行推理以获得更好的小目标检测
python yolov8_sahi.py --source "path/to/your/video.mp4" --save-img --slice-height 512 --slice-width 512
```

此脚本逐帧处理视频，应用 SAHI 的切片和推理逻辑。启用保存时，它会将标注帧导出到 `runs/detect/predict`。在[预测模式文档](https://docs.ultralytics.com/modes/predict/)中了解更多关于使用 Ultralytics 模型进行预测的信息。

## 🛠️ 使用选项

脚本 `yolov8_sahi.py` 接受多个命令行参数来自定义推理过程：

- `--source`：**必需**。输入视频文件路径（如 `"../path/to/video.mp4"`）。
- `--weights`：可选。YOLO11 模型权重文件路径（如 `"yolo11n.pt"`、`"yolo11s.pt"`）。默认为 `"yolo11n.pt"`。你可以下载各种模型或使用自定义训练的模型。查看 [Ultralytics YOLO 模型](https://docs.ultralytics.com/models/)了解更多选项。
- `--save-img`：可选。导出标注帧的标志。保存到 `runs/detect/predict`。
- `--slice-height`：可选。SAHI 每个图像切片的高度。默认为 `1024`。
- `--slice-width`：可选。SAHI 每个图像切片的宽度。默认为 `1024`。

尝试这些选项，特别是切片尺寸，以优化特定[视频处理](https://en.wikipedia.org/wiki/Video_processing)任务和目标对象大小的检测性能。使用适当的[数据集](https://docs.ultralytics.com/datasets/)进行训练也可以显著影响性能。

## ✨ 贡献

欢迎贡献以增强此示例或添加新功能！如果你遇到问题或有建议，请在 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)上提交 issue 或 pull request。查看我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)了解更多详情。
