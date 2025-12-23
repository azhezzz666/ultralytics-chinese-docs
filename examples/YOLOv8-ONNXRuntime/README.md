# YOLOv8 - ONNX Runtime

本仓库提供了使用 [ONNX Runtime](https://onnxruntime.ai/) 运行 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 模型的示例实现。这允许在支持 [ONNX 格式](https://onnx.ai/)的各种硬件平台上进行高效推理。

## ⚙️ 安装

首先，你需要安装 [Python](https://www.python.org/)。然后，安装必要的依赖项。

### 安装所需依赖

克隆仓库并使用 [pip](https://pip.pypa.io/en/stable/) 安装 `requirements.txt` 文件中列出的包：

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/examples/YOLOv8-ONNXRuntime
pip install -r requirements.txt
```

### 安装 ONNX Runtime 后端

你需要根据你的硬件选择适当的 ONNX Runtime 包。

**GPU 加速（NVIDIA）**

如果你有 NVIDIA GPU 并希望利用 CUDA 进行更快的推理，请安装 `onnxruntime-gpu` 包。确保已安装正确的 [NVIDIA 驱动程序](https://www.nvidia.com/Download/index.aspx)和 CUDA 工具包。有关兼容性详情，请参阅官方 [ONNX Runtime GPU 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

```bash
pip install onnxruntime-gpu
```

**仅 CPU**

如果你没有兼容的 NVIDIA GPU 或更喜欢基于 CPU 的推理，请安装标准 `onnxruntime` 包。查看 [ONNX Runtime 安装指南](https://onnxruntime.ai/docs/install/)了解更多选项。

```bash
pip install onnxruntime
```

## 🚀 使用方法

安装依赖项和适当的 ONNX Runtime 后端后，你可以使用提供的 Python 脚本执行推理。

### 导出你的模型

在运行推理之前，你需要一个 ONNX 格式（`.onnx`）的 YOLOv8 模型。你可以使用 Ultralytics CLI 或 Python SDK 导出训练好的 Ultralytics YOLOv8 模型。详细说明请参阅 [Ultralytics 导出文档](https://docs.ultralytics.com/modes/export/)。

导出命令示例：

```bash
yolo export model=yolov8n.pt format=onnx # 将 yolov8n 模型导出为 ONNX
```

### 运行推理

使用 ONNX 模型路径和输入图像执行 `main.py` 脚本。你还可以调整[目标检测](https://docs.ultralytics.com/tasks/detect/)的置信度和[交并比（IoU）](https://www.ultralytics.com/glossary/intersection-over-union-iou)阈值。

```bash
python main.py --model yolov8n.onnx --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

- `--model`：YOLOv8 ONNX 模型文件路径（如 `yolov8n.onnx`）。
- `--img`：输入图像路径（如 `image.jpg`）。
- `--conf-thres`：过滤检测的置信度阈值。只有分数高于此值的检测才会被保留。在[性能指标指南](https://docs.ultralytics.com/guides/yolo-performance-metrics/)中了解更多关于阈值的信息。
- `--iou-thres`：非极大值抑制（NMS）的 IoU 阈值。IoU 大于此阈值的框将被抑制。详情请参阅 [NMS 术语表条目](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)。

脚本将处理图像，执行目标检测，在检测到的目标上绘制边界框，并将输出图像保存为 `output.jpg`。

## 贡献

欢迎贡献以增强此示例或添加新功能！请参阅 [Ultralytics 主仓库](https://github.com/ultralytics/ultralytics)了解贡献指南。如果你遇到问题或有建议，请随时在 [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime) 或 Ultralytics 仓库上提交 issue。
