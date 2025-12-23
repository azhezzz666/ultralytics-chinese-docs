# YOLOv8-Segmentation-ONNXRuntime-Python 演示

本仓库提供了一个 [Python](https://www.python.org/) 演示，用于使用 [ONNX Runtime](https://onnxruntime.ai/) 执行 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 实例分割。它展示了 YOLOv8 模型的互操作性，允许在不需要完整 [PyTorch](https://pytorch.org/) 堆栈的情况下进行推理。这种方法非常适合首选最小依赖的部署场景。在我们的文档中了解更多关于[分割任务](https://docs.ultralytics.com/tasks/segment/)的信息。

## ✨ 功能特性

- **框架无关**：纯粹在 ONNX Runtime 上运行分割推理，无需导入 PyTorch。
- **高效推理**：支持 FP32 和[半精度](https://www.ultralytics.com/glossary/half-precision)（FP16）[ONNX](https://onnx.ai/) 模型，满足不同计算需求并优化[推理延迟](https://www.ultralytics.com/glossary/inference-latency)。
- **易于使用**：使用简单的命令行参数实现直接的模型执行。
- **广泛兼容**：利用 [NumPy](https://numpy.org/) 和 [OpenCV](https://opencv.org/) 进行图像处理，确保在各种环境中的广泛兼容性。

## 🛠️ 安装

使用 pip 安装所需包。你需要 [`ultralytics`](https://github.com/ultralytics/ultralytics) 来导出 YOLOv8-seg ONNX 模型并使用一些实用函数，[`onnxruntime-gpu`](https://pypi.org/project/onnxruntime-gpu/) 用于 GPU 加速推理，以及 [`opencv-python`](https://pypi.org/project/opencv-python/) 用于图像处理。

```bash
pip install ultralytics
pip install onnxruntime-gpu # GPU 支持
# pip install onnxruntime # 仅 CPU 支持
pip install numpy opencv-python
```

## 🚀 快速开始

### 1. 导出 YOLOv8 ONNX 模型

首先，使用 `ultralytics` 包将你的 Ultralytics YOLOv8 分割模型导出为 ONNX 格式。此步骤将 PyTorch 模型转换为适合 ONNX Runtime 的标准化格式。查看我们的[导出文档](https://docs.ultralytics.com/modes/export/)了解更多导出选项和我们的 [ONNX 集成指南](https://docs.ultralytics.com/integrations/onnx/)。

```bash
yolo export model=yolov8s-seg.pt imgsz=640 format=onnx opset=12 simplify
```

### 2. 运行推理

使用导出的 ONNX 模型对图像或视频源执行推理。使用命令行参数指定 ONNX 模型路径和图像源。

```bash
python main.py --model yolov8s-seg.onnx --source path/to/image.jpg
```

### 示例输出

运行命令后，脚本将处理图像，执行分割，并显示叠加了边界框和掩码的结果。

<img src="https://user-images.githubusercontent.com/51357717/279988626-eb74823f-1563-4d58-a8e4-0494025b7c9a.jpg" alt="YOLOv8 分割 ONNX 演示输出" width="800">

## 💡 高级用法

对于更高级的使用场景，如处理视频流或调整推理参数，请参阅 `main.py` 脚本中可用的命令行参数。你可以探索置信度阈值和输入图像大小等选项。

## 🤝 贡献

欢迎贡献以改进此演示！如果你遇到错误、有功能请求或想提交增强（如新算法或改进的处理步骤），请在 [Ultralytics 主仓库](https://github.com/ultralytics/ultralytics)上提交 issue 或 pull request。查看我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)了解更多关于如何参与的详情。

## 📄 许可证

本项目基于 AGPL-3.0 许可证授权。详细信息请参阅 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件或阅读完整的 [AGPL-3.0 许可证文本](https://opensource.org/license/agpl-v3)。

## 🙏 致谢

- 此 YOLOv8-Segmentation-ONNXRuntime-Python 演示由 GitHub 用户 [jamjamjon](https://github.com/jamjamjon) 贡献。
- 感谢 [ONNX Runtime 社区](https://github.com/microsoft/onnxruntime)提供强大高效的推理引擎。

我们希望你觉得此演示有用！欢迎贡献并帮助使其变得更好。
