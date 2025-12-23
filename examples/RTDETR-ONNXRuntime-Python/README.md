# RT-DETR 使用 ONNX Runtime 进行目标检测

本项目演示如何使用 [ONNX Runtime](https://onnxruntime.ai/) 推理引擎在 [Python](https://www.python.org/) 中运行 Ultralytics [RT-DETR 模型](https://docs.ultralytics.com/models/rtdetr/)。它提供了一个简单的示例，用于对已导出为 [ONNX 格式](https://onnx.ai/)的 RT-DETR 模型执行[目标检测](https://docs.ultralytics.com/tasks/detect/)，ONNX 是表示[机器学习模型](https://www.ultralytics.com/glossary/machine-learning-ml)的标准格式。RT-DETR（实时检测 Transformer）提供高效准确的目标检测能力，详情请参阅 [RT-DETR 研究论文](https://arxiv.org/abs/2304.08069)。

## ⚙️ 安装

要开始使用，您需要安装必要的依赖项。请按照以下步骤操作。

### 安装所需依赖项

使用 [pip](https://pip.pypa.io/en/stable/) 和提供的 `requirements.txt` 文件安装核心需求。这将安装标准的 **`onnxruntime`** 包（基于 CPU 的推理）。有关可用执行选项的更多信息，请参阅 [ONNX Runtime 执行提供程序文档](https://onnxruntime.ai/docs/execution-providers/)。

```bash
pip install -r requirements.txt
```

### 安装 `onnxruntime-gpu`（可选）

要使用 NVIDIA GPU 进行加速推理，请安装 **`onnxruntime-gpu`** 包。首先确保您已安装正确的 [NVIDIA 驱动程序](https://www.nvidia.com/Download/index.aspx)和 [CUDA 工具包](https://developer.nvidia.com/cuda-toolkit)。有关详细的兼容性信息和设置说明，请参阅官方 [ONNX Runtime GPU 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

```bash
pip install onnxruntime-gpu
```

## 🚀 使用方法

安装依赖项后，您可以使用 `main.py` 脚本运行推理。

从终端执行脚本，指定 ONNX 模型路径、输入图像以及可选的置信度和 IoU 阈值：

```bash
python main.py --model rtdetr-l.onnx --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

**参数：**

- `--model`：RT-DETR [ONNX 模型文件](https://docs.ultralytics.com/modes/export/)的路径（例如 `rtdetr-l.onnx`）。您可以轻松[导出 Ultralytics 模型](https://docs.ultralytics.com/modes/export/)为 ONNX 格式。在 [Ultralytics 模型](https://docs.ultralytics.com/models/)页面查找更多模型。
- `--img`：输入图像文件的路径（例如 `image.jpg`）。
- `--conf-thres`：用于过滤检测的置信度阈值。只有分数高于此值的检测才会被保留。在我们的 [YOLO 性能指标指南](https://docs.ultralytics.com/guides/yolo-performance-metrics/)中了解更多关于阈值的信息。
- `--iou-thres`：用于[非极大值抑制 (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) 的[交并比 (IoU)](https://www.ultralytics.com/glossary/intersection-over-union-iou) 阈值，用于移除冗余的[边界框](https://www.ultralytics.com/glossary/bounding-box)。

根据您对检测灵敏度和重叠移除的具体要求调整 `--conf-thres` 和 `--iou-thres` 值。

## 🤝 贡献

欢迎贡献以增强此示例！无论是修复 bug、添加新功能、改进文档还是建议优化，您的意见都很有价值。请参阅 Ultralytics [贡献指南](https://docs.ultralytics.com/help/contributing/)了解如何开始的详细信息。您还可以探索[开源项目贡献](https://opensource.guide/how-to-contribute/)的一般指南。感谢您帮助改进 [Ultralytics](https://www.ultralytics.com/) 生态系统及其在 [GitHub](https://github.com/ultralytics/ultralytics) 上的可用资源！
