# 使用 TFLite Runtime 运行 Ultralytics YOLOv8

本指南演示如何使用导出为 [TensorFlow Lite（TFLite）](https://ai.google.dev/edge/litert)格式的 Ultralytics [YOLOv8](https://docs.ultralytics.com/models/yolov8/) 模型执行推理。TFLite 是在移动、嵌入式和物联网设备上部署机器学习模型的热门选择，因为它针对设备端推理进行了优化，具有低延迟和小二进制大小。本示例支持 FP32、FP16 和 INT8 量化的 TFLite 模型。

## ⚙️ 安装

在运行推理之前，你需要安装必要的 TFLite 解释器包。根据你的硬件（CPU 或 GPU）选择适当的包。

### 安装 `tflite-runtime`（推荐用于边缘设备）

`tflite-runtime` 包是一个较小的包，包含使用 TensorFlow Lite 运行推理所需的最低限度，主要是 `Interpreter` Python 类。它非常适合 Raspberry Pi 或 Coral Edge TPU 等资源受限的环境。

```bash
pip install tflite-runtime
```

更多详情请参阅官方 [TFLite Python 快速入门指南](https://ai.google.dev/edge/litert/microcontrollers/python)。

### 安装完整 `tensorflow` 包（CPU 或 GPU）

或者，你可以安装完整的 TensorFlow 包。这包括 TFLite 解释器以及完整的 TensorFlow 库。

- **仅 CPU**：如果你没有 NVIDIA GPU 或不需要 GPU 加速，适合使用。

  ```bash
  pip install tensorflow
  ```

- **GPU 支持**：要利用 NVIDIA GPU 加速以获得更快的推理，请安装带 GPU 支持的 `tensorflow`。确保已安装必要的 [NVIDIA 驱动程序](https://www.nvidia.com/Download/index.aspx)和 CUDA 工具包。

  ```bash
  # 查看 TensorFlow 文档了解特定的 CUDA/cuDNN 版本要求
  pip install tensorflow[and-cuda] # 或按照 TF 网站上的特定说明
  ```

访问官方 [TensorFlow 安装指南](https://www.tensorflow.org/install)获取详细说明，包括 GPU 设置。

## 🚀 使用方法

按照以下步骤使用导出的 YOLOv8 TFLite 模型运行推理。

1.  **将 YOLOv8 模型导出为 TFLite：**
    首先，使用 `yolo export` 命令将训练好的 Ultralytics YOLOv8 模型（如 `yolov8n.pt`）导出为 TFLite 格式。本示例导出 INT8 量化模型以在边缘设备上获得最佳性能。你也可以通过调整 `format` 和量化参数导出 FP32 或 FP16 模型。更多选项请参阅 Ultralytics [导出模式文档](https://docs.ultralytics.com/modes/export/)。

    ```bash
    yolo export model=yolov8n.pt imgsz=640 format=tflite int8=True # 导出 yolov8n_saved_model/yolov8n_full_integer_quant.tflite
    ```

    导出过程将创建一个目录（如 `yolov8n_saved_model`），包含 `.tflite` 模型文件和可能的 `metadata.yaml` 文件（包含类别名称和其他模型详情）。

2.  **运行推理脚本：**
    执行提供的 Python 脚本（`main.py`）对图像执行推理。根据你的特定模型路径、图像源、置信度阈值和 IoU 阈值调整参数。

    ```bash
    python main.py \
      --model yolov8n_saved_model/yolov8n_full_integer_quant.tflite \
      --img image.jpg \
      --conf 0.25 \
      --iou 0.45 \
      --metadata yolov8n_saved_model/metadata.yaml
    ```

    - `--model`：导出的 `.tflite` 模型文件路径。
    - `--img`：用于检测的输入图像路径。
    - `--conf`：检测的最小[置信度阈值](https://www.ultralytics.com/glossary/confidence)（如 0.25）。
    - `--iou`：非极大值抑制（NMS）的[交并比（IoU）](https://www.ultralytics.com/glossary/intersection-over-union-iou)阈值。
    - `--metadata`：导出时生成的 `metadata.yaml` 文件路径（包含类别名称）。

## ✅ 输出

脚本将使用指定的 TFLite 模型处理输入图像，并显示在检测到的目标周围绘制边界框的图像。每个框将标注预测的类别名称和置信度分数。

![显示 YOLOv8 TFLite 检测公交车的输出图像](https://raw.githubusercontent.com/wamiqraza/Attribute-recognition-and-reidentification-Market1501-dataset/refs/heads/main/img/bus.jpg)

本示例提供了在支持 TFLite 的设备上部署 Ultralytics YOLOv8 模型的直接方法，在各种应用中实现高效的**目标检测**。探索不同的[量化](https://www.ultralytics.com/glossary/model-quantization)选项和模型大小，为你的特定用例找到性能和精度之间的最佳平衡。

## 🤝 贡献

欢迎贡献以增强此示例或添加新功能！随时 fork [Ultralytics 仓库](https://github.com/ultralytics/ultralytics)，进行更改并提交 pull request。
