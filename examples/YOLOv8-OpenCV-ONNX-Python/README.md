# Ultralytics YOLOv8 目标检测与 OpenCV 和 ONNX

本示例演示如何使用 [Python](https://www.python.org/) 中的 [OpenCV](https://opencv.org/) 实现 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 目标检测，利用 [ONNX（开放神经网络交换）](https://onnx.ai/)模型格式进行高效推理。

## 🚀 快速开始

按照以下简单步骤在本地机器上运行示例。

1.  **克隆仓库：**
    如果尚未克隆，请克隆 Ultralytics 仓库以访问示例代码：

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics/examples/YOLOv8-OpenCV-ONNX-Python/
    ```

2.  **安装依赖：**
    安装 `requirements.txt` 文件中列出的必要 Python 包。建议使用虚拟环境。

    ```bash
    pip install -r requirements.txt
    ```

3.  **运行检测脚本：**
    执行主 Python 脚本，指定 ONNX 模型路径和输入图像。
    ```bash
    python main.py --model yolov8n.onnx --img image.jpg
    ```
    脚本将使用 `yolov8n.onnx` 模型对 `image.jpg` 执行目标检测并显示结果。

## 🛠️ 导出你的模型

如果你想使用不同的 Ultralytics YOLOv8 模型或你自己训练的模型，首先需要将其导出为 ONNX 格式。

1.  **安装 Ultralytics：**
    如果尚未安装，获取最新的 `ultralytics` 包：

    ```bash
    pip install ultralytics
    ```

2.  **导出模型：**
    使用 `yolo export` 命令将所需模型（如 `yolov8n.pt`）转换为 ONNX。确保指定 `opset=12` 或更高版本以与 OpenCV 的 DNN 模块兼容。你可以在 Ultralytics [导出文档](https://docs.ultralytics.com/modes/export/)中找到更多详情。
    ```bash
    yolo export model=yolov8n.pt imgsz=640 format=onnx opset=12
    ```
    此命令将在工作目录中生成 `yolov8n.onnx` 文件（或你模型对应的名称）。然后你可以将此 `.onnx` 文件与 `main.py` 脚本一起使用。

## 🤝 贡献

欢迎贡献！如果你发现任何问题或有改进建议，请随时在 [Ultralytics 主仓库](https://github.com/ultralytics/ultralytics)上提交 issue 或 pull request。感谢你帮助使 Ultralytics YOLO 变得更好！
