# YOLOv8 LibTorch 推理 C++

本示例演示如何使用 [LibTorch（PyTorch C++ API）](https://docs.pytorch.org/cppdocs/)在 C++ 中执行 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 模型推理。这允许在 C++ 环境中部署 YOLOv8 模型以实现高效执行。

## ⚙️ 依赖项

在继续之前，请确保已安装以下依赖项：

| 依赖项       | 版本     | 资源                                         |
| :----------- | :------- | :------------------------------------------- |
| OpenCV       | >=4.0.0  | [https://opencv.org/](https://opencv.org/)   |
| C++ 标准     | >=17     | [https://isocpp.org/](https://isocpp.org/)   |
| CMake        | >=3.18   | [https://cmake.org/](https://cmake.org/)     |
| Libtorch     | >=1.12.1 | [https://pytorch.org/](https://pytorch.org/) |

你可以从官方 [PyTorch](https://pytorch.org/) 网站下载所需版本的 LibTorch。确保选择与你的系统和 CUDA 版本（如果使用 GPU）对应的正确版本。

## 🚀 使用方法

按照以下步骤运行 C++ 推理示例：

1.  **克隆 Ultralytics 仓库：**
    使用 [Git](https://git-scm.com/) 克隆包含示例代码和必要文件的仓库。

    ```bash
    git clone https://github.com/ultralytics/ultralytics
    ```

2.  **安装 Ultralytics：**
    进入克隆的目录并使用 [pip](https://pip.pypa.io/en/stable/) 安装 `ultralytics` 包。此步骤是导出模型所必需的。详细安装说明请参阅 [Ultralytics 快速入门指南](https://docs.ultralytics.com/quickstart/)。

    ```bash
    cd ultralytics
    pip install .
    ```

3.  **进入示例目录：**
    切换到 C++ LibTorch 推理示例目录。

    ```bash
    cd examples/YOLOv8-LibTorch-CPP-Inference
    ```

4.  **构建项目：**
    创建构建目录，使用 [CMake](https://cmake.org/) 配置项目，然后使用 [Make](https://www.gnu.org/software/make/) 编译。如果 CMake 未自动找到 LibTorch 和 OpenCV 安装，你可能需要指定它们的路径。

    ```bash
    mkdir build
    cd build
    cmake .. # 如需要，添加 -DCMAKE_PREFIX_PATH=/path/to/libtorch;/path/to/opencv
    make
    ```


5.  **运行推理：**
    执行编译后的二进制文件。应用程序将加载导出的 YOLOv8 模型并对示例图像（根目录 `ultralytics` 中包含的 `zidane.jpg`）或视频执行推理。
    ```bash
    ./yolov8_libtorch_inference
    ```

## ✨ 导出 Ultralytics YOLOv8

要将 Ultralytics YOLOv8 模型与 LibTorch 一起使用，首先需要将其导出为 [TorchScript](https://docs.pytorch.org/docs/stable/jit.html) 格式。TorchScript 是一种从 PyTorch 代码创建可序列化和可优化模型的方法。

使用 `ultralytics` 包提供的 `yolo` [命令行界面（CLI）](https://docs.ultralytics.com/usage/cli/)导出模型。例如，要导出输入图像大小为 640x640 的 `yolov8s.pt` 模型：

```bash
yolo export model=yolov8s.pt imgsz=640 format=torchscript
```

此命令将在模型目录中生成 `yolov8s.torchscript` 文件。此文件包含可由 C++ 应用程序使用 LibTorch 加载和执行的序列化模型。有关将模型导出为各种格式的更多详情，请参阅 [Ultralytics 导出文档](https://docs.ultralytics.com/modes/export/)。

## 🤝 贡献

欢迎贡献以增强此示例或添加新功能！请参阅 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing/)了解如何为项目做出贡献。感谢你帮助使 Ultralytics YOLO 成为最佳的视觉 AI 工具！
