# YOLOv8 MNN 推理 C++

欢迎使用 Ultralytics YOLOv8 MNN 推理 C++ 示例！本指南将帮助你开始在 C++ 项目中使用 [Alibaba MNN](https://mnn-docs.readthedocs.io/en/latest/) 推理引擎利用强大的 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 模型。无论你是想在 CPU 硬件上提升性能还是为应用程序增加灵活性，本示例都提供了坚实的基础。在 [Ultralytics 博客](https://www.ultralytics.com/blog)上了解更多关于优化模型和部署策略的信息。

## 🌟 功能特性

- 🚀 **模型格式支持**：原生支持 MNN 格式。
- ⚡ **精度选项**：支持 **FP32**、**FP16**（[半精度](https://www.ultralytics.com/glossary/half-precision)）和 **INT8**（[模型量化](https://www.ultralytics.com/glossary/model-quantization)）精度运行模型，优化性能并减少资源消耗。
- 🔄 **动态形状加载**：轻松处理具有动态输入形状的模型，这是许多[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务的常见需求。
- 📦 **灵活的 API 使用**：可选择 MNN 的高级 [Express API](https://github.com/alibaba/MNN) 获得用户友好的界面，或使用低级 [Interpreter API](https://mnn-docs.readthedocs.io/en/latest/cpp/Interpreter.html) 进行细粒度控制。

## 📋 依赖项

为确保顺利执行，请确保已安装以下依赖项：

| 依赖项                                            | 版本     | 描述                                                                     |
| :------------------------------------------------ | :------- | :----------------------------------------------------------------------- |
| [MNN](https://mnn-docs.readthedocs.io/en/latest/) | >=2.0.0  | 阿里巴巴的核心推理引擎。                                                 |
| [C++](https://en.cppreference.com/w/)             | >=14     | 支持 C++14 特性的现代 C++ 编译器。                                       |
| [CMake](https://cmake.org/documentation/)         | >=3.12.0 | 构建 MNN 和示例所需的跨平台构建系统生成器。                              |
| [OpenCV](https://opencv.org/)                     | 可选     | 用于示例中的图像加载和预处理（与 MNN 一起构建）。                        |

## ⚙️ 构建说明

按照以下步骤构建项目：

1.  克隆 Ultralytics 仓库：

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics/examples/YOLOv8-MNN-CPP
    ```

2.  克隆 [Alibaba MNN 仓库](https://github.com/alibaba/MNN)：

    ```bash
    git clone https://github.com/alibaba/MNN.git
    cd MNN
    ```

3.  构建 MNN 库：

    ```bash
    # 创建构建目录
    mkdir build && cd build

    # 配置 CMake（启用 OpenCV 集成，禁用共享库，启用图像编解码器）
    cmake -DMNN_BUILD_OPENCV=ON -DBUILD_SHARED_LIBS=OFF -DMNN_IMGCODECS=ON ..

    # 构建库（使用 -j 标志进行并行编译）
    make -j$(nproc) # Linux 使用 nproc，macOS 使用 sysctl -n hw.ncpu
    ```

    **注意：** 如果在构建过程中遇到问题，请参阅官方 [MNN 文档](https://mnn-docs.readthedocs.io/en/latest/)获取详细的构建说明和故障排除提示。

4.  将所需的 MNN 库和头文件复制到示例项目目录：

    ```bash
    # 返回示例目录
    cd ../..

    # 如果不存在，创建库和头文件目录
    mkdir -p libs include

    # 复制静态库
    cp MNN/build/libMNN.a libs/                 # 主 MNN 库
    cp MNN/build/express/libMNN_Express.a libs/ # MNN Express API 库
    cp MNN/build/tools/cv/libMNNOpenCV.a libs/  # MNN OpenCV 封装库

    # 复制头文件
    cp -r MNN/include .
    cp -r MNN/tools/cv/include . # MNN OpenCV 封装头文件
    ```

    **注意：**
    - 库文件扩展名（静态库为 `.a`）和路径可能因操作系统（如 Windows 上使用 `.lib`）和构建配置而异。请相应调整命令。
    - 本示例使用静态链接（`.a` 文件）。如果你构建了共享库（`.so`、`.dylib`、`.dll`），请确保它们正确放置或在系统库路径中可访问。

5.  为示例项目创建构建目录并使用 CMake 编译：
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```


## 🔄 导出 YOLOv8 模型

要将你的 Ultralytics YOLOv8 模型与此 C++ 示例一起使用，首先需要将其导出为 MNN 格式。这可以使用 Ultralytics Python 包提供的 `yolo export` 命令轻松完成。

详细说明和选项请参阅 [Ultralytics 导出文档](https://docs.ultralytics.com/modes/export/)。

```bash
# 将 YOLOv8n 模型导出为 MNN 格式，输入大小为 640x640
yolo export model=yolov8n.pt imgsz=640 format=mnn
```

或者，你可以使用 MNN 提供的 `MNNConvert` 工具：

```bash
# 假设 MNNConvert 已构建并在你的 PATH 或 MNN 构建目录中
# 转换 ONNX 模型（首先将 YOLOv8 导出为 ONNX）
yolo export model=yolov8n.pt format=onnx
./MNN/build/MNNConvert -f ONNX --modelFile yolov8n.onnx --MNNModel yolov8n.mnn --bizCode biz
```

有关使用 MNN 工具进行模型转换的更多详情，请参阅 [MNN 转换文档](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html)。

## 🛠️ 使用方法

### Python 中的 Ultralytics CLI（用于对比）

你可以使用 Ultralytics Python 包验证导出的 MNN 模型以进行快速检查。

下载示例图像：

```bash
wget https://ultralytics.com/images/bus.jpg
```

使用 MNN 模型运行预测：

```bash
yolo predict model=yolov8n.mnn source=bus.jpg
```

预期 Python 输出：

```
ultralytics/examples/YOLOv8-MNN-CPP/assets/bus.jpg: 640x640 4 persons, 1 bus, 84.6ms
Speed: 9.7ms preprocess, 128.7ms inference, 12.4ms postprocess per image at shape (1, 3, 640, 640)
Results saved to runs/detect/predict
```

_（注意：速度和具体检测结果可能因硬件和模型版本而异）_

### C++ 中的 MNN Express API

本示例使用高级 Express API 以简化推理代码。

```bash
./build/main yolov8n.mnn bus.jpg
```

预期 C++ Express API 输出：

```
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
Detection: box = {48.63, 399.30, 243.65, 902.90}, class = person, score = 0.86
Detection: box = {22.14, 228.36, 796.07, 749.74}, class = bus, score = 0.86
Detection: box = {669.92, 375.82, 809.86, 874.41}, class = person, score = 0.86
Detection: box = {216.01, 405.24, 346.36, 858.19}, class = person, score = 0.82
Detection: box = {-0.11, 549.41, 62.05, 874.88}, class = person, score = 0.33
Result image write to `mnn_yolov8_cpp.jpg`.
Speed: 35.6ms preprocess, 386.0ms inference, 68.3ms postprocess
```

_（注意：速度和具体检测结果可能因硬件和 MNN 配置而异）_

### C++ 中的 MNN Interpreter API

本示例使用低级 Interpreter API，提供对推理过程的更多控制。

```bash
./build/main_interpreter yolov8n.mnn bus.jpg
```

预期 C++ Interpreter API 输出：

```
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
Detection: box = {48.63, 399.30, 243.65, 902.90}, class = person, score = 0.86
Detection: box = {22.14, 228.36, 796.07, 749.74}, class = bus, score = 0.86
Detection: box = {669.92, 375.82, 809.86, 874.41}, class = person, score = 0.86
Detection: box = {216.01, 405.24, 346.36, 858.19}, class = person, score = 0.82
Result image written to `mnn_yolov8_cpp.jpg`.
Speed: 26.0ms preprocess, 190.9ms inference, 58.9ms postprocess
```

_（注意：速度和具体检测结果可能因硬件和 MNN 配置而异）_

## ❤️ 贡献

我们希望此示例能帮助你轻松地将 Ultralytics YOLOv8 与 MNN 集成到你的 C++ 项目中！欢迎贡献以改进此示例或添加新功能。请参阅 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing/)了解更多关于如何参与的信息。

有关 Ultralytics YOLO 模型和工具的更多指南、教程和文档，请访问 [Ultralytics 主文档](https://docs.ultralytics.com/)。祝编码愉快！🚀
