# YOLOv8 OpenVINO 推理 C++

欢迎使用 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) OpenVINO 推理 C++ 示例！本指南将帮助你开始在 C++ 项目中使用 [Intel OpenVINO™ 工具包](https://docs.openvino.ai/)和 [OpenCV API](https://docs.opencv.org/) 利用强大的 YOLOv8 模型。无论你是想在 Intel 硬件上提升性能还是为应用程序增加灵活性，本示例都提供了坚实的基础。在 [Ultralytics 博客](https://www.ultralytics.com/blog)上了解更多关于优化模型的信息。

## 🌟 功能特性

- 🚀 **模型格式支持**：兼容 [ONNX](https://onnx.ai/) 和 [OpenVINO 中间表示（IR）](https://docs.openvino.ai/2023.3/openvino_docs_MO_DG_IR_and_opsets.html)格式。查看 [Ultralytics ONNX 集成](https://docs.ultralytics.com/integrations/onnx/)了解更多详情。
- ⚡ **精度选项**：支持 **FP32**、**FP16**（[半精度](https://www.ultralytics.com/glossary/half-precision)）和 **INT8**（[量化](https://www.ultralytics.com/glossary/model-quantization)）精度运行模型以优化性能。
- 🔄 **动态形状加载**：轻松处理具有动态输入形状的模型，这是许多[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务的常见需求。

## 📋 依赖项

为确保顺利执行，请确保已安装以下依赖项：

| 依赖项                                                | 版本     |
| ----------------------------------------------------- | -------- |
| [OpenVINO](https://docs.openvino.ai/latest/home.html) | >=2023.3 |
| [OpenCV](https://opencv.org/)                         | >=4.5.0  |
| [C++](https://en.cppreference.com/w/)                 | >=14     |
| [CMake](https://cmake.org/documentation/)             | >=3.12.0 |

## ⚙️ 构建说明

按照以下步骤构建项目：

1.  克隆 Ultralytics 仓库：

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics/examples/YOLOv8-OpenVINO-CPP-Inference
    ```

2.  创建构建目录并使用 CMake 编译项目：
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

## 🛠️ 使用方法

构建完成后，你可以使用编译后的可执行文件对图像运行[推理](https://www.ultralytics.com/glossary/real-time-inference)。提供模型文件路径（OpenVINO IR 的 `.xml` 或 `.onnx`）和图像路径：

```bash
# 使用 OpenVINO IR 模型的示例
./detect path/to/your/model.xml path/to/your/image.jpg

# 使用 ONNX 模型的示例
./detect path/to/your/model.onnx path/to/your/image.jpg
```

此命令将使用指定的 YOLOv8 模型处理图像并显示[目标检测](https://www.ultralytics.com/glossary/object-detection)结果。探索各种 [Ultralytics 解决方案](https://docs.ultralytics.com/solutions/)了解实际应用。

## 🔄 导出 YOLOv8 模型

要将你的 Ultralytics YOLOv8 模型与此 C++ 示例一起使用，首先需要将其导出为 OpenVINO IR 或 ONNX 格式。使用 Ultralytics Python 包中提供的 `yolo export` 命令。详细说明请参阅[导出模式文档](https://docs.ultralytics.com/modes/export/)。

```bash
# 导出为 OpenVINO 格式（生成 .xml 和 .bin 文件）
yolo export model=yolov8s.pt imgsz=640 format=openvino

# 导出为 ONNX 格式
yolo export model=yolov8s.pt imgsz=640 format=onnx
```

有关导出和优化 OpenVINO 模型的更多详情，请参阅 [Ultralytics OpenVINO 集成指南](https://docs.ultralytics.com/integrations/openvino/)。

## 📸 截图

### 使用 OpenVINO 模型运行

![运行 OpenVINO 模型](https://github.com/ultralytics/ultralytics/assets/76827698/2d7cf201-3def-4357-824c-12446ccf85a9)

### 使用 ONNX 模型运行

![运行 ONNX 模型](https://github.com/ultralytics/ultralytics/assets/76827698/9b90031c-cc81-4cfb-8b34-c619e09035a7)

## ❤️ 贡献

我们希望此示例能帮助你轻松地将 YOLOv8 与 OpenVINO 和 OpenCV 集成到你的 C++ 项目中。欢迎贡献以改进此示例或添加新功能！请参阅 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing/)了解更多信息。访问 [Ultralytics 主文档](https://docs.ultralytics.com/)获取更多指南和资源。祝编码愉快！🚀
