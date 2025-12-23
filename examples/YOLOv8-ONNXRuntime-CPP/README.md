# YOLOv8 ONNX Runtime C++ 示例

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white">

本示例提供了使用 [C++](https://isocpp.org/) 执行 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 模型推理的实用指南，利用 [ONNX Runtime](https://onnxruntime.ai/) 和 [OpenCV](https://opencv.org/) 库的能力。它专为希望将 YOLOv8 集成到 C++ 应用程序中以实现高效目标检测的开发者设计。

## ✨ 优势

- **部署友好**：非常适合在工业和生产环境中部署。
- **性能**：在 CPU 和 [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) 上都比 OpenCV 的 DNN 模块提供更快的[推理延迟](https://www.ultralytics.com/glossary/inference-latency)。
- **加速**：支持使用 [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) 的 FP32 和 [FP16（半精度）](https://www.ultralytics.com/glossary/half-precision)推理加速。

## ☕ 注意

由于 Ultralytics 的最新更新，YOLOv8 模型现在包含 `Transpose` 操作，使其输出形状与 YOLOv5 对齐。这允许本项目中的 C++ 代码无缝地为导出到 [ONNX 格式](https://onnx.ai/)的 YOLOv5、YOLOv7 和 YOLOv8 模型运行推理。

## 📦 导出 YOLOv8 模型

你可以将训练好的 [Ultralytics YOLO](https://docs.ultralytics.com/) 模型导出为本项目所需的 ONNX 格式。使用 Ultralytics `export` 模式完成此操作。

### Python

```python
from ultralytics import YOLO

# 加载 YOLOv8 模型（如 yolov8n.pt）
model = YOLO("yolov8n.pt")

# 将模型导出为 ONNX 格式
# 建议使用 opset=12 以确保兼容性
# simplify=True 优化模型图
# dynamic=False 确保固定输入大小，通常更适合 C++ 部署
# imgsz=640 设置输入图像大小
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
print("模型成功导出为 yolov8n.onnx")
```

### CLI

```bash
# 使用命令行导出模型
yolo export model=yolov8n.pt format=onnx opset=12 simplify=True dynamic=False imgsz=640
```

有关导出模型的更多详情，请参阅 [Ultralytics 导出文档](https://docs.ultralytics.com/modes/export/)。

## 📦 导出 YOLOv8 FP16 模型

为了在兼容硬件（如 NVIDIA GPU）上获得更高性能，你可以将导出的 FP32 ONNX 模型转换为 FP16。

```python
import onnx
from onnxconverter_common import (
    float16,
)  # 确保已安装 onnxconverter-common：pip install onnxconverter-common

# 加载你的 FP32 ONNX 模型
fp32_model_path = "yolov8n.onnx"
model = onnx.load(fp32_model_path)

# 将模型转换为 FP16
model_fp16 = float16.convert_float_to_float16(model)

# 保存 FP16 模型
fp16_model_path = "yolov8n_fp16.onnx"
onnx.save(model_fp16, fp16_model_path)
print(f"模型已转换并保存到 {fp16_model_path}")
```


## 📂 下载 COCO YAML 文件

本示例使用 YAML 文件中定义的类别名称。你需要 `coco.yaml` 文件，它对应标准 [COCO 数据集](https://docs.ultralytics.com/datasets/detect/coco/)类别。直接下载：

- [下载 coco.yaml](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml)

将此文件保存在你计划运行可执行文件的同一目录中，或在 C++ 代码中相应调整路径。

## ⚙️ 依赖项

确保已安装以下依赖项：

| 依赖项                                                               | 版本          | 备注                                                                                                                                                                        |
| :------------------------------------------------------------------- | :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ONNX Runtime](https://onnxruntime.ai/docs/install/)                 | >=1.14.1      | 下载预构建二进制文件或从源代码构建。如果使用 CUDA，确保是 GPU 版本。                                                                                                        |
| [OpenCV](https://opencv.org/releases/)                               | >=4.0.0       | 图像加载和预处理所需。                                                                                                                                                      |
| C++ 编译器                                                           | C++17 支持    | 需要 `<filesystem>` 等特性。（[GCC](https://gcc.gnu.org/)、[Clang](https://clang.llvm.org/)、[MSVC](https://visualstudio.microsoft.com/vs/features/cplusplus/)）            |
| [CMake](https://cmake.org/download/)                                 | >=3.18        | 跨平台构建系统生成器。建议 3.18+ 版本以更好地发现 CUDA 支持。                                                                                                               |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)（可选）    | >=11.4, <12.0 | 通过 ONNX Runtime 的 CUDA 执行提供程序进行 GPU 加速所需。**必须是 CUDA 11.x**。                                                                                             |
| [cuDNN](https://developer.nvidia.com/cudnn)（需要 CUDA）             | =8.x          | CUDA 执行提供程序所需。**必须是 cuDNN 8.x**，与你的 CUDA 11.x 版本兼容。                                                                                                    |

**重要说明：**

1.  **C++17**：此要求源于使用 C++17 中引入的 `<filesystem>` 库进行路径处理。
2.  **CUDA/cuDNN 版本**：ONNX Runtime 的 CUDA 执行提供程序目前有严格的版本要求（CUDA 11.x，cuDNN 8.x）。查看最新的 [ONNX Runtime 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)了解这些约束的任何更新。使用不兼容的版本将导致运行时错误。

## 🛠️ 构建说明

1.  **克隆仓库：**

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics/examples/YOLOv8-ONNXRuntime-CPP
    ```

2.  **创建构建目录：**

    ```bash
    mkdir build && cd build
    ```

3.  **使用 CMake 配置：**
    运行 CMake 生成构建文件。你**必须**使用 `ONNXRUNTIME_ROOT` 指定 ONNX Runtime 安装目录的路径。根据你下载或构建 ONNX Runtime 的位置调整路径。

    ```bash
    # Linux/macOS 示例（根据需要调整路径）
    cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime

    # Windows 示例（根据需要调整路径，使用反斜杠或正斜杠）
    cmake .. -DONNXRUNTIME_ROOT="C:/path/to/onnxruntime"
    ```

    **CMake 选项：**
    - `-DONNXRUNTIME_ROOT=<path>`：**（必需）** 解压的 ONNX Runtime 库路径。
    - `-DCMAKE_BUILD_TYPE=Release`：（可选）以 Release 模式构建以进行优化。
    - 如果 CMake 难以找到 OpenCV，你可能需要设置 `-DOpenCV_DIR=/path/to/opencv/build`。

4.  **构建项目：**
    使用 CMake 生成的构建工具（如 Make、Ninja、Visual Studio）。

    ```bash
    # 使用 Make（Linux/macOS 常用）
    make

    # 使用 CMake 的通用构建命令（适用于 Make、Ninja 等）
    cmake --build . --config Release
    ```

5.  **定位可执行文件：**
    编译后的可执行文件（如 `yolov8_onnxruntime_cpp`）将位于 `build` 目录中。

## 🚀 使用方法

运行前，请确保：

- 导出的 `.onnx` 模型文件（如 `yolov8n.onnx`）可访问。
- `coco.yaml` 文件可访问。
- ONNX Runtime 和 OpenCV 所需的任何共享库都在系统 PATH 中或可执行文件可访问。

修改 `main.cpp` 文件（或创建配置机制）以设置参数：

```cpp
// 根据需要更改参数
// 注意你的设备和 onnx 模型类型（fp32 或 fp16）
DL_INIT_PARAM params;
params.rectConfidenceThreshold = 0.1;
params.iouThreshold = 0.5;
params.modelPath = "yolov8n.onnx";
params.imgSize = { 640, 640 };
params.cudaEnable = true;
params.modelType = YOLO_DETECT_V8;
yoloDetector->CreateSession(params);
Detector(yoloDetector);
```

从 `build` 目录运行可执行文件：

```bash
./yolov8_onnxruntime_cpp
```

## 🤝 贡献

欢迎贡献！如果你发现任何问题或有改进建议，请随时在 [Ultralytics 主仓库](https://github.com/ultralytics/ultralytics)上提交 issue 或 pull request。
