# YOLO 系列 ONNXRuntime Rust 演示 - 核心 YOLO 任务

本仓库提供了一个 [Rust](https://rust-lang.org/) 演示，展示了使用 [ONNXRuntime](https://github.com/microsoft/onnxruntime) 执行 [Ultralytics YOLO](https://docs.ultralytics.com/) 系列的关键任务，包括[分类](https://docs.ultralytics.com/tasks/classify/)、[分割](https://docs.ultralytics.com/tasks/segment/)、[检测](https://docs.ultralytics.com/tasks/detect/)、[姿态估计](https://docs.ultralytics.com/tasks/pose/)和定向边界框（[OBB](https://docs.ultralytics.com/tasks/obb/)）检测。支持多种 YOLO 模型（v5 到 11）和多种计算机视觉任务。

## ✨ 简介

- 本示例利用了 [ONNX Runtime](https://onnxruntime.ai/) 和流行 YOLO 模型的最新版本。
- 我们使用 [usls crate](https://github.com/jamjamjon/usls/tree/main) 来简化 Rust 中的 YOLO 模型推理，提供高效的数据加载、可视化和优化的推理性能。这使开发者能够轻松地将最先进的目标检测集成到他们的 Rust 应用程序中。

## 🚀 功能特性

- **广泛的模型兼容性**：支持多种 YOLO 版本，包括 [YOLOv5](https://docs.ultralytics.com/models/yolov5/)、[YOLOv6](https://docs.ultralytics.com/models/yolov6/)、[YOLOv7](https://docs.ultralytics.com/models/yolov7/)、[YOLOv8](https://docs.ultralytics.com/models/yolov8/)、[YOLOv9](https://docs.ultralytics.com/models/yolov9/)、[YOLOv10](https://docs.ultralytics.com/models/yolov10/)、[YOLO11](https://docs.ultralytics.com/models/yolo11/)、[YOLO-World](https://docs.ultralytics.com/models/yolo-world/)、[RT-DETR](https://docs.ultralytics.com/models/rtdetr/) 等。
- **多任务覆盖**：包含 `分类`、`分割`、`检测`、`姿态` 和 `OBB` 任务示例。
- **精度灵活性**：无缝支持 `FP16` 和 `FP32` 精度的 [ONNX 模型](https://docs.ultralytics.com/integrations/onnx/)。
- **执行提供程序**：支持 `CPU`、[CUDA](https://developer.nvidia.com/cuda-toolkit)、[CoreML](https://developer.apple.com/documentation/coreml) 和 [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) 加速计算。
- **动态输入形状**：动态调整可变的 `batch`、`width` 和 `height` 维度，实现灵活的模型输入。
- **灵活的数据加载**：`DataLoader` 组件处理图像、文件夹、视频和实时视频流。
- **实时显示和视频导出**：`Viewer` 提供实时帧可视化和视频导出功能，类似于 OpenCV 的 `imshow()` 和 `imwrite()`。
- **增强的标注和可视化**：`Annotator` 支持全面的结果渲染，包括水平边界框（HBB）、定向边界框（OBB）、多边形、掩码、关键点和文本标签。

## 🛠️ 设置说明

### 1. ONNXRuntime 链接

<details>
<summary>你有两种方式链接 ONNXRuntime 库：</summary>

- **方式 1：手动链接**
  - 详细设置说明请参阅 [ONNX Runtime 链接文档](https://ort.pyke.io/setup/linking)。
  - **Linux 或 macOS**：
    1. 从官方 [Releases 页面](https://github.com/microsoft/onnxruntime/releases)下载适当的 ONNX Runtime 包。
    2. 通过导出 `ORT_DYLIB_PATH` 环境变量设置库路径，指向下载的库文件：
       ```bash
       # 示例路径，请替换为你的实际路径
       export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so.1.19.0
       ```

- **方式 2：自动下载**
  - 使用 Cargo 的 `--features auto` 标志让构建脚本自动处理库下载：
    ```bash
    cargo run -r --example yolo --features auto
    ```

</details>

### 2. [可选] 安装 CUDA、CuDNN 和 TensorRT

- CUDA 执行提供程序需要 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 版本 `12.x`。
- TensorRT 执行提供程序需要 CUDA `12.x` 和 [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) `10.x`。确保 [cuDNN](https://developer.nvidia.com/cudnn) 也正确安装。

### 3. [可选] 安装 ffmpeg

要启用视频帧查看和保存视频推理结果，请安装 `rust-ffmpeg` crate 的依赖。按照此处的说明操作：
[https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building#dependencies](https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building#dependencies)

## ▶️ 快速开始

使用 Cargo 运行示例。`--` 分隔 Cargo 参数和示例参数。

```bash
# 运行自定义模型（如 YOLOv8 检测）
cargo run -r -- --task detect --ver v8 --nc 6 --model path/to/your/model.onnx

# 分类示例
cargo run -r -- --task classify --ver v5 --scale s --width 224 --height 224 --nc 1000  # YOLOv5 分类
cargo run -r -- --task classify --ver v8 --scale n --width 224 --height 224 --nc 1000  # YOLOv8 分类
cargo run -r -- --task classify --ver v11 --scale n --width 224 --height 224 --nc 1000 # YOLO11 分类

# 检测示例
cargo run -r -- --task detect --ver v5 --scale n     # YOLOv5 检测
cargo run -r -- --task detect --ver v6 --scale n     # YOLOv6 检测
cargo run -r -- --task detect --ver v7 --scale t     # YOLOv7 检测
cargo run -r -- --task detect --ver v8 --scale n     # YOLOv8 检测
cargo run -r -- --task detect --ver v9 --scale t     # YOLOv9 检测
cargo run -r -- --task detect --ver v10 --scale n    # YOLOv10 检测
cargo run -r -- --task detect --ver v11 --scale n    # YOLO11 检测
cargo run -r -- --task detect --ver rtdetr --scale l # RT-DETR 检测

# 姿态示例
cargo run -r -- --task pose --ver v8 --scale n  # YOLOv8 姿态估计
cargo run -r -- --task pose --ver v11 --scale n # YOLO11 姿态估计

# 分割示例
cargo run -r -- --task segment --ver v5 --scale n                              # YOLOv5 分割
cargo run -r -- --task segment --ver v8 --scale n                              # YOLOv8 分割
cargo run -r -- --task segment --ver v11 --scale n                             # YOLO11 分割
cargo run -r -- --task segment --ver v8 --model path/to/FastSAM-s-dyn-f16.onnx # FastSAM 分割

# OBB（定向边界框）示例
cargo run -r -- --ver v8 --task obb --scale n --width 1024 --height 1024 --source images/dota.png  # YOLOv8-OBB
cargo run -r -- --ver v11 --task obb --scale n --width 1024 --height 1024 --source images/dota.png # YOLO11-OBB
```

**使用 `cargo run -- --help` 查看所有可用选项。**

更多详细信息和高级用法，请参阅 [usls-yolo 示例文档](https://github.com/jamjamjon/usls/tree/main/examples/yolo)。

## 🤝 贡献

欢迎贡献！如果你想改进此演示或添加新功能，请随时在仓库上提交 issue 或 pull request。你的参与有助于让 Ultralytics 生态系统对每个人都更好。查看 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing/)了解更多详情。
