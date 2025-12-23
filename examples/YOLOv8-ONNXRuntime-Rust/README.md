# YOLOv8-ONNXRuntime-Rust 全部关键 YOLO 任务

本仓库提供了一个 Rust 演示，用于使用 [ONNXRuntime](https://onnxruntime.ai/) 执行 Ultralytics YOLOv8 任务，包括[分类](https://docs.ultralytics.com/tasks/classify/)、[分割](https://docs.ultralytics.com/tasks/segment/)、[检测](https://docs.ultralytics.com/tasks/detect/)、[姿态估计](https://docs.ultralytics.com/tasks/pose/)和[定向边界框（OBB）](https://docs.ultralytics.com/tasks/obb/)检测。

## ✨ 最近更新

- 添加了 YOLOv8-OBB 演示。
- 将 ONNXRuntime 依赖更新到 1.19.x。

新更新的 YOLOv8 示例代码位于[此仓库](https://github.com/jamjamjon/usls/tree/main/examples/yolo)。

## 🚀 功能特性

- 支持 `分类`、`分割`、`检测`、`姿态（关键点）检测` 和 `OBB` 任务。
- 支持 `FP16` 和 `FP32` [ONNX](https://onnx.ai/) 模型。
- 支持 `CPU`、`CUDA` 和 `TensorRT` 执行提供程序以加速计算。
- 支持动态输入形状（`batch`、`width`、`height`）。

## 🛠️ 安装

### 1. 安装 Rust

请按照官方 Rust 安装指南操作：[https://www.rust-lang.org/tools/install](https://rust-lang.org/tools/install/)。

### 2. ONNXRuntime 链接

- #### 详细设置说明请参阅 [ORT 文档](https://ort.pyke.io/setup/linking)。

- #### Linux 或 macOS 用户：
  - 从 [Releases 页面](https://github.com/microsoft/onnxruntime/releases)下载 ONNX Runtime 包。
  - 通过导出 `ORT_DYLIB_PATH` 环境变量设置库路径：
    ```bash
    export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so.1.19.0 # 根据需要调整版本/路径
    ```

### 3. [可选] 安装 CUDA、CuDNN 和 TensorRT

- CUDA 执行提供程序需要 [CUDA](https://developer.nvidia.com/cuda-toolkit) v11.6+。
- TensorRT 执行提供程序需要 CUDA v11.4+ 和 [TensorRT](https://developer.nvidia.com/tensorrt) v8.4+。你可能还需要 [cuDNN](https://developer.nvidia.com/cudnn)。

## ▶️ 快速开始

### 1. 导出 Ultralytics YOLOv8 ONNX 模型

首先，安装 Ultralytics 包：

```bash
pip install -U ultralytics
```

然后，将所需的 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 模型导出为 ONNX 格式。详情请参阅[导出文档](https://docs.ultralytics.com/modes/export/)。

```bash
# 导出具有动态形状的 ONNX 模型（推荐以获得灵活性）
yolo export model=yolov8m.pt format=onnx simplify dynamic
yolo export model=yolov8m-cls.pt format=onnx simplify dynamic
yolo export model=yolov8m-pose.pt format=onnx simplify dynamic
yolo export model=yolov8m-seg.pt format=onnx simplify dynamic
# yolo export model=yolov8m-obb.pt format=onnx simplify dynamic # 如需要添加 OBB 导出

# 导出具有固定形状的 ONNX 模型（如果不需要动态形状）
# yolo export model=yolov8m.pt format=onnx simplify
# yolo export model=yolov8m-cls.pt format=onnx simplify
# yolo export model=yolov8m-pose.pt format=onnx simplify
# yolo export model=yolov8m-seg.pt format=onnx simplify
# yolo export model=yolov8m-obb.pt format=onnx simplify
```

### 2. 运行推理

此命令将使用 CPU 在源图像上使用指定的 ONNX 模型执行推理。

```bash
cargo run --release -- --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```


#### 使用 GPU 加速

设置 `--cuda` 以使用 CUDA 执行提供程序在 NVIDIA GPU 上进行更快的推理。

```bash
cargo run --release -- --cuda --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

设置 `--trt` 以使用 TensorRT 执行提供程序。你还可以同时设置 `--fp16` 以利用 TensorRT FP16 引擎获得更高速度，特别是在兼容硬件上。

```bash
cargo run --release -- --trt --fp16 --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

#### 指定设备和批量大小

设置 `--device_id` 以选择特定的 GPU 设备。如果指定的设备 ID 无效（例如，只有一个 GPU 时设置 `device_id 1`），`ort` 将自动回退到 `CPU` 执行提供程序而不会崩溃。

```bash
cargo run --release -- --cuda --device_id 0 --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

设置 `--batch` 以使用特定批量大小执行推理。

```bash
cargo run --release -- --cuda --batch 2 --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

如果你使用 `--trt` 和具有动态批量维度导出的模型，你可以使用 `--batch-min`、`--batch` 和 `--batch-max` 显式指定 TensorRT 优化的最小、最优和最大批量大小。详情请参阅 [TensorRT 执行提供程序文档](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#explicit-shape-range-for-dynamic-shape-input)。

#### 动态图像大小

设置 `--height` 和 `--width` 以使用动态图像大小执行推理。**注意：** ONNX 模型必须以动态输入形状导出（`dynamic=True`）。

```bash
cargo run --release -- --cuda --width 480 --height 640 --model MODEL_PATH_dynamic.onnx --source SOURCE_IMAGE.jpg
```

#### 性能分析

设置 `--profile` 以测量推理管道每个阶段（预处理、H2D 传输、推理、D2H 传输、后处理）消耗的时间。**注意：** 模型通常需要几次"预热"运行（1-3 次迭代）才能达到最佳性能。确保运行足够多次以获得稳定的性能评估。

```bash
cargo run --release -- --trt --fp16 --profile --model MODEL_PATH.onnx --source SOURCE_IMAGE.jpg
```

示例性能分析输出（yolov8m.onnx，batch=1，3 次运行，trt，fp16，RTX 3060Ti）：

```text
==> 0 # 预热运行
[Model Preprocess]: 12.75788ms
[ORT H2D]: 237.118µs
[ORT Inference]: 507.895469ms
[ORT D2H]: 191.655µs
[Model Inference]: 508.34589ms
[Model Postprocess]: 1.061122ms
==> 1 # 稳定运行
[Model Preprocess]: 13.658655ms
[ORT H2D]: 209.975µs
[ORT Inference]: 5.12372ms
[ORT D2H]: 182.389µs
[Model Inference]: 5.530022ms
[Model Postprocess]: 1.04851ms
==> 2 # 稳定运行
[Model Preprocess]: 12.475332ms
[ORT H2D]: 246.127µs
[ORT Inference]: 5.048432ms
[ORT D2H]: 187.117µs
[Model Inference]: 5.493119ms
[Model Postprocess]: 1.040906ms
```

#### 其他选项

- `--conf`：检测的置信度阈值 \[默认：0.3]。
- `--iou`：非极大值抑制（NMS）的 IoU（交并比）阈值 \[默认：0.45]。
- `--kconf`：关键点的置信度阈值（姿态估计中）\[默认：0.55]。
- `--plot`：使用随机 RGB 颜色绘制推理结果并将输出图像保存到 `runs` 目录。

你可以通过运行以下命令查看所有可用的命令行参数：

```bash
# 如果尚未克隆仓库
# git clone https://github.com/ultralytics/ultralytics
# cd ultralytics/examples/YOLOv8-ONNXRuntime-Rust

cargo run --release -- --help
```

## 🖼️ 示例

![Ultralytics YOLO 任务](https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png)

### 分类

在 `CPU` 上运行具有特定图像大小（`--height 224 --width 224`）的动态形状 ONNX 分类模型。绘制的结果图像将保存在 `runs` 目录中。

```bash
cargo run --release -- --model ../assets/weights/yolov8m-cls-dyn.onnx --source ../assets/images/dog.jpg --height 224 --width 224 --plot --profile
```

### 目标检测

使用 `CUDA` 执行提供程序和动态图像大小（`--height 640 --width 480`）。

```bash
cargo run --release -- --cuda --model ../assets/weights/yolov8m-dynamic.onnx --source ../assets/images/bus.jpg --plot --height 640 --width 480
```

### 姿态检测

使用 `TensorRT` 执行提供程序。

```bash
cargo run --release -- --trt --model ../assets/weights/yolov8m-pose.onnx --source ../assets/images/bus.jpg --plot
```

### 实例分割

使用 `TensorRT` 执行提供程序和 FP16 模型（`--fp16`）。

```bash
cargo run --release -- --trt --fp16 --model ../assets/weights/yolov8m-seg.onnx --source ../assets/images/0172.jpg --plot
```

## 🤝 贡献

欢迎贡献！如果你发现任何问题或有改进建议，请随时在 [Ultralytics 主仓库](https://github.com/ultralytics/ultralytics)上提交 issue 或 pull request。
