# YOLOv8/YOLOv5 C++ 推理与 OpenCV DNN

本示例演示如何使用 [OpenCV DNN 模块](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)在 C++ 中执行 Ultralytics YOLOv8 和 YOLOv5 模型推理。

## 🛠️ 使用方法

按照以下步骤设置和运行 C++ 推理示例：

```bash
# 1. 克隆 Ultralytics 仓库
git clone https://github.com/ultralytics/ultralytics
cd ultralytics

# 2. 安装 Ultralytics Python 包（导出模型需要）
pip install .

# 3. 进入 C++ 示例目录
cd examples/YOLOv8-CPP-Inference

# 4. 导出模型：添加 yolov8*.onnx 和/或 yolov5*.onnx 模型（见下方导出说明）
#    将导出的 ONNX 模型放在当前目录（YOLOv8-CPP-Inference）。

# 5. 更新源代码：编辑 main.cpp 并将 'projectBasePath' 变量
#    设置为你系统上 'YOLOv8-CPP-Inference' 目录的绝对路径。
#    示例：std::string projectBasePath = "/path/to/your/ultralytics/examples/YOLOv8-CPP-Inference";

# 6. 配置 OpenCV DNN 后端（可选 - CUDA）：
#    - 默认 CMakeLists.txt 尝试使用 CUDA 进行 OpenCV DNN 的 GPU 加速。
#    - 如果你的 OpenCV 构建不支持 CUDA/cuDNN，或你想要 CPU 推理，
#      请从 CMakeLists.txt 中移除 CUDA 相关行。

# 7. 构建项目
mkdir build
cd build
cmake ..
make

# 8. 运行推理可执行文件
./Yolov8CPPInference
```

## ✨ 导出 YOLOv8 和 YOLOv5 模型

你需要将训练好的 PyTorch 模型导出为 [ONNX](https://onnx.ai/) 格式才能与 OpenCV DNN 一起使用。

**导出 Ultralytics YOLOv8 模型：**

使用 Ultralytics CLI 导出。确保指定所需的 `imgsz` 和 `opset`。为了与本示例兼容，建议使用 `opset=12`。

```bash
yolo export model=yolov8s.pt imgsz=640,480 format=onnx opset=12 # 示例：640x480 分辨率
```

**导出 YOLOv5 模型：**

使用 YOLOv5 仓库结构中的 `export.py` 脚本（包含在克隆的 `ultralytics` 仓库中）。

```bash
# 假设你在克隆后位于 'ultralytics' 基础目录
python export.py --weights yolov5s.pt --imgsz 640 480 --include onnx --opset 12 # 示例：640x480 分辨率
```

将生成的 `.onnx` 文件（如 `yolov8s.onnx`、`yolov5s.onnx`）放入 `ultralytics/examples/YOLOv8-CPP-Inference/` 目录。

**示例输出：**

_yolov8s.onnx:_

![YOLOv8 ONNX 输出](https://user-images.githubusercontent.com/40023722/217356132-a4cecf2e-2729-4acb-b80a-6559022d7707.png)

_yolov5s.onnx:_

![YOLOv5 ONNX 输出](https://user-images.githubusercontent.com/40023722/217357005-07464492-d1da-42e3-98a7-fc753f87d5e6.png)

## 📝 注意事项

- 本仓库使用 [OpenCV DNN API](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html) 运行 YOLOv5 和 Ultralytics YOLOv8 的 [ONNX](https://onnx.ai/) 导出模型。
- 虽然未经明确测试，但如果 YOLOv6 和 YOLOv7 等其他 YOLO 架构的 ONNX 导出格式兼容，理论上也可能适用。
- 示例模型以矩形分辨率（640x480）导出，但代码应能处理以不同分辨率导出的模型。如果输入图像的宽高比与模型训练分辨率不同，特别是对于正方形 `imgsz` 导出，请考虑使用[信箱填充](https://docs.ultralytics.com/modes/predict/#letterbox)等技术。
- `main` 分支版本包含使用 [Qt](https://www.qt.io/) 的简单 GUI 封装。但核心逻辑位于 `Inference` 类（`inference.h`、`inference.cpp`）中。
- `Inference` 类的一个关键部分演示了如何处理 YOLOv5 和 YOLOv8 模型之间的输出差异，有效地转置 YOLOv8 的输出格式以匹配 YOLOv5 预期的结构，实现一致的后处理。

## 🤝 贡献

欢迎贡献！如果你发现任何问题或有改进建议，请随时提交 issue 或 pull request。查看我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)了解更多详情。
