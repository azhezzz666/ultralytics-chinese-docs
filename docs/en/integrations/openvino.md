---
comments: true
description: 学习如何将 YOLO11 模型导出为 OpenVINO 格式，以获得高达 3 倍的 CPU 加速，以及在 Intel GPU 和 NPU 上的硬件加速。
keywords: YOLO11, OpenVINO, 模型导出, Intel, AI 推理, CPU 加速, GPU 加速, NPU, 深度学习
---

# Intel OpenVINO 导出

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ecosystem.avif" alt="OpenVINO 生态系统">

在本指南中，我们介绍如何将 YOLO11 模型导出为 [OpenVINO](https://docs.openvino.ai/) 格式，该格式可以提供高达 3 倍的 [CPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) 加速，以及在 Intel [GPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) 和 [NPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) 硬件上加速 YOLO 推理。

OpenVINO 是 Open Visual Inference & [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Optimization 工具包的缩写，是一个用于优化和部署 AI 推理模型的综合工具包。尽管名称中包含 Visual，OpenVINO 还支持各种其他任务，包括语言、音频、时间序列等。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/AvFh-oTGDaw"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何将 Ultralytics YOLO11 导出为 Intel OpenVINO 格式以实现更快推理 🚀
</p>

## 用法示例

将 YOLO11n 模型导出为 OpenVINO 格式并使用导出的模型运行推理。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11n PyTorch 模型
        model = YOLO("yolo11n.pt")

        # 导出模型
        model.export(format="openvino")  # 创建 'yolo11n_openvino_model/'

        # 加载导出的 OpenVINO 模型
        ov_model = YOLO("yolo11n_openvino_model/")

        # 运行推理
        results = ov_model("https://ultralytics.com/images/bus.jpg")

        # 使用指定设备运行推理，可用设备: ["intel:gpu", "intel:npu", "intel:cpu"]
        results = ov_model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 OpenVINO 格式
        yolo export model=yolo11n.pt format=openvino # 创建 'yolo11n_openvino_model/'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg'

        # 使用指定设备运行推理，可用设备: ["intel:gpu", "intel:npu", "intel:cpu"]
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg' device="intel:gpu"
        ```

## 导出参数

| 参数 | 类型 | 默认值 | 描述 |
| ---------- | ---------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format` | `str` | `'openvino'` | 导出模型的目标格式，定义与各种部署环境的兼容性。 |
| `imgsz` | `int` 或 `tuple` | `640` | 模型输入所需的图像大小。可以是整数（用于正方形图像）或元组 `(height, width)`（用于特定尺寸）。 |
| `half` | `bool` | `False` | 启用 FP16（半精度）量化，减小模型大小并可能在支持的硬件上加速推理。 |
| `int8` | `bool` | `False` | 激活 INT8 量化，进一步压缩模型并加速推理，同时[准确率](https://www.ultralytics.com/glossary/accuracy)损失最小，主要用于边缘设备。 |
| `dynamic` | `bool` | `False` | 允许动态输入大小，增强处理不同图像尺寸的灵活性。 |
| `nms` | `bool` | `False` | 添加非极大值抑制（NMS），对于准确高效的检测后处理至关重要。 |
| `batch` | `int` | `1` | 指定导出模型批量推理大小或导出模型在 `predict` 模式下将并发处理的最大图像数量。 |
| `data` | `str` | `'coco8.yaml'` | [数据集](https://docs.ultralytics.com/datasets/)配置文件的路径（默认：`coco8.yaml`），对量化至关重要。 |
| `fraction` | `float` | `1.0` | 指定用于 INT8 量化校准的数据集比例。允许在完整数据集的子集上进行校准，在实验或资源有限时很有用。如果未指定且启用了 INT8，将使用完整数据集。 |

有关导出过程的更多详细信息，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

!!! warning

    OpenVINO™ 与大多数 Intel® 处理器兼容，但为确保最佳性能：

    1. 验证 OpenVINO™ 支持
        使用 [Intel 的兼容性列表](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html)检查你的 Intel® 芯片是否被 OpenVINO™ 官方支持。

    2. 识别你的加速器
        通过查阅 [Intel 的硬件指南](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html)确定你的处理器是否包含集成 NPU（神经处理单元）或 GPU（集成 GPU）。

    3. 安装最新驱动程序
        如果你的芯片支持 NPU 或 GPU 但 OpenVINO™ 未检测到它，你可能需要安装或更新相关驱动程序。按照[驱动程序安装说明](https://medium.com/openvino-toolkit/how-to-run-openvino-on-a-linux-ai-pc-52083ce14a98)启用完全加速。

    通过遵循这三个步骤，你可以确保 OpenVINO™ 在你的 Intel® 硬件上以最佳方式运行。

## OpenVINO 的优势

1. **性能**：OpenVINO 通过利用 Intel CPU、集成和独立 GPU 以及 FPGA 的能力提供高性能推理。
2. **支持异构执行**：OpenVINO 提供 API，可以编写一次并在任何支持的 Intel 硬件（CPU、GPU、FPGA、VPU 等）上部署。
3. **模型优化器**：OpenVINO 提供模型优化器，可以从流行的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)框架（如 PyTorch、[TensorFlow](https://www.ultralytics.com/glossary/tensorflow)、TensorFlow Lite、Keras、ONNX、PaddlePaddle 和 Caffe）导入、转换和优化模型。
4. **易于使用**：该工具包附带超过 [80 个教程 notebooks](https://github.com/openvinotoolkit/openvino_notebooks)（包括 [YOLOv8 优化](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization)），教授工具包的不同方面。

## OpenVINO 导出结构

当你将模型导出为 OpenVINO 格式时，会生成一个包含以下内容的目录：

1. **XML 文件**：描述网络拓扑。
2. **BIN 文件**：包含权重和偏置的二进制数据。
3. **映射文件**：保存原始模型输出张量到 OpenVINO 张量名称的映射。

你可以使用这些文件通过 OpenVINO 推理引擎运行推理。


## 在部署中使用 OpenVINO 导出

一旦你的模型成功导出为 OpenVINO 格式，你有两个主要选项来运行推理：

1. 使用 `ultralytics` 包，它提供高级 API 并封装了 OpenVINO Runtime。

2. 使用原生 `openvino` 包，以获得对推理行为的更高级或自定义控制。

### 使用 Ultralytics 进行推理

ultralytics 包允许你通过 predict 方法轻松使用导出的 OpenVINO 模型运行推理。你还可以使用 device 参数指定目标设备（例如 `intel:gpu`、`intel:npu`、`intel:cpu`）。

```python
from ultralytics import YOLO

# 加载导出的 OpenVINO 模型
ov_model = YOLO("yolo11n_openvino_model/")  # 你导出的 OpenVINO 模型路径
# 使用导出的模型运行推理
ov_model.predict(device="intel:gpu")  # 指定你想要运行推理的设备
```

这种方法非常适合快速原型设计或在不需要完全控制推理管道时进行部署。

### 使用 OpenVINO Runtime 进行推理

OpenVINO Runtime 为所有支持的 Intel 硬件上的推理提供统一的 API。它还提供高级功能，如跨 Intel 硬件的负载均衡和异步执行。有关运行推理的更多信息，请参阅 [YOLO11 notebooks](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov11-optimization)。

请记住，你需要 XML 和 BIN 文件以及任何特定于应用的设置（如输入大小、归一化的缩放因子等）来正确设置和使用 Runtime 中的模型。

在你的部署应用中，通常会执行以下步骤：

1. 通过创建 `core = Core()` 初始化 OpenVINO。
2. 使用 `core.read_model()` 方法加载模型。
3. 使用 `core.compile_model()` 函数编译模型。
4. 准备输入（图像、文本、音频等）。
5. 使用 `compiled_model(input_data)` 运行推理。

有关更详细的步骤和代码片段，请参阅 [OpenVINO 文档](https://docs.openvino.ai/)或 [API 教程](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-api/openvino-api.ipynb)。

## OpenVINO YOLO11 基准测试

Ultralytics 团队在各种模型格式和[精度](https://www.ultralytics.com/glossary/precision)下对 YOLO11 进行了基准测试，评估了在与 OpenVINO 兼容的不同 Intel 设备上的速度和准确率。

!!! note

    以下基准测试结果仅供参考，可能会根据系统的确切硬件和软件配置以及运行基准测试时系统的当前工作负载而有所不同。

    所有基准测试都使用 `openvino` Python 包版本 [2025.1.0](https://pypi.org/project/openvino/2025.1.0/) 运行。

### Intel Core CPU

Intel® Core® 系列是 Intel 的一系列高性能处理器。该系列包括 Core i3（入门级）、Core i5（中端）、Core i7（高端）和 Core i9（极致性能）。每个系列都满足不同的计算需求和预算，从日常任务到要求苛刻的专业工作负载。随着每一代新产品的推出，性能、能效和功能都得到了改进。

以下基准测试在第 12 代 Intel® Core® i9-12900KS CPU 上以 FP32 精度运行。

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-corei9.avif" alt="Core CPU 基准测试">
</div>

### Intel® Core™ Ultra

Intel® Core™ Ultra™ 系列代表了高性能计算的新基准，旨在满足现代用户不断变化的需求——从游戏玩家和创作者到利用 AI 的专业人士。这一新一代产品线不仅仅是传统的 CPU 系列；它在单个芯片中结合了强大的 CPU 核心、集成的高性能 GPU 功能和专用的神经处理单元（NPU），为多样化和密集型计算工作负载提供统一的解决方案。

Intel® Core Ultra™ 架构的核心是混合设计，可在传统处理任务、GPU 加速工作负载和 AI 驱动操作中实现卓越性能。NPU 的加入增强了设备端 AI 推理，在广泛的应用中实现更快、更高效的机器学习和数据处理。

Core Ultra™ 系列包括针对不同性能需求量身定制的各种型号，从节能设计到标有"H"标识的高功率变体——非常适合需要强大计算能力的笔记本电脑和紧凑型外形设备。在整个产品线中，用户受益于 CPU、GPU 和 NPU 集成的协同作用，提供卓越的效率、响应能力和多任务处理能力。

## 总结

在本指南中，我们介绍了如何将 Ultralytics YOLO11 模型导出为 OpenVINO 格式。OpenVINO 通过利用 Intel 硬件的能力提供高性能推理，是在 Intel 平台上部署 YOLO11 模型的绝佳选择。

有关更多信息和高级用法，请参阅 [OpenVINO 官方文档](https://docs.openvino.ai/)。

## 常见问题

### 如何将 Ultralytics YOLO11 模型导出为 OpenVINO 格式？

要将 Ultralytics YOLO11 模型导出为 OpenVINO 格式，请按照以下步骤操作：

```python
from ultralytics import YOLO

# 加载 YOLO11 模型
model = YOLO("yolo11n.pt")

# 导出为 OpenVINO 格式
model.export(format="openvino")  # 创建 'yolo11n_openvino_model/'
```

### 使用 OpenVINO 进行 YOLO11 推理有什么优势？

OpenVINO 为 YOLO11 推理提供了几个关键优势：

- 在 Intel CPU 上高达 3 倍的加速
- 支持 Intel GPU 和 NPU 硬件加速
- INT8 和 FP16 量化支持
- 跨 Intel 硬件的异构执行
- 模型优化和压缩工具

### OpenVINO 支持哪些 Intel 硬件？

OpenVINO 支持广泛的 Intel 硬件：

- **CPU**：Intel Core、Xeon 处理器
- **GPU**：Intel 集成和独立 GPU（Arc 系列）
- **NPU**：Intel AI Boost NPU（Core Ultra 系列）
- **VPU**：Intel 视觉处理单元

### 如何在 Intel GPU 或 NPU 上运行 YOLO11 推理？

使用 device 参数指定目标硬件：

```python
from ultralytics import YOLO

model = YOLO("yolo11n_openvino_model/")

# 在 Intel GPU 上运行
results = model("image.jpg", device="intel:gpu")

# 在 Intel NPU 上运行
results = model("image.jpg", device="intel:npu")

# 在 Intel CPU 上运行
results = model("image.jpg", device="intel:cpu")
```
