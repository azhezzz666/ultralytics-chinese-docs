---
comments: true
description: 将 YOLO11 模型导出为 ExecuTorch 格式，以便在移动和边缘设备上进行高效的设备端推理。针对 iOS、Android 和嵌入式系统优化您的 AI 模型。
keywords: Ultralytics, YOLO11, ExecuTorch, 模型导出, PyTorch, 边缘 AI, 移动部署, 设备端推理, XNNPACK, 嵌入式系统
---

# 使用 ExecuTorch 在移动和边缘设备上部署 YOLO11

在智能手机、平板电脑和嵌入式系统等边缘设备上部署计算机视觉模型需要一个优化的运行时，以平衡性能和资源限制。ExecuTorch 是 PyTorch 的边缘计算解决方案，可为 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 模型实现高效的设备端推理。

本指南概述了如何将 Ultralytics YOLO 模型导出为 ExecuTorch 格式，使您能够在移动和边缘设备上以优化的性能部署模型。

## 为什么导出为 ExecuTorch？

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/assets/releases/download/v0.0.0/executorch-pipeline.avif" alt="PyTorch ExecuTorch 概述">
</p>

[ExecuTorch](https://docs.pytorch.org/executorch/) 是 PyTorch 的端到端解决方案，用于在移动和边缘设备上实现设备端推理能力。ExecuTorch 以可移植性和高效性为目标构建，可用于在各种计算平台上运行 PyTorch 程序。

## ExecuTorch 的主要特性

ExecuTorch 为在边缘设备上部署 Ultralytics YOLO 模型提供了几个强大的功能：

- **可移植模型格式**：ExecuTorch 使用 `.pte`（PyTorch ExecuTorch）格式，该格式针对资源受限设备上的大小和加载速度进行了优化。

- **XNNPACK 后端**：与 XNNPACK 的默认集成在移动 CPU 上提供高度优化的推理，无需专用硬件即可提供出色的性能。

- **量化支持**：内置支持量化技术，以减小模型大小并提高推理速度，同时保持准确性。

- **内存效率**：优化的内存管理减少了运行时内存占用，使其适合 RAM 有限的设备。

- **模型元数据**：导出的模型在单独的 YAML 文件中包含元数据（图像大小、类名等），便于集成。

## ExecuTorch 的部署选项

ExecuTorch 模型可以部署在各种边缘和移动平台上：

- **移动应用程序**：在 iOS 和 Android 应用程序上以原生性能部署，在移动应用中实现实时目标检测。

- **嵌入式系统**：在 Raspberry Pi、NVIDIA Jetson 和其他基于 ARM 的系统等嵌入式 Linux 设备上以优化的性能运行。

- **边缘 AI 设备**：使用自定义委托在专用边缘 AI 硬件上部署以加速推理。

- **物联网设备**：集成到物联网设备中进行设备端推理，无需云连接。

## 将 Ultralytics YOLO11 模型导出为 ExecuTorch

将 Ultralytics YOLO11 模型转换为 ExecuTorch 格式可在移动和边缘设备上实现高效部署。

### 安装

ExecuTorch 导出需要 Python 3.10 或更高版本以及特定的依赖项：

!!! tip "安装"

    === "CLI"

        ```bash
        # 安装 Ultralytics 包
        pip install ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [YOLO11 安装指南](../quickstart.md)。在安装 YOLO11 所需的包时，如果遇到任何困难，请查阅我们的[常见问题指南](../guides/yolo-common-issues.md)以获取解决方案和提示。

### 使用

将 YOLO11 模型导出为 ExecuTorch 非常简单：

!!! example "使用"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 ExecuTorch 格式
        model.export(format="executorch")  # 创建 'yolo11n_executorch_model' 目录

        executorch_model = YOLO("yolo11n_executorch_model")

        results = executorch_model.predict("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 ExecuTorch 格式
        yolo export model=yolo11n.pt format=executorch # 创建 'yolo11n_executorch_model' 目录

        # 使用导出的模型运行推理
        yolo predict model=yolo11n_executorch_model source=https://ultralytics.com/images/bus.jpg
        ```

    ExecuTorch 导出生成一个包含 `.pte` 文件和元数据的目录。在您的移动或嵌入式应用程序中使用 ExecuTorch 运行时加载 `.pte` 模型并执行推理。

### 导出参数

导出为 ExecuTorch 格式时，您可以指定以下参数：

| 参数     | 类型            | 默认值  | 描述                               |
| -------- | --------------- | ------- | ---------------------------------- |
| `imgsz`  | `int` 或 `list` | `640`   | 模型输入的图像尺寸（高度、宽度）   |
| `device` | `str`           | `'cpu'` | 用于导出的设备（`'cpu'`）          |

### 输出结构

ExecuTorch 导出创建一个包含模型和元数据的目录：

```text
yolo11n_executorch_model/
├── yolo11n.pte              # ExecuTorch 模型文件
└── metadata.yaml            # 模型元数据（类别、图像大小等）
```


## 使用导出的 ExecuTorch 模型

导出模型后，您需要使用 ExecuTorch 运行时将其集成到目标应用程序中。

### 移动集成

对于移动应用程序（iOS/Android），您需要：

1. **添加 ExecuTorch 运行时**：在您的移动项目中包含 ExecuTorch 运行时库
2. **加载模型**：在您的应用程序中加载 `.pte` 文件
3. **运行推理**：处理图像并获取预测

iOS 集成示例（Objective-C/C++）：

```objc
// iOS 使用 C++ API 进行模型加载和推理
// 有关完整示例，请参阅 https://pytorch.org/executorch/stable/using-executorch-ios.html

#include <executorch/extension/module/module.h>

using namespace ::executorch::extension;

// 加载模型
Module module("/path/to/yolo11n.pte");

// 创建输入张量
float input[1 * 3 * 640 * 640];
auto tensor = from_blob(input, {1, 3, 640, 640});

// 运行推理
const auto result = module.forward(tensor);
```

Android 集成示例（Kotlin）：

```kotlin
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

// 加载模型
val module = Module.load("/path/to/yolo11n.pte")

// 准备输入张量
val inputTensor = Tensor.fromBlob(floatData, longArrayOf(1, 3, 640, 640))
val inputEValue = EValue.from(inputTensor)

// 运行推理
val outputs = module.forward(inputEValue)
val scores = outputs[0].toTensor().dataAsFloatArray
```

### 嵌入式 Linux

对于嵌入式 Linux 系统，使用 ExecuTorch C++ API：

```cpp
#include <executorch/extension/module/module.h>

// 加载模型
auto module = torch::executor::Module("yolo11n.pte");

// 准备输入
std::vector<float> input_data = preprocessImage(image);
auto input_tensor = torch::executor::Tensor(input_data, {1, 3, 640, 640});

// 运行推理
auto outputs = module.forward({input_tensor});
```

有关将 ExecuTorch 集成到您的应用程序的更多详细信息，请访问 [ExecuTorch 文档](https://docs.pytorch.org/executorch/)。

## 性能优化

### 模型大小优化

要减小部署的模型大小：

- **使用较小的模型**：从 YOLO11n（nano）开始以获得最小的占用空间
- **降低输入分辨率**：使用较小的图像尺寸（例如 `imgsz=320` 或 `imgsz=416`）
- **量化**：应用量化技术（在未来的 ExecuTorch 版本中支持）

### 推理速度优化

为了更快的推理：

- **XNNPACK 后端**：默认的 XNNPACK 后端提供优化的 CPU 推理
- **硬件加速**：使用平台特定的委托（例如 iOS 的 CoreML）
- **批处理**：尽可能处理多张图像

## 基准测试

Ultralytics 团队对 YOLO11 模型进行了基准测试，比较了 PyTorch 和 ExecuTorch 之间的速度和准确性。

!!! tip "性能"

    === "Raspberry Pi 5"

        | 模型    | 格式        | 状态 | 大小 (MB) | metrics/mAP50-95(B) | 推理时间 (ms/im) |
        | ------- | ----------- | ---- | --------- | ------------------- | ---------------- |
        | YOLO11n | PyTorch     | ✅   | 5.4       | 0.5060              | 337.67           |
        | YOLO11n | ExecuTorch  | ✅   | 11        | 0.5080              | 167.28           |
        | YOLO11s | PyTorch     | ✅   | 19        | 0.5770              | 928.80           |
        | YOLO11s | ExecuTorch  | ✅   | 37        | 0.5780              | 388.31           |

    === "更多设备即将推出！"

    !!! note

        推理时间不包括预处理/后处理。

## 故障排除

### 常见问题

**问题**：`Python 版本错误`

**解决方案**：ExecuTorch 需要 Python 3.10 或更高版本。升级您的 Python 安装：

```bash
# 使用 conda
conda create -n executorch python=3.10
conda activate executorch
```

**问题**：`首次运行时导出失败`

**解决方案**：ExecuTorch 可能需要在首次使用时下载和编译组件。确保您已：

```bash
pip install --upgrade executorch
```

**问题**：`ExecuTorch 模块的导入错误`

**解决方案**：确保 ExecuTorch 已正确安装：

```bash
pip install executorch --force-reinstall
```

如需更多故障排除帮助，请访问 [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues) 或 [ExecuTorch 文档](https://docs.pytorch.org/executorch/stable/getting-started-setup.html)。

## 总结

将 YOLO11 模型导出为 ExecuTorch 格式可在移动和边缘设备上实现高效部署。通过 PyTorch 原生集成、跨平台支持和优化的性能，ExecuTorch 是边缘 AI 应用的绝佳选择。

主要要点：

- ExecuTorch 提供具有出色性能的 PyTorch 原生边缘部署
- 使用 `format='executorch'` 参数可轻松导出
- 模型通过 XNNPACK 后端针对移动 CPU 进行优化
- 支持 iOS、Android 和嵌入式 Linux 平台
- 需要 Python 3.10+ 和 FlatBuffers 编译器

## 常见问题

### 如何将 YOLO11 模型导出为 ExecuTorch 格式？

使用 Python 或 CLI 将 YOLO11 模型导出为 ExecuTorch：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="executorch")
```

或

```bash
yolo export model=yolo11n.pt format=executorch
```

### ExecuTorch 导出的系统要求是什么？

ExecuTorch 导出需要：

- Python 3.10 或更高版本
- `executorch` 包（通过 `pip install executorch` 安装）
- PyTorch（随 ultralytics 自动安装）

注意：在首次导出期间，ExecuTorch 将自动下载和编译必要的组件，包括 FlatBuffers 编译器。

### 我可以直接在 Python 中使用 ExecuTorch 模型运行推理吗？

ExecuTorch 模型（`.pte` 文件）设计用于使用 ExecuTorch 运行时在移动和边缘设备上部署。它们不能直接使用 `YOLO()` 在 Python 中加载进行推理。您需要使用 ExecuTorch 运行时库将它们集成到目标应用程序中。

### ExecuTorch 支持哪些平台？

ExecuTorch 支持：

- **移动**：iOS 和 Android
- **嵌入式 Linux**：Raspberry Pi、NVIDIA Jetson 和其他 ARM 设备
- **桌面**：Linux、macOS 和 Windows（用于开发）

### ExecuTorch 与 TFLite 在移动部署方面相比如何？

ExecuTorch 和 TFLite 都非常适合移动部署：

- **ExecuTorch**：更好的 PyTorch 集成、原生 PyTorch 工作流程、不断发展的生态系统
- **TFLite**：更成熟、更广泛的硬件支持、更多部署示例

如果您已经在使用 PyTorch 并希望获得原生部署路径，请选择 ExecuTorch。如果需要最大兼容性和成熟的工具，请选择 TFLite。

### 我可以将 ExecuTorch 模型与 GPU 加速一起使用吗？

可以！ExecuTorch 通过各种后端支持硬件加速：

- **移动 GPU**：通过 Vulkan、Metal 或 OpenCL 委托
- **NPU/DSP**：通过平台特定的委托
- **默认**：XNNPACK 用于优化的 CPU 推理

有关后端特定设置，请参阅 [ExecuTorch 文档](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html)。
