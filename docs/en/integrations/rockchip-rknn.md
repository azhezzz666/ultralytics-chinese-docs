---
comments: true
description: 学习如何将 YOLO11 模型导出为 RKNN 格式，以便在 Rockchip 平台上高效部署并获得增强的性能。
keywords: YOLO11, RKNN, 模型导出, Ultralytics, Rockchip, 机器学习, 模型部署, 计算机视觉, 深度学习, 边缘 AI, NPU, 嵌入式设备
---

# Ultralytics YOLO11 模型的 Rockchip RKNN 导出

在嵌入式设备上部署计算机视觉模型时，特别是那些由 Rockchip 处理器驱动的设备，拥有兼容的模型格式至关重要。将 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 模型导出为 RKNN 格式可确保在 Rockchip 硬件上获得优化的性能和兼容性。本指南将引导您将 YOLO11 模型转换为 RKNN 格式，从而在 Rockchip 平台上实现高效部署。

<p align="center">
  <img width="50%" src="https://github.com/ultralytics/assets/releases/download/v0.0.0/rockchip-rknn.avif" alt="RKNN">
</p>

!!! note

    本指南已在基于 Rockchip RK3588 的 [Radxa Rock 5B](https://radxa.com/products/rock5/5b/) 和基于 Rockchip RK3566 的 [Radxa Zero 3W](https://radxa.com/products/zeros/zero3w/) 上进行测试。预计可在其他支持 [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) 的 Rockchip 设备上运行，如 RK3576、RK3568、RK3562、RV1103、RV1106、RV1103B、RV1106B、RV1126B 和 RK2118。

## 什么是 Rockchip？

Rockchip 以提供多功能且节能的解决方案而闻名，设计先进的片上系统（SoC），为各种消费电子产品、工业应用和 AI 技术提供动力。凭借基于 ARM 的架构、内置神经处理单元（NPU）和高分辨率多媒体支持，Rockchip SoC 为平板电脑、智能电视、物联网系统和[边缘 AI 应用](https://www.ultralytics.com/blog/understanding-the-real-world-applications-of-edge-ai)等设备提供尖端性能。Radxa、ASUS、Pine64、Orange Pi、Odroid、Khadas 和 Banana Pi 等公司提供基于 Rockchip SoC 的各种产品，进一步扩展了其在不同市场的影响力。

## RKNN 工具包

[RKNN 工具包](https://github.com/airockchip/rknn-toolkit2)是 Rockchip 提供的一套工具和库，用于促进深度学习模型在其硬件平台上的部署。RKNN（Rockchip Neural Network）是这些工具使用的专有格式。RKNN 模型旨在充分利用 Rockchip NPU（神经处理单元）提供的硬件加速，确保在 RK3588、RK3566、RV1103、RV1106 等 Rockchip 驱动的系统上实现高性能 AI 任务。

## RKNN 模型的主要特性

RKNN 模型为在 Rockchip 平台上部署提供了多项优势：

- **针对 NPU 优化**：RKNN 模型专门针对 Rockchip 的 NPU 进行优化，确保最大性能和效率。
- **低延迟**：RKNN 格式最小化推理延迟，这对于边缘设备上的实时应用至关重要。
- **平台特定定制**：RKNN 模型可以针对特定的 Rockchip 平台进行定制，从而更好地利用硬件资源。
- **能效**：通过利用专用 NPU 硬件，RKNN 模型比基于 CPU 或 GPU 的处理消耗更少的电力，延长便携设备的电池寿命。

## 在 Rockchip 硬件上刷写操作系统

获得 Rockchip 设备后的第一步是刷写操作系统，以便硬件可以启动到工作环境。在本指南中，我们将指向我们测试的两个设备（Radxa Rock 5B 和 Radxa Zero 3W）的入门指南。

- [Radxa Rock 5B 入门指南](https://docs.radxa.com/en/rock5/rock5b)
- [Radxa Zero 3W 入门指南](https://docs.radxa.com/en/zero/zero3)

## 导出到 RKNN：转换您的 YOLO11 模型

将 Ultralytics YOLO11 模型导出为 RKNN 格式并使用导出的模型运行推理。

!!! note

    请确保使用基于 X86 的 Linux PC 导出模型到 RKNN，因为不支持在 Rockchip 设备（ARM64）上导出。

### 安装

要安装所需的包，请运行：

!!! Tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 所需的包
        pip install ultralytics
        ```

有关安装过程的详细说明和最佳实践，请查看我们的 [Ultralytics 安装指南](../quickstart.md)。在为 YOLO11 安装所需包时，如果遇到任何困难，请参阅我们的[常见问题指南](../guides/yolo-common-issues.md)获取解决方案和提示。

### 使用方法

!!! note

    目前仅支持检测模型的导出。未来将支持更多模型。

!!! example "使用方法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 RKNN 格式
        # 'name' 可以是 rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118 之一
        model.export(format="rknn", name="rk3588")  # 创建 '/yolo11n_rknn_model'
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 RKNN 格式
        # 'name' 可以是 rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118 之一
        yolo export model=yolo11n.pt format=rknn name=rk3588 # 创建 '/yolo11n_rknn_model'
        ```

### 导出参数

| 参数       | 类型             | 默认值     | 描述                                                                                                                             |
| -------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'rknn'`   | 导出模型的目标格式，定义与各种部署环境的兼容性。                                      |
| `imgsz`  | `int` 或 `tuple` | `640`      | 模型输入所需的图像尺寸。可以是整数（正方形图像）或元组 `(height, width)` 指定特定尺寸。       |
| `batch`  | `int`            | `1`        | 指定导出模型的批量推理大小或导出模型在 `predict` 模式下并发处理的最大图像数量。 |
| `name`   | `str`            | `'rk3588'` | 指定 Rockchip 型号（rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118）                         |
| `device` | `str`            | `None`     | 指定导出设备：GPU (`device=0`)，CPU (`device=cpu`)。                                                               |

!!! tip

    导出到 RKNN 时请确保使用 x86 Linux 机器。

有关导出过程的更多详情，请访问 [Ultralytics 导出文档页面](../modes/export.md)。

## 部署导出的 YOLO11 RKNN 模型

成功将 Ultralytics YOLO11 模型导出为 RKNN 格式后，下一步是在 Rockchip 设备上部署这些模型。

### 安装

要安装所需的包，请运行：

!!! Tip "安装"

    === "CLI"

        ```bash
        # 安装 YOLO11 所需的包
        pip install ultralytics
        ```

### 使用方法

!!! example "使用方法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载导出的 RKNN 模型
        rknn_model = YOLO("./yolo11n_rknn_model")

        # 运行推理
        results = rknn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 使用导出的模型运行推理
        yolo predict model='./yolo11n_rknn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    如果您遇到日志消息指示 RKNN 运行时版本与 RKNN 工具包版本不匹配且推理失败，请将 `/usr/lib/librknnrt.so` 替换为官方 [librknnrt.so 文件](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so)。

    ![RKNN 导出截图](https://github.com/ultralytics/assets/releases/download/v0.0.0/rknn-npu-log.avif)

## 实际应用

搭载 YOLO11 RKNN 模型的 Rockchip 设备可用于各种应用：

- **智能监控**：部署高效的目标检测系统用于安全监控，功耗低。
- **工业自动化**：直接在嵌入式设备上实现质量控制和缺陷检测。
- **零售分析**：实时跟踪客户行为和库存管理，无需依赖云端。
- **智慧农业**：使用[农业计算机视觉](https://www.ultralytics.com/solutions/ai-in-agriculture)监测作物健康和检测害虫。
- **自主机器人**：在资源受限的平台上实现基于视觉的导航和障碍物检测。

## 基准测试

以下 YOLO11 基准测试由 Ultralytics 团队在基于 Rockchip RK3588 的 Radxa Rock 5B 上使用 `rknn` 模型格式运行，测量速度和准确率。

!!! tip "性能"

    | 模型    | 格式   | 状态 | 大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
    | ------- | ------ | ------ | --------- | ----------- | ---------------------- |
    | YOLO11n | `rknn` | ✅     | 7.4       | 0.505       | 71.5                   |
    | YOLO11s | `rknn` | ✅     | 20.7      | 0.578       | 98.9                   |
    | YOLO11m | `rknn` | ✅     | 41.9      | 0.629       | 235.3                  |
    | YOLO11l | `rknn` | ✅     | 53.3      | 0.633       | 282.0                  |
    | YOLO11x | `rknn` | ✅     | 114.6     | 0.687       | 679.2                  |

    使用 `ultralytics 8.3.152` 进行基准测试

    !!! note

        上述基准测试的验证使用 COCO128 数据集完成。推理时间不包括预处理/后处理。

## 总结

在本指南中，您学习了如何将 Ultralytics YOLO11 模型导出为 RKNN 格式，以增强其在 Rockchip 平台上的部署。您还了解了 RKNN 工具包以及使用 RKNN 模型进行边缘 AI 应用的具体优势。

[Ultralytics YOLO11](https://www.ultralytics.com/blog/all-you-need-to-know-about-ultralytics-yolo11-and-its-applications) 与 Rockchip NPU 技术的结合为在嵌入式设备上运行高级计算机视觉任务提供了高效的解决方案。这种方法以最小的功耗和高性能实现实时[目标检测](https://www.ultralytics.com/blog/a-guide-to-deep-dive-into-object-detection-in-2025)和其他视觉 AI 应用。

有关使用的更多详情，请访问 [RKNN 官方文档](https://github.com/airockchip/rknn-toolkit2)。

此外，如果您想了解更多关于其他 Ultralytics YOLO11 集成的信息，请访问我们的[集成指南页面](../integrations/index.md)。您将在那里找到大量有用的资源和见解。

## 常见问题

### 如何将我的 Ultralytics YOLO 模型导出为 RKNN 格式？

您可以使用 Ultralytics Python 包中的 `export()` 方法或通过命令行界面（CLI）轻松将 Ultralytics YOLO 模型导出为 RKNN 格式。确保使用基于 x86 的 Linux PC 进行导出过程，因为不支持 ARM64 设备（如 Rockchip）进行此操作。您可以使用 `name` 参数指定目标 Rockchip 平台，如 `rk3588`、`rk3566` 等。此过程生成一个优化的 RKNN 模型，可在您的 Rockchip 设备上部署，利用其神经处理单元（NPU）进行加速推理。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载您的 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 导出为特定 Rockchip 平台的 RKNN 格式
        model.export(format="rknn", name="rk3588")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=rknn name=rk3588
        ```

### 在 Rockchip 设备上使用 RKNN 模型有什么好处？

RKNN 模型专门设计用于利用 Rockchip 神经处理单元（NPU）的硬件加速能力。与在相同硬件上运行 ONNX 或 TensorFlow Lite 等通用模型格式相比，这种优化可显著提高推理速度并降低延迟。使用 RKNN 模型可以更有效地利用设备资源，从而降低功耗并提高整体性能，这对于边缘设备上的实时应用尤为重要。通过将 Ultralytics YOLO 模型转换为 RKNN，您可以在 RK3588、RK3566 等 Rockchip SoC 驱动的设备上获得最佳性能。

### 我可以在 NVIDIA 或 Google 等其他制造商的设备上部署 RKNN 模型吗？

RKNN 模型专门针对 Rockchip 平台及其集成 NPU 进行优化。虽然您可以在技术上使用软件模拟在其他平台上运行 RKNN 模型，但您将无法从 Rockchip 设备提供的硬件加速中受益。为了在其他平台上获得最佳性能，建议将 Ultralytics YOLO 模型导出为专门为这些平台设计的格式，例如用于 NVIDIA GPU 的 TensorRT 或用于 Google Edge TPU 的 [TensorFlow Lite](https://docs.ultralytics.com/integrations/tflite/)。Ultralytics 支持导出到多种格式，确保与各种硬件加速器的兼容性。

### 哪些 Rockchip 平台支持 RKNN 模型部署？

Ultralytics YOLO 导出到 RKNN 格式支持多种 Rockchip 平台，包括流行的 RK3588、RK3576、RK3566、RK3568、RK3562、RV1103、RV1106、RV1103B、RV1106B 和 RK2118。这些平台常见于 Radxa、ASUS、Pine64、Orange Pi、Odroid、Khadas 和 Banana Pi 等制造商的设备中。这种广泛的支持确保您可以在各种 Rockchip 驱动的设备上部署优化的 RKNN 模型，从单板计算机到工业系统，充分利用其 AI 加速能力来增强计算机视觉应用的性能。

### RKNN 模型在 Rockchip 设备上与其他格式相比性能如何？

由于针对 Rockchip NPU 的优化，RKNN 模型在 Rockchip 设备上通常优于 ONNX 或 TensorFlow Lite 等其他格式。例如，在 Radxa Rock 5B（RK3588）上的基准测试显示，RKNN 格式的 [YOLO11n](https://www.ultralytics.com/blog/all-you-need-to-know-about-ultralytics-yolo11-and-its-applications) 实现了 99.5 ms/图像的推理时间，明显快于其他格式。这种性能优势在各种 YOLO11 模型尺寸中保持一致，如[基准测试部分](#基准测试)所示。通过利用专用 NPU 硬件，RKNN 模型最小化延迟并最大化吞吐量，使其成为 Rockchip 边缘设备上实时应用的理想选择。
