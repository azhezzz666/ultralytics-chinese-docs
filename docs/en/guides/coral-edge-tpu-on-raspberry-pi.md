---
comments: true
description: 学习如何使用 Coral Edge TPU 和 Ultralytics YOLO11 提升树莓派的机器学习性能。按照我们详细的设置和安装指南操作。
keywords: Coral Edge TPU, 树莓派, YOLO11, Ultralytics, TensorFlow Lite, 机器学习推理, 机器学习, AI, 安装指南, 设置教程
---

# 在树莓派上使用 Coral Edge TPU 和 Ultralytics YOLO11 🚀

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/edge-tpu-usb-accelerator-and-pi.avif" alt="带有 USB Edge TPU 加速器的树莓派单板计算机">
</p>

## 什么是 Coral Edge TPU？

Coral Edge TPU 是一款紧凑型设备，可为您的系统添加 Edge TPU 协处理器。它能够为 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite 模型实现低功耗、高性能的机器学习推理。在 [Coral Edge TPU 主页](https://developers.google.com/coral)了解更多信息。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/w4yHORvDBw0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Google Coral Edge TPU 在树莓派上运行推理
</p>

## 使用 Coral Edge TPU 提升树莓派模型性能

许多人希望在嵌入式或移动设备（如树莓派）上运行他们的模型，因为这些设备非常节能，可用于许多不同的应用。然而，即使使用 [ONNX](../integrations/onnx.md) 或 [OpenVINO](../integrations/openvino.md) 等格式，这些设备上的推理性能通常也很差。Coral Edge TPU 是解决这个问题的绝佳方案，因为它可以与树莓派一起使用，大大加速推理性能。

## 在树莓派上使用 TensorFlow Lite 的 Edge TPU（新版）⭐

Coral 关于如何在树莓派上使用 Edge TPU 的[现有指南](https://gweb-coral-full.uc.r.appspot.com/docs/accelerator/get-started/)已经过时，当前的 Coral Edge TPU 运行时构建版本不再与当前的 TensorFlow Lite 运行时版本兼容。此外，Google 似乎已经完全放弃了 Coral 项目，2021 年至 2025 年期间没有任何更新。本指南将向您展示如何在树莓派单板计算机（SBC）上使用最新版本的 TensorFlow Lite 运行时和更新的 Coral Edge TPU 运行时来使 Edge TPU 正常工作。

## 前提条件

- [树莓派 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)（建议 2GB 或更多内存）或[树莓派 5](https://www.raspberrypi.com/products/raspberry-pi-5/)（推荐）
- [树莓派操作系统](https://www.raspberrypi.com/software/) Bullseye/Bookworm（64 位）带桌面（推荐）
- [Coral USB 加速器](https://developers.google.com/coral)
- 用于导出 Ultralytics [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型的非 ARM 平台

## 安装步骤

本指南假设您已经安装了可用的树莓派操作系统，并已安装 `ultralytics` 及所有依赖项。要安装 `ultralytics`，请访问[快速入门指南](../quickstart.md)进行设置，然后再继续此处。

### 安装 Edge TPU 运行时

首先，我们需要安装 Edge TPU 运行时。有许多不同的版本可用，因此您需要为您的操作系统选择正确的版本。
高频版本以更高的时钟速度运行 Edge TPU，从而提高性能。但是，这可能导致 Edge TPU 热节流，因此建议有某种散热机制。

| 树莓派操作系统 | 高频模式 | 要下载的版本                        |
| --------------- | :-----------------: | ------------------------------------------ |
| Bullseye 32位  |         否          | `libedgetpu1-std_ ... .bullseye_armhf.deb` |
| Bullseye 64位  |         否          | `libedgetpu1-std_ ... .bullseye_arm64.deb` |
| Bullseye 32位  |         是         | `libedgetpu1-max_ ... .bullseye_armhf.deb` |
| Bullseye 64位  |         是         | `libedgetpu1-max_ ... .bullseye_arm64.deb` |
| Bookworm 32位  |         否          | `libedgetpu1-std_ ... .bookworm_armhf.deb` |
| Bookworm 64位  |         否          | `libedgetpu1-std_ ... .bookworm_arm64.deb` |
| Bookworm 32位  |         是         | `libedgetpu1-max_ ... .bookworm_armhf.deb` |
| Bookworm 64位  |         是         | `libedgetpu1-max_ ... .bookworm_arm64.deb` |

[从这里下载最新版本](https://github.com/feranick/libedgetpu/releases)。

下载文件后，您可以使用以下命令安装它：

```bash
sudo dpkg -i path/to/package.deb
```

安装运行时后，将 Coral Edge TPU 插入树莓派的 USB 3.0 端口，以便新的 `udev` 规则生效。

???+ warning "重要"

    如果您已经安装了 Coral Edge TPU 运行时，请使用以下命令卸载它。

    ```bash
    # 如果您安装了标准版本
    sudo apt remove libedgetpu1-std

    # 如果您安装了高频版本
    sudo apt remove libedgetpu1-max
    ```

## 导出到 Edge TPU

要使用 Edge TPU，您需要将模型转换为兼容格式。建议在 Google Colab、x86_64 Linux 机器上运行导出，使用官方 [Ultralytics Docker 容器](docker-quickstart.md)，或使用 [Ultralytics HUB](../hub/quickstart.md)，因为 Edge TPU 编译器在 ARM 上不可用。有关可用参数，请参阅[导出模式](../modes/export.md)。

!!! example "导出模型"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/model.pt")  # 加载官方模型或自定义模型

        # 导出模型
        model.export(format="edgetpu")
        ```

    === "命令行"

        ```bash
        yolo export model=path/to/model.pt format=edgetpu # 导出官方模型或自定义模型
        ```

导出的模型将保存在 `<model_name>_saved_model/` 文件夹中，文件名为 `<model_name>_full_integer_quant_edgetpu.tflite`。确保文件名以 `_edgetpu.tflite` 后缀结尾；否则，Ultralytics 将无法检测到您正在使用 Edge TPU 模型。

## 运行模型

在实际运行模型之前，您需要安装正确的库。

如果您已经安装了 TensorFlow，请使用以下命令卸载它：

```bash
pip uninstall tensorflow tensorflow-aarch64
```

然后安装或更新 `tflite-runtime`：

```bash
pip install -U tflite-runtime
```

现在您可以使用以下代码运行推理：

!!! example "运行模型"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/<model_name>_full_integer_quant_edgetpu.tflite")  # 加载官方模型或自定义模型

        # 运行预测
        model.predict("path/to/source.png")
        ```

    === "命令行"

        ```bash
        yolo predict model=path/to/MODEL_NAME_full_integer_quant_edgetpu.tflite source=path/to/source.png # 加载官方模型或自定义模型
        ```

在[预测](../modes/predict.md)页面查找完整预测模式详细信息的综合信息。

!!! note "使用多个 Edge TPU 进行推理"

    如果您有多个 Edge TPU，可以使用以下代码选择特定的 TPU。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/<model_name>_full_integer_quant_edgetpu.tflite")  # 加载官方模型或自定义模型

        # 运行预测
        model.predict("path/to/source.png")  # 推理默认使用第一个 TPU

        model.predict("path/to/source.png", device="tpu:0")  # 选择第一个 TPU

        model.predict("path/to/source.png", device="tpu:1")  # 选择第二个 TPU
        ```

## 基准测试

!!! tip "基准测试"

    使用树莓派操作系统 Bookworm 64 位和 USB Coral Edge TPU 进行测试。

    !!! note

        显示的是推理时间，不包括预处理/后处理。

    === "树莓派 4B 2GB"

        | 图像尺寸 | 模型   | 标准推理时间 (ms) | 高频推理时间 (ms) |
        |------------|---------|------------------------------|------------------------------------|
        | 320        | YOLOv8n | 32.2                         | 26.7                               |
        | 320        | YOLOv8s | 47.1                         | 39.8                               |
        | 512        | YOLOv8n | 73.5                         | 60.7                               |
        | 512        | YOLOv8s | 149.6                        | 125.3                              |

    === "树莓派 5 8GB"

        | 图像尺寸 | 模型   | 标准推理时间 (ms) | 高频推理时间 (ms) |
        |------------|---------|------------------------------|------------------------------------|
        | 320        | YOLOv8n | 22.2                         | 16.7                               |
        | 320        | YOLOv8s | 40.1                         | 32.2                               |
        | 512        | YOLOv8n | 53.5                         | 41.6                               |
        | 512        | YOLOv8s | 132.0                        | 103.3                              |

    平均而言：

    - 树莓派 5 在标准模式下比树莓派 4B 快 22%。
    - 树莓派 5 在高频模式下比树莓派 4B 快 30.2%。
    - 高频模式比标准模式快 28.4%。

## 常见问题

### 什么是 Coral Edge TPU，它如何增强树莓派与 Ultralytics YOLO11 的性能？

Coral Edge TPU 是一款紧凑型设备，旨在为您的系统添加 Edge TPU 协处理器。该协处理器能够实现低功耗、高性能的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)推理，特别针对 TensorFlow Lite 模型进行了优化。当与树莓派一起使用时，Edge TPU 可以加速机器学习模型推理，显著提升性能，特别是对于 Ultralytics YOLO11 模型。您可以在其[主页](https://developers.google.com/coral)上了解更多关于 Coral Edge TPU 的信息。

### 如何在树莓派上安装 Coral Edge TPU 运行时？

要在树莓派上安装 Coral Edge TPU 运行时，请从[此链接](https://github.com/feranick/libedgetpu/releases)下载适合您树莓派操作系统版本的 `.deb` 包。下载后，使用以下命令安装：

```bash
sudo dpkg -i path/to/package.deb
```

确保按照[安装步骤](#安装步骤)部分中概述的步骤卸载任何以前的 Coral Edge TPU 运行时版本。

### 我可以将 Ultralytics YOLO11 模型导出为与 Coral Edge TPU 兼容的格式吗？

是的，您可以将 Ultralytics YOLO11 模型导出为与 Coral Edge TPU 兼容的格式。建议在 Google Colab、x86_64 Linux 机器上或使用 [Ultralytics Docker 容器](docker-quickstart.md)执行导出。您也可以使用 [Ultralytics HUB](../hub/quickstart.md) 进行导出。以下是使用 Python 和命令行导出模型的方法：

!!! example "导出模型"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/model.pt")  # 加载官方模型或自定义模型

        # 导出模型
        model.export(format="edgetpu")
        ```

    === "命令行"

        ```bash
        yolo export model=path/to/model.pt format=edgetpu # 导出官方模型或自定义模型
        ```

有关更多信息，请参阅[导出模式](../modes/export.md)文档。

### 如果我的树莓派上已经安装了 TensorFlow，但我想使用 tflite-runtime，该怎么办？

如果您的树莓派上安装了 TensorFlow 并需要切换到 `tflite-runtime`，您需要先卸载 TensorFlow：

```bash
pip uninstall tensorflow tensorflow-aarch64
```

然后，使用以下命令安装或更新 `tflite-runtime`：

```bash
pip install -U tflite-runtime
```

有关详细说明，请参阅[运行模型](#运行模型)部分。

### 如何使用 Coral Edge TPU 在树莓派上运行导出的 YOLO11 模型进行推理？

将 YOLO11 模型导出为 Edge TPU 兼容格式后，您可以使用以下代码片段运行推理：

!!! example "运行模型"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/edgetpu_model.tflite")  # 加载官方模型或自定义模型

        # 运行预测
        model.predict("path/to/source.png")
        ```

    === "命令行"

        ```bash
        yolo predict model=path/to/edgetpu_model.tflite source=path/to/source.png # 加载官方模型或自定义模型
        ```

有关完整预测模式功能的详细信息，请参阅[预测页面](../modes/predict.md)。
