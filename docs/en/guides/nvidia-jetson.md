---
comments: true
description: 学习如何在 NVIDIA Jetson 设备上部署 Ultralytics YOLO11，包含详细指南。探索性能基准测试并最大化 AI 能力。
keywords: Ultralytics, YOLO11, NVIDIA Jetson, JetPack, AI 部署, 性能基准测试, 嵌入式系统, 深度学习, TensorRT, 计算机视觉
---

# 快速入门指南：NVIDIA Jetson 与 Ultralytics YOLO11

本综合指南提供了在 [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) 设备上部署 Ultralytics YOLO11 的详细演练。此外，它还展示了性能基准测试，以展示 YOLO11 在这些小巧而强大的设备上的能力。

!!! tip "新产品支持"

    我们已使用最新的 [NVIDIA Jetson AGX Thor 开发者套件](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor)更新了本指南，该套件提供高达 2070 FP4 TFLOPS 的 AI 计算能力和 128 GB 内存，功耗可在 40 W 到 130 W 之间配置。它提供比 NVIDIA Jetson AGX Orin 高 7.5 倍以上的 AI 计算能力，能效提高 3.5 倍，可无缝运行最流行的 AI 模型。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/BPYkGt3odNk"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 NVIDIA Jetson 设备上使用 Ultralytics YOLO11
</p>

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/nvidia-jetson-ecosystem.avif" alt="NVIDIA Jetson 生态系统">

!!! note

    本指南已在运行最新稳定 JetPack 版本 [JP7.0](https://developer.nvidia.com/embedded/jetpack/downloads) 的 [NVIDIA Jetson AGX Thor 开发者套件 (Jetson T5000)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor)、运行 JetPack 版本 [JP6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62) 的 [NVIDIA Jetson AGX Orin 开发者套件 (64GB)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin)、运行 JetPack 版本 [JP6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61) 的 [NVIDIA Jetson Orin Nano Super 开发者套件](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit)、基于 NVIDIA Jetson Orin NX 16GB 运行 JetPack 版本 [JP6.0](https://developer.nvidia.com/embedded/jetpack-sdk-60)/ JetPack 版本 [JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) 的 [Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html) 以及基于 NVIDIA Jetson Nano 4GB 运行 JetPack 版本 [JP4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461) 的 [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html) 上进行了测试。预计可在所有 NVIDIA Jetson 硬件产品线上运行，包括最新和旧版设备。

## 什么是 NVIDIA Jetson？

NVIDIA Jetson 是一系列嵌入式计算板，旨在将加速 AI（人工智能）计算带到边缘设备。这些紧凑而强大的设备基于 NVIDIA 的 GPU 架构构建，可以直接在设备上运行复杂的 AI 算法和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，而无需依赖[云计算](https://www.ultralytics.com/glossary/cloud-computing)资源。Jetson 板通常用于机器人、自动驾驶车辆、工业自动化以及其他需要以低延迟和高效率在本地执行 AI 推理的应用。此外，这些板基于 ARM64 架构，与传统 GPU 计算设备相比功耗更低。

## NVIDIA Jetson 系列比较

[NVIDIA Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) 是基于 NVIDIA Blackwell 架构的 NVIDIA Jetson 系列的最新迭代，与前几代相比，AI 性能大幅提升。下表比较了生态系统中的几款 Jetson 设备。

|                   | Jetson AGX Thor(T5000)                                           | Jetson AGX Orin 64GB                                              | Jetson Orin NX 16GB                                              | Jetson Orin Nano Super                                        | Jetson AGX Xavier                                           | Jetson Xavier NX                                              | Jetson Nano                                   |
| ----------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| AI 性能    | 2070 TFLOPS                                                      | 275 TOPS                                                          | 100 TOPS                                                         | 67 TOPS                                                       | 32 TOPS                                                     | 21 TOPS                                                       | 472 GFLOPS                                    |
| GPU               | 2560 核 NVIDIA Blackwell 架构 GPU，96 个 Tensor Core | 2048 核 NVIDIA Ampere 架构 GPU，64 个 Tensor Core     | 1024 核 NVIDIA Ampere 架构 GPU，32 个 Tensor Core    | 1024 核 NVIDIA Ampere 架构 GPU，32 个 Tensor Core | 512 核 NVIDIA Volta 架构 GPU，64 个 Tensor Core | 384 核 NVIDIA Volta™ 架构 GPU，48 个 Tensor Core | 128 核 NVIDIA Maxwell™ 架构 GPU    |
| GPU 最大频率 | 1.57 GHz                                                         | 1.3 GHz                                                           | 918 MHz                                                          | 1020 MHz                                                      | 1377 MHz                                                    | 1100 MHz                                                      | 921MHz                                        |
| CPU               | 14 核 Arm® Neoverse®-V3AE 64 位 CPU 1MB L2 + 16MB L3        | 12 核 NVIDIA Arm® Cortex A78AE v8.2 64 位 CPU 3MB L2 + 6MB L3 | 8 核 NVIDIA Arm® Cortex A78AE v8.2 64 位 CPU 2MB L2 + 4MB L3 | 6 核 Arm® Cortex®-A78AE v8.2 64 位 CPU 1.5MB L2 + 4MB L3 | 8 核 NVIDIA Carmel Arm®v8.2 64 位 CPU 8MB L2 + 4MB L3   | 6 核 NVIDIA Carmel Arm®v8.2 64 位 CPU 6MB L2 + 4MB L3     | 四核 Arm® Cortex®-A57 MPCore 处理器 |
| CPU 最大频率 | 2.6 GHz                                                          | 2.2 GHz                                                           | 2.0 GHz                                                          | 1.7 GHz                                                       | 2.2 GHz                                                     | 1.9 GHz                                                       | 1.43GHz                                       |
| 内存            | 128GB 256 位 LPDDR5X 273GB/s                                    | 64GB 256 位 LPDDR5 204.8GB/s                                     | 16GB 128 位 LPDDR5 102.4GB/s                                    | 8GB 128 位 LPDDR5 102 GB/s                                   | 32GB 256 位 LPDDR4x 136.5GB/s                              | 8GB 128 位 LPDDR4x 59.7GB/s                                  | 4GB 64 位 LPDDR4 25.6GB/s                    |

有关更详细的比较表，请访问[官方 NVIDIA Jetson 页面](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems)的**比较规格**部分。

## 什么是 NVIDIA JetPack？

为 Jetson 模块提供支持的 [NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack) 是最全面的解决方案，提供完整的开发环境，用于构建端到端加速 AI 应用并缩短上市时间。JetPack 包括带有引导加载程序、Linux 内核、Ubuntu 桌面环境的 Jetson Linux，以及用于加速 GPU 计算、多媒体、图形和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)的完整库集。它还包括用于主机计算机和开发套件的示例、文档和开发工具，并支持更高级别的 SDK，如用于流视频分析的 [DeepStream](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson/)、用于机器人的 Isaac 和用于对话式 AI 的 Riva。

## 将 JetPack 刷入 NVIDIA Jetson

获得 NVIDIA Jetson 设备后的第一步是将 NVIDIA JetPack 刷入设备。有几种不同的方法可以刷入 NVIDIA Jetson 设备。

1. 如果您拥有官方 NVIDIA 开发套件（如 Jetson AGX Thor 开发者套件），您可以[下载镜像并准备可启动 USB 盘以将 JetPack 刷入随附的 SSD](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html)。
2. 如果您拥有官方 NVIDIA 开发套件（如 Jetson Orin Nano 开发者套件），您可以[下载镜像并准备带有 JetPack 的 SD 卡以启动设备](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)。
3. 如果您拥有任何其他 NVIDIA 开发套件，您可以[使用 SDK Manager 将 JetPack 刷入设备](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)。
4. 如果您拥有 Seeed Studio reComputer J4012 设备，您可以[将 JetPack 刷入随附的 SSD](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/)；如果您拥有 Seeed Studio reComputer J1020 v2 设备，您可以[将 JetPack 刷入 eMMC/SSD](https://wiki.seeedstudio.com/reComputer_J2021_J202_Flash_Jetpack/)。
5. 如果您拥有任何其他由 NVIDIA Jetson 模块驱动的第三方设备，建议按照[命令行刷入](https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/IN/QuickStart.html)进行操作。

!!! note

    对于上述方法 1、4 和 5，刷入系统并启动设备后，请在设备终端输入"sudo apt update && sudo apt install nvidia-jetpack -y"以安装所需的所有剩余 JetPack 组件。

## 基于 Jetson 设备的 JetPack 支持

下表突出显示了不同 NVIDIA Jetson 设备支持的 NVIDIA JetPack 版本。

|                   | JetPack 4 | JetPack 5 | JetPack 6 | JetPack 7 |
| ----------------- | --------- | --------- | --------- | --------- |
| Jetson Nano       | ✅        | ❌        | ❌        | ❌        |
| Jetson TX2        | ✅        | ❌        | ❌        | ❌        |
| Jetson Xavier NX  | ✅        | ✅        | ❌        | ❌        |
| Jetson AGX Xavier | ✅        | ✅        | ❌        | ❌        |
| Jetson AGX Orin   | ❌        | ✅        | ✅        | ❌        |
| Jetson Orin NX    | ❌        | ✅        | ✅        | ❌        |
| Jetson Orin Nano  | ❌        | ✅        | ✅        | ❌        |
| Jetson AGX Thor   | ❌        | ❌        | ❌        | ✅        |

## 使用 Docker 快速入门

在 NVIDIA Jetson 上开始使用 Ultralytics YOLO11 的最快方法是使用为 Jetson 预构建的 Docker 镜像运行。请参考上表，根据您拥有的 Jetson 设备选择 JetPack 版本。

=== "JetPack 4"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack4
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 5"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack5
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 6"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack6
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 7"

    即将推出。

完成后，跳转到[在 NVIDIA Jetson 上使用 TensorRT 部分](#在-nvidia-jetson-上使用-tensorrt)。

## 从原生安装开始

如果不使用 Docker 进行原生安装，请参考以下步骤。


### 在 JetPack 7.0 上运行

#### 安装 Ultralytics 包

这里我们将在 Jetson 上安装带有可选依赖项的 Ultralytics 包，以便我们可以将 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型导出为其他不同格式。我们将主要关注 [NVIDIA TensorRT 导出](../integrations/tensorrt.md)，因为 TensorRT 将确保我们从 Jetson 设备获得最大性能。

1. 更新软件包列表，安装 pip 并升级到最新版本

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. 安装带有可选依赖项的 `ultralytics` pip 包

    ```bash
    pip install ultralytics[export]
    ```

3. 重启设备

    ```bash
    sudo reboot
    ```

#### 安装 PyTorch 和 Torchvision

上述 ultralytics 安装将安装 Torch 和 Torchvision。但是，通过 pip 安装的这两个包与配备 JetPack 7.0 和 CUDA 13 的 Jetson AGX Thor 不兼容。因此，我们需要手动安装它们。

根据 JP7.0 安装 `torch` 和 `torchvision`

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### 安装 `onnxruntime-gpu`

PyPI 上托管的 [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) 包没有适用于 Jetson 的 `aarch64` 二进制文件。因此我们需要手动安装此包。某些导出需要此包。

这里我们将下载并安装支持 `Python3.12` 的 `onnxruntime-gpu 1.24.0`。

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
```

### 在 JetPack 6.1 上运行

#### 安装 Ultralytics 包

这里我们将在 Jetson 上安装带有可选依赖项的 Ultralytics 包，以便我们可以将 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型导出为其他不同格式。我们将主要关注 [NVIDIA TensorRT 导出](../integrations/tensorrt.md)，因为 TensorRT 将确保我们从 Jetson 设备获得最大性能。

1. 更新软件包列表，安装 pip 并升级到最新版本

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. 安装带有可选依赖项的 `ultralytics` pip 包

    ```bash
    pip install ultralytics[export]
    ```

3. 重启设备

    ```bash
    sudo reboot
    ```

#### 安装 PyTorch 和 Torchvision

上述 ultralytics 安装将安装 Torch 和 Torchvision。但是，通过 pip 安装的这两个包与基于 ARM64 架构的 Jetson 平台不兼容。因此，我们需要手动安装预构建的 PyTorch pip wheel 并从源代码编译或安装 Torchvision。

根据 JP6.1 安装 `torch 2.5.0` 和 `torchvision 0.20`

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

!!! note

    访问 [PyTorch for Jetson 页面](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 获取适用于不同 JetPack 版本的所有不同版本的 PyTorch。有关 PyTorch、Torchvision 兼容性的更详细列表，请访问 [PyTorch 和 Torchvision 兼容性页面](https://github.com/pytorch/vision)。

安装 [`cuSPARSELt`](https://developer.nvidia.com/cusparselt-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network) 以修复 `torch 2.5.0` 的依赖问题

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

#### 安装 `onnxruntime-gpu`

PyPI 上托管的 [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) 包没有适用于 Jetson 的 `aarch64` 二进制文件。因此我们需要手动安装此包。某些导出需要此包。

您可以在 [Jetson Zoo ONNX Runtime 兼容性矩阵](https://elinux.org/Jetson_Zoo#ONNX_Runtime) 中找到所有可用的 `onnxruntime-gpu` 包——按 JetPack 版本、Python 版本和其他兼容性详细信息组织。

对于支持 `Python 3.10` 的 **JetPack 6**，您可以安装 `onnxruntime-gpu 1.23.0`：

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

或者，安装 `onnxruntime-gpu 1.20.0`：

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

!!! note

    `onnxruntime-gpu` 会自动将 numpy 版本回退到最新版本。因此我们需要重新安装 numpy 到 `1.23.5` 以修复问题，执行：

    `pip install numpy==1.23.5`

### 在 JetPack 5.1.2 上运行

#### 安装 Ultralytics 包

这里我们将在 Jetson 上安装带有可选依赖项的 Ultralytics 包，以便我们可以将 PyTorch 模型导出为其他不同格式。我们将主要关注 [NVIDIA TensorRT 导出](../integrations/tensorrt.md)，因为 TensorRT 将确保我们从 Jetson 设备获得最大性能。

1. 更新软件包列表，安装 pip 并升级到最新版本

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. 安装带有可选依赖项的 `ultralytics` pip 包

    ```bash
    pip install ultralytics[export]
    ```

3. 重启设备

    ```bash
    sudo reboot
    ```

#### 安装 PyTorch 和 Torchvision

上述 ultralytics 安装将安装 Torch 和 Torchvision。但是，通过 pip 安装的这两个包与基于 ARM64 架构的 Jetson 平台不兼容。因此，我们需要手动安装预构建的 PyTorch pip wheel 并从源代码编译或安装 Torchvision。

1. 卸载当前安装的 PyTorch 和 Torchvision

    ```bash
    pip uninstall torch torchvision
    ```

2. 根据 JP5.1.2 安装 `torch 2.2.0` 和 `torchvision 0.17.2`

    ```bash
    pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.2.0-cp38-cp38-linux_aarch64.whl
    pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl
    ```

!!! note

    访问 [PyTorch for Jetson 页面](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 获取适用于不同 JetPack 版本的所有不同版本的 PyTorch。有关 PyTorch、Torchvision 兼容性的更详细列表，请访问 [PyTorch 和 Torchvision 兼容性页面](https://github.com/pytorch/vision)。

#### 安装 `onnxruntime-gpu`

PyPI 上托管的 [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) 包没有适用于 Jetson 的 `aarch64` 二进制文件。因此我们需要手动安装此包。某些导出需要此包。

您可以在 [Jetson Zoo ONNX Runtime 兼容性矩阵](https://elinux.org/Jetson_Zoo#ONNX_Runtime) 中找到所有可用的 `onnxruntime-gpu` 包——按 JetPack 版本、Python 版本和其他兼容性详细信息组织。这里我们将下载并安装支持 `Python3.8` 的 `onnxruntime-gpu 1.17.0`。

```bash
wget https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
```

!!! note

    `onnxruntime-gpu` 会自动将 numpy 版本回退到最新版本。因此我们需要重新安装 numpy 到 `1.23.5` 以修复问题，执行：

    `pip install numpy==1.23.5`

## 在 NVIDIA Jetson 上使用 TensorRT

在 Ultralytics 支持的所有模型导出格式中，TensorRT 在 NVIDIA Jetson 设备上提供最高的推理性能，使其成为我们对 Jetson 部署的首选推荐。有关设置说明和高级用法，请参阅我们的[专用 TensorRT 集成指南](../integrations/tensorrt.md)。

### 转换模型为 TensorRT 并运行推理

PyTorch 格式的 YOLO11n 模型被转换为 TensorRT，以使用导出的模型运行推理。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11n PyTorch 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 TensorRT
        model.export(format="engine")  # 创建 'yolo11n.engine'

        # 加载导出的 TensorRT 模型
        trt_model = YOLO("yolo11n.engine")

        # 运行推理
        results = trt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 TensorRT 格式
        yolo export model=yolo11n.pt format=engine # 创建 'yolo11n.engine'

        # 使用导出的模型运行推理
        yolo predict model=yolo11n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    访问[导出页面](../modes/export.md#arguments)以获取将模型导出为不同模型格式时的其他参数

### 使用 NVIDIA 深度学习加速器 (DLA)

[NVIDIA 深度学习加速器 (DLA)](https://developer.nvidia.com/deep-learning-accelerator) 是内置于 NVIDIA Jetson 设备中的专用硬件组件，可优化深度学习推理的能效和性能。通过从 GPU 卸载任务（释放 GPU 用于更密集的处理），DLA 使模型能够以更低的功耗运行，同时保持高吞吐量，非常适合嵌入式系统和实时 AI 应用。

以下 Jetson 设备配备了 DLA 硬件：

| Jetson 设备            | DLA 核心数 | DLA 最大频率 |
| ------------------------ | --------- | ----------------- |
| Jetson AGX Orin 系列   | 2         | 1.6 GHz           |
| Jetson Orin NX 16GB      | 2         | 614 MHz           |
| Jetson Orin NX 8GB       | 1         | 614 MHz           |
| Jetson AGX Xavier 系列 | 2         | 1.4 GHz           |
| Jetson Xavier NX 系列  | 2         | 1.1 GHz           |

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11n PyTorch 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为启用 DLA 的 TensorRT（仅适用于 FP16 或 INT8）
        model.export(format="engine", device="dla:0", half=True)  # dla:0 或 dla:1 对应 DLA 核心

        # 加载导出的 TensorRT 模型
        trt_model = YOLO("yolo11n.engine")

        # 运行推理
        results = trt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为启用 DLA 的 TensorRT 格式（仅适用于 FP16 或 INT8）
        # 一旦在导出时指定了 DLA 核心编号，推理时将使用相同的核心
        yolo export model=yolo11n.pt format=engine device="dla:0" half=True # dla:0 或 dla:1 对应 DLA 核心

        # 在 DLA 上使用导出的模型运行推理
        yolo predict model=yolo11n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    使用 DLA 导出时，某些层可能不支持在 DLA 上运行，将回退到 GPU 执行。这种回退可能会引入额外的延迟并影响整体推理性能。因此，DLA 的主要设计目的不是减少与完全在 GPU 上运行的 TensorRT 相比的推理延迟。相反，其主要目的是增加吞吐量并提高能效。

## NVIDIA Jetson YOLO11 基准测试

YOLO11 基准测试由 Ultralytics 团队在 11 种不同的模型格式上运行，测量速度和[准确率](https://www.ultralytics.com/glossary/accuracy)：PyTorch、TorchScript、ONNX、OpenVINO、TensorRT、TF SavedModel、TF GraphDef、TF Lite、MNN、NCNN、ExecuTorch。基准测试在 NVIDIA Jetson AGX Thor 开发者套件、NVIDIA Jetson AGX Orin 开发者套件 (64GB)、NVIDIA Jetson Orin Nano Super 开发者套件和由 Jetson Orin NX 16GB 驱动的 Seeed Studio reComputer J4012 设备上运行，使用 FP32 [精度](https://www.ultralytics.com/glossary/precision)和默认输入图像尺寸 640。

### 比较图表

尽管所有模型导出都可以在 NVIDIA Jetson 上运行，但我们在下面的比较图表中仅包含了 **PyTorch、TorchScript、TensorRT**，因为它们利用了 Jetson 上的 GPU 并保证产生最佳结果。所有其他导出仅使用 CPU，性能不如上述三种。您可以在此图表后的部分找到所有导出的基准测试。

#### NVIDIA Jetson AGX Thor 开发者套件

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-agx-thor-benchmarks-coco128.avif" alt="Jetson AGX Thor 基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.3.226 进行基准测试</figcaption>
</figure>

#### NVIDIA Jetson AGX Orin 开发者套件 (64GB)

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-agx-orin-benchmarks-coco128.avif" alt="Jetson AGX Orin 基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.3.157 进行基准测试</figcaption>
</figure>

#### NVIDIA Jetson Orin Nano Super 开发者套件

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-orin-nano-super-benchmarks-coco128.avif" alt="Jetson Orin Nano Super 基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.3.157 进行基准测试</figcaption>
</figure>

#### NVIDIA Jetson Orin NX 16GB

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-orin-nx-16-benchmarks-coco128.avif" alt="Jetson Orin NX 16GB 基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.3.157 进行基准测试</figcaption>
</figure>


### 详细比较表

下表展示了五种不同模型（YOLO11n、YOLO11s、YOLO11m、YOLO11l、YOLO11x）在 11 种不同格式（PyTorch、TorchScript、ONNX、OpenVINO、TensorRT、TF SavedModel、TF GraphDef、TF Lite、MNN、NCNN、ExecuTorch）上的基准测试结果，给出了每种组合的状态、大小、mAP50-95(B) 指标和推理时间。

#### NVIDIA Jetson AGX Thor 开发者套件

!!! tip "性能"

    === "YOLO11n"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.5070      | 4.1                    |
        | TorchScript     | ✅      | 10.5              | 0.5083      | 3.61                   |
        | ONNX            | ✅      | 10.2              | 0.5076      | 4.8                    |
        | OpenVINO        | ✅      | 10.4              | 0.5058      | 16.48                  |
        | TensorRT (FP32) | ✅      | 12.6              | 0.5077      | 1.70                   |
        | TensorRT (FP16) | ✅      | 7.7               | 0.5075      | 1.20                   |
        | TensorRT (INT8) | ✅      | 6.2               | 0.4858      | 1.29                   |
        | TF SavedModel   | ✅      | 25.7              | 0.5076      | 40.35                  |
        | TF GraphDef     | ✅      | 10.3              | 0.5076      | 40.55                  |
        | TF Lite         | ✅      | 10.3              | 0.5075      | 206.74                 |
        | MNN             | ✅      | 10.1              | 0.5075      | 23.47                  |
        | NCNN            | ✅      | 10.2              | 0.5041      | 22.05                  |
        | ExecuTorch      | ✅      | 10.2              | 0.5075      | 34.28                  |

    === "YOLO11s"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.5770      | 6.10                  |
        | TorchScript     | ✅      | 36.6              | 0.5783      | 5.33                   |
        | ONNX            | ✅      | 36.3              | 0.5783      | 7.01                   |
        | OpenVINO        | ✅      | 36.4              | 0.5809      | 33.08                  |
        | TensorRT (FP32) | ✅      | 40.1              | 0.5784      | 2.57                   |
        | TensorRT (FP16) | ✅      | 20.8              | 0.5796      | 1.55                   |
        | TensorRT (INT8) | ✅      | 12.7              | 0.5514      | 1.50                   |
        | TF SavedModel   | ✅      | 90.8              | 0.5782      | 80.55                  |
        | TF GraphDef     | ✅      | 36.3              | 0.5782      | 80.82                  |
        | TF Lite         | ✅      | 36.3              | 0.5782      | 615.29                 |
        | MNN             | ✅      | 36.2              | 0.5790      | 54.12                  |
        | NCNN            | ✅      | 36.3              | 0.5806      | 40.76                  |
        | ExecuTorch      | ✅      | 36.2              | 0.5782      | 67.21                  |

    === "YOLO11m"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.6250      | 11.4                   |
        | TorchScript     | ✅      | 77.3              | 0.6304      | 10.16                  |
        | ONNX            | ✅      | 76.9              | 0.6304      | 12.35                  |
        | OpenVINO        | ✅      | 77.1              | 0.6284      | 77.81                  |
        | TensorRT (FP32) | ✅      | 80.7              | 0.6305      | 5.29                   |
        | TensorRT (FP16) | ✅      | 41.3              | 0.6294      | 2.42                   |
        | TensorRT (INT8) | ✅      | 23.7              | 0.6133      | 2.20                   |
        | TF SavedModel   | ✅      | 192.4             | 0.6306      | 184.66                 |
        | TF GraphDef     | ✅      | 76.9              | 0.6306      | 187.91                 |
        | TF Lite         | ✅      | 76.9              | 0.6306      | 1845.09                |
        | MNN             | ✅      | 76.8              | 0.6298      | 143.52                 |
        | NCNN            | ✅      | 76.9              | 0.6308      | 95.86                  |
        | ExecuTorch      | ✅      | 76.9              | 0.6306      | 167.94                 |

    === "YOLO11l"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.6370      | 14.0                   |
        | TorchScript     | ✅      | 97.6              | 0.6409      | 13.77                  |
        | ONNX            | ✅      | 97.0              | 0.6410      | 16.37                  |
        | OpenVINO        | ✅      | 97.3              | 0.6377      | 98.86                  |
        | TensorRT (FP32) | ✅      | 101.0             | 0.6396      | 6.71                   |
        | TensorRT (FP16) | ✅      | 51.5              | 0.6358      | 3.26                   |
        | TensorRT (INT8) | ✅      | 29.7              | 0.6190      | 3.21                   |
        | TF SavedModel   | ✅      | 242.7             | 0.6409      | 246.93                 |
        | TF GraphDef     | ✅      | 97.0              | 0.6409      | 251.84                 |
        | TF Lite         | ✅      | 97.0              | 0.6409      | 2383.45                |
        | MNN             | ✅      | 96.9              | 0.6361      | 176.53                 |
        | NCNN            | ✅      | 97.0              | 0.6373      | 118.05                 |
        | ExecuTorch      | ✅      | 97.0              | 0.6409      | 211.46                 |

    === "YOLO11x"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.6990      | 21.70                  |
        | TorchScript     | ✅      | 218.1             | 0.6900      | 20.99                  |
        | ONNX            | ✅      | 217.5             | 0.6900      | 24.07                  |
        | OpenVINO        | ✅      | 217.8             | 0.6872      | 187.33                 |
        | TensorRT (FP32) | ✅      | 220.0             | 0.6902      | 11.70                  |
        | TensorRT (FP16) | ✅      | 114.6             | 0.6881      | 5.10                   |
        | TensorRT (INT8) | ✅      | 59.9              | 0.6857      | 4.53                   |
        | TF SavedModel   | ✅      | 543.9             | 0.6900      | 489.91                 |
        | TF GraphDef     | ✅      | 217.5             | 0.6900      | 503.21                 |
        | TF Lite         | ✅      | 217.5             | 0.6900      | 5164.31                |
        | MNN             | ✅      | 217.3             | 0.6905      | 350.37                 |
        | NCNN            | ✅      | 217.5             | 0.6901      | 230.63                 |
        | ExecuTorch      | ✅      | 217.4             | 0.6900      | 419.9                  |

    使用 Ultralytics 8.3.226 进行基准测试

    !!! note

        推理时间不包括预处理/后处理。

#### NVIDIA Jetson AGX Orin 开发者套件 (64GB)

!!! tip "性能"

    === "YOLO11n"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.5101      | 9.40                   |
        | TorchScript     | ✅      | 10.5              | 0.5083      | 11.00                  |
        | ONNX            | ✅      | 10.2              | 0.5077      | 48.32                  |
        | OpenVINO        | ✅      | 10.4              | 0.5058      | 27.24                  |
        | TensorRT (FP32) | ✅      | 12.1              | 0.5085      | 3.93                   |
        | TensorRT (FP16) | ✅      | 8.3               | 0.5063      | 2.55                   |
        | TensorRT (INT8) | ✅      | 5.4               | 0.4719      | 2.18                   |
        | TF SavedModel   | ✅      | 25.9              | 0.5077      | 66.87                  |
        | TF GraphDef     | ✅      | 10.3              | 0.5077      | 65.68                  |
        | TF Lite         | ✅      | 10.3              | 0.5077      | 272.92                 |
        | MNN             | ✅      | 10.1              | 0.5059      | 36.33                  |
        | NCNN            | ✅      | 10.2              | 0.5031      | 28.51                  |

    === "YOLO11s"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.5783      | 12.10                  |
        | TorchScript     | ✅      | 36.5              | 0.5782      | 11.01                  |
        | ONNX            | ✅      | 36.3              | 0.5782      | 107.54                 |
        | OpenVINO        | ✅      | 36.4              | 0.5810      | 55.03                  |
        | TensorRT (FP32) | ✅      | 38.1              | 0.5781      | 6.52                   |
        | TensorRT (FP16) | ✅      | 21.4              | 0.5803      | 3.65                   |
        | TensorRT (INT8) | ✅      | 12.1              | 0.5735      | 2.81                   |
        | TF SavedModel   | ✅      | 91.0              | 0.5782      | 132.73                 |
        | TF GraphDef     | ✅      | 36.4              | 0.5782      | 134.96                 |
        | TF Lite         | ✅      | 36.3              | 0.5782      | 798.21                 |
        | MNN             | ✅      | 36.2              | 0.5777      | 82.35                  |
        | NCNN            | ✅      | 36.2              | 0.5784      | 56.07                  |

    === "YOLO11m"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.6265      | 22.20                  |
        | TorchScript     | ✅      | 77.3              | 0.6307      | 21.47                  |
        | ONNX            | ✅      | 76.9              | 0.6307      | 270.89                 |
        | OpenVINO        | ✅      | 77.1              | 0.6284      | 129.10                 |
        | TensorRT (FP32) | ✅      | 78.8              | 0.6306      | 12.53                  |
        | TensorRT (FP16) | ✅      | 41.9              | 0.6305      | 6.25                   |
        | TensorRT (INT8) | ✅      | 23.2              | 0.6291      | 4.69                   |
        | TF SavedModel   | ✅      | 192.7             | 0.6307      | 299.95                 |
        | TF GraphDef     | ✅      | 77.1              | 0.6307      | 310.58                 |
        | TF Lite         | ✅      | 77.0              | 0.6307      | 2400.54                |
        | MNN             | ✅      | 76.8              | 0.6308      | 213.56                 |
        | NCNN            | ✅      | 76.8              | 0.6284      | 141.18                 |

    === "YOLO11l"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.6364      | 27.70                   |
        | TorchScript     | ✅      | 97.6              | 0.6399      | 27.94                  |
        | ONNX            | ✅      | 97.0              | 0.6409      | 345.47                 |
        | OpenVINO        | ✅      | 97.3              | 0.6378      | 161.93                 |
        | TensorRT (FP32) | ✅      | 99.1              | 0.6406      | 16.11                  |
        | TensorRT (FP16) | ✅      | 52.6              | 0.6376      | 8.08                   |
        | TensorRT (INT8) | ✅      | 30.8              | 0.6208      | 6.12                   |
        | TF SavedModel   | ✅      | 243.1             | 0.6409      | 390.78                 |
        | TF GraphDef     | ✅      | 97.2              | 0.6409      | 398.76                 |
        | TF Lite         | ✅      | 97.1              | 0.6409      | 3037.05                |
        | MNN             | ✅      | 96.9              | 0.6372      | 265.46                 |
        | NCNN            | ✅      | 96.9              | 0.6364      | 179.68                 |

    === "YOLO11x"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.7005      | 44.40                  |
        | TorchScript     | ✅      | 218.1             | 0.6898      | 47.49                  |
        | ONNX            | ✅      | 217.5             | 0.6900      | 682.98                 |
        | OpenVINO        | ✅      | 217.8             | 0.6876      | 298.15                 |
        | TensorRT (FP32) | ✅      | 219.6             | 0.6904      | 28.50                  |
        | TensorRT (FP16) | ✅      | 112.2             | 0.6887      | 13.55                  |
        | TensorRT (INT8) | ✅      | 60.0              | 0.6574      | 9.40                   |
        | TF SavedModel   | ✅      | 544.3             | 0.6900      | 749.85                 |
        | TF GraphDef     | ✅      | 217.7             | 0.6900      | 753.86                 |
        | TF Lite         | ✅      | 217.6             | 0.6900      | 6603.27                |
        | MNN             | ✅      | 217.3             | 0.6868      | 519.77                 |
        | NCNN            | ✅      | 217.3             | 0.6849      | 298.58                 |

    使用 Ultralytics 8.3.157 进行基准测试

    !!! note

        推理时间不包括预处理/后处理。


#### NVIDIA Jetson Orin Nano Super 开发者套件

!!! tip "性能"

    === "YOLO11n"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.5101      | 13.70                  |
        | TorchScript     | ✅      | 10.5              | 0.5082      | 13.69                  |
        | ONNX            | ✅      | 10.2              | 0.5081      | 14.47                  |
        | OpenVINO        | ✅      | 10.4              | 0.5058      | 56.66                  |
        | TensorRT (FP32) | ✅      | 12.0              | 0.5081      | 7.44                   |
        | TensorRT (FP16) | ✅      | 8.2               | 0.5061      | 4.53                   |
        | TensorRT (INT8) | ✅      | 5.4               | 0.4825      | 3.70                   |
        | TF SavedModel   | ✅      | 25.9              | 0.5077      | 116.23                 |
        | TF GraphDef     | ✅      | 10.3              | 0.5077      | 114.92                 |
        | TF Lite         | ✅      | 10.3              | 0.5077      | 340.75                 |
        | MNN             | ✅      | 10.1              | 0.5059      | 76.26                  |
        | NCNN            | ✅      | 10.2              | 0.5031      | 45.03                  |

    === "YOLO11s"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.5790      | 20.90                  |
        | TorchScript     | ✅      | 36.5              | 0.5781      | 21.22                  |
        | ONNX            | ✅      | 36.3              | 0.5781      | 25.07                  |
        | OpenVINO        | ✅      | 36.4              | 0.5810      | 122.98                 |
        | TensorRT (FP32) | ✅      | 37.9              | 0.5783      | 13.02                  |
        | TensorRT (FP16) | ✅      | 21.8              | 0.5779      | 6.93                   |
        | TensorRT (INT8) | ✅      | 12.2              | 0.5735      | 5.08                   |
        | TF SavedModel   | ✅      | 91.0              | 0.5782      | 250.65                 |
        | TF GraphDef     | ✅      | 36.4              | 0.5782      | 252.69                 |
        | TF Lite         | ✅      | 36.3              | 0.5782      | 998.68                 |
        | MNN             | ✅      | 36.2              | 0.5781      | 188.01                 |
        | NCNN            | ✅      | 36.2              | 0.5784      | 101.37                 |

    === "YOLO11m"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.6266      | 46.50                  |
        | TorchScript     | ✅      | 77.3              | 0.6307      | 47.95                  |
        | ONNX            | ✅      | 76.9              | 0.6307      | 53.06                  |
        | OpenVINO        | ✅      | 77.1              | 0.6284      | 301.63                 |
        | TensorRT (FP32) | ✅      | 78.8              | 0.6305      | 27.86                  |
        | TensorRT (FP16) | ✅      | 41.7              | 0.6309      | 13.50                  |
        | TensorRT (INT8) | ✅      | 23.2              | 0.6291      | 9.12                   |
        | TF SavedModel   | ✅      | 192.7             | 0.6307      | 622.24                 |
        | TF GraphDef     | ✅      | 77.1              | 0.6307      | 628.74                 |
        | TF Lite         | ✅      | 77.0              | 0.6307      | 2997.93                |
        | MNN             | ✅      | 76.8              | 0.6299      | 509.96                 |
        | NCNN            | ✅      | 76.8              | 0.6284      | 292.99                 |

    === "YOLO11l"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.6364      | 56.50                  |
        | TorchScript     | ✅      | 97.6              | 0.6409      | 62.51                  |
        | ONNX            | ✅      | 97.0              | 0.6399      | 68.35                  |
        | OpenVINO        | ✅      | 97.3              | 0.6378      | 376.03                 |
        | TensorRT (FP32) | ✅      | 99.2              | 0.6396      | 35.59                  |
        | TensorRT (FP16) | ✅      | 52.1              | 0.6361      | 17.48                  |
        | TensorRT (INT8) | ✅      | 30.9              | 0.6207      | 11.87                  |
        | TF SavedModel   | ✅      | 243.1             | 0.6409      | 807.47                 |
        | TF GraphDef     | ✅      | 97.2              | 0.6409      | 822.88                 |
        | TF Lite         | ✅      | 97.1              | 0.6409      | 3792.23                |
        | MNN             | ✅      | 96.9              | 0.6372      | 631.16                 |
        | NCNN            | ✅      | 96.9              | 0.6364      | 350.46                 |

    === "YOLO11x"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.7005      | 90.00                  |
        | TorchScript     | ✅      | 218.1             | 0.6901      | 113.40                 |
        | ONNX            | ✅      | 217.5             | 0.6901      | 122.94                 |
        | OpenVINO        | ✅      | 217.8             | 0.6876      | 713.1                  |
        | TensorRT (FP32) | ✅      | 219.5             | 0.6904      | 66.93                  |
        | TensorRT (FP16) | ✅      | 112.2             | 0.6892      | 32.58                  |
        | TensorRT (INT8) | ✅      | 61.5              | 0.6612      | 19.90                  |
        | TF SavedModel   | ✅      | 544.3             | 0.6900      | 1605.4                 |
        | TF GraphDef     | ✅      | 217.8             | 0.6900      | 2961.8                 |
        | TF Lite         | ✅      | 217.6             | 0.6900      | 8234.86                |
        | MNN             | ✅      | 217.3             | 0.6893      | 1254.18                |
        | NCNN            | ✅      | 217.3             | 0.6849      | 725.50                 |

    使用 Ultralytics 8.3.157 进行基准测试

    !!! note

        推理时间不包括预处理/后处理。

#### NVIDIA Jetson Orin NX 16GB

!!! tip "性能"

    === "YOLO11n"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.5101      | 12.90                  |
        | TorchScript     | ✅      | 10.5              | 0.5082      | 13.17                  |
        | ONNX            | ✅      | 10.2              | 0.5081      | 15.43                  |
        | OpenVINO        | ✅      | 10.4              | 0.5058      | 39.80                  |
        | TensorRT (FP32) | ✅      | 11.8              | 0.5081      | 7.94                   |
        | TensorRT (FP16) | ✅      | 8.1               | 0.5085      | 4.73                   |
        | TensorRT (INT8) | ✅      | 5.4               | 0.4786      | 3.90                   |
        | TF SavedModel   | ✅      | 25.9              | 0.5077      | 88.48                  |
        | TF GraphDef     | ✅      | 10.3              | 0.5077      | 86.67                  |
        | TF Lite         | ✅      | 10.3              | 0.5077      | 302.55                 |
        | MNN             | ✅      | 10.1              | 0.5059      | 52.73                  |
        | NCNN            | ✅      | 10.2              | 0.5031      | 32.04                  |

    === "YOLO11s"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.5790      | 21.70                  |
        | TorchScript     | ✅      | 36.5              | 0.5781      | 22.71                  |
        | ONNX            | ✅      | 36.3              | 0.5781      | 26.49                  |
        | OpenVINO        | ✅      | 36.4              | 0.5810      | 84.73                  |
        | TensorRT (FP32) | ✅      | 37.8              | 0.5783      | 13.77                  |
        | TensorRT (FP16) | ✅      | 21.2              | 0.5796      | 7.31                   |
        | TensorRT (INT8) | ✅      | 12.0              | 0.5735      | 5.33                   |
        | TF SavedModel   | ✅      | 91.0              | 0.5782      | 185.06                 |
        | TF GraphDef     | ✅      | 36.4              | 0.5782      | 186.45                 |
        | TF Lite         | ✅      | 36.3              | 0.5782      | 882.58                 |
        | MNN             | ✅      | 36.2              | 0.5775      | 126.36                 |
        | NCNN            | ✅      | 36.2              | 0.5784      | 66.73                  |

    === "YOLO11m"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.6266      | 45.00                  |
        | TorchScript     | ✅      | 77.3              | 0.6307      | 51.87                  |
        | ONNX            | ✅      | 76.9              | 0.6307      | 56.00                  |
        | OpenVINO        | ✅      | 77.1              | 0.6284      | 202.69                 |
        | TensorRT (FP32) | ✅      | 78.7              | 0.6305      | 30.38                  |
        | TensorRT (FP16) | ✅      | 41.8              | 0.6302      | 14.48                  |
        | TensorRT (INT8) | ✅      | 23.2              | 0.6291      | 9.74                   |
        | TF SavedModel   | ✅      | 192.7             | 0.6307      | 445.58                 |
        | TF GraphDef     | ✅      | 77.1              | 0.6307      | 460.94                 |
        | TF Lite         | ✅      | 77.0              | 0.6307      | 2653.65                |
        | MNN             | ✅      | 76.8              | 0.6308      | 339.38                 |
        | NCNN            | ✅      | 76.8              | 0.6284      | 187.64                 |

    === "YOLO11l"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.6364      | 56.60                  |
        | TorchScript     | ✅      | 97.6              | 0.6409      | 66.72                  |
        | ONNX            | ✅      | 97.0              | 0.6399      | 71.92                  |
        | OpenVINO        | ✅      | 97.3              | 0.6378      | 254.17                 |
        | TensorRT (FP32) | ✅      | 99.2              | 0.6406      | 38.89                  |
        | TensorRT (FP16) | ✅      | 51.9              | 0.6363      | 18.59                  |
        | TensorRT (INT8) | ✅      | 30.9              | 0.6207      | 12.60                  |
        | TF SavedModel   | ✅      | 243.1             | 0.6409      | 575.98                 |
        | TF GraphDef     | ✅      | 97.2              | 0.6409      | 583.79                 |
        | TF Lite         | ✅      | 97.1              | 0.6409      | 3353.41                |
        | MNN             | ✅      | 96.9              | 0.6367      | 421.33                 |
        | NCNN            | ✅      | 96.9              | 0.6364      | 228.26                 |

    === "YOLO11x"

        | 格式          | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.7005      | 98.50                  |
        | TorchScript     | ✅      | 218.1             | 0.6901      | 123.03                 |
        | ONNX            | ✅      | 217.5             | 0.6901      | 129.55                 |
        | OpenVINO        | ✅      | 217.8             | 0.6876      | 483.44                 |
        | TensorRT (FP32) | ✅      | 219.6             | 0.6904      | 75.92                  |
        | TensorRT (FP16) | ✅      | 112.1             | 0.6885      | 35.78                  |
        | TensorRT (INT8) | ✅      | 61.6              | 0.6592      | 21.60                  |
        | TF SavedModel   | ✅      | 544.3             | 0.6900      | 1120.43                |
        | TF GraphDef     | ✅      | 217.7             | 0.6900      | 1172.35                |
        | TF Lite         | ✅      | 217.6             | 0.6900      | 7283.63                |
        | MNN             | ✅      | 217.3             | 0.6877      | 840.16                 |
        | NCNN            | ✅      | 217.3             | 0.6849      | 474.41                 |

    使用 Ultralytics 8.3.157 进行基准测试

    !!! note

        推理时间不包括预处理/后处理。

[探索 Seeed Studio 在不同版本 NVIDIA Jetson 硬件上运行的更多基准测试工作](https://www.seeedstudio.com/blog/2023/03/30/yolov8-performance-benchmarks-on-nvidia-jetson-devices/)。

## 复现我们的结果

要在所有导出[格式](../modes/export.md)上复现上述 Ultralytics 基准测试，请运行以下代码：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11n PyTorch 模型
        model = YOLO("yolo11n.pt")

        # 在 COCO128 数据集上对所有导出格式进行 YOLO11n 速度和准确率基准测试
        results = model.benchmark(data="coco128.yaml", imgsz=640)
        ```

    === "CLI"

        ```bash
        # 在 COCO128 数据集上对所有导出格式进行 YOLO11n 速度和准确率基准测试
        yolo benchmark model=yolo11n.pt data=coco128.yaml imgsz=640
        ```

    请注意，基准测试结果可能会因系统的确切硬件和软件配置以及运行基准测试时系统的当前工作负载而有所不同。为获得最可靠的结果，请使用包含大量图像的数据集，例如 `data='coco.yaml'`（5000 张验证图像）。

## 使用 NVIDIA Jetson 时的最佳实践

使用 NVIDIA Jetson 时，有几个最佳实践可以遵循，以便在运行 YOLO11 的 NVIDIA Jetson 上实现最大性能。

1. 启用 MAX 功率模式

    在 Jetson 上启用 MAX 功率模式将确保所有 CPU、GPU 核心都已开启。

    ```bash
    sudo nvpmodel -m 0
    ```

2. 启用 Jetson 时钟

    启用 Jetson 时钟将确保所有 CPU、GPU 核心都以最大频率运行。

    ```bash
    sudo jetson_clocks
    ```

3. 安装 Jetson Stats 应用程序

    我们可以使用 jetson stats 应用程序来监控系统组件的温度并检查其他系统详细信息，如查看 CPU、GPU、RAM 利用率、更改功率模式、设置为最大时钟、检查 JetPack 信息

    ```bash
    sudo apt update
    sudo pip install jetson-stats
    sudo reboot
    jtop
    ```

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/jetson-stats-application.avif" alt="Jetson Stats">

## 后续步骤

如需进一步学习和支持，请参阅 [Ultralytics YOLO11 文档](../index.md)。

## 常见问题

### 如何在 NVIDIA Jetson 设备上部署 Ultralytics YOLO11？

在 NVIDIA Jetson 设备上部署 Ultralytics YOLO11 是一个简单的过程。首先，使用 NVIDIA JetPack SDK 刷入您的 Jetson 设备。然后，使用预构建的 Docker 镜像进行快速设置，或手动安装所需的软件包。每种方法的详细步骤可以在[使用 Docker 快速入门](#使用-docker-快速入门)和[从原生安装开始](#从原生安装开始)部分找到。

### 我可以期望 YOLO11 模型在 NVIDIA Jetson 设备上有什么性能基准？

YOLO11 模型已在各种 NVIDIA Jetson 设备上进行了基准测试，显示出显著的性能改进。例如，TensorRT 格式提供最佳的推理性能。[详细比较表](#详细比较表)部分的表格提供了跨不同模型格式的性能指标（如 mAP50-95 和推理时间）的全面视图。

### 为什么我应该使用 TensorRT 在 NVIDIA Jetson 上部署 YOLO11？

强烈建议使用 TensorRT 在 NVIDIA Jetson 上部署 YOLO11 模型，因为它具有最佳性能。它通过利用 Jetson 的 GPU 功能来加速推理，确保最大效率和速度。在[在 NVIDIA Jetson 上使用 TensorRT](#在-nvidia-jetson-上使用-tensorrt) 部分了解更多关于如何转换为 TensorRT 并运行推理的信息。

### 如何在 NVIDIA Jetson 上安装 PyTorch 和 Torchvision？

要在 NVIDIA Jetson 上安装 PyTorch 和 Torchvision，首先卸载可能通过 pip 安装的任何现有版本。然后，手动安装适用于 Jetson ARM64 架构的兼容 PyTorch 和 Torchvision 版本。此过程的详细说明在[安装 PyTorch 和 Torchvision](#安装-pytorch-和-torchvision) 部分提供。

### 使用 YOLO11 时在 NVIDIA Jetson 上最大化性能的最佳实践是什么？

要在使用 YOLO11 的 NVIDIA Jetson 上最大化性能，请遵循以下最佳实践：

1. 启用 MAX 功率模式以利用所有 CPU 和 GPU 核心。
2. 启用 Jetson 时钟以最大频率运行所有核心。
3. 安装 Jetson Stats 应用程序以监控系统指标。

有关命令和其他详细信息，请参阅[使用 NVIDIA Jetson 时的最佳实践](#使用-nvidia-jetson-时的最佳实践)部分。
