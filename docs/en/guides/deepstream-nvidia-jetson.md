---
comments: true
description: 学习如何使用 TensorRT 和 DeepStream SDK 在 NVIDIA Jetson 设备上部署 Ultralytics YOLO11。探索性能基准测试并最大化 AI 能力。
keywords: Ultralytics, YOLO11, NVIDIA Jetson, JetPack, AI 部署, 嵌入式系统, 深度学习, TensorRT, DeepStream SDK, 计算机视觉
---

# 使用 DeepStream SDK 和 TensorRT 在 NVIDIA Jetson 上运行 Ultralytics YOLO11

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hvGqrVT2wPg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 Jetson Orin NX 上使用 NVIDIA Deepstream 运行 Ultralytics YOLO11 模型 🚀
</p>

本综合指南详细介绍了如何使用 DeepStream SDK 和 TensorRT 在 [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) 设备上部署 Ultralytics YOLO11。这里我们使用 TensorRT 来最大化 Jetson 平台上的推理性能。

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/deepstream-nvidia-jetson.avif" alt="NVIDIA Jetson 上的 DeepStream">

!!! note

    本指南已在运行最新稳定 JetPack 版本 [JP6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61) 的 [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit)、
    基于 NVIDIA Jetson Orin NX 16GB 运行 JetPack 版本 [JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) 的 [Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html) 以及基于 NVIDIA Jetson Nano 4GB 运行 JetPack 版本 [JP4.6.4](https://developer.nvidia.com/jetpack-sdk-464) 的 [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html) 上测试通过。预计可在所有 NVIDIA Jetson 硬件产品线上运行，包括最新和旧版设备。

## 什么是 NVIDIA DeepStream？

[NVIDIA 的 DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) 是一个基于 GStreamer 的完整流分析工具包，用于基于 AI 的多传感器处理、视频、音频和图像理解。它非常适合构建 IVA（智能视频分析）应用和服务的视觉 AI 开发人员、软件合作伙伴、初创公司和 OEM。您现在可以创建包含[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)和其他复杂处理任务（如跟踪、视频编码/解码和视频渲染）的流处理管道。这些管道能够对视频、图像和传感器数据进行实时分析。DeepStream 的多平台支持为您提供了一种更快、更简单的方式来开发本地、边缘和云端的视觉 AI 应用和服务。

## 前提条件

在开始遵循本指南之前：

- 访问我们的文档[快速入门指南：使用 Ultralytics YOLO11 的 NVIDIA Jetson](nvidia-jetson.md)，在您的 NVIDIA Jetson 设备上设置 Ultralytics YOLO11
- 根据 JetPack 版本安装 [DeepStream SDK](https://developer.nvidia.com/deepstream-getting-started)
    - 对于 JetPack 4.6.4，安装 [DeepStream 6.0.1](https://docs.nvidia.com/metropolis/deepstream/6.0.1/dev-guide/text/DS_Quickstart.html)
    - 对于 JetPack 5.1.3，安装 [DeepStream 6.3](https://docs.nvidia.com/metropolis/deepstream/6.3/dev-guide/text/DS_Quickstart.html)
    - 对于 JetPack 6.1，安装 [DeepStream 7.1](https://docs.nvidia.com/metropolis/deepstream/7.0/dev-guide/text/DS_Overview.html)

!!! tip

    在本指南中，我们使用 Debian 包方法将 DeepStream SDK 安装到 Jetson 设备。您也可以访问 [Jetson 上的 DeepStream SDK（存档）](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)来访问旧版本的 DeepStream。

## YOLO11 的 DeepStream 配置

这里我们使用 [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) GitHub 仓库，其中包含对 YOLO 模型的 NVIDIA DeepStream SDK 支持。我们感谢 marcoslucianops 的贡献！

1.  安装 Ultralytics 及必要的依赖项

    ```bash
    cd ~
    pip install -U pip
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics
    pip install -e ".[export]" onnxslim
    ```

2.  克隆 DeepStream-Yolo 仓库

    ```bash
    cd ~
    git clone https://github.com/marcoslucianops/DeepStream-Yolo
    ```

3.  将 `DeepStream-Yolo/utils` 目录中的 `export_yolo11.py` 文件复制到 `ultralytics` 文件夹

    ```bash
    cp ~/DeepStream-Yolo/utils/export_yolo11.py ~/ultralytics
    cd ultralytics
    ```

4.  从 [YOLO11 发布页面](https://github.com/ultralytics/assets/releases)下载您选择的 Ultralytics YOLO11 检测模型（.pt）。这里我们使用 [yolo11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)。

    ```bash
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
    ```

    !!! note

        您也可以使用[自定义训练的 YOLO11 模型](https://docs.ultralytics.com/modes/train/)。

5.  将模型转换为 ONNX

    ```bash
    python3 export_yolo11.py -w yolo11s.pt
    ```

    !!! note "将以下参数传递给上述命令"

        对于 DeepStream 5.1，移除 `--dynamic` 参数并使用 `opset` 12 或更低版本。默认 `opset` 是 17。

        ```bash
        --opset 12
        ```

        更改推理尺寸（默认：640）

        ```bash
        -s SIZE
        --size SIZE
        -s HEIGHT WIDTH
        --size HEIGHT WIDTH
        ```

        1280 的示例：

        ```bash
        -s 1280
        或
        -s 1280 1280
        ```

        简化 ONNX 模型（DeepStream >= 6.0）

        ```bash
        --simplify
        ```

        使用动态批量大小（DeepStream >= 6.1）

        ```bash
        --dynamic
        ```

        使用静态批量大小（例如批量大小 = 4）

        ```bash
        --batch 4
        ```

6.  将生成的 `.onnx` 模型文件和 `labels.txt` 文件复制到 `DeepStream-Yolo` 文件夹

    ```bash
    cp yolo11s.pt.onnx labels.txt ~/DeepStream-Yolo
    cd ~/DeepStream-Yolo
    ```

7.  根据安装的 JetPack 版本设置 CUDA 版本

    对于 JetPack 4.6.4：

    ```bash
    export CUDA_VER=10.2
    ```

    对于 JetPack 5.1.3：

    ```bash
    export CUDA_VER=11.4
    ```

    对于 JetPack 6.1：

    ```bash
    export CUDA_VER=12.6
    ```

8.  编译库

    ```bash
    make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
    ```

9.  根据您的模型编辑 `config_infer_primary_yolo11.txt` 文件（对于具有 80 个类别的 YOLO11s）

    ```bash
    [property]
    ...
    onnx-file=yolo11s.pt.onnx
    ...
    num-detected-classes=80
    ...
    ```

10. 编辑 `deepstream_app_config` 文件

    ```bash
    ...
    [primary-gie]
    ...
    config-file=config_infer_primary_yolo11.txt
    ```

11. 您还可以在 `deepstream_app_config` 文件中更改视频源。这里加载了一个默认视频文件

    ```bash
    ...
    [source0]
    ...
    uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
    ```

### 运行推理

```bash
deepstream-app -c deepstream_app_config.txt
```

!!! note

    在开始推理之前，生成 TensorRT 引擎文件需要很长时间。请耐心等待。

<div align=center><img width=1000 src="https://github.com/ultralytics/docs/releases/download/0/yolov8-with-deepstream.avif" alt="使用 deepstream 的 YOLO11"></div>

!!! tip

    如果您想将模型转换为 FP16 精度，只需在 `config_infer_primary_yolo11.txt` 中设置 `model-engine-file=model_b1_gpu0_fp16.engine` 和 `network-mode=2`

## INT8 校准

如果您想使用 INT8 精度进行推理，需要按照以下步骤操作：

!!! note

    目前 INT8 不适用于 TensorRT 10.x。本指南的这一部分已在 TensorRT 8.x 上测试，预计可以正常工作。

1.  设置 `OPENCV` 环境变量

    ```bash
    export OPENCV=1
    ```

2.  编译库

    ```bash
    make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
    ```

3.  对于 COCO 数据集，下载 [val2017](http://images.cocodataset.org/zips/val2017.zip)，解压并移动到 `DeepStream-Yolo` 文件夹

4.  为校准图像创建新目录

    ```bash
    mkdir calibration
    ```

5.  运行以下命令从 COCO 数据集中随机选择 1000 张图像进行校准

    ```bash
    for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do
      cp ${jpg} calibration/
    done
    ```

    !!! note

        NVIDIA 建议至少使用 500 张图像以获得良好的[准确率](https://www.ultralytics.com/glossary/accuracy)。在此示例中，选择了 1000 张图像以获得更好的准确率（更多图像 = 更高准确率）。您可以通过 **head -1000** 设置。例如，对于 2000 张图像，使用 **head -2000**。此过程可能需要很长时间。

6.  创建包含所有选定图像的 `calibration.txt` 文件

    ```bash
    realpath calibration/*jpg > calibration.txt
    ```

7.  设置环境变量

    ```bash
    export INT8_CALIB_IMG_PATH=calibration.txt
    export INT8_CALIB_BATCH_SIZE=1
    ```

    !!! note

        更高的 INT8_CALIB_BATCH_SIZE 值将带来更高的准确率和更快的校准速度。根据您的 GPU 内存进行设置。

8.  更新 `config_infer_primary_yolo11.txt` 文件

    从

    ```bash
    ...
    model-engine-file=model_b1_gpu0_fp32.engine
    #int8-calib-file=calib.table
    ...
    network-mode=0
    ...
    ```

    改为

    ```bash
    ...
    model-engine-file=model_b1_gpu0_int8.engine
    int8-calib-file=calib.table
    ...
    network-mode=1
    ...
    ```

### 运行推理

```bash
deepstream-app -c deepstream_app_config.txt
```

## 多流设置

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/wWmXKIteRLA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO11 在 Jetson Nano 上使用 DeepStream SDK 运行多个流 🎉
</p>

要在单个 DeepStream 应用程序下设置多个流，请对 `deepstream_app_config.txt` 文件进行以下更改：

1. 根据您想要的流数量更改行和列以构建网格显示。例如，对于 4 个流，我们可以添加 2 行和 2 列。

    ```bash
    [tiled-display]
    rows=2
    columns=2
    ```

2. 设置 `num-sources=4` 并为所有四个流添加 `uri` 条目。

    ```bash
    [source0]
    enable=1
    type=3
    uri=path/to/video1.jpg
    uri=path/to/video2.jpg
    uri=path/to/video3.jpg
    uri=path/to/video4.jpg
    num-sources=4
    ```

### 运行推理

```bash
deepstream-app -c deepstream_app_config.txt
```

<div align=center><img width=1000 src="https://github.com/ultralytics/docs/releases/download/0/multistream-setup.avif" alt="多流设置"></div>

## 基准测试结果

以下基准测试总结了 YOLO11 模型在 NVIDIA Jetson Orin NX 16GB 上以 640x640 输入尺寸在不同 TensorRT 精度级别下的性能表现。

### 比较图表

<div align=center><img width=1000 src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-deepstream-benchmarks.avif" alt="Jetson DeepStream 基准测试图表"></div>

### 详细比较表

!!! tip "性能"

    === "YOLO11n"

        | 格式          | 状态 | 推理时间 (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 8.64                   |
        | TensorRT (FP16) | ✅      | 5.27                   |
        | TensorRT (INT8) | ✅      | 4.54                   |

    === "YOLO11s"

        | 格式          | 状态 | 推理时间 (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 14.53                  |
        | TensorRT (FP16) | ✅      | 7.91                   |
        | TensorRT (INT8) | ✅      | 6.05                   |

    === "YOLO11m"

        | 格式          | 状态 | 推理时间 (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 32.05                  |
        | TensorRT (FP16) | ✅      | 15.55                  |
        | TensorRT (INT8) | ✅      | 10.43                  |

    === "YOLO11l"

        | 格式          | 状态 | 推理时间 (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 39.68                  |
        | TensorRT (FP16) | ✅      | 19.88                  |
        | TensorRT (INT8) | ✅      | 13.64                  |

    === "YOLO11x"

        | 格式          | 状态 | 推理时间 (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 80.65                  |
        | TensorRT (FP16) | ✅      | 39.06                  |
        | TensorRT (INT8) | ✅      | 22.83                  |

## 致谢

本指南最初由我们在 Seeed Studio 的朋友 Lakshantha 和 Elaine 创建。

## 常见问题

### 如何在 NVIDIA Jetson 设备上设置 Ultralytics YOLO11？

要在 [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) 设备上设置 Ultralytics YOLO11，您首先需要安装与您的 JetPack 版本兼容的 [DeepStream SDK](https://developer.nvidia.com/deepstream-getting-started)。按照我们的[快速入门指南](nvidia-jetson.md)中的分步说明为 YOLO11 部署配置您的 NVIDIA Jetson。

### 在 NVIDIA Jetson 上使用 TensorRT 与 YOLO11 有什么好处？

将 TensorRT 与 YOLO11 一起使用可以优化模型进行推理，显著降低延迟并提高 NVIDIA Jetson 设备上的吞吐量。TensorRT 通过层融合、精度校准和内核自动调优提供高性能、低延迟的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)推理。这带来了更快、更高效的执行，对于视频分析和自主机器等实时应用特别有用。

### 我可以在不同的 NVIDIA Jetson 硬件上使用 DeepStream SDK 运行 Ultralytics YOLO11 吗？

是的，使用 DeepStream SDK 和 TensorRT 部署 Ultralytics YOLO11 的指南与整个 NVIDIA Jetson 产品线兼容。这包括使用 [JetPack 5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) 的 Jetson Orin NX 16GB 和使用 [JetPack 4.6.4](https://developer.nvidia.com/jetpack-sdk-464) 的 Jetson Nano 4GB 等设备。有关详细步骤，请参阅 [YOLO11 的 DeepStream 配置](#yolo11-的-deepstream-配置)部分。

### 如何将 YOLO11 模型转换为 ONNX 以用于 DeepStream？

要将 YOLO11 模型转换为 ONNX 格式以便与 DeepStream 一起部署，请使用 [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) 仓库中的 `utils/export_yolo11.py` 脚本。

以下是示例命令：

```bash
python3 utils/export_yolo11.py -w yolo11s.pt --opset 12 --simplify
```

有关模型转换的更多详细信息，请查看我们的[模型导出部分](../modes/export.md)。

### YOLO 在 NVIDIA Jetson Orin NX 上的性能基准是什么？

YOLO11 模型在 NVIDIA Jetson Orin NX 16GB 上的性能因 TensorRT 精度级别而异。例如，YOLO11s 模型实现：

- **FP32 精度**：14.6 ms/im，68.5 FPS
- **FP16 精度**：7.94 ms/im，126 FPS
- **INT8 精度**：5.95 ms/im，168 FPS

这些基准测试强调了在 NVIDIA Jetson 硬件上使用 TensorRT 优化的 YOLO11 模型的效率和能力。有关更多详细信息，请参阅我们的[基准测试结果](#基准测试结果)部分。
