---
comments: true
description: 学习如何在树莓派上部署 Ultralytics YOLO11，包含全面的指南。获取性能基准测试、设置说明和最佳实践。
keywords: Ultralytics, YOLO11, 树莓派, 设置, 指南, 基准测试, 计算机视觉, 目标检测, NCNN, Docker, 摄像头模块
---

# 快速入门指南：树莓派与 Ultralytics YOLO11

本综合指南提供了在[树莓派](https://www.raspberrypi.com/)设备上部署 Ultralytics YOLO11 的详细演练。此外，它还展示了性能基准测试，以展示 YOLO11 在这些小巧而强大的设备上的能力。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yul4gq_LrOI"
    title="树莓派 5 介绍" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>树莓派 5 更新和改进。
</p>

!!! note

    本指南已在运行最新 [Raspberry Pi OS Bookworm (Debian 12)](https://www.raspberrypi.com/software/operating-systems/) 的树莓派 4 和树莓派 5 上进行了测试。只要安装了相同的 Raspberry Pi OS Bookworm，本指南预计也适用于较旧的树莓派设备（如树莓派 3）。

## 什么是树莓派？

树莓派是一款小巧、经济实惠的单板计算机。它已成为各种项目和应用的热门选择，从业余爱好者的家庭自动化到工业用途。树莓派板能够运行各种操作系统，并提供 GPIO（通用输入/输出）引脚，可轻松与传感器、执行器和其他硬件组件集成。它们有不同规格的型号，但都共享相同的基本设计理念：低成本、紧凑和多功能。

## 树莓派系列比较

|                   | 树莓派 3                         | 树莓派 4                         | 树莓派 5                         |
| ----------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| CPU               | Broadcom BCM2837, Cortex-A53 64 位 SoC | Broadcom BCM2711, Cortex-A72 64 位 SoC | Broadcom BCM2712, Cortex-A76 64 位 SoC |
| CPU 最大频率 | 1.4GHz                                 | 1.8GHz                                 | 2.4GHz                                 |
| GPU               | Videocore IV                           | Videocore VI                           | VideoCore VII                          |
| GPU 最大频率 | 400Mhz                                 | 500Mhz                                 | 800Mhz                                 |
| 内存            | 1GB LPDDR2 SDRAM                       | 1GB, 2GB, 4GB, 8GB LPDDR4-3200 SDRAM   | 4GB, 8GB LPDDR4X-4267 SDRAM            |
| PCIe              | 无                                    | 无                                    | 1xPCIe 2.0 接口                   |
| 最大功耗    | 2.5A@5V                                | 3A@5V                                  | 5A@5V (PD 启用)                     |

## 什么是 Raspberry Pi OS？

[Raspberry Pi OS](https://www.raspberrypi.com/software/)（前身为 Raspbian）是一个基于 Debian GNU/Linux 发行版的类 Unix 操作系统，专为树莓派基金会发布的树莓派系列紧凑型单板计算机设计。Raspberry Pi OS 针对带有 ARM CPU 的树莓派进行了高度优化，使用带有 Openbox 堆叠窗口管理器的修改版 LXDE 桌面环境。Raspberry Pi OS 正在积极开发中，重点是提高尽可能多的 Debian 软件包在树莓派上的稳定性和性能。

## 将 Raspberry Pi OS 刷入树莓派

获得树莓派后的第一件事是将 Raspberry Pi OS 刷入 micro-SD 卡，插入设备并启动操作系统。请按照[树莓派官方入门文档](https://www.raspberrypi.com/documentation/computers/getting-started.html)的详细说明准备您的设备以供首次使用。

## 设置 Ultralytics

有两种方法可以在树莓派上设置 Ultralytics 包来构建您的下一个[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目。您可以使用其中任何一种。

- [使用 Docker 开始](#使用-docker-开始)
- [不使用 Docker 开始](#不使用-docker-开始)

### 使用 Docker 开始

在树莓派上开始使用 Ultralytics YOLO11 的最快方法是使用为树莓派预构建的 Docker 镜像运行。

执行以下命令拉取 Docker 容器并在树莓派上运行。这基于 [arm64v8/debian](https://hub.docker.com/r/arm64v8/debian) Docker 镜像，其中包含 Python3 环境中的 Debian 12 (Bookworm)。

```bash
t=ultralytics/ultralytics:latest-arm64
sudo docker pull $t && sudo docker run -it --ipc=host $t
```

完成后，跳转到[在树莓派上使用 NCNN 部分](#在树莓派上使用-ncnn)。

### 不使用 Docker 开始

#### 安装 Ultralytics 包

这里我们将在树莓派上安装带有可选依赖项的 Ultralytics 包，以便我们可以将 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型导出为其他不同格式。

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

## 在树莓派上使用 NCNN

在 Ultralytics 支持的所有模型导出格式中，[NCNN](https://docs.ultralytics.com/integrations/ncnn/) 在树莓派设备上提供最佳的推理性能，因为 NCNN 针对移动/嵌入式平台（如 ARM 架构）进行了高度优化。

## 转换模型为 NCNN 并运行推理

PyTorch 格式的 YOLO11n 模型被转换为 NCNN，以使用导出的模型运行推理。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11n PyTorch 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 NCNN 格式
        model.export(format="ncnn")  # 创建 'yolo11n_ncnn_model'

        # 加载导出的 NCNN 模型
        ncnn_model = YOLO("yolo11n_ncnn_model")

        # 运行推理
        results = ncnn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 NCNN 格式
        yolo export model=yolo11n.pt format=ncnn # 创建 'yolo11n_ncnn_model'

        # 使用导出的模型运行推理
        yolo predict model='yolo11n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

!!! tip

    有关支持的导出选项的更多详细信息，请访问 [Ultralytics 部署选项文档页面](https://docs.ultralytics.com/guides/model-deployment-options/)。

## 树莓派 5 YOLO11 基准测试

YOLO11 基准测试由 Ultralytics 团队在十种不同的模型格式上运行，测量速度和[准确率](https://www.ultralytics.com/glossary/accuracy)：PyTorch、TorchScript、ONNX、OpenVINO、TF SavedModel、TF GraphDef、TF Lite、PaddlePaddle、MNN、NCNN。基准测试在树莓派 5 上以 FP32 [精度](https://www.ultralytics.com/glossary/precision)和默认输入图像尺寸 640 运行。

### 比较图表

我们仅包含了 YOLO11n 和 YOLO11s 模型的基准测试，因为其他模型尺寸太大，无法在树莓派上运行，且无法提供良好的性能。

<figure style="text-align: center;">
    <img width="800" src="https://github.com/ultralytics/assets/releases/download/v0.0.0/rpi-yolo11-benchmarks-coco128.avif" alt="树莓派 5 上的 YOLO11 基准测试">
    <figcaption style="font-style: italic; color: gray;">使用 Ultralytics 8.3.152 进行基准测试</figcaption>
</figure>

### 详细比较表

下表展示了两种不同模型（YOLO11n、YOLO11s）在十种不同格式（PyTorch、TorchScript、ONNX、OpenVINO、TF SavedModel、TF GraphDef、TF Lite、PaddlePaddle、MNN、NCNN）上的基准测试结果，在树莓派 5 上运行，给出了每种组合的状态、大小、mAP50-95(B) 指标和推理时间。

!!! tip "性能"

    === "YOLO11n"

        | 格式        | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |---------------|--------|-------------------|-------------|------------------------|
        | PyTorch       | ✅      | 5.4               | 0.5101      | 387.63                |
        | TorchScript   | ✅      | 10.5              | 0.5077      | 457.84                |
        | ONNX          | ✅      | 10.2              | 0.5077      | 191.09                |
        | OpenVINO      | ✅      | 10.4              | 0.5058      | 84.76                 |
        | TF SavedModel | ✅      | 25.9              | 0.5077      | 306.94                |
        | TF GraphDef   | ✅      | 10.3              | 0.5077      | 309.82                |
        | TF Lite       | ✅      | 10.3              | 0.5077      | 425.77                |
        | PaddlePaddle  | ✅      | 20.5              | 0.5077      | 463.93                |
        | MNN           | ✅      | 10.1              | 0.5059      | 114.97                |
        | NCNN          | ✅      | 10.2              | 0.5031      | 94.03                 |

    === "YOLO11s"

        | 格式        | 状态 | 磁盘大小 (MB) | mAP50-95(B) | 推理时间 (ms/im) |
        |---------------|--------|-------------------|-------------|------------------------|
        | PyTorch       | ✅      | 18.4              | 0.5791      | 962.69                |
        | TorchScript   | ✅      | 36.5              | 0.5782      | 1181.94               |
        | ONNX          | ✅      | 36.3              | 0.5782      | 449.85                |
        | OpenVINO      | ✅      | 36.4              | 0.5810      | 181.53                |
        | TF SavedModel | ✅      | 91.0              | 0.5782      | 660.62                |
        | TF GraphDef   | ✅      | 36.4              | 0.5782      | 669.23                |
        | TF Lite       | ✅      | 36.3              | 0.5782      | 1093.41               |
        | PaddlePaddle  | ✅      | 72.6              | 0.5782      | 1140.61               |
        | MNN           | ✅      | 36.2              | 0.5805      | 274.63                |
        | NCNN          | ✅      | 36.2              | 0.5784      | 224.20                |

    使用 Ultralytics 8.3.152 进行基准测试

    !!! note

        推理时间不包括预处理/后处理。

## 复现我们的结果

要在所有[导出格式](../modes/export.md)上复现上述 Ultralytics 基准测试，请运行以下代码：

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


## 使用树莓派摄像头

在使用树莓派进行计算机视觉项目时，获取实时视频流以执行推理可能是必不可少的。树莓派上的板载 MIPI CSI 连接器允许您连接官方树莓派摄像头模块。在本指南中，我们使用了[树莓派摄像头模块 3](https://www.raspberrypi.com/products/camera-module-3/) 来获取视频流并使用 YOLO11 模型执行推理。

!!! tip

    了解更多关于[树莓派提供的不同摄像头模块](https://www.raspberrypi.com/documentation/accessories/camera.html)以及[如何开始使用树莓派摄像头模块](https://www.raspberrypi.com/documentation/computers/camera_software.html#introducing-the-raspberry-pi-cameras)。

!!! note

    树莓派 5 使用比树莓派 4 更小的 CSI 连接器（15 针 vs 22 针），因此您需要一根 [15 针转 22 针适配器线缆](https://www.raspberrypi.com/products/camera-cable/)来连接树莓派摄像头。

### 测试摄像头

将摄像头连接到树莓派后执行以下命令。您应该会看到来自摄像头的实时视频流，持续约 5 秒。

```bash
rpicam-hello
```

!!! tip

    在[官方树莓派文档](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-hello)上了解更多关于 `rpicam-hello` 的用法

### 使用摄像头进行推理

有两种方法可以使用树莓派摄像头在 YOLO11 模型上运行推理。

!!! usage

    === "方法 1"

        我们可以使用树莓派 OS 预装的 `picamera2` 来访问摄像头并在 YOLO11 模型上运行推理。

        !!! example

            === "Python"

                ```python
                import cv2
                from picamera2 import Picamera2

                from ultralytics import YOLO

                # 初始化 Picamera2
                picam2 = Picamera2()
                picam2.preview_configuration.main.size = (1280, 720)
                picam2.preview_configuration.main.format = "RGB888"
                picam2.preview_configuration.align()
                picam2.configure("preview")
                picam2.start()

                # 加载 YOLO11 模型
                model = YOLO("yolo11n.pt")

                while True:
                    # 逐帧捕获
                    frame = picam2.capture_array()

                    # 在帧上运行 YOLO11 推理
                    results = model(frame)

                    # 在帧上可视化结果
                    annotated_frame = results[0].plot()

                    # 显示结果帧
                    cv2.imshow("Camera", annotated_frame)

                    # 如果按下 'q' 则退出循环
                    if cv2.waitKey(1) == ord("q"):
                        break

                # 释放资源并关闭窗口
                cv2.destroyAllWindows()
                ```

    === "方法 2"

        我们需要使用 `rpicam-vid` 从连接的摄像头启动 TCP 流，以便我们稍后进行推理时可以使用此流 URL 作为输入。执行以下命令启动 TCP 流。

        ```bash
        rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888
        ```

        在[官方树莓派文档](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-vid)上了解更多关于 `rpicam-vid` 的用法

        !!! example

            === "Python"

                ```python
                from ultralytics import YOLO

                # 加载 YOLO11n PyTorch 模型
                model = YOLO("yolo11n.pt")

                # 运行推理
                results = model("tcp://127.0.0.1:8888")
                ```

            === "CLI"

                ```bash
                yolo predict model=yolo11n.pt source="tcp://127.0.0.1:8888"
                ```

!!! tip

    如果您想更改图像/视频输入类型，请查看我们关于[推理源](https://docs.ultralytics.com/modes/predict/#inference-sources)的文档

## 使用树莓派时的最佳实践

为了在运行 YOLO11 的树莓派上实现最大性能，有几个最佳实践可以遵循。

1. 使用 SSD

    当使用树莓派进行 24x7 持续使用时，建议使用 SSD 作为系统存储，因为 SD 卡无法承受持续写入，可能会损坏。借助树莓派 5 上的板载 PCIe 连接器，您现在可以使用适配器（如 [NVMe Base for Raspberry Pi 5](https://shop.pimoroni.com/products/nvme-base)）连接 SSD。

2. 刷入无 GUI 版本

    刷入 Raspberry Pi OS 时，您可以选择不安装桌面环境（Raspberry Pi OS Lite），这可以节省设备上的一些 RAM，为计算机视觉处理留出更多空间。

3. 超频树莓派

    如果您想在树莓派 5 上运行 Ultralytics YOLO11 模型时获得一点性能提升，您可以将 CPU 从基础的 2.4GHz 超频到 2.9GHz，将 GPU 从 800MHz 超频到 1GHz。如果系统变得不稳定或崩溃，请以 100MHz 为增量降低超频值。确保有适当的散热措施，因为超频会增加热量产生，可能导致热节流。

    a. 升级软件

    ```bash
    sudo apt update && sudo apt dist-upgrade
    ```

    b. 打开配置文件进行编辑

    ```bash
    sudo nano /boot/firmware/config.txt
    ```

    c. 在底部添加以下行

    ```bash
    arm_freq=3000
    gpu_freq=1000
    force_turbo=1
    ```

    d. 按 CTRL + X 保存并退出，然后按 Y，再按 ENTER

    e. 重启树莓派

## 后续步骤

您已成功在树莓派上设置了 YOLO。如需进一步学习和支持，请访问 [Ultralytics YOLO11 文档](../index.md)和 [Kashmir World Foundation](https://www.kashmirworldfoundation.org/)。

## 致谢和引用

本指南最初由 Daan Eeltink 为 Kashmir World Foundation 创建，该组织致力于使用 YOLO 保护濒危物种。我们感谢他们在目标检测技术领域的开创性工作和教育重点。

有关 Kashmir World Foundation 活动的更多信息，您可以访问他们的[网站](https://www.kashmirworldfoundation.org/)。

## 常见问题

### 如何在不使用 Docker 的情况下在树莓派上设置 Ultralytics YOLO11？

要在不使用 Docker 的情况下在树莓派上设置 Ultralytics YOLO11，请按照以下步骤操作：

1. 更新软件包列表并安装 `pip`：
    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```
2. 安装带有可选依赖项的 Ultralytics 包：
    ```bash
    pip install ultralytics[export]
    ```
3. 重启设备以应用更改：
    ```bash
    sudo reboot
    ```

有关详细说明，请参阅[不使用 Docker 开始](#不使用-docker-开始)部分。

### 为什么我应该在树莓派上使用 Ultralytics YOLO11 的 NCNN 格式进行 AI 任务？

Ultralytics YOLO11 的 NCNN 格式针对移动和嵌入式平台进行了高度优化，使其非常适合在树莓派设备上运行 AI 任务。NCNN 通过利用 ARM 架构最大化推理性能，与其他格式相比提供更快、更高效的处理。有关支持的导出选项的更多详细信息，请访问 [Ultralytics 部署选项文档页面](https://docs.ultralytics.com/guides/model-deployment-options/)。

### 如何将 YOLO11 模型转换为 NCNN 格式以在树莓派上使用？

您可以使用 Python 或 CLI 命令将 PyTorch YOLO11 模型转换为 NCNN 格式：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 YOLO11n PyTorch 模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为 NCNN 格式
        model.export(format="ncnn")  # 创建 'yolo11n_ncnn_model'

        # 加载导出的 NCNN 模型
        ncnn_model = YOLO("yolo11n_ncnn_model")

        # 运行推理
        results = ncnn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 将 YOLO11n PyTorch 模型导出为 NCNN 格式
        yolo export model=yolo11n.pt format=ncnn # 创建 'yolo11n_ncnn_model'

        # 使用导出的模型运行推理
        yolo predict model='yolo11n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

有关更多详细信息，请参阅[在树莓派上使用 NCNN](#在树莓派上使用-ncnn) 部分。

### 树莓派 4 和树莓派 5 在运行 YOLO11 方面有哪些硬件差异？

主要差异包括：

- **CPU**：树莓派 4 使用 Broadcom BCM2711, Cortex-A72 64 位 SoC，而树莓派 5 使用 Broadcom BCM2712, Cortex-A76 64 位 SoC。
- **最大 CPU 频率**：树莓派 4 的最大频率为 1.8GHz，而树莓派 5 达到 2.4GHz。
- **内存**：树莓派 4 提供最高 8GB 的 LPDDR4-3200 SDRAM，而树莓派 5 配备 LPDDR4X-4267 SDRAM，有 4GB 和 8GB 两种规格。

这些增强使树莓派 5 上的 YOLO11 模型性能基准测试优于树莓派 4。有关更多详细信息，请参阅[树莓派系列比较](#树莓派系列比较)表。

### 如何设置树莓派摄像头模块以与 Ultralytics YOLO11 配合使用？

有两种方法可以设置树莓派摄像头进行 YOLO11 推理：

1. **使用 `picamera2`**：

    ```python
    import cv2
    from picamera2 import Picamera2

    from ultralytics import YOLO

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    model = YOLO("yolo11n.pt")

    while True:
        frame = picam2.capture_array()
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Camera", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    ```

2. **使用 TCP 流**：

    ```bash
    rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888
    ```

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    results = model("tcp://127.0.0.1:8888")
    ```

有关详细设置说明，请访问[使用摄像头进行推理](#使用摄像头进行推理)部分。
