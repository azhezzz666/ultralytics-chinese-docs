---
comments: true
description: 学习如何轻松设置 Ultralytics Docker 环境，从安装到使用 CPU/GPU 支持运行。按照我们的综合指南获得无缝的容器体验。
keywords: Ultralytics, Docker, 快速入门指南, CPU 支持, GPU 支持, NVIDIA Docker, NVIDIA Container Toolkit, 容器设置, Docker 环境, Docker Hub, Ultralytics 项目
---

# Ultralytics Docker 快速入门指南

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-docker-package-visual.avif" alt="Ultralytics Docker 包视觉">
</p>

本指南全面介绍如何为 Ultralytics 项目设置 Docker 环境。[Docker](https://www.docker.com/) 是一个用于在容器中开发、交付和运行应用程序的平台。它特别有助于确保软件无论部署在哪里都能始终以相同方式运行。有关更多详细信息，请访问 [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 上的 Ultralytics Docker 仓库。

[![Docker 镜像版本](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)
[![Docker 拉取次数](https://img.shields.io/docker/pulls/ultralytics/ultralytics)](https://hub.docker.com/r/ultralytics/ultralytics)

## 您将学到什么

- 设置带有 NVIDIA 支持的 Docker
- 安装 Ultralytics Docker 镜像
- 在 Docker 容器中使用 CPU 或 GPU 支持运行 Ultralytics
- 使用显示服务器与 Docker 显示 Ultralytics 检测结果
- 将本地目录挂载到容器中

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/IYWQZvtOy_Q"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何开始使用 Docker | 在 Docker 中使用 Ultralytics Python 包的实时演示 🎉
</p>

---

## 前提条件

- 确保您的系统上已安装 Docker。如果没有，可以从 [Docker 官网](https://www.docker.com/products/docker-desktop/)下载并安装。
- 确保您的系统有 NVIDIA GPU 并已安装 NVIDIA 驱动程序。
- 如果您使用 NVIDIA Jetson 设备，请确保已安装适当的 JetPack 版本。有关更多详细信息，请参阅 [NVIDIA Jetson 指南](https://docs.ultralytics.com/guides/nvidia-jetson/)。

---

## 设置带有 NVIDIA 支持的 Docker

首先，通过运行以下命令验证 NVIDIA 驱动程序是否正确安装：

```bash
nvidia-smi
```

### 安装 NVIDIA Container Toolkit

现在，让我们安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) 以在 Docker 容器中启用 GPU 支持：

=== "Ubuntu/Debian"

    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```
    更新包列表并安装 nvidia-container-toolkit 包：

    ```bash
    sudo apt-get update
    ```

    安装最新版本的 `nvidia-container-toolkit`：

    ```bash
    sudo apt-get install -y nvidia-container-toolkit \
      nvidia-container-toolkit-base libnvidia-container-tools \
      libnvidia-container1
    ```

    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

### 验证 Docker 的 NVIDIA 运行时

运行 `docker info | grep -i runtime` 确保 `nvidia` 出现在运行时列表中：

```bash
docker info | grep -i runtime
```

---

## 安装 Ultralytics Docker 镜像

Ultralytics 提供多个针对不同平台和用例优化的 Docker 镜像：

- **Dockerfile**：GPU 镜像，适合训练。
- **Dockerfile-arm64**：用于 ARM64 架构，适合[树莓派](raspberry-pi.md)等设备。
- **Dockerfile-cpu**：仅 CPU 版本，用于推理和非 GPU 环境。
- **Dockerfile-jetson-jetpack4**：针对运行 [NVIDIA JetPack 4](https://developer.nvidia.com/embedded/jetpack-sdk-461) 的 [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) 设备优化。
- **Dockerfile-jetson-jetpack5**：针对运行 [NVIDIA JetPack 5](https://developer.nvidia.com/embedded/jetpack-sdk-512) 的 NVIDIA Jetson 设备优化。
- **Dockerfile-jetson-jetpack6**：针对运行 [NVIDIA JetPack 6](https://developer.nvidia.com/embedded/jetpack-sdk-61) 的 NVIDIA Jetson 设备优化。
- **Dockerfile-jupyter**：用于在浏览器中使用 JupyterLab 进行交互式开发。
- **Dockerfile-python**：用于轻量级应用的最小 Python 环境。
- **Dockerfile-conda**：包含 [Miniconda3](https://www.anaconda.com/docs/main) 和通过 Conda 安装的 Ultralytics 包。

拉取最新镜像：

```bash
# 将镜像名称设置为变量
t=ultralytics/ultralytics:latest

# 从 Docker Hub 拉取最新的 Ultralytics 镜像
sudo docker pull $t
```

---

## 在 Docker 容器中运行 Ultralytics

以下是执行 Ultralytics Docker 容器的方法：

### 仅使用 CPU

```bash
# 不使用 GPU 运行
sudo docker run -it --ipc=host $t
```

### 使用 GPU

```bash
# 使用所有 GPU 运行
sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t

# 指定使用哪些 GPU 运行
sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' $t
```

`-it` 标志分配一个伪 TTY 并保持 stdin 打开，允许您与容器交互。`--ipc=host` 标志启用共享主机的 IPC 命名空间，这对于进程间共享内存至关重要。`--gpus` 标志允许容器访问主机的 GPU。

### 关于文件可访问性的说明

要在容器内使用本地机器上的文件，可以使用 Docker 卷：

```bash
# 将本地目录挂载到容器中
sudo docker run -it --ipc=host --runtime=nvidia --gpus all -v /path/on/host:/path/in/container $t
```

将 `/path/on/host` 替换为本地机器上的目录路径，将 `/path/in/container` 替换为 Docker 容器内所需的路径。


## 在 Docker 容器中运行图形用户界面（GUI）应用程序

!!! danger "高度实验性 - 用户自行承担所有风险"

    以下说明是实验性的。与 Docker 容器共享 X11 套接字存在潜在的安全风险。因此，建议仅在受控环境中测试此解决方案。有关更多信息，请参阅这些关于如何使用 `xhost` 的资源<sup>[(1)](http://users.stat.umn.edu/~geyer/secure.html)[(2)](https://linux.die.net/man/1/xhost)</sup>。

Docker 主要用于容器化后台应用程序和 CLI 程序，但它也可以运行图形程序。在 Linux 世界中，两个主要的图形服务器处理图形显示：[X11](https://www.x.org/wiki/)（也称为 X Window System）和 [Wayland](https://en.wikipedia.org/wiki/Wayland_(protocol))。在开始之前，必须确定您当前使用的是哪个图形服务器。运行此命令以找出：

```bash
env | grep -E -i 'x11|xorg|wayland'
```

X11 或 Wayland 显示服务器的设置和配置超出了本指南的范围。如果上述命令没有返回任何内容，则您需要先为系统设置其中一个才能继续。

### 使用 GUI 运行 Docker 容器

!!! example

    === "X11"

        如果您使用 X11，可以运行以下命令允许 Docker 容器访问 X11 套接字：

        ```bash
        xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
          -v /tmp/.X11-unix:/tmp/.X11-unix \
          -v ~/.Xauthority:/root/.Xauthority \
          -it --ipc=host $t
        ```

    === "Wayland"

        对于 Wayland，使用以下命令：

        ```bash
        xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
          -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
          --net=host -it --ipc=host $t
        ```

### 在 Docker 中使用 GUI

现在您可以在 Docker 容器内显示图形应用程序。例如，您可以运行以下 [CLI 命令](../usage/cli.md)来可视化 [YOLO11 模型](../models/yolo11.md)的[预测](../modes/predict.md)结果：

```bash
yolo predict model=yolo11n.pt show=True
```

### 完成 Docker GUI 后

!!! warning "撤销访问权限"

    在两种情况下，完成后不要忘记撤销 Docker 组的访问权限。

    ```bash
    xhost -local:docker
    ```

---

您现在已设置好使用 Docker 的 Ultralytics，并准备好利用其功能。有关替代安装方法，请参阅 [Ultralytics 快速入门文档](../quickstart.md)。

## 常见问题

### 如何使用 Docker 设置 Ultralytics？

要使用 Docker 设置 Ultralytics，首先确保您的系统上已安装 Docker。如果您有 NVIDIA GPU，请安装 [NVIDIA Container Toolkit](#安装-nvidia-container-toolkit) 以启用 GPU 支持。然后，使用以下命令从 Docker Hub 拉取最新的 Ultralytics Docker 镜像：

```bash
sudo docker pull ultralytics/ultralytics:latest
```

有关详细步骤，请参阅我们的 Docker 快速入门指南。

### 使用 Ultralytics Docker 镜像进行机器学习项目有什么好处？

使用 Ultralytics Docker 镜像可确保在不同机器上保持一致的环境，复制相同的软件和依赖项。这对于[跨团队协作](https://www.ultralytics.com/blog/how-ultralytics-integration-can-enhance-your-workflow)、在各种硬件上运行模型以及保持可重现性特别有用。对于基于 GPU 的训练，Ultralytics 提供优化的 Docker 镜像，如用于一般 GPU 使用的 `Dockerfile` 和用于 NVIDIA Jetson 设备的 `Dockerfile-jetson`。探索 [Ultralytics Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 了解更多详情。

### 如何在支持 GPU 的 Docker 容器中运行 Ultralytics YOLO？

首先，确保已安装并配置 [NVIDIA Container Toolkit](#安装-nvidia-container-toolkit)。然后，使用以下命令运行支持 GPU 的 Ultralytics YOLO：

```bash
sudo docker run -it --ipc=host --runtime=nvidia --gpus all ultralytics/ultralytics:latest # 所有 GPU
```

此命令设置一个具有 GPU 访问权限的 Docker 容器。有关更多详细信息，请参阅 Docker 快速入门指南。

### 如何在带有显示服务器的 Docker 容器中可视化 YOLO 预测结果？

要在 Docker 容器中使用 GUI 可视化 YOLO 预测结果，您需要允许 Docker 访问您的显示服务器。对于运行 X11 的系统，命令是：

```bash
xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/.Xauthority:/root/.Xauthority \
  -it --ipc=host ultralytics/ultralytics:latest
```

对于运行 Wayland 的系统，使用：

```bash
xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
  -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
  --net=host -it --ipc=host ultralytics/ultralytics:latest
```

更多信息可在[在 Docker 容器中运行图形用户界面（GUI）应用程序](#在-docker-容器中运行图形用户界面gui应用程序)部分找到。

### 我可以将本地目录挂载到 Ultralytics Docker 容器中吗？

是的，您可以使用 `-v` 标志将本地目录挂载到 Ultralytics Docker 容器中：

```bash
sudo docker run -it --ipc=host --runtime=nvidia --gpus all -v /path/on/host:/path/in/container ultralytics/ultralytics:latest
```

将 `/path/on/host` 替换为本地机器上的目录，将 `/path/in/container` 替换为容器内所需的路径。此设置允许您在容器内使用本地文件。有关更多信息，请参阅[关于文件可访问性的说明](#关于文件可访问性的说明)部分。
