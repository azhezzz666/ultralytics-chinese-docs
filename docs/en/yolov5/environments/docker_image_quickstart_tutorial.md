---
comments: true
description: 学习如何在 Docker 容器中设置和运行 YOLOv5，包含 CPU 和 GPU 环境、挂载卷和使用显示服务器的分步说明。
keywords: YOLOv5, Docker, Ultralytics, 设置, 指南, 教程, 机器学习, 深度学习, AI, GPU, NVIDIA, 容器, X11, Wayland
---

# 在 Docker 中开始使用 YOLOv5 🚀

欢迎阅读 Ultralytics YOLOv5 Docker 快速入门指南！本教程提供在 [Docker](https://www.ultralytics.com/glossary/docker) 容器中设置和运行 [YOLOv5](../../models/yolov5.md) 的分步说明。使用 Docker 使您能够在隔离、一致的环境中运行 YOLOv5，简化跨不同系统的部署和依赖管理。这种方法利用[容器化](https://www.ultralytics.com/glossary/containerization)将应用程序及其依赖项打包在一起。

有关其他设置方法，请考虑我们的 [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>、[GCP 深度学习虚拟机](./google_cloud_quickstart_tutorial.md)或 [Amazon AWS](./aws_quickstart_tutorial.md) 指南。有关 Ultralytics 模型 Docker 使用的一般概述，请参阅 [Ultralytics Docker 快速入门指南](../../guides/docker-quickstart.md)。

## 先决条件

在开始之前，请确保已安装以下内容：

1.  **Docker**：从[官方 Docker 网站](https://docs.docker.com/get-started/get-docker/)下载并安装 Docker。Docker 对于创建和管理容器至关重要。
2.  **NVIDIA 驱动程序**（[GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) 支持所需）：确保已安装 NVIDIA 驱动程序版本 455.23 或更高版本。您可以从 [NVIDIA 网站](https://www.nvidia.com/Download/index.aspx)下载最新驱动程序。
3.  **NVIDIA Container Toolkit**（GPU 支持所需）：此工具包允许 Docker 容器访问主机的 NVIDIA GPU。按照官方 [NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)获取详细说明。

### 设置 NVIDIA Container Toolkit（GPU 用户）

首先，通过运行以下命令验证您的 NVIDIA 驱动程序是否正确安装：

```bash
nvidia-smi
```

此命令应显示有关您的 GPU 和已安装驱动程序版本的信息。

接下来，安装 NVIDIA Container Toolkit。以下命令适用于基于 Debian 的系统（如 Ubuntu）和基于 RHEL 的系统（如 Fedora/CentOS），但请参阅上面链接的官方指南以获取特定于您发行版的说明：

=== "Debian/Ubuntu"

    ```bash
    # 添加 NVIDIA 仓库
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # 安装工具包
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    # 配置 Docker 运行时
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

=== "RHEL/Fedora/CentOS"

    ```bash
    # 添加 NVIDIA 仓库
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
      sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

    # 安装工具包
    sudo yum install -y nvidia-container-toolkit

    # 配置 Docker 运行时
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

### 验证 Docker 的 NVIDIA 运行时

运行 `docker info | grep -i runtime` 以确保 `nvidia` 出现在运行时列表中：

```bash
docker info | grep -i runtime
```

您应该看到 `nvidia` 被列为可用运行时之一。

## 步骤 1：拉取 YOLOv5 Docker 镜像

Ultralytics 在 Docker Hub 上提供官方 YOLOv5 镜像。`latest` 标签跟踪最新的仓库提交，确保您始终获得最新版本。使用以下命令拉取镜像：

```bash
sudo docker pull ultralytics/yolov5:latest
```

您可以在 [Ultralytics YOLOv5 Docker Hub 仓库](https://hub.docker.com/r/ultralytics/yolov5)浏览所有可用镜像。

## 步骤 2：运行 Docker 容器

镜像拉取完成后，您可以将其作为容器运行。

### 仅使用 CPU

要仅使用 CPU 运行交互式容器实例，请使用 `-it` 标志。`--ipc=host` 标志允许共享主机 IPC 命名空间，这对于共享内存访问很重要。

```bash
sudo docker run -it --ipc=host ultralytics/yolov5:latest
```

### 使用 GPU

要在容器内启用 GPU 访问，请使用 `--gpus` 标志。这需要正确安装 NVIDIA Container Toolkit。

```bash
sudo docker run -it --ipc=host --gpus all ultralytics/yolov5:latest
```

有关命令选项的更多详细信息，请参阅 [Docker run 参考文档](https://docs.docker.com/reference/cli/docker/container/run/)。

### 挂载本地目录

要在容器内使用本地文件（数据集、模型权重等），请使用 `-v` 标志将主机目录挂载到容器中：

```bash
sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container ultralytics/yolov5:latest
```

将 `/path/on/host` 替换为您机器上的实际路径，将 `/path/in/container` 替换为 Docker 容器内的所需路径（例如 `/usr/src/datasets`）。

## 步骤 3：在 Docker 容器中使用 YOLOv5 🚀

您现在已经在运行的 YOLOv5 Docker 容器中了！从这里，您可以执行标准的 YOLOv5 命令来完成各种[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)任务，如[目标检测](https://www.ultralytics.com/glossary/object-detection)。

```bash
# 训练模型
python train.py

# 验证模型
python val.py --weights yolov5s.pt

# 运行推理
python detect.py --weights yolov5s.pt --source path/to/images

# 导出模型
python export.py --weights yolov5s.pt --include onnx
```

浏览文档以了解不同模式的详细用法：

- [训练](../tutorials/train_custom_data.md)
- [验证](https://github.com/ultralytics/yolov5/blob/master/val.py)
- [预测](https://github.com/ultralytics/yolov5/blob/master/detect.py)
- [导出](https://github.com/ultralytics/yolov5/blob/master/export.py)

了解更多关于评估指标的信息，如[精确率](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和 [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)。了解不同的导出格式，如 [ONNX](../../integrations/onnx.md)、[CoreML](../../integrations/coreml.md) 和 [TFLite](../../integrations/tflite.md)，并探索各种[模型部署选项](../../guides/model-deployment-options.md)。记得有效管理您的模型权重。

恭喜！您已成功在 Docker 容器中设置并运行 YOLOv5。
