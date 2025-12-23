---
comments: true
description: 学习为 Ultralytics 项目设置 Conda 环境。按照我们的综合指南轻松安装和初始化。
keywords: Ultralytics, Conda, 设置, 安装, 环境, 指南, 机器学习, 数据科学
---

# Ultralytics Conda 快速入门指南

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-conda-package-visual.avif" alt="Ultralytics Conda 包视觉">
</p>

本指南全面介绍如何为 Ultralytics 项目设置 Conda 环境。Conda 是一个开源的包和环境管理系统，为安装包和依赖项提供了 pip 的优秀替代方案。其隔离环境使其特别适合数据科学和[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工作。有关更多详细信息，请访问 [Anaconda](https://anaconda.org/conda-forge/ultralytics) 上的 Ultralytics Conda 包，并查看 [GitHub](https://github.com/conda-forge/ultralytics-feedstock/) 上的 Ultralytics feedstock 仓库以获取包更新。

[![Conda 版本](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics)
[![Conda 下载量](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)
[![Conda 配方](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics)
[![Conda 平台](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

## 您将学到什么

- 设置 Conda 环境
- 通过 Conda 安装 Ultralytics
- 在您的环境中初始化 Ultralytics
- 使用带有 Conda 的 Ultralytics Docker 镜像

---

## 前提条件

- 您的系统上应该已安装 Anaconda 或 Miniconda。如果没有，请从 [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://www.anaconda.com/docs/main) 下载并安装。

---

## 设置 Conda 环境

首先，让我们创建一个新的 Conda 环境。打开终端并运行以下命令：

```bash
conda create --name ultralytics-env python=3.11 -y
```

激活新环境：

```bash
conda activate ultralytics-env
```

---

## 安装 Ultralytics

您可以从 conda-forge 频道安装 Ultralytics 包。执行以下命令：

```bash
conda install -c conda-forge ultralytics
```

### CUDA 环境注意事项

如果您在支持 CUDA 的环境中工作，最好将 `ultralytics`、`pytorch` 和 `pytorch-cuda` 一起安装以解决任何冲突：

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

---

## 使用 Ultralytics

安装 Ultralytics 后，您现在可以开始使用其强大的功能进行[目标检测](https://www.ultralytics.com/glossary/object-detection)、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)等。例如，要对图像进行预测，您可以运行：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # 初始化模型
results = model("path/to/image.jpg")  # 执行推理
results[0].show()  # 显示第一张图像的结果
```

---

## Ultralytics Conda Docker 镜像

如果您更喜欢使用 Docker，Ultralytics 提供包含 Conda 环境的 Docker 镜像。您可以从 [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics) 拉取这些镜像。

拉取最新的 Ultralytics 镜像：

```bash
# 将镜像名称设置为变量
t=ultralytics/ultralytics:latest-conda

# 从 Docker Hub 拉取最新的 Ultralytics 镜像
sudo docker pull $t
```

运行镜像：

```bash
# 在支持 GPU 的容器中运行 Ultralytics 镜像
sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t            # 所有 GPU
sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' $t # 指定 GPU
```

## 使用 Libmamba 加速安装

如果您希望[加速 Conda 中的包安装](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)过程，可以选择使用 `libmamba`，这是一个快速、跨平台、依赖感知的包管理器，作为 Conda 默认求解器的替代方案。

### 如何启用 Libmamba

要启用 `libmamba` 作为 Conda 的求解器，您可以执行以下步骤：

1. 首先，安装 `conda-libmamba-solver` 包。如果您的 Conda 版本是 4.11 或更高版本，可以跳过此步骤，因为 `libmamba` 已默认包含。

    ```bash
    conda install conda-libmamba-solver
    ```

2. 接下来，配置 Conda 使用 `libmamba` 作为求解器：

    ```bash
    conda config --set solver libmamba
    ```

就是这样！您的 Conda 安装现在将使用 `libmamba` 作为求解器，这应该会加快包安装过程。

---

您已成功设置 Conda 环境，安装了 Ultralytics 包，现在可以探索其功能了。有关更高级的教程和示例，请参阅 [Ultralytics 文档](../index.md)。


## 常见问题

### 为 Ultralytics 项目设置 Conda 环境的过程是什么？

为 Ultralytics 项目设置 Conda 环境非常简单，可确保顺畅的包管理。首先，使用以下命令创建一个新的 Conda 环境：

```bash
conda create --name ultralytics-env python=3.11 -y
```

然后，使用以下命令激活新环境：

```bash
conda activate ultralytics-env
```

最后，从 conda-forge 频道安装 Ultralytics：

```bash
conda install -c conda-forge ultralytics
```

### 为什么在 Ultralytics 项目中应该使用 Conda 而不是 pip 来管理依赖项？

Conda 是一个强大的包和环境管理系统，相比 pip 有几个优势。它高效地管理依赖项并确保所有必要的库兼容。Conda 的隔离环境可防止包之间的冲突，这在数据科学和机器学习项目中至关重要。此外，Conda 支持二进制包分发，加快了安装过程。

### 我可以在支持 CUDA 的环境中使用 Ultralytics YOLO 以获得更快的性能吗？

是的，您可以通过使用支持 CUDA 的环境来提高性能。确保将 `ultralytics`、`pytorch` 和 `pytorch-cuda` 一起安装以避免冲突：

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

此设置启用 GPU 加速，这对于[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型训练和推理等密集任务至关重要。有关更多信息，请访问 [Ultralytics 安装指南](../quickstart.md)。

### 使用带有 Conda 环境的 Ultralytics Docker 镜像有什么好处？

使用 Ultralytics Docker 镜像可确保一致且可重现的环境，消除"在我的机器上可以运行"的问题。这些镜像包含预配置的 Conda 环境，简化了设置过程。您可以使用以下命令拉取并运行最新的 Ultralytics Docker 镜像：

```bash
sudo docker pull ultralytics/ultralytics:latest-conda
sudo docker run -it --ipc=host --runtime=nvidia --gpus all ultralytics/ultralytics:latest-conda            # 所有 GPU
sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' ultralytics/ultralytics:latest-conda # 指定 GPU
```

这种方法非常适合在生产环境中部署应用程序或运行复杂的工作流程，无需手动配置。了解更多关于 [Ultralytics Conda Docker 镜像](../quickstart.md)。

### 如何在 Ultralytics 环境中加速 Conda 包安装？

您可以通过使用 `libmamba`（Conda 的快速依赖求解器）来加速包安装过程。首先，安装 `conda-libmamba-solver` 包：

```bash
conda install conda-libmamba-solver
```

然后配置 Conda 使用 `libmamba` 作为求解器：

```bash
conda config --set solver libmamba
```

此设置提供更快、更高效的包管理。有关优化环境的更多技巧，请阅读关于 [libmamba 安装](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)的内容。
