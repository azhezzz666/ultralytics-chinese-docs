---
comments: true
description: 了解如何在 AWS 深度学习实例上设置和运行 Ultralytics YOLOv5。按照我们的综合指南快速且经济高效地开始。
keywords: YOLOv5, AWS, 深度学习, 机器学习, AWS EC2, YOLOv5 设置, 深度学习实例, AI, 目标检测, Ultralytics
---

# Ultralytics YOLOv5 🚀 在 AWS 深度学习实例上：完整指南

设置高性能[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)环境可能看起来令人生畏，尤其是对于新手。但不用担心！🛠️ 本指南提供了在 AWS 深度学习实例上启动和运行 [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) 的分步演练。通过利用 Amazon Web Services (AWS) 的强大功能，即使是[机器学习 (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) 新手也可以快速且经济高效地开始。AWS 平台的[可扩展性](https://www.ultralytics.com/glossary/scalability)使其非常适合实验和生产[部署](https://docs.ultralytics.com/guides/model-deployment-options/)。

YOLOv5 的其他快速入门选项包括我们的 [Google Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>、[Kaggle 环境](https://www.kaggle.com/models/ultralytics/yolov5) <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>、[GCP 深度学习虚拟机](./google_cloud_quickstart_tutorial.md)，以及我们在 [Docker Hub](https://hub.docker.com/r/ultralytics/yolov5) 上提供的预构建 Docker 镜像 <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker 拉取次数"></a>。

## 步骤 1：AWS 控制台登录

首先创建账户或登录 [AWS 管理控制台](https://aws.amazon.com/console/)。登录后，导航到 **EC2** 服务仪表板，您可以在那里管理您的虚拟服务器（实例）。

![AWS 控制台登录](https://github.com/ultralytics/docs/releases/download/0/aws-console-sign-in.avif)

## 步骤 2：启动您的实例

从 EC2 仪表板，点击 **启动实例** 按钮。这将启动创建根据您需求定制的新虚拟服务器的过程。

![启动实例按钮](https://github.com/ultralytics/docs/releases/download/0/launch-instance-button.avif)

### 选择正确的 Amazon 机器映像 (AMI)

选择正确的 AMI 至关重要。这决定了您实例的操作系统和预装软件。在搜索栏中输入"[深度学习](https://aws.amazon.com/ai/machine-learning/amis/)"并选择最新的基于 Ubuntu 的深度学习 AMI（除非您对不同操作系统有特定要求）。Amazon 的深度学习 AMI 预配置了流行的[深度学习框架](https://aws.amazon.com/ai/machine-learning/amis/#Frameworks_and_Interface)（如 YOLOv5 使用的 [PyTorch](https://pytorch.org/)）和必要的 [GPU 驱动程序](https://developer.nvidia.com/cuda-downloads)，大大简化了设置过程。

![选择 AMI](https://github.com/ultralytics/docs/releases/download/0/choose-ami.avif)

### 选择实例类型

对于训练深度学习模型等要求较高的任务，强烈建议选择 GPU 加速实例类型。与 CPU 相比，GPU 可以显著减少模型训练所需的时间。选择实例大小时，确保其内存容量（RAM）足以满足您的模型和数据集需求。

**注意：** 模型和数据集的大小是关键因素。如果您的 ML 任务需要的内存超过所选实例提供的内存，您需要选择更大的实例类型以避免性能问题或错误。

在 [EC2 实例类型页面](https://aws.amazon.com/ec2/instance-types/)上探索可用的 GPU 实例类型，特别是 **加速计算** 类别。

![选择实例类型](https://github.com/ultralytics/docs/releases/download/0/choose-instance-type.avif)

有关监控和优化 GPU 使用的详细信息，请参阅 AWS 关于 [GPU 监控和优化](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-gpu.html)的指南。使用[按需定价](https://aws.amazon.com/ec2/pricing/on-demand/)比较成本，并探索[竞价实例定价](https://aws.amazon.com/ec2/spot/pricing/)的潜在节省。

### 配置您的实例

考虑使用 Amazon EC2 竞价实例以获得更经济高效的方法。竞价实例允许您竞标未使用的 EC2 容量，通常比按需价格有显著折扣。对于需要持久性（即使竞价实例被中断也能保存数据）的任务，选择 **持久请求**。这确保您的存储卷持久保存。

![竞价请求配置](https://github.com/ultralytics/docs/releases/download/0/spot-request.avif)

继续完成实例启动向导的步骤 4-7，配置存储、添加标签、设置安全组（确保从您的 IP 打开 SSH 端口 22），并在点击 **启动** 之前查看您的设置。您还需要创建或选择现有的密钥对以进行安全的 SSH 访问。

## 步骤 3：连接到您的实例

一旦您的实例状态显示为"运行中"，从 EC2 仪表板选择它。点击 **连接** 按钮查看连接选项。在本地终端（如 macOS/Linux 上的 Terminal 或 Windows 上的 PuTTY/WSL）中使用提供的 SSH 命令示例建立安全连接。您需要在启动期间创建或选择的私钥文件（`.pem`）。

![连接到实例](https://github.com/ultralytics/docs/releases/download/0/connect-to-instance.avif)

## 步骤 4：运行 Ultralytics YOLOv5

现在您已通过 SSH 连接，可以设置和运行 YOLOv5。首先，从 [GitHub](https://github.com/ultralytics/yolov5) 克隆官方 YOLOv5 仓库并进入目录。然后，使用 `pip` 安装所需的依赖项。建议使用 [Python](https://www.python.org/) 3.8 或更高版本的环境。当您运行训练或检测等命令时，必要的模型和数据集将自动从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)下载。

```bash
# 克隆 YOLOv5 仓库
git clone https://github.com/ultralytics/yolov5
cd yolov5

# 安装所需包
pip install -r requirements.txt
```

环境准备好后，您可以开始使用 YOLOv5 执行各种任务：

```bash
# 在自定义数据集上训练 YOLOv5 模型（例如 coco128.yaml）
python train.py --data coco128.yaml --weights yolov5s.pt --img 640

# 验证训练模型的性能（精确率、召回率、mAP）（例如 yolov5s.pt）
python val.py --weights yolov5s.pt --data coco128.yaml --img 640

# 使用训练模型对图像或视频运行推理（目标检测）
python detect.py --weights yolov5s.pt --source path/to/your/images_or_videos/ --img 640

# 将训练模型导出为各种格式，如 ONNX、CoreML、TFLite 以进行部署
# 有关更多详情，请参阅 https://docs.ultralytics.com/modes/export/
python export.py --weights yolov5s.pt --include onnx coreml tflite --img 640
```

有关详细指南，请参阅 Ultralytics 文档中的[训练](https://docs.ultralytics.com/modes/train/)、[验证](https://docs.ultralytics.com/modes/val/)、[预测（推理）](https://docs.ultralytics.com/modes/predict/)和[导出](https://docs.ultralytics.com/modes/export/)。

## 可选附加功能：增加交换内存

如果您正在处理非常大的数据集或在训练期间遇到内存限制，增加实例上的交换内存有时会有所帮助。交换空间允许系统使用磁盘空间作为虚拟 RAM。

```bash
# 分配 64GB 交换文件（根据需要调整大小）
sudo fallocate -l 64G /swapfile

# 设置正确的权限
sudo chmod 600 /swapfile

# 将文件设置为 Linux 交换区域
sudo mkswap /swapfile

# 启用交换文件
sudo swapon /swapfile

# 验证交换内存是否激活
free -h
```

恭喜！🎉 您已成功设置 AWS 深度学习实例，安装了 Ultralytics YOLOv5，并准备好执行[目标检测](https://www.ultralytics.com/glossary/object-detection)任务。无论您是使用预训练模型进行实验还是在自己的数据上进行[训练](https://docs.ultralytics.com/modes/train/)，这个强大的设置为您的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目提供了可扩展的基础。如果您遇到任何问题，请查阅广泛的 [AWS 文档](https://docs.aws.amazon.com/)和有用的 Ultralytics 社区资源，如[常见问题](https://docs.ultralytics.com/help/FAQ/)。祝检测愉快！
