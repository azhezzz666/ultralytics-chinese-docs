---
comments: true
description: 掌握在 Google Cloud Platform 深度学习虚拟机上部署 Ultralytics YOLOv5。非常适合 AI 初学者和专家实现高性能目标检测。
keywords: YOLOv5, Google Cloud Platform, GCP, 深度学习虚拟机, 目标检测, AI, 机器学习, 教程, 云计算, GPU 加速, Ultralytics
---

# 掌握在 Google Cloud Platform (GCP) 深度学习虚拟机上部署 YOLOv5

踏上[人工智能 (AI)](https://www.ultralytics.com/glossary/artificial-intelligence-ai) 和[机器学习 (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) 的旅程可能令人兴奋，尤其是当您利用[云计算](https://www.ultralytics.com/glossary/cloud-computing)平台的强大功能和灵活性时。Google Cloud Platform (GCP) 提供了为 ML 爱好者和专业人士量身定制的强大工具。其中一个工具是深度学习虚拟机，预配置用于数据科学和 ML 任务。在本教程中，我们将介绍在 [GCP 深度学习虚拟机](https://docs.cloud.google.com/deep-learning-vm/docs)上设置 [Ultralytics YOLOv5](../../models/yolov5.md) 的过程。无论您是 ML 新手还是经验丰富的从业者，本指南都提供了实现由 YOLOv5 驱动的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型的清晰路径。

🆓 另外，如果您是 GCP 新用户，您很幸运，可以获得 [$300 免费额度优惠](https://cloud.google.com/free/docs/free-cloud-features#free-trial)来启动您的项目。

除了 GCP，还可以探索 YOLOv5 的其他便捷快速入门选项，如我们的 [Google Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"> 提供基于浏览器的体验，或可扩展的 [Amazon AWS](./aws_quickstart_tutorial.md)。此外，容器爱好者可以使用我们在 [Docker Hub](https://hub.docker.com/r/ultralytics/yolov5) 上提供的官方 Docker 镜像 <img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker 拉取次数"> 获得封装环境，请参阅我们的 [Docker 快速入门指南](../../guides/docker-quickstart.md)。

## 步骤 1：创建和配置您的深度学习虚拟机

让我们首先创建一个针对[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)优化的虚拟机：

1.  导航到 [GCP 市场](https://cloud.google.com/marketplace)并选择 **深度学习虚拟机**。
2.  选择 **n1-standard-8** 实例；它提供 8 个 vCPU 和 30 GB 内存的平衡配置，适合许多 ML 任务。
3.  选择一个 [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit)。选择取决于您的工作负载；即使是基本的 T4 GPU 也会显著加速模型训练。
4.  勾选"首次启动时自动安装 NVIDIA GPU 驱动程序？"以实现无缝设置。
5.  分配 300 GB SSD 持久磁盘以防止 I/O 瓶颈。
6.  点击"部署"并允许 GCP 配置您的自定义深度学习虚拟机。

此虚拟机预装了基本工具和框架，包括 [Anaconda](https://www.anaconda.com/) Python 发行版，它方便地捆绑了 YOLOv5 所需的许多依赖项。

![GCP 市场设置深度学习虚拟机的说明](https://github.com/ultralytics/docs/releases/download/0/gcp-deep-learning-vm-setup.avif)

## 步骤 2：为 YOLOv5 准备虚拟机

设置环境后，让我们安装并准备好 YOLOv5：

```bash
# 克隆 YOLOv5 仓库
git clone https://github.com/ultralytics/yolov5
cd yolov5

# 安装依赖
pip install -r requirements.txt
```

此设置过程确保您拥有 Python 3.8.0 或更新版本的环境以及 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 1.8 或更高版本。我们的脚本会自动从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)下载[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)，简化了开始模型训练的过程。

## 步骤 3：在 GCP 虚拟机上训练和部署您的 YOLOv5 模型

设置完成后，您可以在 GCP 虚拟机上使用 YOLOv5 进行[训练](../../modes/train.md)、[验证](../../modes/val.md)、[预测](../../modes/predict.md)和[导出](../../modes/export.md)：

```bash
# 在您的数据集上训练 YOLOv5 模型（例如 yolov5s）
python train.py --data coco128.yaml --weights yolov5s.pt --img 640

# 验证训练模型以检查精确率、召回率和 mAP
python val.py --weights yolov5s.pt --data coco128.yaml

# 使用训练模型对图像或视频运行推理
python detect.py --weights yolov5s.pt --source path/to/your/images_or_videos

# 将训练模型导出为各种格式，如 ONNX、CoreML、TFLite 以进行部署
python export.py --weights yolov5s.pt --include onnx coreml tflite
```

只需几个命令，YOLOv5 就能让您训练针对特定需求定制的自定义[目标检测](https://docs.ultralytics.com/tasks/detect/)模型，或使用预训练权重在各种任务中快速获得结果。导出后探索不同的[模型部署选项](../../guides/model-deployment-options.md)。

![终端命令图像说明在 GCP 深度学习虚拟机上进行模型训练](https://github.com/ultralytics/docs/releases/download/0/terminal-command-model-training.avif)

## 分配交换空间（可选）

如果您正在处理可能超出虚拟机 RAM 的特别大的数据集，请考虑添加交换空间以防止内存错误：

```bash
# 分配 64GB 交换文件
sudo fallocate -l 64G /swapfile

# 设置交换文件的正确权限
sudo chmod 600 /swapfile

# 设置 Linux 交换区域
sudo mkswap /swapfile

# 启用交换文件
sudo swapon /swapfile

# 验证交换空间分配（应显示增加的交换内存）
free -h
```

## 训练自定义数据集

要在 GCP 中使用您的自定义数据集训练 YOLOv5，请按照以下一般步骤操作：

1.  根据 YOLOv5 格式准备您的数据集（图像和相应的标签文件）。有关指导，请参阅我们的[数据集概述](../../datasets/index.md)。
2.  使用 `gcloud compute scp` 或 Web 控制台的 SSH 功能将数据集上传到您的 GCP 虚拟机。
3.  创建一个数据集配置 YAML 文件（`custom_dataset.yaml`），指定训练和验证数据的路径、类别数量和类别名称。
4.  使用您的自定义数据集 YAML 开始[训练过程](../../modes/train.md)，可能从预训练权重开始：

    ```bash
    # 示例：在自定义数据集上训练 YOLOv5s 100 个训练周期
    python train.py --img 640 --batch 16 --epochs 100 --data custom_dataset.yaml --weights yolov5s.pt
    ```

有关准备数据和使用自定义数据集训练的全面说明，请参阅 [Ultralytics YOLOv5 训练文档](../../modes/train.md)。

## 利用云存储

为了高效的数据管理，特别是对于大型数据集或大量实验，将您的 YOLOv5 工作流与 [Google Cloud Storage](https://cloud.google.com/storage) 集成：

```bash
# 确保已安装并初始化 Google Cloud SDK
# 如果未安装：curl https://sdk.cloud.google.com/ | bash
# 然后初始化：gcloud init

# 示例：将数据集从 GCS 存储桶复制到您的虚拟机
gsutil cp -r gs://your-data-bucket/my_dataset ./datasets/

# 示例：将训练模型权重从您的虚拟机复制到 GCS 存储桶
gsutil cp -r ./runs/train/exp/weights gs://your-models-bucket/yolov5_custom_weights/
```

这种方法允许您在云中安全且经济高效地存储大型数据集和训练模型，最大限度地减少虚拟机实例上的存储需求。

## 总结

恭喜！您现在已具备利用 Ultralytics YOLOv5 结合 Google Cloud Platform 计算能力的能力。此设置为您的目标检测项目提供了可扩展性、效率和多功能性。无论是个人探索、学术研究还是构建工业[解决方案](../../solutions/index.md)，您都已迈出了进入云端 AI 和 ML 世界的重要一步。

考虑使用 [Ultralytics HUB](../../hub/index.md) 获得简化的无代码体验来训练和管理您的模型。

记得记录您的进展，与充满活力的 Ultralytics 社区分享见解，并利用 [GitHub 讨论](https://github.com/ultralytics/yolov5/discussions)等资源进行协作和支持。现在，继续使用 YOLOv5 和 GCP 进行创新吧！

想继续提升您的 ML 技能吗？深入阅读我们的[文档](../../quickstart.md)并探索 [Ultralytics 博客](https://www.ultralytics.com/blog)获取更多教程和见解。让您的 AI 冒险继续！
