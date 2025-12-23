---
comments: true
description: 探索全面的 Ultralytics YOLOv5 文档，包含训练、部署和模型优化的分步教程。立即为您的视觉项目赋能！
keywords: YOLOv5, Ultralytics, 目标检测, 计算机视觉, 深度学习, AI, 教程, PyTorch, 模型优化, 机器学习, 神经网络, YOLOv5 教程
---

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/yolo" target="_blank">
      <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov5-splash.avif" alt="Ultralytics YOLOv5 v7.0 横幅">
    </a>
  </p>

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>
<a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 引用"></a>
<a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker 拉取次数"></a>
<br>
<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="在 Gradient 上运行"></a>
<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>
<a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>

</div>

# Ultralytics YOLOv5 综合指南

欢迎阅读 Ultralytics [YOLOv5](https://github.com/ultralytics/yolov5)🚀 文档！Ultralytics YOLOv5 是革命性的"You Only Look Once"[目标检测](https://www.ultralytics.com/glossary/object-detection)模型的第五代版本，旨在实时提供高速、高精度的结果。虽然 YOLOv5 仍然是一个强大的工具，但建议探索其继任者 [Ultralytics YOLOv8](../models/yolov8.md) 以获取最新进展。

基于 [PyTorch](https://pytorch.org/) 构建，这个强大的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)框架因其多功能性、易用性和高性能而广受欢迎。我们的文档将指导您完成安装过程，解释模型的架构细节，展示各种用例，并提供一系列详细教程。这些资源将帮助您充分发挥 YOLOv5 在[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目中的潜力。让我们开始吧！

## 探索与学习

以下是一系列全面的教程，将指导您了解 YOLOv5 的不同方面。

- [训练自定义数据](tutorials/train_custom_data.md) 🚀 推荐：学习如何在自定义数据集上训练 YOLOv5 模型。
- [最佳训练结果技巧](tutorials/tips_for_best_training_results.md) ☘️：发现优化模型训练过程的实用技巧。
- [多 GPU 训练](tutorials/multi_gpu_training.md)：了解如何利用多个 GPU 加速训练。
- [PyTorch Hub](tutorials/pytorch_hub_model_loading.md) 🌟 新功能：学习通过 PyTorch Hub 加载预训练模型。
- [TFLite、ONNX、CoreML、TensorRT 导出](tutorials/model_export.md) 🚀：了解如何将模型导出为不同格式。
- [测试时增强（TTA）](tutorials/test_time_augmentation.md)：探索如何使用 TTA 提高模型预测精度。
- [模型集成](tutorials/model_ensembling.md)：学习组合多个模型以提高性能的策略。
- [模型剪枝/稀疏化](tutorials/model_pruning_and_sparsity.md)：了解剪枝和稀疏化概念，以及如何创建更高效的模型。
- [超参数进化](tutorials/hyperparameter_evolution.md)：发现自动化[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)过程以获得更好的模型性能。
- [冻结层迁移学习](tutorials/transfer_learning_with_frozen_layers.md)：学习如何通过冻结 YOLOv5 中的层来实现[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)。
- [架构摘要](tutorials/architecture_description.md) 🌟 深入了解 YOLOv5 模型的结构细节。阅读 [YOLOv5 v6.0 博客文章](https://www.ultralytics.com/blog/yolov5-v6-0-is-here)获取更多见解。
- [ClearML 日志集成](tutorials/clearml_logging_integration.md) 🌟 学习如何集成 [ClearML](https://clear.ml/) 以在模型训练期间进行高效日志记录。
- [YOLOv5 与 Neural Magic](tutorials/neural_magic_pruning_quantization.md)：了解如何使用 [Neural Magic 的 DeepSparse](https://github.com/neuralmagic/deepsparse/blob/main/README.md) 来剪枝和量化您的 YOLOv5 模型。
- [Comet 日志集成](tutorials/comet_logging_integration.md) 🌟 新功能：探索如何利用 [Comet](https://www.comet.com/site/) 改进模型训练日志记录。

## 支持的环境

Ultralytics 提供一系列即用型环境，每个环境都预装了 [CUDA](https://developer.nvidia.com/cuda)、[CuDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/) 等基本依赖项，以快速启动您的项目。您还可以使用 [Ultralytics HUB](https://www.ultralytics.com/hub) 管理您的模型和数据集。

- **免费 GPU 笔记本**：<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="在 Gradient 上运行"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>
- **Google Cloud**：[GCP 快速入门指南](environments/google_cloud_quickstart_tutorial.md)
- **Amazon**：[AWS 快速入门指南](environments/aws_quickstart_tutorial.md)
- **Azure**：[AzureML 快速入门指南](environments/azureml_quickstart_tutorial.md)
- **Docker**：[Docker 快速入门指南](environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker 拉取次数"></a>

## 项目状态

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

此徽章表示所有 [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) 持续集成（CI）测试均已成功通过。这些 CI 测试严格检查 YOLOv5 在各个关键方面的功能和性能：[训练](https://github.com/ultralytics/yolov5/blob/master/train.py)、[验证](https://github.com/ultralytics/yolov5/blob/master/val.py)、[推理](https://github.com/ultralytics/yolov5/blob/master/detect.py)、[导出](https://github.com/ultralytics/yolov5/blob/master/export.py)和[基准测试](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py)。它们确保在 macOS、Windows 和 Ubuntu 上的一致可靠运行，测试每 24 小时进行一次，并在每次新提交时进行。

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>

## 连接与贡献

您的 YOLOv5 之旅不必孤军奋战。加入我们在 [GitHub](https://github.com/ultralytics/yolov5) 上充满活力的社区，在 [LinkedIn](https://www.linkedin.com/company/ultralytics/) 上与专业人士联系，在 [Twitter](https://twitter.com/ultralytics) 上分享您的成果，并在 [YouTube](https://www.youtube.com/ultralytics?sub_confirmation=1) 上找到教育资源。在 [TikTok](https://www.tiktok.com/@ultralytics) 和 [BiliBili](https://ultralytics.com/bilibili) 上关注我们获取更多精彩内容。

有兴趣贡献吗？我们欢迎各种形式的贡献，从代码改进和错误报告到文档更新。查看我们的[贡献指南](../help/contributing.md)了解更多信息。

我们很期待看到您使用 YOLOv5 的创新方式。深入探索、实验，并革新您的计算机视觉项目！🚀

## 常见问题

### Ultralytics YOLOv5 的主要特点是什么？

Ultralytics YOLOv5 以其高速和高[精度](https://www.ultralytics.com/glossary/accuracy)的目标检测能力而闻名。基于 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 构建，它多功能且用户友好，适用于各种计算机视觉项目。主要特点包括实时推理、支持多种训练技巧（如测试时增强（TTA）和模型集成），以及与 TFLite、ONNX、CoreML 和 TensorRT 等导出格式的兼容性。要深入了解 Ultralytics YOLOv5 如何提升您的项目，请探索我们的 [TFLite、ONNX、CoreML、TensorRT 导出指南](tutorials/model_export.md)。

### 如何在我的数据集上训练自定义 YOLOv5 模型？

在您的数据集上训练自定义 YOLOv5 模型涉及几个关键步骤。首先，以所需格式准备带有标签标注的数据集。然后，配置 YOLOv5 训练参数并使用 `train.py` 脚本开始训练过程。有关此过程的深入教程，请参阅我们的[训练自定义数据指南](tutorials/train_custom_data.md)。它提供分步说明，以确保您的特定用例获得最佳结果。

### 为什么我应该使用 Ultralytics YOLOv5 而不是其他目标检测模型（如 RCNN）？

Ultralytics YOLOv5 优于 [R-CNN](https://www.ultralytics.com/glossary/object-detection-architectures) 等模型，因为它在实时目标检测中具有卓越的速度和精度。YOLOv5 一次性处理整个图像，与 RCNN 涉及多次传递的基于区域的方法相比，速度显著更快。此外，YOLOv5 与各种导出格式的无缝集成和广泛的文档使其成为初学者和专业人士的绝佳选择。在我们的[架构摘要](tutorials/architecture_description.md)中了解更多关于架构优势的信息。

### 如何在训练期间优化 YOLOv5 模型性能？

优化 YOLOv5 模型性能涉及调整各种超参数并结合[数据增强](https://www.ultralytics.com/glossary/data-augmentation)和迁移学习等技术。Ultralytics 提供关于[超参数进化](tutorials/hyperparameter_evolution.md)和[剪枝/稀疏化](tutorials/model_pruning_and_sparsity.md)的全面资源以提高模型效率。您可以在我们的[最佳训练结果技巧指南](tutorials/tips_for_best_training_results.md)中发现实用技巧，该指南提供可操作的见解以在训练期间实现最佳性能。

### 运行 YOLOv5 应用程序支持哪些环境？

Ultralytics YOLOv5 支持多种环境，包括 [Gradient](https://bit.ly/yolov5-paperspace-notebook)、[Google Colab](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) 和 [Kaggle](https://www.kaggle.com/models/ultralytics/yolov5) 上的免费 GPU 笔记本，以及 [Google Cloud](environments/google_cloud_quickstart_tutorial.md)、[Amazon AWS](environments/aws_quickstart_tutorial.md) 和 [Azure](environments/azureml_quickstart_tutorial.md) 等主要云平台。[Docker 镜像](https://hub.docker.com/r/ultralytics/yolov5)也可用于便捷设置。有关设置这些环境的详细指南，请查看我们的[支持的环境](#支持的环境)部分，其中包含每个平台的分步说明。
