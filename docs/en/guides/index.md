---
comments: true
description: 通过 Ultralytics 教程掌握 YOLO，涵盖训练、部署和优化。找到解决方案，改进指标，轻松部署。
keywords: Ultralytics, YOLO, 教程, 指南, 目标检测, 深度学习, PyTorch, 训练, 部署, 优化, 计算机视觉
---

# Ultralytics YOLO 综合教程

欢迎来到 Ultralytics 的 YOLO 指南。我们的综合教程涵盖 YOLO [目标检测](https://www.ultralytics.com/glossary/object-detection)模型的各个方面，从训练和预测到部署。基于 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 构建，YOLO 以其在实时目标检测任务中卓越的速度和[准确性](https://www.ultralytics.com/glossary/accuracy)而脱颖而出。

无论您是[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)初学者还是专家，我们的教程都为您的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目提供关于 YOLO 实现和优化的宝贵见解。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/96NkhsV-W1U"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics YOLO11 指南概览
</p>

## 指南

以下是帮助您掌握 Ultralytics YOLO 各个方面的深入指南汇编。

- [模型测试指南](model-testing.md)：在真实环境中测试计算机视觉模型的全面指南。学习如何验证准确性、可靠性和性能是否符合项目目标。
- [AzureML 快速入门](azureml-quickstart.md)：在 Microsoft Azure [机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)平台上启动和运行 Ultralytics YOLO 模型。学习如何在云端训练、部署和扩展您的目标检测项目。
- [模型部署最佳实践](model-deployment-practices.md)：了解在计算机视觉项目中高效部署模型的技巧和最佳实践，重点关注优化、故障排除和安全性。
- [Conda 快速入门](conda-quickstart.md)：设置 [Conda](https://anaconda.org/conda-forge/ultralytics) 环境的分步指南。学习如何使用 Conda 高效安装和开始使用 Ultralytics 包。
- [数据收集和标注](data-collection-and-annotation.md)：探索收集和标注数据的工具、技术和最佳实践，为计算机视觉模型创建高质量输入。
- [NVIDIA Jetson 上的 DeepStream](deepstream-nvidia-jetson.md)：使用 DeepStream 和 TensorRT 在 NVIDIA Jetson 设备上部署 YOLO 模型的快速入门指南。
- [定义计算机视觉项目目标](defining-project-goals.md)：了解如何有效地为计算机视觉项目定义清晰且可衡量的目标。学习明确定义问题陈述的重要性以及它如何为项目创建路线图。
- [Docker 快速入门](docker-quickstart.md)：使用 [Docker](https://hub.docker.com/r/ultralytics/ultralytics) 设置和使用 Ultralytics YOLO 模型的完整指南。学习如何安装 Docker、管理 GPU 支持，以及在隔离容器中运行 YOLO 模型以实现一致的开发和部署。
- [树莓派上的 Edge TPU](coral-edge-tpu-on-raspberry-pi.md)：[Google Edge TPU](https://developers.google.com/coral) 在[树莓派](https://www.raspberrypi.com/)上加速 YOLO 推理。
- [超参数调优](hyperparameter-tuning.md)：了解如何使用 Tuner 类和遗传进化算法通过微调超参数来优化 YOLO 模型。
- [模型评估和微调洞察](model-evaluation-insights.md)：获取评估和微调计算机视觉模型的策略和最佳实践洞察。了解优化模型以实现最佳结果的迭代过程。
- [隔离分割目标](isolating-segmentation-objects.md)：使用 Ultralytics 分割从图像中提取和/或隔离目标的分步方法和说明。
- [K 折交叉验证](kfold-cross-validation.md)：学习如何使用 K 折交叉验证技术提高模型泛化能力。
- [维护计算机视觉模型](model-monitoring-and-maintenance.md)：了解监控、维护和记录计算机视觉模型的关键实践，以保证准确性、发现异常并减轻数据漂移。
- [模型部署选项](model-deployment-options.md)：YOLO [模型部署](https://www.ultralytics.com/glossary/model-deployment)格式概览，如 ONNX、OpenVINO 和 TensorRT，以及每种格式的优缺点，以指导您的部署策略。
- [模型 YAML 配置指南](model-yaml-config.md)：深入了解 Ultralytics 模型架构定义。探索 YAML 格式，理解模块解析系统，学习如何无缝集成自定义模块。
- [NVIDIA Jetson](nvidia-jetson.md)：在 NVIDIA Jetson 设备上部署 YOLO 模型的快速入门指南。
- [OpenVINO 延迟与吞吐量模式](optimizing-openvino-latency-vs-throughput-modes.md)：学习延迟和吞吐量优化技术，以实现最佳 YOLO 推理性能。
- [预处理标注数据](preprocessing_annotated_data.md)：学习使用 YOLO11 在计算机视觉项目中预处理和增强图像数据，包括归一化、数据集增强、拆分和探索性数据分析（EDA）。
- [树莓派](raspberry-pi.md)：在最新树莓派硬件上运行 YOLO 模型的快速入门教程。
- [ROS 快速入门](ros-quickstart.md)：学习如何将 YOLO 与机器人操作系统（ROS）集成，用于机器人应用中的实时目标检测，包括点云和深度图像。
- [SAHI 切片推理](sahi-tiled-inference.md)：利用 SAHI 的切片推理功能与 YOLO11 进行高分辨率图像目标检测的综合指南。
- [计算机视觉项目步骤](steps-of-a-cv-project.md)：了解计算机视觉项目涉及的关键步骤，包括定义目标、选择模型、准备数据和评估结果。
- [模型训练技巧](model-training-tips.md)：探索优化[批量大小](https://www.ultralytics.com/glossary/batch-size)、使用[混合精度](https://www.ultralytics.com/glossary/mixed-precision)、应用预训练权重等技巧，让计算机视觉模型训练变得轻松。
- [Triton 推理服务器集成](triton-inference-server.md)：深入了解 Ultralytics YOLO11 与 NVIDIA Triton 推理服务器的集成，用于可扩展和高效的深度学习推理部署。
- [使用 Docker 部署到 Vertex AI](vertex-ai-deployment-with-docker.md)：使用 Docker 容器化 YOLO 模型并部署到 Google Cloud Vertex AI 的简化指南——涵盖构建、推送、自动扩展和监控。
- [在终端中查看推理图像](view-results-in-terminal.md)：使用 VSCode 的集成终端在使用远程隧道或 SSH 会话时查看推理结果。
- [YOLO 常见问题](yolo-common-issues.md) ⭐ 推荐：使用 Ultralytics YOLO 模型时最常遇到问题的实用解决方案和故障排除技巧。
- [YOLO 数据增强](yolo-data-augmentation.md)：掌握 YOLO 中从基本变换到高级策略的完整数据增强技术范围，以提高模型鲁棒性和性能。
- [YOLO 性能指标](yolo-performance-metrics.md) ⭐ 必读：了解用于评估 YOLO 模型性能的关键指标，如 mAP、IoU 和 [F1 分数](https://www.ultralytics.com/glossary/f1-score)。包括实际示例和如何提高检测准确性和速度的技巧。
- [YOLO 线程安全推理](yolo-thread-safe-inference.md)：以线程安全方式使用 YOLO 模型进行推理的指南。了解线程安全的重要性和防止竞态条件、确保一致预测的最佳实践。

## 贡献指南

我们欢迎社区贡献！如果您已经掌握了 Ultralytics YOLO 的某个特定方面，而我们的指南尚未涵盖，我们鼓励您分享您的专业知识。撰写指南是回馈社区的好方法，有助于使我们的文档更加全面和用户友好。

要开始，请阅读我们的[贡献指南](../help/contributing.md)，了解如何提交拉取请求（PR）的指导方针。我们期待您的贡献。

## 常见问题

### 如何使用 Ultralytics YOLO 训练自定义目标检测模型？

使用 Ultralytics YOLO 训练自定义目标检测模型非常简单。首先以正确的格式准备数据集并安装 Ultralytics 包。使用以下代码开始训练：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # 加载预训练的 YOLO 模型
        model.train(data="path/to/dataset.yaml", epochs=50)  # 在自定义数据集上训练
        ```

    === "CLI"

        ```bash
        yolo task=detect mode=train model=yolo11n.pt data=path/to/dataset.yaml epochs=50
        ```

有关详细的数据集格式和其他选项，请参阅我们的[模型训练技巧](model-training-tips.md)指南。

### 我应该使用哪些性能指标来评估我的 YOLO 模型？

评估 YOLO 模型性能对于了解其效果至关重要。关键指标包括[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）、[交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou)（IoU）和 F1 分数。这些指标有助于评估目标检测任务的准确性和[精度](https://www.ultralytics.com/glossary/precision)。您可以在我们的 [YOLO 性能指标](yolo-performance-metrics.md)指南中了解更多关于这些指标以及如何改进模型的信息。

### 为什么我应该在计算机视觉项目中使用 Ultralytics HUB？

Ultralytics HUB 是一个无代码平台，简化了 YOLO 模型的管理、训练和部署。它支持无缝集成、实时跟踪和云端训练，非常适合初学者和专业人士。在我们的 [Ultralytics HUB](https://docs.ultralytics.com/hub/) 快速入门指南中了解更多关于其功能以及它如何简化您的工作流程。

### YOLO 模型训练期间常见的问题有哪些，如何解决？

YOLO 模型训练期间的常见问题包括数据格式错误、模型架构不匹配和[训练数据](https://www.ultralytics.com/glossary/training-data)不足。要解决这些问题，请确保数据集格式正确，检查模型版本兼容性，并增强训练数据。有关解决方案的完整列表，请参阅我们的 [YOLO 常见问题](yolo-common-issues.md)指南。

### 如何在边缘设备上部署 YOLO 模型进行实时目标检测？

在 NVIDIA Jetson 和树莓派等边缘设备上部署 YOLO 模型需要将模型转换为兼容格式，如 TensorRT 或 TFLite。按照我们的 [NVIDIA Jetson](nvidia-jetson.md) 和[树莓派](raspberry-pi.md)部署分步指南开始在边缘硬件上进行实时目标检测。这些指南将引导您完成安装、配置和性能优化。
