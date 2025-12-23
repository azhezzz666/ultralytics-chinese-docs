---
comments: true
description: 学习如何逐步在 Amazon SageMaker Endpoints 上部署 Ultralytics 的 YOLO11，从设置到测试，使用 AWS 服务进行强大的实时推理。
keywords: YOLO11, Amazon SageMaker, AWS, Ultralytics, 机器学习, 计算机视觉, 模型部署, AWS CloudFormation, AWS CDK, 实时推理
---

# 在 Amazon SageMaker Endpoints 上部署 YOLO11 指南

在 Amazon SageMaker Endpoints 上部署像 [Ultralytics 的 YOLO11](https://github.com/ultralytics/ultralytics) 这样的高级[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型，为各种[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)应用开辟了广泛的可能性。有效使用这些模型的关键在于理解它们的设置、配置和部署过程。当 YOLO11 与 Amazon SageMaker（AWS 提供的强大且可扩展的机器学习服务）无缝集成时，它变得更加强大。

本指南将逐步带您完成在 Amazon SageMaker Endpoints 上部署 YOLO11 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 模型的过程。您将学习准备 AWS 环境、正确配置模型以及使用 AWS CloudFormation 和 AWS Cloud Development Kit (CDK) 等工具进行部署的基本知识。

## Amazon SageMaker

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/amazon-sagemaker-overview.avif" alt="Amazon SageMaker 概述">
</p>

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) 是 Amazon Web Services (AWS) 的机器学习服务，简化了构建、训练和部署机器学习模型的过程。它提供了广泛的工具来处理机器学习工作流程的各个方面。这包括用于调整模型的自动化功能、大规模训练模型的选项以及将模型部署到生产环境的简单方法。SageMaker 支持流行的机器学习框架，为各种项目提供所需的灵活性。其功能还涵盖数据标注、工作流程管理和性能分析。

## 在 Amazon SageMaker Endpoints 上部署 YOLO11

在 Amazon SageMaker 上部署 YOLO11 让您可以使用其托管环境进行实时推理，并利用自动扩展等功能。请看下面的 AWS 架构。

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/aws-architecture.avif" alt="AWS 架构">
</p>

### 步骤 1：设置您的 AWS 环境

首先，确保您具备以下先决条件：

- AWS 账户：如果您还没有，请注册一个 AWS 账户。
- 配置的 IAM 角色：您需要一个具有 Amazon SageMaker、AWS CloudFormation 和 Amazon S3 必要权限的 IAM 角色。
- AWS CLI：如果尚未安装，请下载并安装 AWS 命令行界面 (CLI)。
- AWS CDK：如果尚未安装，请安装 AWS Cloud Development Kit (CDK)。
- 足够的服务配额：确认您有 `ml.m5.4xlarge` 实例的足够配额。

### 步骤 2：克隆 YOLO11 SageMaker 仓库

```bash
git clone https://github.com/aws-samples/host-yolov8-on-sagemaker-endpoint.git
cd host-yolov8-on-sagemaker-endpoint/yolov8-pytorch-cdk
```

### 步骤 3：设置 CDK 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
pip install --upgrade aws-cdk-lib
```

### 步骤 4：创建 AWS CloudFormation 堆栈

```bash
cdk synth
cdk bootstrap
cdk deploy
```

### 步骤 5：部署 YOLO 模型

创建 AWS CloudFormation 堆栈后，打开笔记本实例，修改 inference.py 文件，然后使用 1_DeployEndpoint.ipynb 笔记本部署端点。

### 步骤 6：测试您的部署

使用 2_TestEndpoint.ipynb 笔记本测试已部署的 SageMaker 端点。

### 步骤 7：监控和管理

使用 Amazon CloudWatch 定期检查 SageMaker 端点的性能和健康状况。

## 总结

本指南带您完成了使用 AWS CloudFormation 和 AWS CDK 在 Amazon SageMaker Endpoints 上部署 YOLO11 的过程。有关更多技术细节，请参阅 [AWS 机器学习博客](https://aws.amazon.com/blogs/machine-learning/hosting-yolov8-pytorch-model-on-amazon-sagemaker-endpoints/)和官方 [Amazon SageMaker 文档](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)。

## 常见问题

### 如何在 Amazon SageMaker Endpoints 上部署 Ultralytics YOLO11 模型？

按照以下步骤操作：设置 AWS 环境、克隆仓库、设置 CDK 环境、部署堆栈，然后部署和测试模型。

### 在 Amazon SageMaker 上部署 YOLO11 的先决条件是什么？

需要 AWS 账户、配置的 IAM 角色、AWS CLI、AWS CDK 和足够的服务配额。

### 为什么应该在 Amazon SageMaker 上使用 Ultralytics YOLO11？

SageMaker 提供可扩展性、与 AWS 服务的集成、易于部署和高性能基础设施。
