---
comments: true
description: 学习如何冻结 YOLOv5 层以实现高效迁移学习，减少资源消耗并加速训练，同时保持精度。
keywords: YOLOv5, 迁移学习, 冻结层, 机器学习, 深度学习, 模型训练, PyTorch, Ultralytics
---

# YOLOv5 中使用冻结层的迁移学习

📚 本指南介绍如何在实现[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)时**冻结** [YOLOv5](https://github.com/ultralytics/yolov5) 🚀 层。迁移学习是一种强大的[机器学习（ML）](https://www.ultralytics.com/glossary/machine-learning-ml)技术，允许你在新数据上快速重新训练模型，而无需从头开始重新训练整个网络。通过冻结初始层的权重，只更新后续层的参数，你可以显著减少计算资源需求和训练时间。然而，这种方法可能会略微影响最终模型的[准确性](https://www.ultralytics.com/glossary/accuracy)。

## 开始之前

首先，克隆 YOLOv5 仓库并安装 [`requirements.txt`](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) 中列出的必要依赖。确保你有安装了 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) 的 [**Python>=3.8.0**](https://www.python.org/) 环境。预训练[模型](https://github.com/ultralytics/yolov5/tree/master/models)和所需[数据集](https://github.com/ultralytics/yolov5/tree/master/data)将自动从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)下载。

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆仓库
cd yolov5
pip install -r requirements.txt # 安装依赖
```

## 层冻结的工作原理

当你冻结[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)中的层时，你阻止它们的参数（权重和偏置）在训练过程中被更新。在 PyTorch 中，这是通过将层张量的 `requires_grad` 属性设置为 `False` 来实现的。因此，在[反向传播](https://www.ultralytics.com/glossary/backpropagation)期间不会为这些层计算梯度，从而节省计算和内存。

以下是 YOLOv5 在其[训练脚本](https://github.com/ultralytics/yolov5/blob/master/train.py)中实现层冻结的方式：

```python
# 冻结指定层
freeze = [f"model.{x}." for x in range(freeze)]  # 根据模块索引定义要冻结的层
for k, v in model.named_parameters():
    v.requires_grad = True  # 确保所有参数最初都是可训练的
    if any(x in k for x in freeze):
        print(f"冻结层: {k}")
        v.requires_grad = False  # 禁用冻结层的梯度计算
```

## 探索模型架构

了解 YOLOv5 模型的结构对于决定冻结哪些层至关重要。你可以使用以下 Python 代码片段检查所有模块及其参数的名称：

```python
# 假设 'model' 是你加载的 YOLOv5 模型实例
for name, param in model.named_parameters():
    print(name)

"""
示例输出:
model.0.conv.conv.weight
model.0.conv.bn.weight
model.0.conv.bn.bias
model.1.conv.weight
model.1.bn.weight
model.1.bn.bias
model.2.cv1.conv.weight
model.2.cv1.bn.weight
...
"""
```

YOLOv5 架构通常由[骨干网络](https://www.ultralytics.com/glossary/backbone)（标准配置如 YOLOv5s/m/l/x 中的第 0-9 层）负责[特征提取](https://www.ultralytics.com/glossary/feature-extraction)，以及头部（其余层）执行[目标检测](https://www.ultralytics.com/glossary/object-detection)。

```yaml
# YOLOv5 v6.0 骨干网络结构示例
backbone:
    # [from, number, module, args]
    - [-1, 1, Conv, [64, 6, 2, 2]]  # 第 0 层: 初始卷积 (P1/2 步长)
    - [-1, 1, Conv, [128, 3, 2]] # 第 1 层: 下采样卷积 (P2/4 步长)
    - [-1, 3, C3, [128]]          # 第 2 层: C3 模块
    - [-1, 1, Conv, [256, 3, 2]] # 第 3 层: 下采样卷积 (P3/8 步长)
    - [-1, 6, C3, [256]]          # 第 4 层: C3 模块
    - [-1, 1, Conv, [512, 3, 2]] # 第 5 层: 下采样卷积 (P4/16 步长)
    - [-1, 9, C3, [512]]          # 第 6 层: C3 模块
    - [-1, 1, Conv, [1024, 3, 2]]# 第 7 层: 下采样卷积 (P5/32 步长)
    - [-1, 3, C3, [1024]]         # 第 8 层: C3 模块
    - [-1, 1, SPPF, [1024, 5]]    # 第 9 层: 空间金字塔池化快速版

# YOLOv5 v6.0 头部结构示例
head:
    - [-1, 1, Conv, [512, 1, 1]] # 第 10 层
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 第 11 层
    - [[-1, 6], 1, Concat, [1]] # 第 12 层: 与骨干网络 P4 连接（来自第 6 层）
    - [-1, 3, C3, [512, False]] # 第 13 层: C3 模块
    # ... 后续头部层用于特征融合和检测
```

## 冻结选项

你可以使用训练命令中的 `--freeze` 参数控制冻结哪些层。此参数指定第一个_未冻结_模块的索引；此索引之前的所有模块都将冻结其权重。如果需要确认哪些索引对应于特定块，可以使用 `model.model`（一个 `nn.Sequential`）检查模块顺序。

### 仅冻结骨干网络

要冻结整个骨干网络（第 0 到 9 层），这在将模型适应新的对象类别同时保留从大型数据集（如 [COCO](https://docs.ultralytics.com/datasets/detect/coco/)）学习的通用特征提取能力时很常见：

```bash
python train.py --weights yolov5m.pt --data your_dataset.yaml --freeze 10
```

当你的目标数据集与原始训练数据（如 COCO）共享相似的低级视觉特征（边缘、纹理）但包含不同的对象类别时，此策略非常有效。

### 冻结除最终检测层外的所有层

要冻结几乎整个网络，只保留最终输出卷积层（`Detect` 模块的一部分，通常是最后一个模块，如 YOLOv5s 中的模块 24）可训练：

```bash
python train.py --weights yolov5m.pt --data your_dataset.yaml --freeze 24
```

当你主要需要调整模型以适应不同数量的输出类别，同时保持绝大多数学习到的特征不变时，此方法很有用。它需要最少的计算资源进行[微调](https://www.ultralytics.com/glossary/fine-tuning)。

## 性能比较

为了说明冻结层的效果，我们在 [Pascal VOC 数据集](https://docs.ultralytics.com/datasets/detect/voc/)上训练 YOLOv5m 50 个[轮次](https://www.ultralytics.com/glossary/epoch)，从官方 COCO 预训练[权重](https://www.ultralytics.com/glossary/model-weights)（`yolov5m.pt`）开始。我们比较了三种场景：训练所有层（`--freeze 0`）、冻结骨干网络（`--freeze 10`）和冻结除最终检测层外的所有层（`--freeze 24`）。

```bash
# 冻结骨干网络训练的示例命令
python train.py --batch 48 --weights yolov5m.pt --data voc.yaml --epochs 50 --cache --img 512 --hyp hyp.finetune.yaml --freeze 10
```

### 精度结果

结果表明，冻结层可以显著加速训练，但可能导致最终 [mAP（平均精度均值）](https://www.ultralytics.com/glossary/mean-average-precision-map)略有下降。训练所有层通常产生最佳精度，而冻结更多层则以可能较低的性能为代价提供更快的训练速度。

![比较不同冻结策略的训练 mAP50 结果](https://github.com/ultralytics/docs/releases/download/0/freezing-training-map50-results.avif)
_训练期间的 mAP50 比较_

![比较不同冻结策略的训练 mAP50-95 结果](https://github.com/ultralytics/docs/releases/download/0/freezing-training-map50-95-results.avif)
_训练期间的 mAP50-95 比较_

<img width="922" alt="性能结果汇总表" src="https://github.com/ultralytics/docs/releases/download/0/table-results.avif">
*性能指标汇总表*

### 资源利用

冻结更多层可以大幅减少 [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) 内存需求和整体利用率。这使得使用冻结层的迁移学习在硬件资源有限时成为一个有吸引力的选择，允许训练更大的模型或使用比其他情况下可能更大的图像尺寸。

![训练期间 GPU 内存分配百分比](https://github.com/ultralytics/docs/releases/download/0/training-gpu-memory-allocated-percent.avif)
_GPU 内存分配（%）_

![训练期间 GPU 内存利用率百分比](https://github.com/ultralytics/docs/releases/download/0/training-gpu-memory-utilization-percent.avif)
_GPU 利用率（%）_

## 何时使用层冻结

在迁移学习期间冻结层在以下几种情况下特别有利：

1.  **计算资源有限**：如果你在 GPU 内存或处理能力方面有限制。
2.  **小数据集**：当你的目标数据集明显小于原始预训练数据集时，冻结有助于防止[过拟合](https://www.ultralytics.com/glossary/overfitting)。
3.  **快速原型设计**：当你需要快速将现有模型适应新任务或领域进行初步评估时。
4.  **相似的特征域**：如果新数据集中的低级特征与模型预训练数据集中的特征非常相似。

在我们的[术语表条目](https://www.ultralytics.com/glossary/transfer-learning)中探索更多关于迁移学习的细微差别，并考虑使用[超参数调优](https://docs.ultralytics.com/guides/hyperparameter-tuning/)等技术来优化性能。

## 支持的环境

Ultralytics 提供各种预装了必要依赖项（如 [CUDA](https://developer.nvidia.com/cuda)、[CuDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/)）的即用环境。

- **免费 GPU 笔记本**：<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="在 Gradient 上运行"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>
- **Google Cloud**：[GCP 快速入门指南](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**：[AWS 快速入门指南](../environments/aws_quickstart_tutorial.md)
- **Azure**：[AzureML 快速入门指南](../environments/azureml_quickstart_tutorial.md)
- **Docker**：[Docker 快速入门指南](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker 拉取次数"></a>

## 项目状态

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 持续集成状态"></a>

此徽章确认所有 [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) 持续集成（CI）测试均已成功通过。这些 CI 测试严格评估 YOLOv5 在关键操作上的功能和性能：[训练](https://github.com/ultralytics/yolov5/blob/master/train.py)、[验证](https://github.com/ultralytics/yolov5/blob/master/val.py)、[推理](https://github.com/ultralytics/yolov5/blob/master/detect.py)、[导出](https://github.com/ultralytics/yolov5/blob/master/export.py)和[基准测试](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py)。它们确保在 macOS、Windows 和 Ubuntu 上的一致可靠运行，每 24 小时自动运行一次，并在每次新代码提交时运行。
