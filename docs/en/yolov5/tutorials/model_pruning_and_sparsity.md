---
comments: true
description: 学习如何对 YOLOv5 模型进行剪枝以提高性能。按照本分步指南有效优化您的 YOLOv5 模型。
keywords: YOLOv5 剪枝, 模型剪枝, YOLOv5 优化, YOLOv5 指南, 机器学习剪枝, 模型稀疏性, 神经网络优化
---

# YOLOv5 中的模型剪枝和稀疏性

📚 本指南介绍如何对 YOLOv5 🚀 模型应用**剪枝**，以创建更高效的网络同时保持性能。

## 什么是模型剪枝？

[模型剪枝](https://www.ultralytics.com/glossary/model-pruning)是一种通过移除不太重要的参数（权重和连接）来减少神经网络大小和复杂性的技术。这个过程创建了一个更高效的模型，具有以下几个优点：

- 减小模型大小，便于在资源受限的设备上部署
- 更快的推理速度，对准确率影响最小
- 更低的内存使用和能耗
- 提高实时应用的整体效率

剪枝通过识别和移除对模型性能贡献最小的参数来工作，从而产生一个具有相似准确率的更轻量级模型。

## 开始之前

克隆仓库并在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)会从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)自动下载。

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆
cd yolov5
pip install -r requirements.txt # 安装
```

## 测试基线性能

在剪枝之前，建立基线性能以进行比较。此命令在图像大小 640 像素下测试 COCO val2017 上的 YOLOv5x。`yolov5x.pt` 是可用的最大和最准确的模型。其他选项有 `yolov5s.pt`、`yolov5m.pt` 和 `yolov5l.pt`，或您自己训练自定义数据集的检查点 `./weights/best.pt`。有关所有可用模型的详细信息，请参阅 README [表格](https://github.com/ultralytics/yolov5#pretrained-checkpoints)。

```bash
python val.py --weights yolov5x.pt --data coco.yaml --img 640 --half
```

输出：

```text
val: data=/content/yolov5/data/coco.yaml, weights=['yolov5x.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.65, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
YOLOv5 🚀 v6.0-224-g4c40933 torch 1.10.0+cu111 CUDA:0 (Tesla V100-SXM2-16GB, 16160MiB)

Fusing layers...
Model Summary: 444 layers, 86705005 parameters, 0 gradients
val: Scanning '/content/datasets/coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupt: 100% 5000/5000 [00:00<?, ?it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 157/157 [01:12<00:00,  2.16it/s]
                 all       5000      36335      0.732      0.628      0.683      0.496
Speed: 0.1ms pre-process, 5.2ms inference, 1.7ms NMS per image at shape (32, 3, 640, 640)  # <--- 基线速度

Evaluating pycocotools mAP... saving runs/val/exp2/yolov5x_predictions.json...
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.507  # <--- 基线 mAP
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.689
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.552
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.559
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.652
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.381
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.630
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.682
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.731
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.829
Results saved to runs/val/exp
```


## 对 YOLOv5x 应用剪枝（30% 稀疏性）

我们可以使用 `utils/torch_utils.py` 中定义的 `torch_utils.prune()` 命令对模型应用剪枝。要测试剪枝后的模型，我们更新 `val.py` 将 YOLOv5x 剪枝到 0.3 稀疏性（30% 的权重设置为零）：

<img width="894" alt="显示将 YOLOv5x 剪枝到 30% 稀疏性的代码截图" src="https://github.com/ultralytics/docs/releases/download/0/sparsity-test-yolov5x-coco.avif">

30% 剪枝输出：

```text
val: data=/content/yolov5/data/coco.yaml, weights=['yolov5x.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.65, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
YOLOv5 🚀 v6.0-224-g4c40933 torch 1.10.0+cu111 CUDA:0 (Tesla V100-SXM2-16GB, 16160MiB)

Fusing layers...
Model Summary: 444 layers, 86705005 parameters, 0 gradients
Pruning model...  0.3 global sparsity
val: Scanning '/content/datasets/coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupt: 100% 5000/5000 [00:00<?, ?it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 157/157 [01:11<00:00,  2.19it/s]
                 all       5000      36335      0.724      0.614      0.671      0.478
Speed: 0.1ms pre-process, 5.2ms inference, 1.7ms NMS per image at shape (32, 3, 640, 640)  # <--- 剪枝速度

Evaluating pycocotools mAP... saving runs/val/exp3/yolov5x_predictions.json...
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.489  # <--- 剪枝 mAP
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.677
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.537
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.334
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.542
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.635
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.370
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.664
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.803
Results saved to runs/val/exp3
```

## 结果分析

从结果中，我们可以观察到：

- **实现 30% 稀疏性**：模型 `nn.Conv2d` 层中 30% 的权重参数现在为零
- **推理时间保持不变**：尽管进行了剪枝，处理速度基本相同
- **性能影响最小**：mAP 仅从 0.507 略微下降到 0.489（仅降低 3.6%）
- **模型大小减小**：剪枝后的模型需要更少的存储内存

这表明剪枝可以显著降低模型复杂性，同时对性能的影响很小，使其成为在资源受限环境中部署的有效优化技术。

## 微调剪枝模型

为获得最佳结果，剪枝后的模型应在剪枝后进行微调以恢复准确率。可以通过以下方式完成：

1. 以所需的稀疏性级别应用剪枝
2. 使用较低的学习率训练剪枝后的模型几个轮次
3. 将微调后的剪枝模型与基线进行评估

此过程帮助剩余参数适应以补偿移除的连接，通常可以恢复大部分或全部原始准确率。

## 支持的环境

Ultralytics 提供一系列开箱即用的环境，每个环境都预装了 [CUDA](https://developer.nvidia.com/cuda)、[CUDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/) 等基本依赖项，以便快速启动您的项目。

- **免费 GPU Notebook**：<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="在 Gradient 上运行"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>
- **Google Cloud**：[GCP 快速入门指南](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**：[AWS 快速入门指南](../environments/aws_quickstart_tutorial.md)
- **Azure**：[AzureML 快速入门指南](../environments/azureml_quickstart_tutorial.md)
- **Docker**：[Docker 快速入门指南](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## 项目状态

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

此徽章表示所有 [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) 持续集成（CI）测试均成功通过。这些 CI 测试严格检查 YOLOv5 在各个关键方面的功能和性能：[训练](https://github.com/ultralytics/yolov5/blob/master/train.py)、[验证](https://github.com/ultralytics/yolov5/blob/master/val.py)、[推理](https://github.com/ultralytics/yolov5/blob/master/detect.py)、[导出](https://github.com/ultralytics/yolov5/blob/master/export.py)和[基准测试](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py)。它们确保在 macOS、Windows 和 Ubuntu 上的一致可靠运行，测试每 24 小时进行一次，并在每次新提交时进行。
