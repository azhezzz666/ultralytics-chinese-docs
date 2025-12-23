---
comments: true
description: 学习如何在测试和推理期间使用 YOLOv5 模型集成来提高 mAP 和召回率，获得更准确的预测。
keywords: YOLOv5, 模型集成, 测试, 推理, mAP, 召回率, Ultralytics, 目标检测, PyTorch
---

# YOLOv5 模型集成

📚 本指南介绍如何在测试和推理期间使用 Ultralytics YOLOv5 🚀 **模型集成**来提高 mAP 和[召回率](https://www.ultralytics.com/glossary/recall)。

来自[集成学习](https://en.wikipedia.org/wiki/Ensemble_learning)：

> 集成建模是一个创建多个不同模型来预测结果的过程，可以使用多种不同的建模算法或使用不同的[训练数据](https://www.ultralytics.com/glossary/training-data)集。然后集成模型聚合每个基础模型的预测，并为未见数据生成一个最终预测。使用集成模型的动机是减少预测的泛化误差。只要基础模型是多样化和独立的，使用集成方法时模型的预测误差就会减少。该方法在做出预测时寻求群体智慧。尽管集成模型在模型内有多个基础模型，但它作为单个模型运行和执行。

## 开始之前

克隆仓库并在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)会从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)自动下载。

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆
cd yolov5
pip install -r requirements.txt # 安装
```

## 正常测试

在集成之前，建立单个模型的基线性能。此命令在图像大小 640 像素下测试 COCO val2017 上的 YOLOv5x。`yolov5x.pt` 是可用的最大和最准确的模型。其他选项有 `yolov5s.pt`、`yolov5m.pt` 和 `yolov5l.pt`，或您自己训练自定义数据集的检查点 `./weights/best.pt`。有关所有可用模型的详细信息，请参阅[预训练检查点表](https://docs.ultralytics.com/models/yolov5/)。

```bash
python val.py --weights yolov5x.pt --data coco.yaml --img 640 --half
```

输出：

```text
val: data=./data/coco.yaml, weights=['yolov5x.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.65, task=val, device=, single_cls=False, augment=False, verbose=False, save_txt=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True
YOLOv5 🚀 v5.0-267-g6a3ee7c torch 1.9.0+cu102 CUDA:0 (Tesla P100-PCIE-16GB, 16280.875MB)

Fusing layers...
Model Summary: 476 layers, 87730285 parameters, 0 gradients

val: Scanning '../datasets/coco/val2017' images and labels...4952 found, 48 missing, 0 empty, 0 corrupted: 100% 5000/5000 [00:01<00:00, 2846.03it/s]
val: New cache created: ../datasets/coco/val2017.cache
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 157/157 [02:30<00:00,  1.05it/s]
                 all       5000      36335      0.746      0.626       0.68       0.49
Speed: 0.1ms pre-process, 22.4ms inference, 1.4ms NMS per image at shape (32, 3, 640, 640)  # <--- 基线速度

Evaluating pycocotools mAP... saving runs/val/exp/yolov5x_predictions.json...
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504  # <--- 基线 mAP
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.688
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.546
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.551
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.382
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.681  # <--- 基线 mAR
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.735
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.826
```


## 集成测试

通过简单地在任何现有的 val.py 或 detect.py 命令中向 `--weights` 参数添加额外模型，可以在测试和推理时将多个预训练模型集成在一起。此示例测试 2 个模型的集成：

- YOLOv5x
- YOLOv5l6

```bash
python val.py --weights yolov5x.pt yolov5l6.pt --data coco.yaml --img 640 --half
```

您可以列出任意数量的检查点，包括自定义权重如 `runs/train/exp5/weights/best.pt`。YOLOv5 将自动运行每个模型，在每张图像的基础上对齐预测，并在执行 NMS 之前平均输出。

输出：

```text
val: data=./data/coco.yaml, weights=['yolov5x.pt', 'yolov5l6.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, task=val, device=, single_cls=False, augment=False, verbose=False, save_txt=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True
YOLOv5 🚀 v5.0-267-g6a3ee7c torch 1.9.0+cu102 CUDA:0 (Tesla P100-PCIE-16GB, 16280.875MB)

Fusing layers...
Model Summary: 476 layers, 87730285 parameters, 0 gradients  # 模型 1
Fusing layers...
Model Summary: 501 layers, 77218620 parameters, 0 gradients  # 模型 2
Ensemble created with ['yolov5x.pt', 'yolov5l6.pt']  # 集成通知

val: Scanning '../datasets/coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100% 5000/5000 [00:00<00:00, 49695545.02it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 157/157 [03:58<00:00,  1.52s/it]
                 all       5000      36335      0.747      0.637      0.692      0.502
Speed: 0.1ms pre-process, 39.5ms inference, 2.0ms NMS per image at shape (32, 3, 640, 640)  # <--- 集成速度

Evaluating pycocotools mAP... saving runs/val/exp3/yolov5x_predictions.json...
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515  # <--- 集成 mAP
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.699
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.557
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.563
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.387
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.689  # <--- 集成 mAR
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.743
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.844
```

## 集成推理

向 `--weights` 参数添加额外模型以运行集成推理：

```bash
python detect.py --weights yolov5x.pt yolov5l6.pt --img 640 --source data/images
```

输出：

```text
YOLOv5 🚀 v5.0-267-g6a3ee7c torch 1.9.0+cu102 CUDA:0 (Tesla P100-PCIE-16GB, 16280.875MB)

Fusing layers...
Model Summary: 476 layers, 87730285 parameters, 0 gradients
Fusing layers...
Model Summary: 501 layers, 77218620 parameters, 0 gradients
Ensemble created with ['yolov5x.pt', 'yolov5l6.pt']

image 1/2 /content/yolov5/data/images/bus.jpg: 640x512 4 persons, 1 bus, 1 tie, Done. (0.063s)
image 2/2 /content/yolov5/data/images/zidane.jpg: 384x640 3 persons, 2 ties, Done. (0.056s)
Results saved to runs/detect/exp2
Done. (0.223s)
```

<img src="https://github.com/ultralytics/docs/releases/download/0/yolo-inference-result.avif" width="500" alt="YOLO 推理结果">

## 模型集成的优势

使用 YOLOv5 进行模型集成有几个优势：

1. **提高准确率**：如上面的示例所示，集成多个模型将 mAP 从 0.504 提高到 0.515，mAR 从 0.681 提高到 0.689。
2. **更好的泛化**：组合不同的模型有助于减少过拟合并提高在各种数据上的性能。
3. **增强鲁棒性**：集成通常对数据中的噪声和异常值更加鲁棒。
4. **互补优势**：不同的模型可能擅长检测不同类型的目标或在不同的环境条件下表现更好。

主要的权衡是推理时间增加，如速度指标所示（单模型 22.4ms vs 集成 39.5ms）。

## 何时使用模型集成

在以下场景中考虑使用模型集成：

- 当准确率比推理速度更重要时
- 对于必须最小化漏检的关键应用
- 处理具有不同光照、遮挡或尺度的挑战性图像时
- 在需要最大性能的竞赛或基准测试期间

对于有严格延迟要求的实时应用，单模型推理可能更合适。

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
