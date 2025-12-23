---
comments: true
description: 学习如何在多个 GPU 上训练 YOLOv5 以获得最佳性能。指南涵盖单机和多机设置以及 DistributedDataParallel。
keywords: YOLOv5, 多 GPU, 机器学习, 深度学习, PyTorch, 数据并行, 分布式数据并行, DDP, 多 GPU 训练
---

# YOLOv5 多 GPU 训练

本指南介绍如何正确使用**多个** GPU 在单台或多台机器上使用 YOLOv5 🚀 训练数据集。

## 开始之前

克隆仓库并在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)会从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)自动下载。

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆
cd yolov5
pip install -r requirements.txt # 安装
```

!!! tip "专业提示！"

    **Docker 镜像**推荐用于所有多 GPU 训练。参见 [Docker 快速入门指南](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

!!! tip "专业提示！"

    `torch.distributed.run` 在 **[PyTorch](https://www.ultralytics.com/glossary/pytorch)>=1.9** 中替代了 `torch.distributed.launch`。详情参见 [PyTorch 分布式文档](https://docs.pytorch.org/docs/stable/distributed.html)。

## 训练

选择一个预训练模型开始训练。这里我们选择 [YOLOv5s](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml)，这是可用的最小和最快的模型。有关所有模型的完整比较，请参阅我们的 README [表格](https://github.com/ultralytics/yolov5#pretrained-checkpoints)。我们将在 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) 数据集上使用多 GPU 训练此模型。

<p align="center"><img width="700" alt="YOLOv5 模型" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-model-comparison.avif"></p>

### 单 GPU

```bash
python train.py --batch 64 --data coco.yaml --weights yolov5s.pt --device 0
```

### 多 GPU [DataParallel](https://docs.pytorch.org/docs/stable/nn.html#torch.nn.DataParallel) 模式（⚠️ 不推荐）

您可以增加 `device` 以在 DataParallel 模式下使用多个 GPU。

```bash
python train.py --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
```

与仅使用 1 个 GPU 相比，此方法速度较慢，几乎不能加速训练。

### 多 GPU [DistributedDataParallel](https://docs.pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) 模式（✅ 推荐）

您需要传递 `python -m torch.distributed.run --nproc_per_node`，然后是常规参数。

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
```

- `--nproc_per_node` 指定您想使用多少个 GPU。在上面的示例中是 2。
- `--batch` 是总批次大小。它将平均分配到每个 GPU。在上面的示例中，每个 GPU 是 64/2=32。

上面的代码将使用 GPU `0... (N-1)`。如果您更喜欢通过环境变量控制设备可见性，也可以在启动命令之前设置 `CUDA_VISIBLE_DEVICES=2,3`（或任何其他列表）。


<details>
  <summary>使用特定 GPU（点击展开）</summary>

您可以通过简单地传递 `--device` 后跟您的特定 GPU 来实现。例如，在下面的代码中，我们将使用 GPU `2,3`。

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights '' --device 2,3
```

</details>

<details>
  <summary>使用 SyncBatchNorm（点击展开）</summary>

[SyncBatchNorm](https://docs.pytorch.org/docs/master/generated/torch.nn.SyncBatchNorm.html) 可以提高多 GPU 训练的[准确率](https://www.ultralytics.com/glossary/accuracy)，但会显著降低训练速度。它**仅**适用于多 GPU DistributedDataParallel 训练。

当**每个** GPU 上的批次大小较小（<= 8）时，最好使用它。

要使用 SyncBatchNorm，只需像下面这样向命令传递 `--sync-bn`：

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights '' --sync-bn
```

</details>

<details>
  <summary>使用多台机器（点击展开）</summary>

这**仅**适用于多 GPU DistributedDataParallel 训练。

在继续之前，请确保所有机器上的文件相同，包括数据集、代码库等。之后，确保机器之间可以相互通信。

您需要选择一台主机（其他机器将与之通信的机器）。记下其地址（`master_addr`）并选择一个端口（`master_port`）。下面的示例中我将使用 `master_addr = 192.168.1.1` 和 `master_port = 1234`。

要使用它，您可以执行以下操作：

```bash
# 在主机 0 上
python -m torch.distributed.run --nproc_per_node G --nnodes N --node_rank 0 --master_addr "192.168.1.1" --master_port 1234 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights ''
```

```bash
# 在机器 R 上
python -m torch.distributed.run --nproc_per_node G --nnodes N --node_rank R --master_addr "192.168.1.1" --master_port 1234 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights ''
```

其中 `G` 是每台机器的 GPU 数量，`N` 是机器数量，`R` 是从 `0...(N-1)` 的机器编号。假设我有两台机器，每台有两个 GPU，则 `G = 2`，`N = 2`，上面的 `R = 1`。

训练不会开始，直到**所有** `N` 台机器都连接。输出只会显示在主机上！

</details>

### 注意事项

- Windows 支持未经测试，推荐使用 Linux。
- `--batch` 必须是 GPU 数量的倍数。
- GPU 0 将比其他 GPU 占用稍多的内存，因为它维护 EMA 并负责检查点等。
- 如果您遇到 `RuntimeError: Address already in use`，可能是因为您同时运行多个训练。要解决此问题，只需通过添加 `--master_port` 使用不同的端口号，如下所示：

    ```bash
    python -m torch.distributed.run --master_port 1234 --nproc_per_node 2 ...
    ```

## 结果

在 [AWS EC2 P4d 实例](../environments/aws_quickstart_tutorial.md)上使用 8x A100 SXM4-40GB 对 YOLOv5l 进行 1 个 COCO [轮次](https://www.ultralytics.com/glossary/epoch)的 DDP 性能分析结果。

<details>
  <summary>性能分析代码</summary>

```bash
# 准备
t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --runtime=nvidia --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
cd .. && rm -rf app && git clone https://github.com/ultralytics/yolov5 -b master app && cd app
cp data/coco.yaml data/coco_profile.yaml

# 性能分析
python train.py --batch-size 16 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0
python -m torch.distributed.run --nproc_per_node 2 train.py --batch-size 32 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0,1
python -m torch.distributed.run --nproc_per_node 4 train.py --batch-size 64 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0,1,2,3
python -m torch.distributed.run --nproc_per_node 8 train.py --batch-size 128 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0,1,2,3,4,5,6,7
```

</details>

| GPU<br>A100 | 批次大小 | CUDA 内存<br><sup>device0 (G)</sup> | COCO<br><sup>训练</sup> | COCO<br><sup>验证</sup> |
| ----------- | -------- | ----------------------------------- | ----------------------- | ----------------------- |
| 1x          | 16       | 26GB                                | 20:39                   | 0:55                    |
| 2x          | 32       | 26GB                                | 11:43                   | 0:57                    |
| 4x          | 64       | 26GB                                | 5:57                    | 0:55                    |
| 8x          | 128      | 26GB                                | 3:09                    | 0:57                    |

如结果所示，使用 [DistributedDataParallel](https://docs.pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) 与多个 GPU 在训练速度上提供了近乎线性的扩展。使用 8 个 GPU，训练完成速度比单个 GPU 快约 6.5 倍，同时保持每个设备相同的内存使用量。

## 常见问题

如果发生错误，请先阅读下面的检查清单！（可能会节省您的时间）

<details>
  <summary>检查清单（点击展开）</summary>

- 您是否正确阅读了这篇文章？
- 您是否尝试重新克隆代码库？代码**每天**都在变化。
- 您是否尝试搜索您的错误？可能有人已经在此仓库或其他地方遇到过并有解决方案。
- 您是否安装了上面列出的所有要求（包括正确的 Python 和 PyTorch 版本）？
- 您是否在下面"环境"部分列出的其他环境中尝试过？
- 您是否尝试使用其他数据集如 coco128 或 coco2017？这将更容易找到根本原因。

如果您完成了以上所有步骤，请随时按照模板提供尽可能多的详细信息来提出 Issue。

</details>

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

## 致谢

我们要感谢 @MagicFrogSJTU 完成了所有繁重的工作，以及 @glenn-jocher 一路指导我们。

## 另请参阅

- [训练模式](https://docs.ultralytics.com/modes/train/) - 了解如何使用 Ultralytics 训练 YOLO 模型
- [超参数调优](https://docs.ultralytics.com/guides/hyperparameter-tuning/) - 优化您的模型性能
- [Docker 快速入门指南](https://docs.ultralytics.com/guides/docker-quickstart/) - 设置您的 Docker 训练环境
