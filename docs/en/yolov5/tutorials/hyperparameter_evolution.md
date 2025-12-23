---
comments: true
description: 学习如何使用遗传算法优化 YOLOv5 超参数以提高训练性能。包含分步说明。
keywords: YOLOv5, 超参数进化, 遗传算法, 机器学习, 优化, Ultralytics, 超参数调优
---

# YOLOv5 超参数进化

📚 本指南介绍 YOLOv5 🚀 的**超参数进化**。超参数进化是一种使用[遗传算法](https://en.wikipedia.org/wiki/Genetic_algorithm)（GA）进行优化的[超参数优化](https://en.wikipedia.org/wiki/Hyperparameter_optimization)方法。

[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)中的超参数控制训练的各个方面，找到它们的最优值可能是一个挑战。传统方法如网格搜索由于以下原因可能很快变得不可行：

1. 高维搜索空间
2. 维度之间的未知相关性
3. 评估每个点的适应度代价昂贵

这使得遗传算法成为超参数搜索的合适候选方法。

## 开始之前

克隆仓库并在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)会从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)自动下载。

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆
cd yolov5
pip install -r requirements.txt # 安装
```

## 1. 初始化超参数

YOLOv5 有大约 30 个用于各种训练设置的超参数。这些在 `/data/hyps` 目录中的 `*.yaml` 文件中定义。更好的初始猜测将产生更好的最终结果，因此在进化之前正确初始化这些值很重要。如果不确定，只需使用默认值，这些值针对从头开始的 YOLOv5 COCO 训练进行了优化。

```yaml
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# 从头开始低增强 COCO 训练的超参数
# python train.py --batch 64 --cfg yolov5n6.yaml --weights '' --data coco.yaml --img 640 --epochs 300 --linear
# 超参数进化教程请参见 https://github.com/ultralytics/yolov5#tutorials

lr0: 0.01 # 初始学习率 (SGD=1E-2, Adam=1E-3)
lrf: 0.01 # 最终 OneCycleLR 学习率 (lr0 * lrf)
momentum: 0.937 # SGD 动量/Adam beta1
weight_decay: 0.0005 # 优化器权重衰减 5e-4
warmup_epochs: 3.0 # 预热轮次（可以是小数）
warmup_momentum: 0.8 # 预热初始动量
warmup_bias_lr: 0.1 # 预热初始偏置学习率
box: 0.05 # 边界框损失增益
cls: 0.5 # 分类损失增益
cls_pw: 1.0 # 分类 BCELoss positive_weight
obj: 1.0 # 目标损失增益（随像素缩放）
obj_pw: 1.0 # 目标 BCELoss positive_weight
iou_t: 0.20 # IoU 训练阈值
anchor_t: 4.0 # 锚框倍数阈值
# anchors: 3  # 每个输出层的锚框数（0 表示忽略）
fl_gamma: 0.0 # focal loss gamma（efficientDet 默认 gamma=1.5）
hsv_h: 0.015 # 图像 HSV-色调增强（比例）
hsv_s: 0.7 # 图像 HSV-饱和度增强（比例）
hsv_v: 0.4 # 图像 HSV-明度增强（比例）
degrees: 0.0 # 图像旋转（+/- 度）
translate: 0.1 # 图像平移（+/- 比例）
scale: 0.5 # 图像缩放（+/- 增益）
shear: 0.0 # 图像剪切（+/- 度）
perspective: 0.0 # 图像透视（+/- 比例），范围 0-0.001
flipud: 0.0 # 图像上下翻转（概率）
fliplr: 0.5 # 图像左右翻转（概率）
mosaic: 1.0 # 图像马赛克（概率）
mixup: 0.0 # 图像混合（概率）
copy_paste: 0.0 # 分割复制粘贴（概率）
```

## 2. 定义适应度

适应度是我们寻求最大化的值。在 YOLOv5 中，我们将默认适应度函数定义为指标的加权组合：`mAP@0.5` 贡献 10% 的权重，`mAP@0.5:0.95` 贡献剩余的 90%，[精确率 (P)](https://www.ultralytics.com/glossary/precision) 和[召回率 (R)](https://www.ultralytics.com/glossary/recall) 不参与。您可以根据需要调整这些，或使用 utils/metrics.py 中的默认适应度定义（推荐）。

```python
def fitness(x):
    """返回模型适应度，作为加权指标 [P, R, mAP@0.5, mAP@0.5:0.95] 的总和。"""
    w = [0.0, 0.0, 0.1, 0.9]  # [P, R, mAP@0.5, mAP@0.5:0.95] 的权重
    return (x[:, :4] * w).sum(1)
```


## 3. 进化

进化是围绕我们寻求改进的基础场景进行的。本示例中的基础场景是使用预训练的 YOLOv5s 对 COCO128 进行 10 个[轮次](https://www.ultralytics.com/glossary/epoch)的[微调](https://www.ultralytics.com/glossary/fine-tuning)。基础场景训练命令是：

```bash
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache
```

要针对**此场景**进化超参数，从**第 1 节**中定义的初始值开始，并最大化**第 2 节**中定义的适应度，请添加 `--evolve`：

```bash
# 单 GPU
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve

# 多 GPU 带延迟
for i in {0..7}; do
  sleep $((30 * i)) # 30 秒延迟（可选）
  echo "启动 GPU $i..."
  nohup python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device $i --evolve > "evolve_gpu_$i.log" &
done

# 持续训练（谨慎使用）
# for i in {0..7}; do
#   sleep $((30 * i))  # 30 秒延迟（可选）
#   echo "在 GPU $i 上启动持续训练..."
#   (
#     while true; do
#       python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device $i --evolve > "evolve_gpu_$i.log"
#     done
#   ) &
# done
```

默认进化设置将运行基础场景 300 次，即 300 代。您可以通过 `--evolve` 参数修改代数，例如 `python train.py --evolve 1000`。

主要的遗传算子是**交叉**和**变异**。在这项工作中使用变异，以 80% 的概率和 0.04 的方差基于所有先前代中最佳父代的组合创建新后代。结果记录到 `runs/evolve/exp/evolve.csv`，每代保存最高适应度后代为 `runs/evolve/hyp_evolved.yaml`：

```yaml
# YOLOv5 超参数进化结果
# 最佳代: 287
# 最后一代: 300
#    metrics/precision,       metrics/recall,      metrics/mAP_0.5, metrics/mAP_0.5:0.95,         val/box_loss,         val/obj_loss,         val/cls_loss
#              0.54634,              0.55625,              0.58201,              0.33665,             0.056451,             0.042892,             0.013441

lr0: 0.01 # 初始学习率 (SGD=1E-2, Adam=1E-3)
lrf: 0.2 # 最终 OneCycleLR 学习率 (lr0 * lrf)
momentum: 0.937 # SGD 动量/Adam beta1
weight_decay: 0.0005 # 优化器权重衰减 5e-4
warmup_epochs: 3.0 # 预热轮次（可以是小数）
warmup_momentum: 0.8 # 预热初始动量
warmup_bias_lr: 0.1 # 预热初始偏置学习率
box: 0.05 # 边界框损失增益
cls: 0.5 # 分类损失增益
cls_pw: 1.0 # 分类 BCELoss positive_weight
obj: 1.0 # 目标损失增益（随像素缩放）
obj_pw: 1.0 # 目标 BCELoss positive_weight
iou_t: 0.20 # IoU 训练阈值
anchor_t: 4.0 # 锚框倍数阈值
# anchors: 3  # 每个输出层的锚框数（0 表示忽略）
fl_gamma: 0.0 # focal loss gamma（efficientDet 默认 gamma=1.5）
hsv_h: 0.015 # 图像 HSV-色调增强（比例）
hsv_s: 0.7 # 图像 HSV-饱和度增强（比例）
hsv_v: 0.4 # 图像 HSV-明度增强（比例）
degrees: 0.0 # 图像旋转（+/- 度）
translate: 0.1 # 图像平移（+/- 比例）
scale: 0.5 # 图像缩放（+/- 增益）
shear: 0.0 # 图像剪切（+/- 度）
perspective: 0.0 # 图像透视（+/- 比例），范围 0-0.001
flipud: 0.0 # 图像上下翻转（概率）
fliplr: 0.5 # 图像左右翻转（概率）
mosaic: 1.0 # 图像马赛克（概率）
mixup: 0.0 # 图像混合（概率）
copy_paste: 0.0 # 分割复制粘贴（概率）
```

我们建议至少进行 300 代进化以获得最佳结果。请注意，**进化通常代价昂贵且耗时**，因为基础场景要训练数百次，可能需要数百或数千个 GPU 小时。

进化完成后，通过将训练指向保存的文件来重用发现的设置，例如 `python train.py --hyp runs/evolve/hyp_evolved.yaml --data your.yaml --weights yolov5s.pt`。

## 4. 可视化

进化完成后，`evolve.csv` 由 `utils.plots.plot_evolve()` 绘制为 `evolve.png`，每个超参数一个子图，显示适应度（y 轴）与超参数值（x 轴）的关系。黄色表示较高的浓度。垂直分布表示参数已被禁用且不会变异。这在 train.py 的 `meta` 字典中可由用户选择，对于固定参数并防止它们进化很有用。

![evolve](https://github.com/ultralytics/docs/releases/download/0/evolve.avif)

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
