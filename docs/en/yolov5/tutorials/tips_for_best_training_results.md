---
comments: true
description: 了解如何使用 YOLOv5 获得最佳 mAP 和训练结果。学习数据集、模型选择和训练设置的最佳实践。
keywords: YOLOv5 训练, mAP, 数据集最佳实践, 模型选择, 训练设置, YOLOv5 指南, YOLOv5 教程, 机器学习
---

# YOLOv5 最佳训练效果技巧

📚 本指南介绍如何使用 YOLOv5 🚀 获得最佳 mAP 和训练结果。

大多数情况下，**只要你的数据集足够大且标注良好**，无需更改模型或训练设置即可获得良好结果。如果一开始没有获得好的结果，可以采取一些步骤来改进，但我们始终建议用户**首先使用所有默认设置进行训练**，然后再考虑任何更改。这有助于建立性能基准并发现需要改进的地方。

如果你对训练结果有疑问，**我们建议你提供尽可能多的信息**以获得有用的回复，包括结果图表（训练损失、验证损失、P、R、mAP）、PR 曲线、[混淆矩阵](https://www.ultralytics.com/glossary/confusion-matrix)、训练马赛克图、测试结果和数据集统计图像（如 labels.png）。所有这些都位于你的 `project/name` 目录中，通常是 `yolov5/runs/train/exp`。

我们为希望在 YOLOv5 训练中获得最佳结果的用户整理了以下完整指南。

## 数据集

- **每类图像数量**：建议每类 ≥ 1500 张图像
- **每类实例数量**：建议每类 ≥ 10000 个实例（标注对象）
- **图像多样性**：必须能代表部署环境。对于实际应用场景，我们建议使用来自不同时间段、不同季节、不同天气、不同光照、不同角度、不同来源（网络抓取、本地采集、不同相机）等的图像。
- **标注一致性**：所有图像中所有类别的所有实例都必须被标注。部分标注将无法正常工作。
- **标注[准确性](https://www.ultralytics.com/glossary/accuracy)**：标注必须紧密包围每个对象。对象与其[边界框](https://www.ultralytics.com/glossary/bounding-box)之间不应存在空隙。不应有对象缺少标注。
- **训练/验证集划分规范**：确保验证集和测试集的图像永远不会出现在训练集中，以避免过于乐观的指标。保持各划分之间的类别分布相似。
- **标注验证**：在训练开始时查看 `train_batch*.jpg` 以验证标注是否正确，即查看[示例](./train_custom_data.md#local-logging)马赛克图。
- **背景图像**：背景图像是不包含任何对象的图像，添加到数据集中以减少误检（FP）。我们建议添加约 0-10% 的背景图像以帮助减少误检（COCO 有 1000 张背景图像作为参考，占总数的 1%）。背景图像不需要标注。

<a href="https://arxiv.org/abs/1405.0312"><img width="800" src="https://github.com/ultralytics/docs/releases/download/0/coco-analysis.avif" alt="COCO 分析"></a>

## 模型选择

较大的模型如 YOLOv5x 和 [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/tag/v5.0) 几乎在所有情况下都能产生更好的结果，但参数更多，需要更多 CUDA 内存进行训练，运行速度也更慢。对于**移动端**部署，我们推荐 YOLOv5s/m；对于**云端**部署，我们推荐 YOLOv5l/x。请参阅我们的 README [表格](https://github.com/ultralytics/yolov5#pretrained-checkpoints)以获取所有模型的完整比较。

<p align="center"><img width="700" alt="YOLOv5 模型" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-model-comparison.avif"></p>

- **从预训练权重开始**：推荐用于中小型数据集（如 [VOC](https://github.com/ultralytics/yolov5/blob/master/data/VOC.yaml)、[VisDrone](https://github.com/ultralytics/yolov5/blob/master/data/VisDrone.yaml)、[GlobalWheat](https://github.com/ultralytics/yolov5/blob/master/data/GlobalWheat2020.yaml)）。将模型名称传递给 `--weights` 参数。模型会自动从[最新 YOLOv5 发布版本](https://github.com/ultralytics/yolov5/releases)下载。

    ```bash
    python train.py --data custom.yaml --weights yolov5s.pt
    python train.py --data custom.yaml --weights yolov5m.pt
    python train.py --data custom.yaml --weights yolov5l.pt
    python train.py --data custom.yaml --weights yolov5x.pt
    python train.py --data custom.yaml --weights custom_pretrained.pt
    ```

- **从头开始训练**：推荐用于大型数据集（如 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml)、[Objects365](https://github.com/ultralytics/yolov5/blob/master/data/Objects365.yaml)、[OIv6](https://storage.googleapis.com/openimages/web/index.html)）。传入你感兴趣的模型架构 YAML 文件，以及空的 `--weights ''` 参数：

    ```bash
    python train.py --data custom.yaml --weights '' --cfg yolov5s.yaml
    python train.py --data custom.yaml --weights '' --cfg yolov5m.yaml
    python train.py --data custom.yaml --weights '' --cfg yolov5l.yaml
    python train.py --data custom.yaml --weights '' --cfg yolov5x.yaml
    ```

## 训练设置

在修改任何内容之前，**首先使用默认设置进行训练以建立性能基准**。完整的 train.py 设置列表可以在 [train.py](https://github.com/ultralytics/yolov5/blob/master/train.py) 的参数解析器中找到。

- **[训练轮数](https://www.ultralytics.com/glossary/epoch)**：从 300 轮开始。如果提前过拟合，可以减少轮数。如果 300 轮后没有发生[过拟合](https://www.ultralytics.com/glossary/overfitting)，可以训练更长时间，如 600、1200 轮等。
- **图像尺寸**：COCO 以原始分辨率 `--img 640` 进行训练，但由于数据集中有大量小目标，以更高分辨率（如 `--img 1280`）训练可能会有所帮助。如果有很多小目标，自定义数据集将受益于以原始或更高分辨率进行训练。最佳推理结果是在与训练相同的 `--img` 下获得的，即如果你以 `--img 1280` 训练，也应该以 `--img 1280` 进行测试和检测。
- **[批量大小](https://www.ultralytics.com/glossary/batch-size)**：使用硬件允许的最大 `--batch-size`。小批量大小会产生较差的[批量归一化](https://www.ultralytics.com/glossary/batch-normalization)统计数据，应该避免。你可以使用 `--batch-size -1` 自动选择 GPU 的最佳批量大小。
- **[学习率](https://www.ultralytics.com/glossary/learning-rate)**：默认学习率调度在大多数情况下效果良好。为了更快收敛，你可以尝试使用 `--cos-lr` 标志启用余弦学习率调度，它会在训练轮数中按余弦曲线逐渐降低学习率。
- **[数据增强](https://www.ultralytics.com/glossary/data-augmentation)**：YOLOv5 包含各种增强技术，如马赛克增强，它将多个训练图像组合在一起。在最后几轮，考虑使用 `--close-mosaic 10` 禁用马赛克增强，这有助于稳定训练。
- **超参数**：默认超参数在 [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) 中。我们建议你首先使用默认超参数进行训练，然后再考虑修改任何参数。一般来说，增加增强超参数会减少和延迟过拟合，允许更长的训练时间和更高的最终 mAP。减少损失分量增益超参数（如 `hyp['obj']`）将有助于减少这些特定损失分量的过拟合。有关优化这些超参数的自动化方法，请参阅我们的[超参数进化教程](./hyperparameter_evolution.md)。
- **[混合精度](https://www.ultralytics.com/glossary/mixed-precision)训练**：使用 `--amp` 启用混合精度训练，可以加速训练并减少内存使用，而不会牺牲模型精度。
- **多 GPU 训练**：如果你有多个 GPU，使用 `--device 0,1,2,3` 将训练分布到多个 GPU 上，可以显著减少训练时间。
- **早停**：使用 `--patience 50` 在验证指标 50 轮内没有改善时停止训练，节省时间并防止过拟合。

## 高级优化技术

- **[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)**：对于专业数据集，从预训练权重开始，在训练过程中逐步解冻层，使模型适应你的特定任务。
- **[模型剪枝](https://www.ultralytics.com/glossary/model-pruning)**：训练后，考虑剪枝模型以移除冗余权重，在不显著损失性能的情况下减小模型大小。
- **[模型集成](https://www.ultralytics.com/glossary/model-ensemble)**：对于关键应用，使用不同配置训练多个模型并组合它们的预测以提高准确性。
- **[测试时增强](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation/)**：在推理时使用 `--augment` 启用 TTA，通过对输入图像的增强版本进行平均来提高预测准确性。

## 延伸阅读

如果你想了解更多，Karpathy 的《训练[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)的秘诀》是一个很好的起点，其中包含适用于所有机器学习领域的优秀训练思路：[https://karpathy.github.io/2019/04/25/recipe/](https://karpathy.github.io/2019/04/25/recipe/)

有关训练设置和配置的更多详细信息，请参阅 [Ultralytics 训练设置文档](https://docs.ultralytics.com/modes/train/)，其中提供了所有可用参数的全面解释。

祝你好运 🍀，如果有任何其他问题，请告诉我们！

## 常见问题

### 如何判断模型是否过拟合？

如果训练损失持续下降而验证损失开始上升，你的模型可能正在过拟合。监控验证 mAP - 如果它在训练损失持续改善的同时趋于平稳或下降，这就是过拟合的迹象。解决方案包括添加更多训练数据、增加数据增强或实施正则化技术。

### YOLOv5 训练的最佳批量大小是多少？

最佳批量大小取决于你的 GPU 内存。较大的批量大小通常提供更好的批量归一化统计数据和训练稳定性。使用硬件能够处理的最大批量大小而不会耗尽内存。你可以使用 `--batch-size -1` 自动确定适合你设置的最佳批量大小。

### 如何加速 YOLOv5 训练？

要加速训练，可以尝试：使用 `--amp` 启用混合精度训练，使用 `--device 0,1,2,3` 使用多个 GPU，使用 `--cache` 缓存数据集，以及优化批量大小。如果绝对精度不是关键，也可以考虑使用较小的模型变体如 YOLOv5s。
