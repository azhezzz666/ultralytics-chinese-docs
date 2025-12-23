---
comments: true
description: 深入探索 Ultralytics 强大的 YOLOv5 架构，了解其模型结构、数据增强技术、训练策略和损失计算方法。
keywords: YOLOv5 架构, 目标检测, Ultralytics, YOLO, 模型结构, 数据增强, 训练策略, 损失计算, 深度学习, 机器学习
---

# Ultralytics YOLOv5 架构

YOLOv5（v6.0/6.1）是 Ultralytics 开发的强大目标检测算法。本文深入探讨 YOLOv5 架构、[数据增强](https://www.ultralytics.com/glossary/data-augmentation)策略、训练方法和损失计算技术。这种全面的理解将帮助您改进目标检测在各个领域的实际应用，包括监控、自动驾驶和[图像识别](https://www.ultralytics.com/glossary/image-recognition)。

## 1. 模型结构

YOLOv5 的架构由三个主要部分组成：

- **主干网络（Backbone）**：这是网络的主体。对于 YOLOv5，主干网络使用 `CSPDarknet53` 结构设计，这是对先前版本中使用的 Darknet 架构的修改。
- **颈部网络（Neck）**：这部分连接主干网络和头部。在 YOLOv5 中，使用了 `SPPF`（空间金字塔池化 - 快速版）和 `PANet`（路径聚合网络）结构。
- **头部网络（Head）**：这部分负责生成最终输出。YOLOv5 为此使用 `YOLOv3 Head`。

模型结构如下图所示。模型结构详情可在 [`models/yolov5l.yaml`](https://github.com/ultralytics/yolov5/blob/master/models/yolov5l.yaml) 中找到。

![yolov5](https://github.com/ultralytics/docs/releases/download/0/yolov5-model-structure.avif)

与前代相比，YOLOv5 引入了一些显著改进：

1. 早期版本中的 `Focus` 结构被 `6x6 Conv2d` 结构取代。这一变化提高了效率 [#4825](https://github.com/ultralytics/yolov5/issues/4825)。
2. `SPP` 结构被 `SPPF` 取代。这一改变使处理速度提高了一倍以上，同时保持相同的输出。

要测试 `SPP` 和 `SPPF` 的速度，可以使用以下代码：

<details>
<summary>SPP 与 SPPF 速度对比示例（点击展开）</summary>

```python
import time

import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self):
        """初始化一个具有三种不同大小最大池化层的 SPP 模块。"""
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(5, 1, padding=2)
        self.maxpool2 = nn.MaxPool2d(9, 1, padding=4)
        self.maxpool3 = nn.MaxPool2d(13, 1, padding=6)

    def forward(self, x):
        """对输入 `x` 应用三个最大池化层并沿通道维度连接结果。"""
        o1 = self.maxpool1(x)
        o2 = self.maxpool2(x)
        o3 = self.maxpool3(x)
        return torch.cat([x, o1, o2, o3], dim=1)


class SPPF(nn.Module):
    def __init__(self):
        """使用特定配置的 MaxPool2d 层初始化 SPPF 模块。"""
        super().__init__()
        self.maxpool = nn.MaxPool2d(5, 1, padding=2)

    def forward(self, x):
        """应用顺序最大池化并将结果与输入张量连接。"""
        o1 = self.maxpool(x)
        o2 = self.maxpool(o1)
        o3 = self.maxpool(o2)
        return torch.cat([x, o1, o2, o3], dim=1)


def main():
    """比较 SPP 和 SPPF 在随机张量 (8, 32, 16, 16) 上的输出和性能。"""
    input_tensor = torch.rand(8, 32, 16, 16)
    spp = SPP()
    sppf = SPPF()
    output1 = spp(input_tensor)
    output2 = sppf(input_tensor)

    print(torch.equal(output1, output2))

    t_start = time.time()
    for _ in range(100):
        spp(input_tensor)
    print(f"SPP 时间: {time.time() - t_start}")

    t_start = time.time()
    for _ in range(100):
        sppf(input_tensor)
    print(f"SPPF 时间: {time.time() - t_start}")


if __name__ == "__main__":
    main()
```

结果：

```
True
SPP 时间: 0.5373051166534424
SPPF 时间: 0.20780706405639648
```

</details>


## 2. 数据增强技术

YOLOv5 采用各种数据增强技术来提高模型的泛化能力并减少[过拟合](https://www.ultralytics.com/glossary/overfitting)。这些技术包括：

- **马赛克增强（Mosaic Augmentation）**：一种图像处理技术，将四张训练图像组合成一张，以鼓励[目标检测](https://www.ultralytics.com/glossary/object-detection)模型更好地处理各种目标尺度和位移。

    ![mosaic](https://github.com/ultralytics/docs/releases/download/0/mosaic-augmentation.avif)

- **复制粘贴增强（Copy-Paste Augmentation）**：一种创新的数据增强方法，从一张图像中复制随机区域并粘贴到另一张随机选择的图像上，有效生成新的训练样本。

    ![copy-paste](https://github.com/ultralytics/docs/releases/download/0/copy-paste.avif)

- **随机仿射变换（Random Affine Transformations）**：包括图像的随机旋转、缩放、平移和剪切。

    ![random-affine](https://github.com/ultralytics/docs/releases/download/0/random-affine-transformations.avif)

- **MixUp 增强**：一种通过对两张图像及其关联标签进行线性组合来创建合成图像的方法。

    ![mixup](https://github.com/ultralytics/docs/releases/download/0/mixup.avif)

- **Albumentations**：一个强大的图像增强库，支持多种增强技术。了解更多关于[使用 Albumentations 增强](https://www.ultralytics.com/blog/using-albumentations-augmentations-to-diversify-your-data)的信息。

- **HSV 增强**：对图像的色调、饱和度和明度进行随机更改。

    ![hsv](https://github.com/ultralytics/docs/releases/download/0/hsv-augmentation.avif)

- **随机水平翻转（Random Horizontal Flip）**：一种随机水平翻转图像的增强方法。

    ![horizontal-flip](https://github.com/ultralytics/docs/releases/download/0/random-horizontal-flip.avif)

## 3. 训练策略

YOLOv5 应用多种复杂的训练策略来增强模型性能，包括：

- **多尺度训练（Multiscale Training）**：在训练过程中，输入图像在原始大小的 0.5 到 1.5 倍范围内随机缩放。
- **自动锚框（AutoAnchor）**：此策略优化先验锚框以匹配自定义数据中真实框的统计特征。
- **预热和余弦学习率调度器（Warmup and Cosine LR Scheduler）**：一种调整[学习率](https://www.ultralytics.com/glossary/learning-rate)以增强模型性能的方法。
- **指数移动平均（EMA）**：一种使用过去步骤参数平均值来稳定训练过程并减少泛化误差的策略。
- **[混合精度](https://www.ultralytics.com/glossary/mixed-precision)训练**：一种以半[精度](https://www.ultralytics.com/glossary/precision)格式执行操作的方法，减少内存使用并提高计算速度。
- **超参数进化（Hyperparameter Evolution）**：一种自动调整超参数以实现最佳性能的策略。了解更多关于[超参数调优](https://docs.ultralytics.com/guides/hyperparameter-tuning/)的信息。

## 4. 附加功能

### 4.1 计算损失

YOLOv5 中的损失计算为三个独立损失分量的组合：

- **类别损失（BCE Loss）**：二元交叉熵损失，衡量分类任务的误差。
- **目标性损失（BCE Loss）**：另一个二元交叉熵损失，计算检测特定网格单元中是否存在目标的误差。
- **位置损失（CIoU Loss）**：完整 IoU 损失，衡量在网格单元内定位目标的误差。

整体[损失函数](https://www.ultralytics.com/glossary/loss-function)表示为：

![loss](https://latex.codecogs.com/svg.image?Loss=\lambda_1L_{cls}+\lambda_2L_{obj}+\lambda_3L_{loc})

### 4.2 平衡损失

三个预测层（`P3`、`P4`、`P5`）的目标性损失权重不同。平衡权重分别为 `[4.0, 1.0, 0.4]`。这种方法确保不同尺度的预测对总损失有适当的贡献。

![obj_loss](https://latex.codecogs.com/svg.image?L_{obj}=4.0\cdot&space;L_{obj}^{small}+1.0\cdot&space;L_{obj}^{medium}+0.4\cdot&space;L_{obj}^{large})

### 4.3 消除网格敏感性

与早期版本的 YOLO 相比，YOLOv5 架构对边界框预测策略进行了一些重要更改。在 YOLOv2 和 YOLOv3 中，边界框坐标直接使用最后一层的激活值预测。

![b_x](<https://latex.codecogs.com/svg.image?b_x=\sigma(t_x)+c_x>)
![b_y](<https://latex.codecogs.com/svg.image?b_y=\sigma(t_y)+c_y>)
![b_w](https://latex.codecogs.com/svg.image?b_w=p_w\cdot&space;e^{t_w})
![b_h](https://latex.codecogs.com/svg.image?b_h=p_h\cdot&space;e^{t_h})

<img src="https://user-images.githubusercontent.com/31005897/158508027-8bf63c28-8290-467b-8a3e-4ad09235001a.png#pic_center" width=40% alt="YOLOv5 网格计算">

然而，在 YOLOv5 中，预测边界框坐标的公式已更新，以减少网格敏感性并防止模型预测无界的边界框尺寸。

计算预测[边界框](https://www.ultralytics.com/glossary/bounding-box)的修订公式如下：

![bx](<https://latex.codecogs.com/svg.image?b_x=(2\cdot\sigma(t_x)-0.5)+c_x>)
![by](<https://latex.codecogs.com/svg.image?b_y=(2\cdot\sigma(t_y)-0.5)+c_y>)
![bw](<https://latex.codecogs.com/svg.image?b_w=p_w\cdot(2\cdot\sigma(t_w))^2>)
![bh](<https://latex.codecogs.com/svg.image?b_h=p_h\cdot(2\cdot\sigma(t_h))^2>)

比较缩放前后的中心点偏移。中心点偏移范围从 (0, 1) 调整为 (-0.5, 1.5)。因此，偏移可以轻松获得 0 或 1。

<img src="https://user-images.githubusercontent.com/31005897/158508052-c24bc5e8-05c1-4154-ac97-2e1ec71f582e.png#pic_center" width=40% alt="YOLOv5 网格缩放">

比较调整前后的高度和宽度缩放比（相对于锚框）。原始 yolo/darknet 边界框方程有一个严重缺陷。宽度和高度完全无界，因为它们只是 out=exp(in)，这是危险的，因为它可能导致梯度失控、不稳定、NaN 损失，最终导致训练完全失败。[参考此问题](https://github.com/ultralytics/yolov5/issues/471#issuecomment-662009779)了解更多详情。

<img src="https://user-images.githubusercontent.com/31005897/158508089-5ac0c7a3-6358-44b7-863e-a6e45babb842.png#pic_center" width=40% alt="YOLOv5 无界缩放">

### 4.4 构建目标

YOLOv5 中的构建目标过程对于训练效率和模型[准确率](https://www.ultralytics.com/glossary/accuracy)至关重要。它涉及将真实框分配给输出图中适当的网格单元，并将它们与适当的锚框匹配。

此过程遵循以下步骤：

- 计算真实框尺寸与每个锚框模板尺寸的比率。

![rw](https://latex.codecogs.com/svg.image?r_w=w_{gt}/w_{at})

![rh](https://latex.codecogs.com/svg.image?r_h=h_{gt}/h_{at})

![rwmax](<https://latex.codecogs.com/svg.image?r_w^{max}=max(r_w,1/r_w)>)

![rhmax](<https://latex.codecogs.com/svg.image?r_h^{max}=max(r_h,1/r_h)>)

![rmax](<https://latex.codecogs.com/svg.image?r^{max}=max(r_w^{max},r_h^{max})>)

![match](https://latex.codecogs.com/svg.image?r^{max}<{\rm&space;anchor_t})

<img src="https://user-images.githubusercontent.com/31005897/158508119-fbb2e483-7b8c-4975-8e1f-f510d367f8ff.png#pic_center" width=70% alt="YOLOv5 IoU 计算">

- 如果计算的比率在阈值内，则将真实框与相应的锚框匹配。

<img src="https://user-images.githubusercontent.com/31005897/158508771-b6e7cab4-8de6-47f9-9abf-cdf14c275dfe.png#pic_center" width=70% alt="YOLOv5 网格重叠">

- 将匹配的锚框分配给适当的单元格，请记住，由于修订的中心点偏移，一个真实框可以分配给多个锚框，因为中心点偏移范围从 (0, 1) 调整为 (-0.5, 1.5)，使得额外匹配成为可能。

<img src="https://user-images.githubusercontent.com/31005897/158508139-9db4e8c2-cf96-47e0-bc80-35d11512f296.png#pic_center" width=70% alt="YOLOv5 锚框选择">

通过这种方式，构建目标过程确保每个真实目标在训练过程中被正确分配和匹配，使 YOLOv5 能够更有效地学习目标检测任务。

## 总结

总之，YOLOv5 代表了实时目标检测模型发展的重大进步。通过整合各种新功能、增强和训练策略，它在性能和效率方面超越了 YOLO 系列的先前版本。

YOLOv5 的主要增强包括使用动态架构、广泛的数据增强技术、创新的训练策略，以及在计算损失和构建目标过程中的重要调整。所有这些创新显著提高了目标检测的准确性和效率，同时保持了高速度，这是 YOLO 模型的标志。
