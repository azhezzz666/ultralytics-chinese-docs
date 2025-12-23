---
comments: true
description: 探索 Ultralytics YOLO - 最新的实时目标检测和图像分割技术。了解其功能并在您的项目中最大化发挥其潜力。
keywords: Ultralytics, YOLO, YOLO11, 目标检测, 图像分割, 深度学习, 计算机视觉, 人工智能, 机器学习, 文档, 教程
---

<div align="center">
<br><br>
<a href="https://www.ultralytics.com/events/yolovision?utm_source=github&utm_medium=org&utm_campaign=yv25_event" target="_blank"><img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
<br><br>
<a href="https://docs.ultralytics.com/zh/">中文</a> ·
<a href="https://docs.ultralytics.com/ko/">한국어</a> ·
<a href="https://docs.ultralytics.com/ja/">日本語</a> ·
<a href="https://docs.ultralytics.com/ru/">Русский</a> ·
<a href="https://docs.ultralytics.com/de/">Deutsch</a> ·
<a href="https://docs.ultralytics.com/fr/">Français</a> ·
<a href="https://docs.ultralytics.com/es/">Español</a> ·
<a href="https://docs.ultralytics.com/pt/">Português</a> ·
<a href="https://docs.ultralytics.com/tr/">Türkçe</a> ·
<a href="https://docs.ultralytics.com/vi/">Tiếng Việt</a> ·
<a href="https://docs.ultralytics.com/ar/">العربية</a>
<br><br>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://clickpy.clickhouse.com/dashboard/ultralytics"><img src="https://static.pepy.tech/badge/ultralytics" alt="Ultralytics Downloads"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="Ultralytics YOLO Citation"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolo11"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
    <a href="https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open Ultralytics In Binder"></a>
<br><br>
</div>

# 首页

隆重介绍 Ultralytics [YOLO11](models/yolo11.md)，这是备受赞誉的实时目标检测和图像分割模型的最新版本。YOLO11 基于[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)和[计算机视觉](https://www.ultralytics.com/blog/everything-you-need-to-know-about-computer-vision-in-2025)领域的尖端技术构建，在速度和[准确率](https://www.ultralytics.com/glossary/accuracy)方面提供无与伦比的性能。其精简的设计使其适用于各种应用场景，并可轻松适配不同的硬件平台，从边缘设备到云端 API。

探索 Ultralytics 文档，这是一个全面的资源，旨在帮助您理解和使用其功能和特性。无论您是经验丰富的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)从业者还是该领域的新手，本中心都旨在最大化 YOLO 在您项目中的潜力。

<div align="center">
  <br>
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

## 从哪里开始

<div class="grid cards" markdown>

- :material-clock-fast:{ .lg .middle } &nbsp; **快速入门**

    ***

    使用 pip 安装 `ultralytics`，几分钟内即可启动并运行，训练 YOLO 模型

    ***

    [:octicons-arrow-right-24: 快速入门](quickstart.md)

- :material-image:{ .lg .middle } &nbsp; **预测**

    ***

    使用 YOLO 对新图像、视频和流进行预测 <br /> &nbsp;

    ***

    [:octicons-arrow-right-24: 了解更多](modes/predict.md)

- :fontawesome-solid-brain:{ .lg .middle } &nbsp; **训练模型**

    ***

    从头开始在您自己的自定义数据集上训练新的 YOLO 模型，或加载并训练预训练模型

    ***

    [:octicons-arrow-right-24: 了解更多](modes/train.md)

- :material-magnify-expand:{ .lg .middle } &nbsp; **探索计算机视觉任务**

    ***

    探索 YOLO 任务，如检测、分割、分类、姿态估计、定向边界框和跟踪 <br /> &nbsp;

    ***

    [:octicons-arrow-right-24: 探索任务](tasks/index.md)

- :rocket:{ .lg .middle } &nbsp; **探索 YOLO11 🚀**

    ***

    了解 Ultralytics 最新的最先进 YOLO11 模型及其功能 <br /> &nbsp;

    ***

    [:octicons-arrow-right-24: YOLO11 模型 🚀](models/yolo11.md)

- :material-select-all:{ .lg .middle } &nbsp; **SAM 3：基于概念的分割一切 🚀 新功能**

    ***

    Meta 最新的 SAM 3 具有可提示概念分割功能 - 使用文本或图像示例分割所有实例

    ***

    [:octicons-arrow-right-24: SAM 3 模型](models/sam-3.md)

- :material-scale-balance:{ .lg .middle } &nbsp; **开源，AGPL-3.0 许可证**

    ***

    Ultralytics 提供两种 YOLO 许可证：AGPL-3.0 和企业版。在 [GitHub](https://github.com/ultralytics/ultralytics) 上探索 YOLO。

    ***

    [:octicons-arrow-right-24: YOLO 许可证](https://www.ultralytics.com/license)

</div>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ZN3nRZT7b24"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何在 <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb" target="_blank">Google Colab</a> 中使用您的自定义数据集训练 YOLO11 模型。
</p>

## YOLO：简史

[YOLO](models/index.md)（You Only Look Once，只看一次）是一种流行的[目标检测](https://www.ultralytics.com/glossary/object-detection)和[图像分割](https://www.ultralytics.com/glossary/image-segmentation)模型，由华盛顿大学的 Joseph Redmon 和 Ali Farhadi 开发。YOLO 于 2015 年推出，因其高速度和高准确率而广受欢迎。

- [YOLOv2](models/index.md) 于 2016 年发布，通过引入批量归一化、锚框和维度聚类改进了原始模型。
- [YOLOv3](models/yolov3.md) 于 2018 年推出，使用更高效的骨干网络、多锚点和空间金字塔池化进一步增强了模型性能。
- [YOLOv4](models/yolov4.md) 于 2020 年发布，引入了 Mosaic [数据增强](https://www.ultralytics.com/glossary/data-augmentation)、新的无锚检测头和新的[损失函数](https://www.ultralytics.com/glossary/loss-function)等创新。
- [YOLOv5](models/yolov5.md) 进一步提升了模型性能，并添加了超参数优化、集成实验跟踪和自动导出到流行导出格式等新功能。
- [YOLOv6](models/yolov6.md) 于 2022 年由[美团](https://www.meituan.com/)开源，被用于该公司的许多自动配送机器人。
- [YOLOv7](models/yolov7.md) 在 COCO 关键点数据集上添加了姿态估计等额外任务。
- [YOLOv8](models/yolov8.md) 由 Ultralytics 于 2023 年发布，引入了新功能和改进以增强性能、灵活性和效率，支持全系列视觉 AI 任务。
- [YOLOv9](models/yolov9.md) 引入了可编程梯度信息（PGI）和广义高效层聚合网络（GELAN）等创新方法。
- [YOLOv10](models/yolov10.md) 由[清华大学](https://www.tsinghua.edu.cn/en/)研究人员使用 [Ultralytics](https://www.ultralytics.com/) [Python 包](https://pypi.org/project/ultralytics/)创建，通过引入消除非极大值抑制（NMS）需求的端到端检测头，提供实时[目标检测](tasks/detect.md)的进步。
- **[YOLO11](models/yolo11.md) 🚀**：Ultralytics 最新的 YOLO 模型在多项任务中提供最先进（SOTA）的性能，包括[目标检测](tasks/detect.md)、[分割](tasks/segment.md)、[姿态估计](tasks/pose.md)、[跟踪](modes/track.md)和[分类](tasks/classify.md)，可在各种 AI 应用和领域中部署。
- **[YOLO26](models/yolo26.md) ⚠️ 即将推出**：Ultralytics 下一代 YOLO 模型，针对边缘部署进行优化，具有端到端无 NMS 推理功能。

## YOLO 许可证：Ultralytics YOLO 如何授权？

Ultralytics 提供两种许可选项以适应不同的使用场景：

- **AGPL-3.0 许可证**：这种经 [OSI 批准](https://opensource.org/license/agpl-v3)的开源许可证非常适合学生和爱好者，促进开放协作和知识共享。有关更多详细信息，请参阅 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件。
- **企业许可证**：专为商业用途设计，此许可证允许将 Ultralytics 软件和 AI 模型无缝集成到商业产品和服务中，绕过 AGPL-3.0 的开源要求。如果您的场景涉及将我们的解决方案嵌入商业产品，请通过 [Ultralytics 授权许可](https://www.ultralytics.com/license)联系我们。

我们的许可策略旨在确保对开源项目的任何改进都能回馈社区。我们相信开源，我们的使命是确保我们的贡献能够以造福所有人的方式被使用和扩展。

## 目标检测的演进

目标检测多年来已经有了显著的发展，从传统的计算机视觉技术到先进的深度学习模型。[YOLO 系列模型](https://www.ultralytics.com/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models)一直处于这一演进的前沿，不断突破实时目标检测的可能性边界。

YOLO 的独特方法将目标检测视为单一回归问题，在一次评估中直接从完整图像预测[边界框](https://www.ultralytics.com/glossary/bounding-box)和类别概率。这种革命性的方法使 YOLO 模型比以前的两阶段检测器快得多，同时保持高准确率。

每个新版本的 YOLO 都引入了架构改进和创新技术，提升了各种指标的性能。YOLO11 延续了这一传统，融入了计算机视觉研究的最新进展，为实际应用提供更好的速度-准确率权衡。

## 常见问题

### 什么是 Ultralytics YOLO，它如何改进目标检测？

Ultralytics YOLO 是备受赞誉的 YOLO（You Only Look Once）系列实时目标检测和图像分割的最新进展。它在之前版本的基础上引入了新功能和改进，以增强性能、灵活性和效率。YOLO 支持各种[视觉 AI 任务](tasks/index.md)，如检测、分割、姿态估计、跟踪和分类。其最先进的架构确保了卓越的速度和准确率，使其适用于各种应用，包括边缘设备和云端 API。

### 如何开始安装和设置 YOLO？

开始使用 YOLO 既快速又简单。您可以使用 [pip](https://pypi.org/project/ultralytics/) 安装 Ultralytics 包，几分钟内即可启动并运行。以下是基本安装命令：

!!! example "使用 pip 安装"

    === "CLI"

        ```bash
        pip install -U ultralytics
        ```

有关全面的分步指南，请访问我们的[快速入门](quickstart.md)页面。该资源将帮助您完成安装说明、初始设置和运行您的第一个模型。

### 如何在我的数据集上训练自定义 YOLO 模型？

在您的数据集上训练自定义 YOLO 模型涉及几个详细步骤：

1. 准备您的标注数据集。
2. 在 YAML 文件中配置训练参数。
3. 使用 `yolo TASK train` 命令开始训练。（每个 `TASK` 都有自己的参数）

以下是目标检测任务的示例代码：

!!! example "目标检测任务的训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO 模型（您可以选择 n、s、m、l 或 x 版本）
        model = YOLO("yolo11n.pt")

        # 在您的自定义数据集上开始训练
        model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从命令行训练 YOLO 模型
        yolo detect train data=path/to/dataset.yaml epochs=100 imgsz=640
        ```

有关详细的操作指南，请查看我们的[训练模型](modes/train.md)指南，其中包含优化训练过程的示例和技巧。

### Ultralytics YOLO 有哪些可用的许可选项？

Ultralytics 为 YOLO 提供两种许可选项：

- **AGPL-3.0 许可证**：这种开源许可证非常适合教育和非商业用途，促进开放协作。
- **企业许可证**：专为商业应用设计，允许将 Ultralytics 软件无缝集成到商业产品中，而不受 AGPL-3.0 许可证的限制。

有关更多详细信息，请访问我们的[许可证](https://www.ultralytics.com/license)页面。

### 如何使用 Ultralytics YOLO 进行实时目标跟踪？

Ultralytics YOLO 支持高效且可定制的多目标跟踪。要使用跟踪功能，您可以使用 `yolo track` 命令，如下所示：

!!! example "视频目标跟踪示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO 模型
        model = YOLO("yolo11n.pt")

        # 开始跟踪视频中的目标
        # 您也可以使用实时视频流或网络摄像头输入
        model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # 从命令行对视频执行目标跟踪
        # 您可以指定不同的源，如网络摄像头（0）或 RTSP 流
        yolo track source=path/to/video.mp4
        ```

有关设置和运行目标跟踪的详细指南，请查看我们的[跟踪模式](modes/track.md)文档，其中解释了实时场景中的配置和实际应用。
