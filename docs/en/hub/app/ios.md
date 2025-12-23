---
comments: true
description: 探索 Ultralytics iOS App，在您的 iPhone 或 iPad 上运行 YOLO 模型。通过 Apple 神经引擎实现快速、实时的目标检测。
keywords: Ultralytics, iOS App, YOLO 模型, 实时目标检测, Apple 神经引擎, Core ML, FP16 量化, INT8 量化, 机器学习
---

# Ultralytics iOS App：使用 YOLO 模型进行实时目标检测

<a href="https://www.ultralytics.com/hub" target="_blank">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-android-app-detection.avif" alt="Ultralytics HUB 预览图"></a>
<br>
<div align="center">
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
  <br>
  <br>
  <a href="https://apps.apple.com/xk/app/ultralytics-hub/id1583935240" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/master/app/app-store.svg" width="15%" alt="Apple App store"></a>
</div>

Ultralytics iOS App 是一款强大的工具，允许您直接在 iPhone 或 iPad 上运行 YOLO 模型进行实时目标检测。此应用利用 Apple 神经引擎和 Core ML 进行模型优化和加速，实现快速高效的目标检测。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/AIvrQ7y0aLo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics HUB App 入门指南（iOS 和 Android）
</p>

## 量化和加速

为了在您的 iOS 设备上实现实时性能，YOLO 模型被量化为 FP16 或 INT8 [精度](https://www.ultralytics.com/glossary/precision)。量化是一个降低模型权重和偏置数值精度的过程，从而减小模型大小和所需的计算量。这使得推理时间更快，同时不会显著影响模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。

### FP16 量化

FP16（或半精度）量化将模型的 32 位浮点数转换为 16 位浮点数。这将模型大小减半并加速推理过程，同时在准确率和性能之间保持良好的平衡。

### INT8 量化

INT8（或 8 位整数）量化通过将模型的 32 位浮点数转换为 8 位整数，进一步减小模型大小和计算需求。这种量化方法可以带来显著的加速，但可能会导致准确率略有下降。

## Apple 神经引擎

Apple 神经引擎 (ANE) 是集成在 Apple A 系列和 M 系列芯片中的专用硬件组件。它专为加速[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)任务而设计，特别是[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)，允许更快、更高效地执行您的 YOLO 模型。

通过将量化的 YOLO 模型与 Apple 神经引擎相结合，Ultralytics iOS App 在您的 iOS 设备上实现实时目标检测，同时不会影响准确率或性能。

| 发布年份 | iPhone 名称                                          | 芯片名称                                              | 制程工艺 | ANE TOPs |
| -------- | ---------------------------------------------------- | ----------------------------------------------------- | -------- | -------- |
| 2017     | [iPhone X](https://en.wikipedia.org/wiki/IPhone_X)   | [A11 Bionic](https://en.wikipedia.org/wiki/Apple_A11) | 10 nm    | 0.6      |
| 2018     | [iPhone XS](https://en.wikipedia.org/wiki/IPhone_XS) | [A12 Bionic](https://en.wikipedia.org/wiki/Apple_A12) | 7 nm     | 5        |
| 2019     | [iPhone 11](https://en.wikipedia.org/wiki/IPhone_11) | [A13 Bionic](https://en.wikipedia.org/wiki/Apple_A13) | 7 nm     | 6        |
| 2020     | [iPhone 12](https://en.wikipedia.org/wiki/IPhone_12) | [A14 Bionic](https://en.wikipedia.org/wiki/Apple_A14) | 5 nm     | 11       |
| 2021     | [iPhone 13](https://en.wikipedia.org/wiki/IPhone_13) | [A15 Bionic](https://en.wikipedia.org/wiki/Apple_A15) | 5 nm     | 15.8     |
| 2022     | [iPhone 14](https://en.wikipedia.org/wiki/IPhone_14) | [A16 Bionic](https://en.wikipedia.org/wiki/Apple_A16) | 4 nm     | 17.0     |
| 2023     | [iPhone 15](https://en.wikipedia.org/wiki/IPhone_15) | [A17 Pro](https://en.wikipedia.org/wiki/Apple_A17)    | 3 nm     | 35.0     |

请注意，此列表包含 2017 年以来的 iPhone 型号，ANE TOPs 值为近似值。

## CoreML 集成

Ultralytics iOS App 利用 [CoreML](https://docs.ultralytics.com/integrations/coreml/)（Apple 的基础机器学习框架）来优化 iOS 设备上的 YOLO 模型。CoreML 提供多项优势：

- **设备端处理**：所有推理都在您的设备上本地进行，确保数据隐私并消除对互联网连接的需求
- **硬件加速**：自动利用 Apple 神经引擎、CPU 和 GPU 以获得最佳性能
- **无缝集成**：与 iOS 相机和系统框架原生配合

CoreML 将 YOLO 模型转换为针对 Apple 设备优化的格式，允许高效执行同时保持检测准确率。

## 开始使用 Ultralytics iOS App

要开始使用 Ultralytics iOS App，请按照以下步骤操作：

1. 从 [App Store](https://apps.apple.com/xk/app/ultralytics-hub/id1583935240) 下载 Ultralytics App。

2. 在您的 iOS 设备上启动应用并使用您的 Ultralytics 账户登录。如果您还没有账户，请在 [Ultralytics HUB](https://hub.ultralytics.com/) 创建一个。

3. 登录后，您将看到已训练的 YOLO 模型列表。选择一个模型用于目标检测。

4. 授予应用访问设备摄像头的权限。

5. 将设备的摄像头对准您想要检测的目标。应用将在检测到目标时实时显示边界框和类别标签。

6. 探索应用的设置以调整检测阈值、启用或禁用特定目标类别等。

使用 Ultralytics iOS App，您现在可以在 iPhone 或 iPad 上利用 YOLO 模型进行实时目标检测，由 Apple 神经引擎提供支持，并通过 FP16 或 INT8 量化进行优化。
