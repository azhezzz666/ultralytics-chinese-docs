---
comments: true
description: 探索 Ultralytics HUB App，在 iOS 和 Android 设备上运行 YOLO 模型，通过硬件加速实现实时目标检测。
keywords: Ultralytics HUB App, YOLO 模型, 移动应用, iOS, Android, 硬件加速, YOLOv5, YOLOv8, YOLO11, 神经引擎, GPU, NNAPI, 实时目标检测
---

# Ultralytics HUB App

<a href="https://www.ultralytics.com/hub" target="_blank">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub.avif" alt="Ultralytics HUB 预览图"></a>
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
  <a href="https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/master/app/google-play.svg" width="15%" alt="Google Play store"></a>&nbsp;
</div>

欢迎使用 Ultralytics HUB App！这款强大的移动应用程序允许您直接在 [iOS](https://apps.apple.com/xk/app/ultralytics-hub/id1583935240) 和 [Android](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app) 设备上运行 Ultralytics YOLO 模型，包括 YOLOv5、YOLOv8 和 YOLO11。HUB App 利用硬件加速功能，如 iOS 上的 Apple 神经引擎 (ANE) 或 Android GPU 和[神经网络](https://www.ultralytics.com/glossary/neural-network-nn) API (NNAPI) 代理，在您的移动设备上实现令人印象深刻的实时性能。

## 功能

- **运行 Ultralytics YOLO 模型**：在您的移动设备上体验 YOLO 模型的强大功能，用于实时[目标检测](https://www.ultralytics.com/glossary/object-detection)、[图像分割](https://www.ultralytics.com/glossary/image-segmentation)和[图像识别](https://www.ultralytics.com/glossary/image-recognition)任务。
- **硬件加速**：受益于 iOS 设备上的 Apple ANE 或 Android GPU 和 NNAPI 代理，获得优化的性能和效率。
- **自定义模型训练**：使用 [Ultralytics HUB 平台](https://www.ultralytics.com/hub)训练自定义模型，并使用 HUB App 实时预览。
- **移动兼容性**：HUB App 支持 iOS 和 Android 设备，将 YOLO 模型的强大功能带给广泛的用户。
- **实时性能**：在现代设备上实现高达每秒 30 帧的令人印象深刻的推理速度。
- **模型量化**：模型通过 FP16 或 INT8 量化进行优化，以实现更快的移动推理，同时不会显著损失准确率。

## 开始使用

开始使用 Ultralytics HUB App 非常简单：

1. 从 [App Store](https://apps.apple.com/xk/app/ultralytics-hub/id1583935240)（iOS）或 [Google Play](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app)（Android）下载应用。
2. 使用您的 Ultralytics 账户登录（如果没有账户，请创建一个）。
3. 选择预训练模型或您的自定义模型之一。
4. 使用设备的摄像头开始实时检测目标。

## App 文档

- [**iOS**](ios.md)：了解在 Apple 神经引擎上加速的 YOLO CoreML 模型，适用于 iPhone 和 iPad。
- [**Android**](android.md)：探索 Android 移动设备上的 TFLite 加速。

## 与 Ultralytics HUB 集成

Ultralytics HUB App 与 [Ultralytics HUB 平台](https://docs.ultralytics.com/hub/)完全集成，允许您：

- 无需编码知识即可在云端训练自定义模型
- 在一个地方管理您的数据集、项目和模型
- 直接在移动设备上预览和测试训练好的模型
- 跨不同平台部署模型用于各种应用

立即下载 Ultralytics HUB App 到您的移动设备，释放 YOLO 模型的移动端潜力。有关使用 Ultralytics HUB 平台训练、部署和使用自定义模型的更多信息，请查看我们全面的 [HUB 文档](../index.md)。

## 常见问题

### 我可以在 Ultralytics HUB App 上运行哪些模型？

Ultralytics HUB App 支持运行 YOLOv5、YOLOv8 和 YOLO11 模型。您可以使用预训练模型或使用 [Ultralytics HUB 平台](https://www.ultralytics.com/blog/how-to-train-and-deploy-yolo11-using-ultralytics-hub)训练自己的自定义模型。

### 模型性能如何针对移动设备进行优化？

模型通过量化（FP16 或 INT8）进行优化，并利用硬件加速功能，如 iOS 设备上的 Apple 神经引擎或 Android 设备上的 GPU 和 NNAPI 代理。这使得在保持良好准确率的同时实现实时推理。

### 我可以在应用上使用自定义训练的模型吗？

可以！您可以使用 [Ultralytics HUB 云训练](https://docs.ultralytics.com/hub/cloud-training/)功能训练自定义模型，然后直接部署到 HUB App 上，在移动设备上进行测试和使用。
