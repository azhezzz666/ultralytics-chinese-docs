---
comments: true
description: 使用 Ultralytics 在 Android 上体验实时目标检测。利用 YOLO 模型进行高效快速的目标识别。立即下载！
keywords: Ultralytics, Android 应用, 实时目标检测, YOLO 模型, TensorFlow Lite, FP16 量化, INT8 量化, 硬件代理, 移动 AI, 下载应用
---

# Ultralytics Android App：使用 YOLO 模型进行实时目标检测

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
  <a href="https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/master/app/google-play.svg" width="15%" alt="Google Play store"></a>&nbsp;
</div>

Ultralytics Android App 是一款强大的工具，允许您直接在 Android 设备上运行 YOLO 模型进行实时目标检测。此应用利用 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite 进行模型优化，并使用各种硬件代理进行加速，实现快速高效的目标检测。

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

为了在您的 Android 设备上实现实时性能，YOLO 模型被量化为 FP16 或 INT8 [精度](https://www.ultralytics.com/glossary/precision)。量化是一个降低模型权重和偏置数值精度的过程，从而减小模型大小和所需的计算量。这使得推理时间更快，同时不会显著影响模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。

### FP16 量化

FP16（或半精度）量化将模型的 32 位浮点数转换为 16 位浮点数。这将模型大小减半并加速推理过程，同时在准确率和性能之间保持良好的平衡。

### INT8 量化

INT8（或 8 位整数）量化通过将模型的 32 位浮点数转换为 8 位整数，进一步减小模型大小和计算需求。这种量化方法可以带来显著的加速，但由于数值精度较低，可能会导致[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) 略有下降。

!!! tip "INT8 模型中的 mAP 下降"

    INT8 模型中降低的数值精度可能会在量化过程中导致一些信息丢失，这可能导致 mAP 略有下降。然而，考虑到 INT8 量化带来的显著性能提升，这种权衡通常是可以接受的。

## 代理和性能差异

Android 设备上有不同的代理可用于加速模型推理。这些代理包括 CPU、[GPU](https://ai.google.dev/edge/litert/android/gpu)、[Hexagon](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide) 和 [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)。这些代理的性能因设备的硬件供应商、产品线和设备中使用的特定芯片组而异。

1. **CPU**：默认选项，在大多数设备上具有合理的性能。
2. **GPU**：利用设备的 GPU 进行更快的推理。它可以在具有强大 GPU 的设备上提供显著的性能提升。
3. **Hexagon**：利用高通的 Hexagon DSP 进行更快、更高效的处理。此选项在配备高通骁龙处理器的设备上可用。
4. **NNAPI**：Android [神经网络](https://www.ultralytics.com/glossary/neural-network-nn) API (NNAPI) 作为在 Android 设备上运行机器学习模型的抽象层。NNAPI 可以利用各种硬件加速器，如 CPU、GPU 和专用 AI 芯片（例如 Google 的 Edge TPU 或 Pixel Neural Core）。

以下是显示主要供应商、其产品线、热门设备和支持的代理的表格：

| 供应商                                    | 产品线                                                                               | 热门设备                                                                                                                                                                     | 支持的代理               |
| ----------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| [高通](https://www.qualcomm.com/)         | [骁龙（如 800 系列）](https://www.qualcomm.com/snapdragon/overview)                  | [三星 Galaxy S21](https://www.samsung.com/us/mobile/phones/galaxy-s/)、[一加 9](https://www.oneplus.com/9)、[Google Pixel 6](https://store.google.com/product/pixel_6)       | CPU、GPU、Hexagon、NNAPI |
| [三星](https://www.samsung.com/)          | [Exynos（如 Exynos 2100）](https://www.samsung.com/semiconductor/minisite/exynos/)   | [三星 Galaxy S21（全球版）](https://www.samsung.com/us/mobile/phones/galaxy-s/)                                                                                              | CPU、GPU、NNAPI          |
| [联发科](https://i.mediatek.com/)         | [天玑（如天玑 1200）](https://i.mediatek.com/dimensity-1200)                         | [Realme GT](https://www.realme.com/global/realme-gt)、[小米 Redmi Note](https://www.mi.com/global/phone/redmi/note-list)                                                     | CPU、GPU、NNAPI          |
| [海思](https://www.hisilicon.com/cn)      | [麒麟（如麒麟 990）](https://www.hisilicon.com/en/products/Kirin)                    | [华为 P40 Pro](https://consumer.huawei.com/en/phones/)、[华为 Mate 30 Pro](https://consumer.huawei.com/en/phones/)                                                           | CPU、GPU、NNAPI          |
| [NVIDIA](https://www.nvidia.com/)         | [Tegra（如 Tegra X1）](https://developer.nvidia.com/content/tegra-x1)                | [NVIDIA Shield TV](https://www.nvidia.com/en-us/shield/shield-tv/)、[任天堂 Switch](https://www.nintendo.com/switch/)                                                        | CPU、GPU、NNAPI          |

请注意，上述设备列表并非详尽无遗，可能因具体芯片组和设备型号而异。请始终在目标设备上测试您的模型以确保兼容性和最佳性能。

请记住，代理的选择会影响性能和模型兼容性。例如，某些模型可能无法与某些代理配合使用，或者某个代理可能在特定设备上不可用。因此，在目标设备上测试您的模型和所选代理以获得最佳结果至关重要。

## 开始使用 Ultralytics Android App

要开始使用 Ultralytics Android App，请按照以下步骤操作：

1. 从 [Google Play 商店](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app)下载 Ultralytics App。

2. 在您的 Android 设备上启动应用并使用您的 Ultralytics 账户登录。如果您还没有账户，请在 <https://hub.ultralytics.com/> 创建一个。

3. 登录后，您将看到已训练的 YOLO 模型列表。选择一个模型用于目标检测。

4. 授予应用访问设备摄像头的权限。

5. 将设备的摄像头对准您想要检测的目标。应用将在检测到目标时实时显示边界框和类别标签。

6. 探索应用的设置以调整检测阈值、启用或禁用特定目标类别等。

使用 Ultralytics Android App，您现在可以在指尖体验使用 YOLO 模型进行实时目标检测的强大功能。尽情探索应用的功能并优化其设置以适应您的特定用例。
