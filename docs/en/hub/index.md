---
comments: true
description: 探索 Ultralytics HUB，一站式训练和部署 YOLO 模型的网络工具。使用预训练模型和用户友好功能快速入门。
keywords: Ultralytics HUB, YOLO 模型, 训练 YOLO, YOLOv5, YOLOv8, YOLO11, 目标检测, 模型部署, 机器学习, 深度学习, AI 工具, 数据集上传, 模型训练
---

# Ultralytics HUB

<div align="center">
<a href="https://www.ultralytics.com/hub" target="_blank"><img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub.avif" alt="Ultralytics HUB banner"></a>
<a href="https://docs.ultralytics.com/zh/hub/">中文</a> |
<a href="https://docs.ultralytics.com/ko/hub/">한국어</a> |
<a href="https://docs.ultralytics.com/ja/hub/">日本語</a> |
<a href="https://docs.ultralytics.com/ru/hub/">Русский</a> |
<a href="https://docs.ultralytics.com/de/hub/">Deutsch</a> |
<a href="https://docs.ultralytics.com/fr/hub/">Français</a> |
<a href="https://docs.ultralytics.com/es/hub/">Español</a> |
<a href="https://docs.ultralytics.com/pt/hub/">Português</a> |
<a href="https://docs.ultralytics.com/tr/hub/">Türkçe</a> |
<a href="https://docs.ultralytics.com/vi/hub/">Tiếng Việt</a> |
<a href="https://docs.ultralytics.com/ar/hub/">العربية</a>
<br>
<br>

<a href="https://github.com/ultralytics/hub/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/hub/actions/workflows/ci.yml/badge.svg" alt="CI CPU"></a> <a href="https://colab.research.google.com/github/ultralytics/hub/blob/main/hub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

</div>

👋 来自 [Ultralytics](https://www.ultralytics.com/) 团队的问候！过去几个月我们一直在努力推出 [Ultralytics HUB](https://www.ultralytics.com/hub)，这是一个全新的网络工具，可以在一个地方训练和部署所有 YOLOv5、YOLOv8 和 YOLO11 🚀 模型！

我们希望这里的资源能帮助您充分利用 HUB。请浏览 HUB <a href="https://docs.ultralytics.com/">文档</a>了解详情，在 <a href="https://github.com/ultralytics/hub/issues/new/choose">GitHub</a> 上提交问题获取支持，并加入我们的 <a href="https://discord.com/invite/ultralytics">Discord</a> 社区进行问答和讨论！

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

## 简介

[Ultralytics HUB](https://www.ultralytics.com/hub) 设计为用户友好且直观，允许用户快速上传数据集并训练新的 YOLO 模型。它还提供多种预训练模型供选择，使用户能够轻松入门。模型训练完成后，可以在 [Ultralytics HUB App](app/index.md) 中轻松预览，然后部署用于实时分类、[目标检测](https://www.ultralytics.com/glossary/object-detection)和[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)任务。

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/qE-dfbB5Sis"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics HUB 在自定义数据集上训练 Ultralytics YOLO11 | HUB 数据集 🚀
</p>

我们希望这里的资源能帮助您充分利用 HUB。请浏览 HUB <a href="https://docs.ultralytics.com/hub/">文档</a>了解详情，在 <a href="https://github.com/ultralytics/hub/issues/new/choose">GitHub</a> 上提交问题获取支持，并加入我们的 <a href="https://discord.com/invite/ultralytics">Discord</a> 社区进行问答和讨论！

- [**快速入门**](quickstart.md)：在几秒钟内开始训练和部署模型。
- [**数据集**](datasets.md)：了解如何准备和上传数据集。
- [**项目**](projects.md)：将模型分组到项目中以改善组织。
- [**模型**](models.md)：训练模型并导出为各种格式进行部署。
- [**Pro**](pro.md)：成为 Pro 用户提升您的体验。
- [**云训练**](cloud-training.md)：了解如何使用我们的云训练解决方案训练模型。
- [**推理 API**](inference-api.md)：了解如何使用我们的推理 API。
- [**团队**](teams.md)：与您的团队轻松协作。
- [**集成**](integrations.md)：探索不同的集成选项。
- [**Ultralytics HUB App**](app/index.md)：了解 Ultralytics HUB App，它允许您直接在移动设备上运行模型。
    - [**iOS**](app/ios.md)：探索 iPhone 和 iPad 上的 CoreML 加速。
    - [**Android**](app/android.md)：探索 Android 设备上的 TFLite 加速。

## 常见问题

### 如何开始使用 Ultralytics HUB 训练 YOLO 模型？

要开始使用 [Ultralytics HUB](https://www.ultralytics.com/hub)，请按照以下步骤操作：

1. **注册**：在 [Ultralytics HUB](https://www.ultralytics.com/hub) 上创建账户。
2. **上传数据集**：导航到[数据集](datasets.md)部分上传您的自定义数据集。
3. **训练模型**：转到[模型](models.md)部分，选择预训练的 YOLOv5、YOLOv8 或 YOLO11 模型开始训练。
4. **部署模型**：训练完成后，使用 [Ultralytics HUB App](app/index.md) 预览和部署模型用于实时任务。

有关详细指南，请参阅[快速入门](quickstart.md)页面。

### 与其他 AI 平台相比，使用 Ultralytics HUB 有哪些优势？

[Ultralytics HUB](https://www.ultralytics.com/hub) 提供多项独特优势：

- **用户友好界面**：直观的设计，便于数据集上传和模型训练。
- **预训练模型**：访问各种预训练 YOLO 模型，包括 YOLOv5、YOLOv8 和 YOLO11。
- **云训练**：无缝的云训练功能，详见[云训练](cloud-training.md)页面。
- **实时部署**：使用 [Ultralytics HUB App](app/index.md) 轻松部署模型用于实时应用。
- **团队协作**：通过[团队](teams.md)功能与您的团队高效协作。
- **无代码解决方案**：无需编写一行代码即可训练和部署高级计算机视觉模型。

在我们的 [Ultralytics HUB 博客](https://www.ultralytics.com/blog/ultralytics-hub-a-game-changer-for-computer-vision)中了解更多优势。

### 我可以在移动设备上使用 Ultralytics HUB 进行目标检测吗？

是的，Ultralytics HUB 支持在移动设备上进行目标检测。您可以使用 Ultralytics HUB App 在 iOS 和 Android 设备上运行 YOLO 模型。更多详情：

- **iOS**：在 [iOS](app/ios.md) 部分了解 iPhone 和 iPad 上的 CoreML 加速。
- **Android**：在 [Android](app/android.md) 部分探索 Android 设备上的 TFLite 加速。

### 如何在 Ultralytics HUB 中管理和组织我的项目？

Ultralytics HUB 允许您高效地管理和组织项目。您可以将模型分组到项目中以便更好地组织。了解更多：

- 访问[项目](projects.md)页面获取创建、编辑和管理项目的详细说明。
- 使用[团队](teams.md)功能与团队成员协作和共享资源。

### Ultralytics HUB 提供哪些集成？

Ultralytics HUB 与各种平台无缝集成，以增强您的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工作流程。一些关键集成包括：

- **Roboflow**：用于数据集管理和模型训练。在[集成](integrations.md)页面了解更多。
- **Google Colab**：使用 Google Colab 的云环境高效训练模型。详细步骤请参阅 [Google Colab](https://docs.ultralytics.com/integrations/google-colab/) 部分。
- **Weights & Biases**：用于增强实验跟踪和可视化。探索 [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/) 集成。

有关完整的集成列表，请参阅[集成](integrations.md)页面。
