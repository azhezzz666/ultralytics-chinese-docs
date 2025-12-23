---
description: 了解 Ultralytics 如何收集和使用匿名数据来增强 YOLO Python 包，同时优先考虑用户隐私和控制。
keywords: Ultralytics, 数据收集, YOLO, Python 包, Google Analytics, Sentry, 隐私, 匿名数据, 用户控制, 崩溃报告
---

# Ultralytics Python 包的数据收集

## 概述

[Ultralytics](https://www.ultralytics.com/) 致力于不断增强用户体验和我们 Python 包的功能，包括我们开发的先进 YOLO 模型。我们的方法涉及收集匿名使用统计数据和崩溃报告，帮助我们识别改进机会并确保软件的可靠性。本透明度文档概述了我们收集的数据、其目的以及您对此数据收集的选择。

## 匿名 Google Analytics

[Google Analytics](https://developers.google.com/analytics) 是 Google 提供的网络分析服务，用于跟踪和报告网站流量。它允许我们收集有关如何使用我们的 Python 包的数据，这对于做出有关设计和功能的明智决策至关重要。

### 我们收集的内容

- **使用指标**：这些指标帮助我们了解包的使用频率和方式、哪些功能受欢迎以及通常使用的命令行参数。
- **系统信息**：我们收集有关您计算环境的一般非可识别信息，以确保我们的包在各种系统上表现良好。
- **性能数据**：了解我们模型在训练、验证和推理期间的性能有助于我们识别优化机会。

有关 Google Analytics 和[数据隐私](https://www.ultralytics.com/glossary/data-privacy)的更多信息，请访问 [Google Analytics 隐私](https://support.google.com/analytics/answer/6004245)。

### 我们如何使用这些数据

- **功能改进**：使用指标的洞察指导我们提高用户满意度和界面设计。
- **优化**：性能数据帮助我们微调模型，以在不同的硬件和软件配置中获得更好的效率和速度。
- **趋势分析**：通过研究使用趋势，我们可以预测并响应社区不断变化的需求。

### 隐私考虑

我们采取多项措施确保您委托给我们的数据的隐私和安全：

- **匿名化**：我们配置 Google Analytics 以匿名化收集的数据，这意味着不会收集任何个人身份信息 (PII)。您可以放心使用我们的服务，您的个人详细信息将保持私密。
- **聚合**：数据仅以聚合形式进行分析。这种做法确保可以观察到模式，而不会泄露任何单个用户的活动。
- **不收集图像数据**：Ultralytics 不收集、处理或查看任何训练或推理图像。

## Sentry 崩溃报告

[Sentry](https://sentry.io/welcome/) 是一款以开发人员为中心的错误跟踪软件，有助于实时识别、诊断和解决问题，确保应用程序的稳健性和可靠性。在我们的包中，它通过提供崩溃报告的洞察发挥着关键作用，显著有助于软件的稳定性和持续改进。

!!! note

    只有当 `sentry-sdk` Python 包预先安装在您的系统上时，才会激活通过 Sentry 的崩溃报告。此包不包含在 `ultralytics` 的先决条件中，也不会由 Ultralytics 自动安装。

### 我们收集的内容

如果 `sentry-sdk` Python 包预先安装在您的系统上，崩溃事件可能会发送以下信息：

- **崩溃日志**：关于崩溃时应用程序状态的详细报告，这对我们的调试工作至关重要。
- **错误消息**：我们记录包运行期间生成的错误消息，以快速了解和解决潜在问题。

要了解更多关于 Sentry 如何处理数据的信息，请访问 [Sentry 的隐私政策](https://sentry.io/privacy/)。

### 我们如何使用这些数据

- **调试**：分析崩溃日志和错误消息使我们能够快速识别和纠正软件错误。
- **稳定性指标**：通过持续监控崩溃，我们旨在提高包的稳定性和可靠性。

### 隐私考虑

- **敏感信息**：我们确保崩溃日志清除任何个人身份或敏感用户数据，保护您信息的机密性。
- **受控收集**：我们的崩溃报告机制经过精心校准，仅收集故障排除所需的内容，同时尊重用户隐私。

通过详细说明用于数据收集的工具并提供带有其各自隐私页面 URL 的额外背景信息，用户可以全面了解我们的做法，强调透明度和对用户隐私的尊重。

## 禁用数据收集

我们相信为用户提供对其数据的完全控制。默认情况下，我们的包配置为收集分析和崩溃报告，以帮助改善所有用户的体验。但是，我们尊重某些用户可能更愿意选择退出此数据收集。

要选择退出发送分析和崩溃报告，您只需在 YOLO 设置中设置 `sync=False`。这确保不会从您的机器传输任何数据到我们的分析工具。

### 检查设置

要深入了解您设置的当前配置，您可以直接查看它们：

!!! example "查看设置"

    === "Python"

        您可以使用 Python 查看您的设置。首先从 `ultralytics` 模块导入 `settings` 对象。使用以下命令打印和返回设置：
        ```python
        from ultralytics import settings

        # 查看所有设置
        print(settings)

        # 返回分析和崩溃报告设置
        value = settings["sync"]
        ```

    === "CLI"

        或者，命令行界面允许您使用简单的命令检查设置：
        ```bash
        yolo settings
        ```

### 修改设置

Ultralytics 允许用户轻松修改其设置。可以通过以下方式进行更改：

!!! example "更新设置"

    === "Python"

        在 Python 环境中，调用 `settings` 对象的 `update` 方法来更改设置：
        ```python
        from ultralytics import settings

        # 禁用分析和崩溃报告
        settings.update({"sync": False})

        # 将设置重置为默认值
        settings.reset()
        ```

    === "CLI"

        如果您更喜欢使用命令行界面，以下命令将允许您修改设置：
        ```bash
        # 禁用分析和崩溃报告
        yolo settings sync=False

        # 将设置重置为默认值
        yolo settings reset
        ```

`sync=False` 设置将阻止任何数据发送到 Google Analytics 或 Sentry。您的设置将在使用 Ultralytics 包的所有会话中得到尊重，并保存到磁盘以供将来会话使用。

## 对隐私的承诺

Ultralytics 非常重视用户隐私。我们根据以下原则设计数据收集实践：

- **透明度**：我们对收集的数据及其使用方式持开放态度。
- **控制**：我们让用户完全控制其数据。
- **安全**：我们采用行业标准的安全措施来保护我们收集的数据。

## 问题或疑虑

如果您对我们的数据收集实践有任何问题或疑虑，请通过我们的[联系表单](https://www.ultralytics.com/contact)或 [support@ultralytics.com](mailto:support@ultralytics.com) 联系我们。我们致力于确保用户在使用我们的包时对其隐私感到知情和自信。

## 常见问题

### Ultralytics 如何确保其收集的数据的隐私？

Ultralytics 通过几项关键措施优先考虑用户隐私。首先，通过 Google Analytics 和 Sentry 收集的所有数据都是匿名的，以确保不会收集任何个人身份信息 (PII)。其次，数据以聚合形式进行分析，允许我们观察模式而不识别单个用户活动。最后，我们不收集任何训练或推理图像，进一步保护用户数据。这些措施符合我们对透明度和隐私的承诺。有关更多详细信息，请访问我们的[隐私考虑](#隐私考虑)部分。

### Ultralytics 使用 Google Analytics 收集哪些类型的数据？

Ultralytics 使用 Google Analytics 收集三种主要类型的数据：

- **使用指标**：包括 YOLO Python 包的使用频率和方式、首选功能以及通常使用的命令行参数。
- **系统信息**：关于运行包的计算环境的一般非可识别信息。
- **性能数据**：与训练、验证和推理期间模型性能相关的指标。

这些数据帮助我们增强用户体验并优化软件性能。在[匿名 Google Analytics](#匿名-google-analytics) 部分了解更多信息。

### 如何在 Ultralytics YOLO 包中禁用数据收集？

要选择退出数据收集，您只需在 YOLO 设置中设置 `sync=False`。此操作会停止任何分析或崩溃报告的传输。您可以使用 Python 或 CLI 方法禁用数据收集：

!!! example "更新设置"

    === "Python"

        ```python
        from ultralytics import settings

        # 禁用分析和崩溃报告
        settings.update({"sync": False})

        # 将设置重置为默认值
        settings.reset()
        ```

    === "CLI"

        ```bash
        # 禁用分析和崩溃报告
        yolo settings sync=False

        # 将设置重置为默认值
        yolo settings reset
        ```

有关修改设置的更多详细信息，请参阅[修改设置](#修改设置)部分。

### Ultralytics YOLO 中的 Sentry 崩溃报告如何工作？

如果预先安装了 `sentry-sdk` 包，每当发生崩溃事件时，Sentry 都会收集详细的崩溃日志和错误消息。这些数据帮助我们及时诊断和解决问题，提高 YOLO Python 包的稳健性和可靠性。收集的崩溃日志会清除任何个人身份信息以保护用户隐私。有关更多信息，请查看 [Sentry 崩溃报告](#sentry-崩溃报告)部分。

### 我可以检查 Ultralytics YOLO 中当前的数据收集设置吗？

是的，您可以轻松查看当前设置以了解数据收集偏好的配置。使用以下方法检查这些设置：

!!! example "查看设置"

    === "Python"

        ```python
        from ultralytics import settings

        # 查看所有设置
        print(settings)

        # 返回分析和崩溃报告设置
        value = settings["sync"]
        ```

    === "CLI"

        ```bash
        yolo settings
        ```

有关更多详细信息，请参阅[检查设置](#检查设置)部分。
