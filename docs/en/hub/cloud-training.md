---
comments: true
description: 探索 Ultralytics HUB 云训练，轻松进行模型训练。升级到 Pro 并一键开始训练。立即简化您的工作流程！
keywords: Ultralytics HUB, 云训练, 模型训练, Pro 计划, 简易 AI 设置
---

# Ultralytics HUB 云训练

我们听取了大量需求和广泛兴趣，很高兴推出 [Ultralytics HUB](https://www.ultralytics.com/hub) 云训练，为我们的 [Pro](./pro.md) 用户提供一键训练体验！

[Ultralytics HUB](https://www.ultralytics.com/hub) [Pro](./pro.md) 用户可以使用我们的云训练解决方案在自定义数据集上微调 [Ultralytics HUB](https://www.ultralytics.com/hub) 模型，使模型训练过程变得简单。告别复杂的设置，迎接 [Ultralytics HUB](https://www.ultralytics.com/hub) 直观界面带来的简化工作流程。

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ie3vLUDNYZo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>新功能 🌟 介绍 Ultralytics HUB 云训练
</p>

## 训练模型

要使用 Ultralytics 云训练训练模型，您需要[升级](./pro.md#如何升级)到 [Pro 计划](./pro.md)。

按照[模型](./models.md)页面的[训练模型](./models.md#训练模型)说明操作，直到到达**训练模型**对话框的第三步（[训练](./models.md#3-训练)）。到达此步骤后，只需选择训练时长（轮次或定时）、训练实例、支付方式，然后点击**开始训练**按钮。就是这样！

![Ultralytics HUB 训练模型对话框截图，箭头指向云训练选项和开始训练按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog.avif)

??? note "注意"

    在此步骤，您可以选择关闭**训练模型**对话框，稍后从模型页面开始训练模型。

    ![Ultralytics HUB 模型页面截图，箭头指向开始训练卡片](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-model-page-start-training.avif)

大多数情况下，您将使用轮次训练。轮次数量可以在此步骤调整（如果训练尚未开始），表示数据集需要经过训练、标注和测试循环的次数。基于轮次数量的确切定价难以确定，因此我们只允许使用[账户余额](./pro.md#管理您的账户余额)支付方式。

!!! note "注意"

    使用轮次训练时，您的[账户余额](./pro.md#管理您的账户余额)需要至少 5.00 美元才能开始训练。如果余额不足，您可以直接从此步骤充值。

    ![Ultralytics HUB 训练模型对话框截图，箭头指向充值按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog-top-up.avif)

!!! note "注意"

    使用轮次训练时，[账户余额](./pro.md#管理您的账户余额)在每个[轮次](https://www.ultralytics.com/glossary/epoch)后扣除。

    此外，每个轮次后，我们会检查您是否有足够的[账户余额](./pro.md#管理您的账户余额)进行下一个轮次。如果您没有足够的[账户余额](./pro.md#管理您的账户余额)进行下一个轮次，我们将停止训练会话，允许您从保存的最后一个检查点恢复训练模型。

    ![Ultralytics HUB 模型页面截图，箭头指向恢复训练按钮](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-resume-training-button.avif)

或者，您可以使用定时训练。此选项允许您设置训练时长。在这种情况下，我们可以确定确切的定价。您可以预付或使用[账户余额](./pro.md#管理您的账户余额)支付。

如果您有足够的[账户余额](./pro.md#管理您的账户余额)，可以使用[账户余额](./pro.md#管理您的账户余额)支付方式。

![Ultralytics HUB 训练模型对话框截图，箭头指向开始训练按钮](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-start-training.avif)

如果您没有足够的[账户余额](./pro.md#管理您的账户余额)，将无法使用[账户余额](./pro.md#管理您的账户余额)支付方式。您可以预付或直接从此步骤充值。

![Ultralytics HUB 训练模型对话框截图，箭头指向立即支付按钮](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-train-model-pay-now-button.avif)

在训练会话开始之前，初始化过程会启动配备 GPU 资源的专用实例，这有时可能需要一段时间，具体取决于当前的需求和 GPU 资源的可用性。

![Ultralytics HUB 模型页面初始化过程截图](https://github.com/ultralytics/docs/releases/download/0/model-page-initialization-process.avif)

!!! note "注意"

    在初始化过程中（训练会话开始之前）不会扣除账户余额。

训练会话开始后，您可以监控每个步骤的进度。

如果需要，您可以点击**停止训练**按钮停止训练。

![Ultralytics HUB 正在训练的模型页面截图，箭头指向停止训练按钮](https://github.com/ultralytics/docs/releases/download/0/model-page-training-stop-button.avif)

!!! note "注意"

    您可以从保存的最后一个检查点恢复训练模型。

    ![Ultralytics HUB 模型页面截图，箭头指向恢复训练按钮](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-resume-training-button.avif)

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/H3qL8ImCSV8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics HUB 暂停和恢复模型训练
</p>

!!! note "注意"

    遗憾的是，目前您只能使用 Ultralytics 云一次训练一个模型。

    ![Ultralytics HUB 训练模型对话框截图，显示 Ultralytics 云不可用](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog-1.avif)

## 计费

在训练期间或训练后，您可以点击**计费**选项卡查看模型的费用。此外，您可以点击**下载**按钮下载费用报告。

![Ultralytics HUB 模型页面计费选项卡截图，箭头指向计费选项卡和下载按钮](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-billing-tab.avif)
