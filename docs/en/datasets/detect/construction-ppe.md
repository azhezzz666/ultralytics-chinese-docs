---
comments: true
description: 探索 Construction-PPE，一个专门用于检测真实建筑工地中安全帽、背心、手套、靴子和护目镜的数据集。包括合规和不合规场景，用于 AI 驱动的安全监控。
keywords: Construction-PPE, PPE 数据集, 安全合规, 建筑工人, 目标检测, YOLO11, 工作场所安全, 计算机视觉
---

# Construction-PPE 数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-construction-ppe-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开 Construction-PPE 数据集"></a>

Construction-PPE 数据集旨在通过检测安全帽、背心、手套、靴子和护目镜等基本防护装备以及缺失装备的标注来提高建筑工地的安全合规性。该数据集从真实建筑环境中策划，包括合规和不合规案例，使其成为训练监控工作场所安全的 AI 模型的宝贵资源。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/lFaVnrhMmaE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在个人防护装备数据集上训练 Ultralytics YOLO | 建筑领域的视觉 AI 👷
</p>

## 数据集结构

Construction-PPE 数据集分为三个主要子集：

- **训练集**：包含标注建筑图像的主要集合，展示具有完整和部分 PPE 使用情况的工人。
- **验证集**：用于在 PPE 检测和合规监控期间微调和评估模型性能的指定子集。
- **测试集**：保留用于评估最终模型在检测 PPE 和识别合规问题方面有效性的独立子集。

每张图像都以 [Ultralytics YOLO](../detect/index.md/#what-is-the-ultralytics-yolo-dataset-format-and-how-to-structure-it) 格式标注，确保与最先进的[目标检测](../../tasks/detect.md)和[跟踪](../../modes/track.md)流水线兼容。

该数据集提供 **11 个类别**，分为正面（佩戴 PPE）和负面（缺失 PPE）类别。这种双正/负结构使模型能够检测正确佩戴的装备**并**识别安全违规。


## 商业价值

- 建筑业仍然是世界上最危险的行业之一，2023/2024 年英国 123 起与工作相关的**致命伤害**中有超过 51 起发生在建筑业。然而，问题不再是缺乏监管，42% 的建筑工人承认并不总是遵守流程。
- 建筑业已经受到广泛的健康与安全（HSE）标准框架的约束，但 HSE 团队在持续执行方面面临挑战。HSE 团队通常人手不足，需要平衡文书工作和审计，缺乏实时监控繁忙且不断变化环境每个角落的能力。
- 这就是基于计算机视觉的个人防护装备（PPE）检测变得非常有价值的地方。通过自动检查工人是否佩戴**安全帽、背心和其他个人防护装备**，您可以确保 HSE 规则不仅存在，而且在所有工地得到有效且一致的执行。除了合规性之外，计算机视觉还通过揭示工作人员遵守安全实践的程度来提供风险的领先指标，使组织能够发现合规性下降趋势并在事故发生之前预防。
- 作为额外好处，个人防护装备检测还可以识别未经授权的工地入侵者，因为**那些没有配备适当安全装备的人**是第一个触发通知的。最终，PPE 检测是一个简单但强大的计算机视觉用例，提供全面监督、可操作的见解和标准化报告，使建筑公司能够降低风险、保护工人并保障其项目。

## 应用

Construction-PPE 为各种以安全为重点的计算机视觉应用提供支持：

- **自动化合规监控**：训练 AI 模型即时检查工人是否佩戴所需的安全装备，如安全帽、背心或手套，降低工地风险。
- **工作场所安全分析**：跟踪 PPE 使用情况随时间的变化，发现频繁违规，并生成见解以改善安全文化。
- **智能监控系统**：将检测模型与摄像头连接，在 PPE 缺失时发送实时警报，在事故发生之前预防。
- **机器人和自主系统**：使无人机或机器人能够在大型工地执行 PPE 检查，支持更快、更安全的检查。
- **研究和教育**：为探索工作场所安全和人-物交互的学生和研究人员提供真实世界数据集。

## 数据集 YAML

Construction-PPE 数据集包含一个 YAML 配置文件，定义了训练和验证图像路径以及完整的目标类别列表。您可以直接在 Ultralytics 仓库中访问 `construction-ppe.yaml` 文件：[https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml)

!!! example "ultralytics/cfg/datasets/construction-ppe.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/construction-ppe.yaml"
    ```

## 使用方法

您可以在 Construction-PPE 数据集上训练 YOLO11n 模型 100 个训练周期，图像尺寸为 640。以下示例展示如何快速入门。有关更多选项和高级配置，请参阅[训练指南](../../modes/train.md)。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 在 Construction-PPE 数据集上训练模型
        model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=construction-ppe.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## 示例图像和标注

该数据集捕获了不同环境、光照条件和姿势下的建筑工人。包括**合规**和**不合规**案例。

![Construction-PPE 数据集示例图像，展示合规和不合规安全装备检测](https://github.com/ultralytics/docs/releases/download/0/construction-ppe-dataset-sample.avif)

## 许可和归属

Construction-PPE 在 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)下开发和发布，支持开源研究和具有适当归属的商业应用。

如果您在研究中使用此数据集，请引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Dalvi_Construction_PPE_Dataset_2025,
            author = {Mrunmayee Dalvi and Niyati Singh and Sahil Bhingarde and Ketaki Chalke},
            title = {Construction-PPE: Personal Protective Equipment Detection Dataset},
            month = {January},
            year = {2025},
            version = {1.0.0},
            license = {AGPL-3.0},
            url = {https://docs.ultralytics.com/datasets/detect/construction-ppe/},
            publisher = {Ultralytics}
        }
        ```

## 常见问题

### Construction-PPE 数据集有什么独特之处？

与通用建筑数据集不同，Construction-PPE 明确包含**缺失装备类别**。这种双标签方法使模型不仅能够检测 PPE，还能实时标记违规行为。

### 包含哪些目标类别？

该数据集涵盖安全帽、背心、手套、靴子、护目镜和工人，以及它们的"缺失 PPE"对应类别。这确保了全面的合规覆盖。

### 如何使用 Construction-PPE 数据集训练 YOLO 模型？

要使用 Construction-PPE 数据集训练 YOLO11 模型，您可以使用以下代码片段：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=construction-ppe.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

### 此数据集适合实际应用吗？

是的。图像是从不同条件下的真实建筑工地策划的。这使其对于构建可部署的工作场所安全监控系统非常有效。

### 在 AI 项目中使用 Construction-PPE 数据集有什么好处？

该数据集支持个人防护装备的实时检测，帮助监控建筑工地的工人安全。通过佩戴和缺失装备的类别，它支持能够自动标记安全违规、生成合规见解和降低风险的 AI 系统。它还为开发工作场所安全、机器人和学术研究中的计算机视觉解决方案提供了实用资源。
