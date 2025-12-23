---
comments: true
description: 学习如何为 Ultralytics YOLO 开源仓库做贡献。遵循拉取请求、行为准则和错误报告的指南。
keywords: Ultralytics, YOLO, 开源, 贡献, 拉取请求, 行为准则, 错误报告, GitHub, CLA, Google 风格文档字符串, AGPL-3.0
---

# 为 Ultralytics 开源项目做贡献

欢迎！我们很高兴您考虑为我们的 [Ultralytics](https://www.ultralytics.com/) [开源](https://github.com/ultralytics)项目做贡献。您的参与不仅有助于提高我们仓库的质量，还使整个[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区受益。本指南提供了清晰的指南和最佳实践，帮助您入门。

[![Ultralytics 开源贡献者](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yMR7BgwHQ3g"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何为 Ultralytics 仓库做贡献 | Ultralytics 模型、数据集和文档 🚀
</p>

## 🤝 行为准则

为确保每个人都有一个欢迎和包容的环境，所有贡献者必须遵守我们的[行为准则](https://docs.ultralytics.com/help/code-of-conduct/)。**尊重**、**善良**和**专业**是我们社区的核心。

## 🚀 通过拉取请求贡献

我们非常感谢以[拉取请求 (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) 形式的贡献。为使审查过程尽可能顺利，请按照以下步骤操作：

1. **[Fork 仓库](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)**：首先将相关的 Ultralytics 仓库（例如 [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)）fork 到您的 GitHub 账户。
2. **[创建分支](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop)**：在您 fork 的仓库中创建一个新分支，使用清晰、描述性的名称反映您的更改（例如 `fix-issue-123`、`add-feature-xyz`）。
3. **进行更改**：实施您的改进或修复。确保您的代码遵循项目的风格指南，不会引入新的错误或警告。
4. **测试更改**：在提交之前，在本地测试您的更改以确认它们按预期工作且不会导致[回归](https://en.wikipedia.org/wiki/Software_regression)。如果您引入新功能，请添加测试。
5. **[提交更改](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop)**：使用简洁且描述性的提交消息提交更改。如果您的更改解决了特定问题，请包含问题编号（例如 `Fix #123: 修正了计算错误。`）。
6. **[创建拉取请求](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)**：从您的分支向原始 Ultralytics 仓库的 `main` 分支提交拉取请求。提供清晰的标题和详细描述，解释更改的目的和范围。

### 📝 CLA 签署

在我们合并您的拉取请求之前，您必须签署我们的[贡献者许可协议 (CLA)](https://docs.ultralytics.com/help/CLA/)。此法律协议确保您的贡献得到适当许可，允许项目继续在 [AGPL-3.0 许可](https://www.ultralytics.com/legal/agpl-3-0-software-license)下分发。

提交拉取请求后，CLA 机器人将指导您完成签署过程。要签署 CLA，只需在您的 PR 中添加评论：

```text
I have read the CLA Document and I sign the CLA
```

### ✍️ Google 风格文档字符串

添加新函数或类时，请包含 [Google 风格文档字符串](https://google.github.io/styleguide/pyguide.html)以提供清晰、标准化的文档。始终将输入和输出 `types` 括在括号中（例如 `(bool)`、`(np.ndarray)`）。

!!! example "文档字符串示例"

    === "Google 风格"

        此示例说明了标准的 Google 风格文档字符串格式。注意它如何清晰地分离函数描述、参数、返回值和示例以获得最大可读性。

        ```python
        def example_function(arg1, arg2=4):
            """演示 Google 风格文档字符串的示例函数。

            Args:
                arg1 (int): 第一个参数。
                arg2 (int): 第二个参数。

            Returns:
                (bool): 如果参数相等则为 True，否则为 False。

            Examples:
                >>> example_function(4, 4)  # True
                >>> example_function(1, 2)  # False
            """
            return arg1 == arg2
        ```

    === "Google 风格命名返回"

        此示例演示如何记录命名返回变量。使用命名返回可以使您的代码更具自文档性且更易于理解，特别是对于复杂函数。

        ```python
        def example_function(arg1, arg2=4):
            """演示 Google 风格文档字符串的示例函数。

            Args:
                arg1 (int): 第一个参数。
                arg2 (int): 第二个参数。

            Returns:
                equals (bool): 如果参数相等则为 True，否则为 False。

            Examples:
                >>> example_function(4, 4)  # True
            """
            equals = arg1 == arg2
            return equals
        ```

    === "Google 风格多返回值"

        此示例展示如何记录返回多个值的函数。每个返回值应单独记录，包含其自己的类型和描述以保持清晰。

        ```python
        def example_function(arg1, arg2=4):
            """演示 Google 风格文档字符串的示例函数。

            Args:
                arg1 (int): 第一个参数。
                arg2 (int): 第二个参数。

            Returns:
                equals (bool): 如果参数相等则为 True，否则为 False。
                added (int): 两个输入参数的和。

            Examples:
                >>> equals, added = example_function(2, 2)  # True, 4
            """
            equals = arg1 == arg2
            added = arg1 + arg2
            return equals, added
        ```

    === "带类型提示的 Google 风格"

        此示例将 Google 风格文档字符串与 Python 类型提示结合。使用类型提示时，您可以在文档字符串参数部分省略类型信息，因为它已在函数签名中指定。

        ```python
        def example_function(arg1: int, arg2: int = 4) -> bool:
            """演示 Google 风格文档字符串的示例函数。

            Args:
                arg1: 第一个参数。
                arg2: 第二个参数。

            Returns:
                如果参数相等则为 True，否则为 False。

            Examples:
                >>> example_function(1, 1)  # True
            """
            return arg1 == arg2
        ```

    === "单行"

        对于较小或较简单的函数，单行文档字符串可能就足够了。这些应该是简洁但完整的句子，以大写字母开头并以句号结尾。

        ```python
        def example_small_function(arg1: int, arg2: int = 4) -> bool:
            """带有单行文档字符串的示例函数。"""
            return arg1 == arg2
        ```

### ✅ GitHub Actions CI 测试

所有拉取请求必须通过 [GitHub Actions](https://github.com/features/actions) [持续集成](https://docs.ultralytics.com/help/CI/) (CI) 测试才能合并。这些测试包括代码检查、单元测试和其他检查，以确保您的更改符合项目的质量标准。查看 CI 输出并解决出现的任何问题。

## ✨ 代码贡献最佳实践

在为 Ultralytics 项目贡献代码时，请牢记以下最佳实践：

- **避免代码重复**：尽可能重用现有代码并最小化不必要的参数。
- **进行较小、集中的更改**：专注于有针对性的修改而不是大规模更改。
- **尽可能简化**：寻找简化代码或删除不必要部分的机会。
- **考虑兼容性**：在进行更改之前，考虑它们是否可能破坏使用 Ultralytics 的现有代码。
- **使用一致的格式**：[Ruff Formatter](https://github.com/astral-sh/ruff) 等工具可以帮助保持风格一致性。
- **添加适当的测试**：为新功能包含[测试](https://docs.ultralytics.com/guides/model-testing/)以确保它们按预期工作。

## 👀 审查拉取请求

审查拉取请求是另一种有价值的贡献方式。审查 PR 时：

- **检查单元测试**：验证 PR 是否包含新功能或更改的测试。
- **审查文档更新**：确保[文档](https://docs.ultralytics.com/)已更新以反映更改。
- **评估性能影响**：考虑更改可能如何影响[性能](https://docs.ultralytics.com/guides/yolo-performance-metrics/)。
- **验证 CI 测试**：确认所有[持续集成测试](https://docs.ultralytics.com/help/CI/)都通过。
- **提供建设性反馈**：对任何问题或疑虑提供具体、清晰的反馈。
- **认可努力**：承认作者的工作以保持积极的协作氛围。

## 🐞 报告错误

我们非常重视错误报告，因为它们帮助我们提高项目的质量和可靠性。通过 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) 报告错误时：

- **检查现有问题**：首先搜索该错误是否已被报告。
- **提供[最小可复现示例](https://docs.ultralytics.com/help/minimum-reproducible-example/)**：创建一个小的、独立的代码片段，可以一致地复现问题。这对于高效调试至关重要。
- **描述环境**：指定您的操作系统、Python 版本、相关库版本（例如 [`torch`](https://pytorch.org/)、[`ultralytics`](https://github.com/ultralytics/ultralytics)）和硬件（[CPU](https://en.wikipedia.org/wiki/Central_processing_unit)/[GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit)）。
- **解释预期与实际行为**：清楚说明您期望发生什么以及实际发生了什么。包括任何错误消息或回溯。

## 📜 许可证

Ultralytics 为其仓库使用 [GNU Affero 通用公共许可证 v3.0 (AGPL-3.0)](https://www.ultralytics.com/legal/agpl-3-0-software-license)。此许可证促进软件开发中的[开放性](https://en.wikipedia.org/wiki/Openness)、[透明度](https://www.ultralytics.com/glossary/transparency-in-ai)和[协作改进](https://en.wikipedia.org/wiki/Collaborative_software)。它确保所有用户都有自由使用、修改和共享软件，培养强大的协作和创新社区。

我们鼓励所有贡献者熟悉 [AGPL-3.0 许可证](https://opensource.org/license/agpl-v3)的条款，以有效且合乎道德地为 Ultralytics 开源社区做贡献。

## 结论

感谢您有兴趣为 [Ultralytics](https://www.ultralytics.com/) [开源](https://github.com/ultralytics) YOLO 项目做贡献。您的参与对于塑造我们软件的未来和建立充满活力的创新和协作社区至关重要。无论您是增强代码、报告错误还是建议新功能，您的贡献都是无价的。

我们很高兴看到您的想法变为现实，并感谢您致力于推进[目标检测](https://www.ultralytics.com/glossary/object-detection)技术。让我们一起在这个激动人心的开源之旅中继续成长和创新。

## 常见问题

### 为什么我应该为 Ultralytics YOLO 开源仓库做贡献？

为 Ultralytics YOLO 开源仓库做贡献可以改进软件，使其对整个社区更加强大和功能丰富。贡献可以包括代码增强、错误修复、文档改进和新功能实现。此外，贡献允许您与该领域的其他熟练开发人员和专家合作，提高您自己的技能和声誉。有关如何入门的详细信息，请参阅[通过拉取请求贡献](#通过拉取请求贡献)部分。

### 如何签署 Ultralytics YOLO 的贡献者许可协议 (CLA)？

要签署贡献者许可协议 (CLA)，请在提交拉取请求后按照 CLA 机器人提供的说明操作。此过程确保您的贡献在 AGPL-3.0 许可下得到适当许可，维护开源项目的法律完整性。在您的拉取请求中添加评论：

```text
I have read the CLA Document and I sign the CLA
```

有关更多信息，请参阅 [CLA 签署](#cla-签署)部分。

### 什么是 Google 风格文档字符串，为什么 Ultralytics YOLO 贡献需要它们？

Google 风格文档字符串为函数和类提供清晰、简洁的文档，提高代码可读性和可维护性。这些文档字符串使用特定的格式规则概述函数的目的、参数和返回值。在为 Ultralytics YOLO 做贡献时，遵循 Google 风格文档字符串可确保您的添加内容有良好的文档记录且易于理解。有关示例和指南，请访问 [Google 风格文档字符串](#google-风格文档字符串)部分。

### 如何确保我的更改通过 GitHub Actions CI 测试？

在您的拉取请求可以合并之前，它必须通过所有 GitHub Actions 持续集成 (CI) 测试。这些测试包括代码检查、单元测试和其他检查，以确保代码符合项目的质量标准。查看 CI 输出并修复任何问题。有关 CI 过程和故障排除提示的详细信息，请参阅 [GitHub Actions CI 测试](#github-actions-ci-测试)部分。

### 如何在 Ultralytics YOLO 仓库中报告错误？

要报告错误，请提供清晰简洁的[最小可复现示例](https://docs.ultralytics.com/help/minimum-reproducible-example/)以及您的错误报告。这有助于开发人员快速识别和修复问题。确保您的示例最小但足以复现问题。有关报告错误的更详细步骤，请参阅[报告错误](#报告错误)部分。

### 如果我在项目中使用 Ultralytics YOLO，AGPL-3.0 许可证意味着什么？

如果您在项目中使用 Ultralytics YOLO 代码或模型（在 AGPL-3.0 下许可），AGPL-3.0 许可证要求您的整个项目（衍生作品）也必须在 AGPL-3.0 下许可，并且其完整源代码必须公开可用。这确保了软件的开源性质在其衍生作品中得到保留。如果您无法满足这些要求，您需要获得[企业许可证](https://www.ultralytics.com/license)。有关详细信息，请参阅[开源您的项目](#open-sourcing-your-yolo-project-under-agpl-30)部分。
