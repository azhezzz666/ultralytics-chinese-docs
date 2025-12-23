---
comments: true
description: 了解 Ultralytics CI 操作、Docker 部署、断链检查、CodeQL 分析和 PyPI 发布，以确保高质量代码。
keywords: Ultralytics, 持续集成, CI, Docker 部署, CodeQL, PyPI 发布, 代码质量, 自动化测试
---

# 持续集成 (CI)

持续集成 (CI) 是软件开发的重要方面，涉及自动集成和测试更改。CI 使我们能够通过在开发过程中尽早且频繁地发现问题来维护高质量代码。在 Ultralytics，我们使用各种 CI 测试来确保代码库的质量和完整性。

## CI 操作

以下是我们 CI 操作的简要说明：

- **[CI](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml)**：这是我们的主要 CI 测试，涉及运行单元测试、代码检查，有时还包括更全面的测试，具体取决于仓库。
- **[Docker 部署](https://github.com/ultralytics/ultralytics/actions/workflows/docker.yml)**：此测试使用 Docker 检查项目的部署，以确保 Dockerfile 和相关脚本正常工作。
- **[断链检查](https://github.com/ultralytics/ultralytics/actions/workflows/links.yml)**：此测试扫描代码库中 markdown 或 HTML 文件中的任何断开或失效链接。
- **[CodeQL](https://github.com/ultralytics/ultralytics/actions/workflows/codeql.yaml)**：CodeQL 是 GitHub 的一个工具，对我们的代码执行语义分析，帮助发现潜在的安全漏洞并维护高质量代码。
- **[PyPI 发布](https://github.com/ultralytics/ultralytics/actions/workflows/publish.yml)**：此测试检查项目是否可以打包并发布到 PyPI 而不会出现任何错误。

每个徽章显示相应仓库 `main` 分支上相应 CI 测试的最后一次运行状态。如果测试失败，徽章将显示"failing"状态；如果通过，将显示"passing"状态。

如果您注意到测试失败，如果您能通过相应仓库中的 GitHub issue 报告，将非常有帮助。

请记住，成功的 CI 测试并不意味着一切都是完美的。在部署或合并更改之前，始终建议手动审查代码。

## 代码覆盖率

代码覆盖率是一个指标，表示测试运行时执行的代码库百分比。它提供了关于测试如何充分执行代码的洞察，对于识别应用程序中未测试的部分至关重要。高代码覆盖率百分比通常与较低的错误可能性相关。但是，重要的是要理解代码覆盖率并不能保证没有缺陷。它仅表示代码的哪些部分已被测试执行。

### 与 [codecov.io](https://about.codecov.io/) 集成

在 Ultralytics，我们已将仓库与 [codecov.io](https://about.codecov.io/) 集成，这是一个流行的在线平台，用于测量和可视化代码覆盖率。Codecov 提供详细的洞察、提交之间的覆盖率比较，以及直接在代码上的可视化覆盖，指示哪些行被覆盖。

通过与 Codecov 集成，我们旨在通过关注可能容易出错或需要进一步测试的区域来维护和提高代码质量。

### 覆盖率结果

为了快速了解 `ultralytics` Python 包的代码覆盖率状态，我们包含了 `ultralytics` 覆盖率结果的徽章和旭日图可视化。这些图像显示了测试覆盖的代码百分比，提供了我们测试工作的一目了然的指标。有关完整详细信息，请参阅 [https://codecov.io/github/ultralytics/ultralytics](https://app.codecov.io/github/ultralytics/ultralytics)。

| 仓库                                                | 代码覆盖率                                                                                                                                           |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ultralytics](https://github.com/ultralytics/ultralytics) | [![codecov](https://codecov.io/gh/ultralytics/ultralytics/branch/main/graph/badge.svg?token=HHW7IIVFVY)](https://codecov.io/gh/ultralytics/ultralytics) |

在下面的旭日图中，最内圈是整个项目，从中心向外是文件夹，最后是单个文件。每个切片的大小和颜色分别表示语句数量和覆盖率。

<a href="https://app.codecov.io/github/ultralytics/ultralytics">
    <img src="https://codecov.io/gh/ultralytics/ultralytics/branch/main/graphs/sunburst.svg?token=HHW7IIVFVY" alt="Ultralytics Codecov 图像">
</a>

## 常见问题

### Ultralytics 中的持续集成 (CI) 是什么？

Ultralytics 中的持续集成 (CI) 涉及自动集成和测试代码更改，以确保高质量标准。我们的 CI 设置包括运行[单元测试、代码检查和全面测试](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml)。此外，我们执行 [Docker 部署](https://github.com/ultralytics/ultralytics/actions/workflows/docker.yml)、[断链检查](https://github.com/ultralytics/ultralytics/actions/workflows/links.yml)、用于安全漏洞的 [CodeQL 分析](https://github.com/ultralytics/ultralytics/actions/workflows/codeql.yaml)和 [PyPI 发布](https://github.com/ultralytics/ultralytics/actions/workflows/publish.yml)以打包和分发我们的软件。

### Ultralytics 如何检查文档和代码中的断链？

Ultralytics 使用特定的 CI 操作来[检查断链](https://github.com/ultralytics/ultralytics/actions/workflows/links.yml)，扫描我们的 markdown 和 HTML 文件。这有助于通过扫描和识别失效或断开的链接来维护文档的完整性，确保用户始终可以访问准确和有效的资源。

### 为什么 CodeQL 分析对 Ultralytics 代码库很重要？

[CodeQL 分析](https://github.com/ultralytics/ultralytics/actions/workflows/codeql.yaml)对 Ultralytics 至关重要，因为它执行语义代码分析以发现潜在的安全漏洞并维护高质量标准。通过 CodeQL，我们可以主动识别和缓解代码中的风险，帮助我们提供强大且安全的[软件解决方案](https://www.ultralytics.com/solutions)。

### Ultralytics 如何使用 Docker 进行部署？

Ultralytics 使用 Docker 通过专用的 CI 操作验证项目的部署。此过程确保我们的 [Dockerfile 和相关脚本](https://github.com/ultralytics/ultralytics/actions/workflows/docker.yml)正常运行，允许一致且可重现的部署环境，这对于可扩展和可靠的 AI 解决方案至关重要。

### 自动化 PyPI 发布在 Ultralytics 中的作用是什么？

自动化 [PyPI 发布](https://github.com/ultralytics/ultralytics/actions/workflows/publish.yml)确保我们的项目可以无错误地打包和发布。此步骤对于分发 Ultralytics 的 Python 包至关重要，允许用户通过 [Python 包索引 (PyPI)](https://pypi.org/project/ultralytics/) 轻松安装和使用我们的工具。

### Ultralytics 如何测量代码覆盖率，为什么它很重要？

Ultralytics 使用 [codecov.io](https://about.codecov.io/) 测量代码覆盖率，提供测试执行代码百分比的洞察。高代码覆盖率可以表明经过良好测试的代码，可能减少未检测到的错误的可能性。覆盖率结果通过徽章和旭日图可视化，帮助开发人员识别需要更多测试的区域。
