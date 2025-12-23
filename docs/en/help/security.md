---
description: 了解 Ultralytics 用于保护用户数据和系统的安全措施和工具。了解我们如何使用 Snyk、CodeQL、Dependabot 等解决漏洞。
keywords: Ultralytics 安全政策, Snyk 扫描, CodeQL 扫描, Dependabot 警报, 密钥扫描, 漏洞报告, GitHub 安全, 开源安全
---

# Ultralytics 安全政策

在 [Ultralytics](https://www.ultralytics.com/)，用户数据和系统的安全至关重要。为确保我们[开源项目](https://github.com/ultralytics)的安全性，我们实施了多项措施来检测和防止安全漏洞。

## Snyk 扫描

我们使用 [Snyk](https://snyk.io/advisor/python/ultralytics) 对 Ultralytics 仓库进行全面的安全扫描。Snyk 强大的扫描功能不仅限于依赖项检查；它还检查我们的代码和 Dockerfile 中的各种漏洞。通过主动识别和解决这些问题，我们确保为用户提供更高级别的安全性和可靠性。

[![ultralytics](https://snyk.io/advisor/python/ultralytics/badge.svg)](https://snyk.io/advisor/python/ultralytics)

## GitHub CodeQL 扫描

我们的安全策略包括 GitHub 的 [CodeQL](https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql) 扫描。CodeQL 深入分析我们的代码库，通过分析代码的语义结构来识别复杂漏洞，如 SQL 注入和 XSS。这种高级分析确保了潜在安全风险的早期检测和解决。

[![CodeQL](https://github.com/ultralytics/ultralytics/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/github-code-scanning/codeql)

## GitHub Dependabot 警报

[Dependabot](https://docs.github.com/en/code-security/dependabot) 已集成到我们的工作流程中，用于监控依赖项中的已知漏洞。当在我们的某个依赖项中发现漏洞时，Dependabot 会向我们发出警报，以便快速采取明智的修复措施。

## GitHub 密钥扫描警报

我们使用 GitHub [密钥扫描](https://docs.github.com/en/code-security/secret-scanning/managing-alerts-from-secret-scanning)警报来检测意外推送到我们仓库的敏感数据，如凭据和私钥。这种早期检测机制有助于防止潜在的安全漏洞和数据泄露。

## 私密漏洞报告

我们启用了私密漏洞报告功能，允许用户谨慎地报告潜在的安全问题。这种方法促进了负责任的披露，确保漏洞得到安全高效的处理。

如果您怀疑或发现我们任何仓库中的安全漏洞，请立即告知我们。您可以通过我们的[联系表单](https://www.ultralytics.com/contact)或 [security@ultralytics.com](mailto:security@ultralytics.com) 直接联系我们。我们的安全团队将尽快调查并回复。

我们感谢您帮助保持所有 Ultralytics 开源项目的安全。

## 常见问题

### Ultralytics 实施了哪些安全措施来保护用户数据？

Ultralytics 采用全面的安全策略来保护用户数据和系统。主要措施包括：

- **Snyk 扫描**：进行安全扫描以检测代码和 Dockerfile 中的漏洞。
- **GitHub CodeQL**：分析代码语义以检测复杂漏洞，如 SQL 注入。
- **Dependabot 警报**：监控依赖项中的已知漏洞并发送警报以便快速修复。
- **密钥扫描**：检测代码仓库中的敏感数据（如凭据或私钥）以防止数据泄露。
- **私密漏洞报告**：为用户提供安全渠道以谨慎报告潜在的安全问题。

这些工具确保主动识别和解决安全问题，增强整体系统安全性。有关更多详细信息，请浏览上述各节或联系安全团队咨询任何问题。

### Ultralytics 如何使用 Snyk 进行安全扫描？

Ultralytics 使用 [Snyk](https://snyk.io/advisor/python/ultralytics) 对其仓库进行全面的安全扫描。Snyk 不仅限于基本的依赖项检查，还检查代码和 Dockerfile 中的各种漏洞。通过主动识别和解决潜在的安全问题，Snyk 帮助确保 Ultralytics 的开源项目保持安全可靠。

要查看 Snyk 徽章并了解更多关于其部署的信息，请查看 [Snyk 扫描部分](#snyk-扫描)。

### 什么是 CodeQL，它如何增强 Ultralytics 的安全性？

[CodeQL](https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql) 是一种通过 GitHub 集成到 Ultralytics 工作流程中的安全分析工具。它深入分析代码库以识别复杂漏洞，如 SQL 注入和跨站脚本攻击 (XSS)。CodeQL 分析代码的语义结构，提供高级安全保障，确保潜在风险的早期检测和缓解。

有关 CodeQL 使用方式的更多信息，请访问 [GitHub CodeQL 扫描部分](#github-codeql-扫描)。

### Dependabot 如何帮助维护 Ultralytics 的代码安全？

[Dependabot](https://docs.github.com/en/code-security/dependabot) 是一种自动化工具，用于监控和管理依赖项中的已知漏洞。当 Dependabot 在 Ultralytics 项目依赖项中检测到漏洞时，它会发送警报，使团队能够快速解决和缓解问题。这确保了依赖项保持安全和最新，最大限度地降低潜在的安全风险。

有关更多详细信息，请浏览 [GitHub Dependabot 警报部分](#github-dependabot-警报)。

### Ultralytics 如何处理私密漏洞报告？

Ultralytics 鼓励用户通过私密渠道报告潜在的安全问题。用户可以通过[联系表单](https://www.ultralytics.com/contact)或发送电子邮件至 [security@ultralytics.com](mailto:security@ultralytics.com) 谨慎地报告漏洞。这确保了负责任的披露，并允许安全团队安全高效地调查和解决漏洞。

有关私密漏洞报告的更多信息，请参阅[私密漏洞报告部分](#私密漏洞报告)。
