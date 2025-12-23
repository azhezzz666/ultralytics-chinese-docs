<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 📚 Ultralytics 文档

欢迎来到 Ultralytics 文档，这是您理解和使用我们最先进的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工具和模型（包括 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/)）的综合资源。这些文档会持续维护并部署到 [https://docs.ultralytics.com](https://docs.ultralytics.com/) 以便于访问。

[![pages-build-deployment](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment)
[![Check Broken links](https://github.com/ultralytics/docs/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/links.yml)
[![Check Domains](https://github.com/ultralytics/docs/actions/workflows/check_domains.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/check_domains.yml)
[![Ultralytics Actions](https://github.com/ultralytics/docs/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/format.yml)

<a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

## 🛠️ 安装

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

要以开发者模式安装 `ultralytics` 包（允许您直接修改源代码），请确保您的系统已安装 [Git](https://git-scm.com/) 和 [Python](https://www.python.org/) 3.8 或更高版本。然后按照以下步骤操作：

1.  使用 Git 将 `ultralytics` 仓库克隆到本地：

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2.  进入克隆仓库的根目录：

    ```bash
    cd ultralytics
    ```

3.  使用 [pip](https://pip.pypa.io/en/stable/) 以可编辑模式（`-e`）安装包及其开发依赖项（`[dev]`）：

    ```bash
    pip install -e '.[dev]'
    ```

    此命令安装 `ultralytics` 包，使源代码的更改能立即反映在您的环境中，非常适合开发使用。

## 🚀 本地构建和预览

`mkdocs serve` 命令用于构建并在本地提供 [MkDocs](https://www.mkdocs.org/) 文档服务。这在开发和测试期间非常有用，可以预览更改效果。

```bash
mkdocs serve
```

- **命令说明：**
    - `mkdocs`：MkDocs 主命令行界面工具。
    - `serve`：用于构建并在本地提供文档站点服务的子命令。
- **注意：**
    - `mkdocs serve` 包含实时重载功能，当您保存文档文件的更改时，会自动更新浏览器中的预览。
    - 要停止本地服务器，只需在终端中按 `CTRL+C`。

## 🌍 构建和预览多语言版本

如果您的文档支持多种语言，请按照以下步骤构建和预览所有版本：

1.  使用 Git 暂存所有新建或修改的语言 Markdown（`.md`）文件：

    ```bash
    git add docs/**/*.md -f
    ```

2.  将所有语言版本构建到 `/site` 目录。此脚本确保包含相关的根级文件并清除之前的构建：

    ```bash
    # 清除现有的 /site 目录以防止冲突
    rm -rf site

    # 使用主配置文件构建默认语言站点
    mkdocs build -f docs/mkdocs.yml

    # 遍历每个语言特定的配置文件并构建其站点
    for file in docs/mkdocs_*.yml; do
      echo "正在使用 $file 构建 MkDocs 站点"
      mkdocs build -f "$file"
    done
    ```

3.  要在本地预览完整的多语言站点，进入构建输出目录并启动一个简单的 [Python HTTP 服务器](https://docs.python.org/3/library/http.server.html)：
    ```bash
    cd site
    python -m http.server
    # 在您喜欢的浏览器中打开 http://localhost:8000
    ```
    在 `http://localhost:8000` 访问实时预览站点。

## 📤 部署文档站点

要部署您的 MkDocs 文档站点，请选择一个托管提供商并配置您的部署方式。常见选项包括 [GitHub Pages](https://pages.github.com/)、GitLab Pages 或其他静态站点托管服务。

- 在 `mkdocs.yml` 文件中配置部署设置。
- 使用托管提供商推荐的工作流程（例如在 CI 中运行 `mkdocs build` 或使用 `mkdocs gh-deploy` 部署到 GitHub Pages）来发布生成的 `site/` 目录。

* **GitHub Pages 部署示例：**
  如果部署到 GitHub Pages，您可以使用内置命令：

    ```bash
    mkdocs gh-deploy
    ```

    部署后，如果您希望使用个性化 URL，可能需要在仓库设置页面中更新"自定义域名"设置。

    ![GitHub Pages 自定义域名设置](https://github.com/ultralytics/docs/releases/download/0/github-pages-custom-domain-setting.avif)

- 有关各种部署方法的详细说明，请参阅官方 [MkDocs 部署文档指南](https://www.mkdocs.org/user-guide/deploying-your-docs/)。

## 💡 贡献

我们非常重视开源社区对 Ultralytics 项目的贡献。您的参与有助于推动创新！请查看我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)了解如何参与的详细信息。您也可以通过我们的[调查问卷](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)分享您的反馈和想法。衷心感谢 🙏 所有贡献者的奉献和支持！

![Ultralytics 开源贡献者](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)

我们期待您的贡献！

## 📜 许可证

Ultralytics 文档提供两种许可选项以适应不同的使用场景：

- **AGPL-3.0 许可证**：适合参与学术研究和开放协作的学生、研究人员和爱好者。完整详情请参阅 [LICENSE](https://github.com/ultralytics/docs/blob/main/LICENSE) 文件。此许可证鼓励将改进回馈给社区。
- **企业许可证**：专为商业应用设计，此许可证允许将 Ultralytics 软件和 [AI 模型](https://docs.ultralytics.com/models/)无缝集成到商业产品和服务中。有关获取企业许可证的更多信息，请访问 [Ultralytics 许可](https://www.ultralytics.com/license)。

## ✉️ 联系我们

如需报告文档相关的错误、功能请求和其他问题，请使用 [GitHub Issues](https://github.com/ultralytics/docs/issues)。如需讨论、提问和社区支持，请加入我们的 [Discord 服务器](https://discord.com/invite/ultralytics)与同行和 Ultralytics 团队交流！

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
</div>
