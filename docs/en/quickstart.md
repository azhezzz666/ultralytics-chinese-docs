---
comments: true
description: 了解如何使用 pip、conda 或 Docker 安装 Ultralytics。按照我们的分步指南无缝设置 Ultralytics YOLO。
keywords: Ultralytics, YOLO11, 安装 Ultralytics, pip, conda, Docker, GitHub, 机器学习, 目标检测
---

# 安装 Ultralytics

Ultralytics 提供多种安装方法，包括 pip、conda 和 Docker。您可以通过 `ultralytics` pip 包安装 YOLO 以获取最新稳定版本，或者克隆 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)以获取最新版本。Docker 也是一个选择，可以在隔离的容器中运行包，避免本地安装。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics YOLO 快速入门指南
</p>

!!! example "安装"

    <p align="left" style="margin-bottom: -20px;">![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)<p>

    === "Pip 安装（推荐）"

        通过运行 `pip install -U ultralytics` 使用 pip 安装或更新 `ultralytics` 包。有关 `ultralytics` 包的更多详细信息，请访问 [Python 包索引 (PyPI)](https://pypi.org/project/ultralytics/)。

        [![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
        [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)

        ```bash
        # 从 PyPI 安装或升级 ultralytics 包
        pip install -U ultralytics
        ```

        您也可以直接从 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)安装 `ultralytics`。如果您想要最新的开发版本，这会很有用。确保您的系统上安装了 Git 命令行工具，然后运行：

        ```bash
        # 从 GitHub 安装 ultralytics 包
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```

    === "Conda 安装"

        Conda 可以作为 pip 的替代包管理器使用。有关更多详细信息，请访问 [Anaconda](https://anaconda.org/conda-forge/ultralytics)。用于更新 conda 包的 Ultralytics feedstock 仓库可在 [GitHub](https://github.com/conda-forge/ultralytics-feedstock/) 上找到。

        [![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics)
        [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)
        [![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics)
        [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # 使用 conda 安装 ultralytics 包
        conda install -c conda-forge ultralytics
        ```

        !!! note

            如果您在 CUDA 环境中安装，最佳做法是在同一命令中安装 `ultralytics`、`pytorch` 和 `pytorch-cuda`。这允许 conda 包管理器解决任何冲突。或者，如有必要，最后安装 `pytorch-cuda` 以覆盖特定于 CPU 的 `pytorch` 包。
            ```bash
            # 使用 conda 一起安装所有包
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Conda Docker 镜像

        Ultralytics Conda Docker 镜像也可在 [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 上获取。这些镜像基于 [Miniconda3](https://www.anaconda.com/docs/main)，提供了在 Conda 环境中开始使用 `ultralytics` 的简单方法。

        ```bash
        # 将镜像名称设置为变量
        t=ultralytics/ultralytics:latest-conda

        # 从 Docker Hub 拉取最新的 ultralytics 镜像
        sudo docker pull $t

        # 在支持 GPU 的容器中运行 ultralytics 镜像
        sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t            # 所有 GPU
        sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' $t # 指定 GPU
        ```

    === "Git 克隆"

        如果您有兴趣参与开发或希望使用最新源代码进行实验，请克隆 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)。克隆后，进入目录并使用 pip 以可编辑模式 `-e` 安装包。

        [![GitHub last commit](https://img.shields.io/github/last-commit/ultralytics/ultralytics?logo=github)](https://github.com/ultralytics/ultralytics)
        [![GitHub commit activity](https://img.shields.io/github/commit-activity/t/ultralytics/ultralytics)](https://github.com/ultralytics/ultralytics)

        ```bash
        # 克隆 ultralytics 仓库
        git clone https://github.com/ultralytics/ultralytics

        # 进入克隆的目录
        cd ultralytics

        # 以可编辑模式安装包用于开发
        pip install -e .
        ```

    === "Docker"

        使用 Docker 在隔离的容器中执行 `ultralytics` 包，确保在各种环境中的一致性能。通过从 [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 选择官方 `ultralytics` 镜像之一，您可以避免本地安装的复杂性并获得经过验证的工作环境。Ultralytics 提供五个主要支持的 Docker 镜像，每个都设计为高兼容性和高效率：

        [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)
        [![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics)](https://hub.docker.com/r/ultralytics/ultralytics)

        - **Dockerfile：** 推荐用于训练的 GPU 镜像。
        - **Dockerfile-arm64：** 针对 ARM64 架构优化，适合在树莓派和其他基于 ARM64 的平台上部署。
        - **Dockerfile-cpu：** 基于 Ubuntu 的纯 CPU 版本，适合推理和没有 GPU 的环境。
        - **Dockerfile-jetson：** 为 [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) 设备定制，集成了针对这些平台优化的 GPU 支持。
        - **Dockerfile-python：** 仅包含 Python 和必要依赖项的最小镜像，非常适合轻量级应用和开发。
        - **Dockerfile-conda：** 基于 Miniconda3，包含 `ultralytics` 包的 conda 安装。

        以下是获取最新镜像并执行它的命令：

        ```bash
        # 将镜像名称设置为变量
        t=ultralytics/ultralytics:latest

        # 从 Docker Hub 拉取最新的 ultralytics 镜像
        sudo docker pull $t

        # 在支持 GPU 的容器中运行 ultralytics 镜像
        sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t            # 所有 GPU
        sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' $t # 指定 GPU
        ```

        上述命令使用最新的 `ultralytics` 镜像初始化 Docker 容器。`-it` 标志分配一个伪 TTY 并保持 stdin 打开，允许与容器交互。`--ipc=host` 标志将 IPC（进程间通信）命名空间设置为主机，这对于进程之间共享内存至关重要。`--gpus all` 标志启用对容器内所有可用 GPU 的访问，这对于需要 GPU 计算的任务至关重要。

        注意：要在容器内使用本地机器上的文件，请使用 Docker 卷将本地目录挂载到容器中：

        ```bash
        # 将本地目录挂载到容器内的目录
        sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container $t
        ```

        将 `/path/on/host` 替换为本地机器上的目录路径，将 `/path/in/container` 替换为 Docker 容器内所需的路径。

        有关高级 Docker 用法，请探索 [Ultralytics Docker 指南](guides/docker-quickstart.md)。

有关依赖项列表，请参阅 `ultralytics` [pyproject.toml](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) 文件。请注意，上述所有示例都会安装所有必需的依赖项。

!!! tip

    [PyTorch](https://www.ultralytics.com/glossary/pytorch) 要求因操作系统和 CUDA 要求而异，因此请先按照 [PyTorch](https://pytorch.org/get-started/locally/) 上的说明安装 PyTorch。

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="PyTorch 安装说明" src="https://github.com/ultralytics/docs/releases/download/0/pytorch-installation-instructions.avif">
    </a>

## 自定义安装方法

虽然标准安装方法涵盖了大多数用例，但您可能需要更定制的设置。这可能涉及安装特定的包版本、省略可选依赖项，或替换包，例如将 `opencv-python` 替换为适用于服务器环境的无 GUI 版本 `opencv-python-headless`。

!!! example "自定义方法"

    === "方法 1：不安装依赖项（`--no-deps`）"

        您可以使用 pip 的 `--no-deps` 标志安装不带任何依赖项的 `ultralytics` 包核心。这需要您之后手动安装所有必要的依赖项。

        1.  **安装 `ultralytics` 核心：**
            ```bash
            pip install ultralytics --no-deps
            ```

        2.  **手动安装依赖项：** 您需要安装 `pyproject.toml` 文件中列出的所有必需包，根据需要替换或修改版本。对于无头 OpenCV 示例：
            ```bash
            # 安装其他核心依赖项
            pip install torch torchvision numpy matplotlib polars pyyaml pillow psutil requests scipy ultralytics-thop

            # 安装无头 OpenCV 而不是默认版本
            pip install opencv-python-headless
            ```

        !!! warning "依赖项管理"

            此方法提供完全控制，但需要仔细管理依赖项。通过参考 `ultralytics` `pyproject.toml` 文件，确保所有必需的包都以兼容的版本安装。

    === "方法 2：从自定义 Fork 安装"

        如果您需要持久的自定义修改（如始终使用 `opencv-python-headless`），您可以 fork Ultralytics 仓库，对 `pyproject.toml` 或其他代码进行更改，然后从您的 fork 安装。

        1.  **Fork** [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)到您自己的 GitHub 账户。
        2.  **克隆**您的 fork 到本地：
            ```bash
            git clone https://github.com/YOUR_USERNAME/ultralytics.git
            cd ultralytics
            ```
        3.  **创建新分支**用于您的更改：
            ```bash
            git checkout -b custom-opencv
            ```
        4.  **修改 `pyproject.toml`：** 在文本编辑器中打开 `pyproject.toml`，将包含 `"opencv-python>=4.6.0"` 的行替换为 `"opencv-python-headless>=4.6.0"`（根据需要调整版本）。
        5.  **提交并推送**您的更改：
            ```bash
            git add pyproject.toml
            git commit -m "Switch to opencv-python-headless"
            git push origin custom-opencv
            ```
        6.  使用 `git+https` 语法通过 pip **安装**，指向您的分支：
            ```bash
            pip install git+https://github.com/YOUR_USERNAME/ultralytics.git@custom-opencv
            ```

        此方法确保每当您从此特定 URL 安装时都使用您的自定义依赖项集。有关在 `requirements.txt` 文件中使用此方法，请参阅方法 4。

    === "方法 3：本地克隆、修改和安装"

        与用于开发的标准"Git 克隆"方法类似，您可以在本地克隆仓库，在安装*之前*修改依赖项文件，然后以可编辑模式安装。

        1.  **克隆** Ultralytics 仓库：
            ```bash
            git clone https://github.com/ultralytics/ultralytics
            cd ultralytics
            ```
        2.  **修改 `pyproject.toml`：** 编辑文件以进行所需的更改。例如，使用 `sed`（在 Linux/macOS 上）或文本编辑器将 `opencv-python` 替换为 `opencv-python-headless`。
            *使用 `sed`（首先验证 `pyproject.toml` 中的确切行）：*
            ```bash
            # 示例：替换以 "opencv-python..." 开头的行
            # 根据当前文件内容仔细调整模式
            sed -i'' -e 's/^\s*"opencv-python>=.*",/"opencv-python-headless>=4.8.0",/' pyproject.toml
            ```
            *或手动编辑 `pyproject.toml`* 将 `"opencv-python>=...` 更改为 `"opencv-python-headless>=..."`。
        3.  以可编辑模式（`-e`）**安装**包。Pip 现在将使用您修改的 `pyproject.toml` 来解析和安装依赖项：
            ```bash
            pip install -e .
            ```

        此方法对于在提交之前测试对依赖项或构建配置的本地更改，或设置特定的开发环境非常有用。

    === "方法 4：使用 `requirements.txt`"

        如果您使用 `requirements.txt` 文件管理项目依赖项，您可以直接在其中指定您的自定义 Ultralytics fork。这确保设置项目的任何人都能获得您的特定版本及其修改的依赖项（如 `opencv-python-headless`）。

        1.  **创建或编辑 `requirements.txt`：** 添加一行指向您的自定义 fork 和分支（如方法 2 中准备的）。
            ```text title="requirements.txt"
            # 核心依赖项
            numpy
            matplotlib
            polars
            pyyaml
            Pillow
            psutil
            requests>=2.23.0
            torch>=1.8.0 # 或特定版本/变体
            torchvision>=0.9.0 # 或特定版本/变体

            # 从特定 git 提交或分支安装 ultralytics
            # 将 YOUR_USERNAME 和 custom-branch 替换为您的详细信息
            git+https://github.com/YOUR_USERNAME/ultralytics.git@custom-branch

            # 其他项目依赖项
            flask
            # ... 等等
            ```
            *注意：您不需要在此处列出您的自定义 `ultralytics` fork 已经需要的依赖项（如 `opencv-python-headless`），因为 pip 将根据 fork 的 `pyproject.toml` 安装它们。*
        2.  从文件**安装**依赖项：
            ```bash
            pip install -r requirements.txt
            ```

        此方法与标准 Python 项目依赖项管理工作流程无缝集成，同时允许您将 `ultralytics` 固定到您的自定义 Git 源。

## 通过 CLI 使用 Ultralytics

Ultralytics 命令行界面（CLI）允许简单的单行命令，无需 Python 环境。CLI 不需要自定义或 Python 代码；使用 `yolo` 命令从终端运行所有任务。有关从命令行使用 YOLO 的更多信息，请参阅 [CLI 指南](usage/cli.md)。

!!! example

    === "语法"

        Ultralytics `yolo` 命令使用以下语法：
        ```bash
        yolo TASK MODE ARGS
        ```
        - `TASK`（可选）是 ([detect](tasks/detect.md), [segment](tasks/segment.md), [classify](tasks/classify.md), [pose](tasks/pose.md), [obb](tasks/obb.md)) 之一
        - `MODE`（必需）是 ([train](modes/train.md), [val](modes/val.md), [predict](modes/predict.md), [export](modes/export.md), [track](modes/track.md), [benchmark](modes/benchmark.md)) 之一
        - `ARGS`（可选）是 `arg=value` 对，如 `imgsz=640`，用于覆盖默认值。

        在完整的[配置指南](usage/cfg.md)中查看所有 `ARGS`，或使用 `yolo cfg` CLI 命令。

    === "训练"

        以初始学习率 0.01 训练检测模型 10 个[训练周期](https://www.ultralytics.com/glossary/epoch)：
        ```bash
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
        ```

    === "预测"

        使用预训练的分割模型以图像尺寸 320 预测 YouTube 视频：
        ```bash
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "验证"

        以批次大小 1 和图像尺寸 640 验证预训练的检测模型：
        ```bash
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
        ```

    === "导出"

        以图像尺寸 224x128 将 YOLO11n 分类模型导出为 ONNX 格式（不需要 TASK）：
        ```bash
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128
        ```

    === "计数"

        使用 YOLO11 在视频或实时流中计数目标：
        ```bash
        yolo solutions count show=True

        yolo solutions count source="path/to/video.mp4" # 指定视频文件路径
        ```

    === "健身监测"

        使用 YOLO11 姿态模型监测健身运动：
        ```bash
        yolo solutions workout show=True

        yolo solutions workout source="path/to/video.mp4" # 指定视频文件路径

        # 使用关键点进行腹部锻炼
        yolo solutions workout kpts="[5, 11, 13]" # 左侧
        yolo solutions workout kpts="[6, 12, 14]" # 右侧
        ```

    === "队列"

        使用 YOLO11 在指定队列或区域中计数目标：
        ```bash
        yolo solutions queue show=True

        yolo solutions queue source="path/to/video.mp4" # 指定视频文件路径

        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" # 配置队列坐标
        ```

    === "Streamlit 推理"

        使用 [Streamlit](https://docs.ultralytics.com/reference/solutions/streamlit_inference/) 在 Web 浏览器中执行目标检测、实例分割或姿态估计：
        ```bash
        yolo solutions inference

        yolo solutions inference model="path/to/model.pt" # 使用 Ultralytics Python 包微调的模型
        ```

    === "特殊命令"

        运行特殊命令以查看版本、查看设置、运行检查等：
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        yolo solutions help
        ```

!!! warning

    参数必须作为 `arg=value` 对传递，用等号 `=` 分隔并用空格分隔。不要使用 `--` 参数前缀或参数之间的逗号 `,`。

    - `yolo predict model=yolo11n.pt imgsz=640 conf=0.25`  ✅
    - `yolo predict model yolo11n.pt imgsz 640 conf 0.25`  ❌（缺少 `=`）
    - `yolo predict model=yolo11n.pt, imgsz=640, conf=0.25`  ❌（不要使用 `,`）
    - `yolo predict --model yolo11n.pt --imgsz 640 --conf 0.25`  ❌（不要使用 `--`）
    - `yolo solution model=yolo11n.pt imgsz=640 conf=0.25` ❌（使用 `solutions`，而不是 `solution`）

[CLI 指南](usage/cli.md){ .md-button }

## 通过 Python 使用 Ultralytics

Ultralytics YOLO Python 接口提供与 Python 项目的无缝集成，使加载、运行和处理模型输出变得容易。Python 接口设计简单，允许用户快速实现[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和分类。这使得 YOLO Python 接口成为将这些功能整合到 Python 项目中的宝贵工具。

例如，用户只需几行代码就可以加载模型、训练它、评估其性能并将其导出为 ONNX 格式。探索 [Python 指南](usage/python.md)以了解更多关于在 Python 项目中使用 YOLO 的信息。

!!! example

    ```python
    from ultralytics import YOLO

    # 从头创建新的 YOLO 模型
    model = YOLO("yolo11n.yaml")

    # 加载预训练的 YOLO 模型（推荐用于训练）
    model = YOLO("yolo11n.pt")

    # 使用 'coco8.yaml' 数据集训练模型 3 个周期
    results = model.train(data="coco8.yaml", epochs=3)

    # 在验证集上评估模型性能
    results = model.val()

    # 使用模型对图像执行目标检测
    results = model("https://ultralytics.com/images/bus.jpg")

    # 将模型导出为 ONNX 格式
    success = model.export(format="onnx")
    ```

[Python 指南](usage/python.md){.md-button .md-button--primary}

## Ultralytics 设置

Ultralytics 库包含一个 `SettingsManager`，用于对实验进行精细控制，允许用户轻松访问和修改设置。这些设置存储在环境用户配置目录中的 JSON 文件中，可以在 Python 环境中或通过命令行界面（CLI）查看或修改。

### 检查设置

要查看当前设置配置：

!!! example "查看设置"

    === "Python"

        通过从 `ultralytics` 模块导入 `settings` 对象来使用 Python 查看您的设置。使用以下命令打印和返回设置：
        ```python
        from ultralytics import settings

        # 查看所有设置
        print(settings)

        # 返回特定设置
        value = settings["runs_dir"]
        ```

    === "CLI"

        命令行界面允许您使用以下命令检查设置：
        ```bash
        yolo settings
        ```

### 修改设置

Ultralytics 使修改设置变得简单，方法如下：

!!! example "更新设置"

    === "Python"

        在 Python 中，使用 `settings` 对象的 `update` 方法：
        ```python
        from ultralytics import settings

        # 更新一个设置
        settings.update({"runs_dir": "/path/to/runs"})

        # 更新多个设置
        settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

        # 将设置重置为默认值
        settings.reset()
        ```

    === "CLI"

        使用命令行界面修改设置：
        ```bash
        # 更新一个设置
        yolo settings runs_dir='/path/to/runs'

        # 更新多个设置
        yolo settings runs_dir='/path/to/runs' tensorboard=False

        # 将设置重置为默认值
        yolo settings reset
        ```

### 理解设置

下表概述了 Ultralytics 中可调整的设置，包括示例值、数据类型和描述。

| 名称               | 示例值                | 数据类型 | 描述                                                                                                      |
| ------------------ | --------------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| `settings_version` | `'0.0.4'`             | `str`    | Ultralytics _settings_ 版本（与 Ultralytics [pip] 版本不同）                                              |
| `datasets_dir`     | `'/path/to/datasets'` | `str`    | 存储数据集的目录                                                                                          |
| `weights_dir`      | `'/path/to/weights'`  | `str`    | 存储模型权重的目录                                                                                        |
| `runs_dir`         | `'/path/to/runs'`     | `str`    | 存储实验运行的目录                                                                                        |
| `uuid`             | `'a1b2c3d4'`          | `str`    | 当前设置的唯一标识符                                                                                      |
| `sync`             | `True`                | `bool`   | 将分析和崩溃同步到 [Ultralytics HUB] 的选项                                                               |
| `api_key`          | `''`                  | `str`    | [Ultralytics HUB] API 密钥                                                                                |
| `clearml`          | `True`                | `bool`   | 使用 [ClearML] 日志记录的选项                                                                             |
| `comet`            | `True`                | `bool`   | 使用 [Comet ML] 进行实验跟踪和可视化的选项                                                                |
| `dvc`              | `True`                | `bool`   | 使用 [DVC 进行实验跟踪]和版本控制的选项                                                                   |
| `hub`              | `True`                | `bool`   | 使用 [Ultralytics HUB] 集成的选项                                                                         |
| `mlflow`           | `True`                | `bool`   | 使用 [MLFlow] 进行实验跟踪的选项                                                                          |
| `neptune`          | `True`                | `bool`   | 使用 [Neptune] 进行实验跟踪的选项                                                                         |
| `raytune`          | `True`                | `bool`   | 使用 [Ray Tune] 进行[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)的选项        |
| `tensorboard`      | `True`                | `bool`   | 使用 [TensorBoard] 进行可视化的选项                                                                       |
| `wandb`            | `True`                | `bool`   | 使用 [Weights & Biases] 日志记录的选项                                                                    |
| `vscode_msg`       | `True`                | `bool`   | 当检测到 VS Code 终端时，启用下载 [Ultralytics-Snippets] 扩展的提示                                       |

随着项目或实验的进展，重新审视这些设置以确保最佳配置。

## 常见问题

### 如何使用 pip 安装 Ultralytics？

使用 pip 安装 Ultralytics：

```bash
pip install -U ultralytics
```

这将从 [PyPI](https://pypi.org/project/ultralytics/) 安装最新稳定版本的 `ultralytics` 包。要直接从 GitHub 安装开发版本：

```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```

确保您的系统上安装了 Git 命令行工具。

### 我可以使用 conda 安装 Ultralytics YOLO 吗？

是的，使用 conda 安装 Ultralytics YOLO：

```bash
conda install -c conda-forge ultralytics
```

此方法是 pip 的绝佳替代方案，确保与其他包的兼容性。对于 CUDA 环境，一起安装 `ultralytics`、`pytorch` 和 `pytorch-cuda` 以解决冲突：

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

有关更多说明，请参阅 [Conda 快速入门指南](guides/conda-quickstart.md)。

### 使用 Docker 运行 Ultralytics YOLO 有什么优势？

Docker 为 Ultralytics YOLO 提供了一个隔离、一致的环境，确保跨系统的平稳性能并避免本地安装的复杂性。官方 Docker 镜像可在 [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) 上获取，提供 GPU、CPU、ARM64、[NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) 和 Conda 变体。要拉取并运行最新镜像：

```bash
# 从 Docker Hub 拉取最新的 ultralytics 镜像
sudo docker pull ultralytics/ultralytics:latest

# 在支持 GPU 的容器中运行 ultralytics 镜像
sudo docker run -it --ipc=host --runtime=nvidia --gpus all ultralytics/ultralytics:latest
```

有关详细的 Docker 说明，请参阅 [Docker 快速入门指南](guides/docker-quickstart.md)。

### 如何克隆 Ultralytics 仓库进行开发？

克隆 Ultralytics 仓库并设置开发环境：

```bash
# 克隆 ultralytics 仓库
git clone https://github.com/ultralytics/ultralytics

# 进入克隆的目录
cd ultralytics

# 以可编辑模式安装包用于开发
pip install -e .
```

这允许为项目做贡献或使用最新源代码进行实验。有关详细信息，请访问 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)。

### 为什么应该使用 Ultralytics YOLO CLI？

Ultralytics YOLO CLI 简化了运行目标检测任务而无需 Python 代码，可以直接从终端使用单行命令进行训练、验证和预测。基本语法是：

```bash
yolo TASK MODE ARGS
```

例如，训练检测模型：

```bash
yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
```

在完整的 [CLI 指南](usage/cli.md)中探索更多命令和使用示例。

<!-- 文章链接 -->

[Ultralytics HUB]: https://hub.ultralytics.com
[API Key]: https://hub.ultralytics.com/settings?tab=api+keys
[pip]: https://pypi.org/project/ultralytics/
[DVC 进行实验跟踪]: https://dvc.org/doc/dvclive/ml-frameworks/yolo
[Comet ML]: https://bit.ly/yolov8-readme-comet
[Ultralytics HUB]: https://hub.ultralytics.com
[ClearML]: ./integrations/clearml.md
[MLFlow]: ./integrations/mlflow.md
[Neptune]: https://neptune.ai/
[Tensorboard]: ./integrations/tensorboard.md
[Ray Tune]: ./integrations/ray-tune.md
[Weights & Biases]: ./integrations/weights-biases.md
[Ultralytics-Snippets]: ./integrations/vscode.md
