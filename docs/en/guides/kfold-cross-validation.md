---
comments: true
description: 学习如何使用 Ultralytics YOLO 为目标检测数据集实现 K 折交叉验证。提高模型的可靠性和鲁棒性。
keywords: Ultralytics, YOLO, K 折交叉验证, 目标检测, sklearn, pandas, PyYAML, 机器学习, 数据集划分
---

# 使用 Ultralytics 进行 K 折交叉验证

## 简介

本综合指南说明了如何在 Ultralytics 生态系统中为[目标检测](https://www.ultralytics.com/glossary/object-detection)数据集实现 K 折交叉验证。我们将利用 YOLO 检测格式和关键 Python 库（如 sklearn、pandas 和 PyYAML）来指导您完成必要的设置、生成特征向量的过程以及执行 K 折数据集划分。

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/k-fold-cross-validation-overview.avif" alt="K 折交叉验证概述">
</p>

无论您的项目涉及水果检测数据集还是自定义数据源，本教程旨在帮助您理解和应用 K 折交叉验证，以增强[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型的可靠性和鲁棒性。虽然本教程使用 `k=5` 折，但请记住，最佳折数可能因您的数据集和项目具体情况而异。

让我们开始吧。

## 设置

- 您的标注应采用 [YOLO 检测格式](../datasets/detect/index.md)。

- 本指南假设标注文件在本地可用。

- 对于我们的演示，我们使用[水果检测](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection/code)数据集。
    - 该数据集共包含 8479 张图像。
    - 它包含 6 个类别标签，每个类别的总实例数如下所示。

| 类别标签 | 实例数量 |
| :------- | :------: |
| 苹果     |   7049   |
| 葡萄     |   7202   |
| 菠萝     |   1613   |
| 橙子     |  15549   |
| 香蕉     |   3536   |
| 西瓜     |   1976   |

- 必需的 Python 包包括：
    - `ultralytics`
    - `sklearn`
    - `pandas`
    - `pyyaml`

- 本教程使用 `k=5` 折。但是，您应该为您的特定数据集确定最佳折数。

1. 为您的项目创建一个新的 Python 虚拟环境（`venv`）并激活它。使用 `pip`（或您首选的包管理器）安装：
    - Ultralytics 库：`pip install -U ultralytics`。或者，您可以克隆官方[仓库](https://github.com/ultralytics/ultralytics)。
    - Scikit-learn、pandas 和 PyYAML：`pip install -U scikit-learn pandas pyyaml`。

2. 验证您的标注是否采用 [YOLO 检测格式](../datasets/detect/index.md)。
    - 对于本教程，所有标注文件都位于 `Fruit-Detection/labels` 目录中。

## 为目标检测数据集生成特征向量

1. 首先为以下步骤创建一个新的 `example.py` Python 文件。

2. 继续检索数据集的所有标签文件。

    ```python
    from pathlib import Path

    dataset_path = Path("./Fruit-detection")  # 将其替换为您自定义数据的 'path/to/dataset'
    labels = sorted(dataset_path.rglob("*labels/*.txt"))  # 'labels' 中的所有数据
    ```

3. 现在，读取数据集 YAML 文件的内容并提取类别标签的索引。

    ```python
    import yaml

    yaml_file = "path/to/data.yaml"  # 包含数据目录和名称字典的数据 YAML
    with open(yaml_file, encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = sorted(classes.keys())
    ```

4. 初始化一个空的 `pandas` DataFrame。

    ```python
    import pandas as pd

    index = [label.stem for label in labels]  # 使用基本文件名作为 ID（无扩展名）
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)
    ```

5. 统计标注文件中每个类别标签的实例数量。

    ```python
    from collections import Counter

    for label in labels:
        lbl_counter = Counter()

        with open(label) as lf:
            lines = lf.readlines()

        for line in lines:
            # YOLO 标签的类别使用每行第一个位置的整数
            lbl_counter[int(line.split(" ", 1)[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)  # 将 `nan` 值替换为 `0.0`
    ```

6. 以下是填充后的 DataFrame 的示例视图：

    ```
                                                           0    1    2    3    4    5
    '0000a16e4b057580_jpg.rf.00ab48988370f64f5ca8ea4...'  0.0  0.0  0.0  0.0  0.0  7.0
    '0000a16e4b057580_jpg.rf.7e6dce029fb67f01eb19aa7...'  0.0  0.0  0.0  0.0  0.0  7.0
    '0000a16e4b057580_jpg.rf.bc4d31cdcbe229dd022957a...'  0.0  0.0  0.0  0.0  0.0  7.0
    '00020ebf74c4881c_jpg.rf.508192a0a97aa6c4a3b6882...'  0.0  0.0  0.0  1.0  0.0  0.0
    '00020ebf74c4881c_jpg.rf.5af192a2254c8ecc4188a25...'  0.0  0.0  0.0  1.0  0.0  0.0
     ...                                                  ...  ...  ...  ...  ...  ...
    'ff4cd45896de38be_jpg.rf.c4b5e967ca10c7ced3b9e97...'  0.0  0.0  0.0  0.0  0.0  2.0
    'ff4cd45896de38be_jpg.rf.ea4c1d37d2884b3e3cbce08...'  0.0  0.0  0.0  0.0  0.0  2.0
    'ff5fd9c3c624b7dc_jpg.rf.bb519feaa36fc4bf630a033...'  1.0  0.0  0.0  0.0  0.0  0.0
    'ff5fd9c3c624b7dc_jpg.rf.f0751c9c3aa4519ea3c9d6a...'  1.0  0.0  0.0  0.0  0.0  0.0
    'fffe28b31f2a70d4_jpg.rf.7ea16bd637ba0711c53b540...'  0.0  6.0  0.0  0.0  0.0  0.0
    ```

行索引标签文件，每个对应数据集中的一张图像，列对应您的类别标签索引。每行代表一个伪特征向量，包含数据集中每个类别标签的计数。这种数据结构使得能够将 [K 折交叉验证](https://www.ultralytics.com/glossary/cross-validation)应用于目标检测数据集。

## K 折数据集划分

1. 现在我们将使用 `sklearn.model_selection` 中的 `KFold` 类来生成数据集的 `k` 个划分。
    - 重要提示：
        - 设置 `shuffle=True` 确保类别在划分中随机分布。
        - 通过设置 `random_state=M`（其中 `M` 是选定的整数），您可以获得可重复的结果。

    ```python
    import random

    from sklearn.model_selection import KFold

    random.seed(0)  # 为了可重复性
    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # 设置 random_state 以获得可重复的结果

    kfolds = list(kf.split(labels_df))
    ```

2. 数据集现在已被划分为 `k` 折，每折都有一个 `train` 和 `val` 索引列表。我们将构建一个 DataFrame 来更清晰地显示这些结果。

    ```python
    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"
    ```

3. 现在我们将计算每折的类别标签分布，作为 `val` 中存在的类别与 `train` 中存在的类别的比率。

    ```python
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # 为避免除以零，我们在分母中添加一个小值 (1E-7)
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio
    ```

    理想情况是每个划分和跨类别的所有类别比率都相当相似。然而，这将取决于您数据集的具体情况。

4. 接下来，我们为每个划分创建目录和数据集 YAML 文件。

    ```python
    import datetime

    supported_extensions = [".jpg", ".jpeg", ".png"]

    # 初始化一个空列表来存储图像文件路径
    images = []

    # 循环遍历支持的扩展名并收集图像文件
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

    # 创建必要的目录和数据集 YAML 文件
    save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # 创建目录
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # 创建数据集 YAML 文件
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )
    ```

5. 最后，将图像和标签复制到每个划分的相应目录（'train' 或 'val'）。
    - **注意：**此部分代码所需的时间将根据数据集的大小和系统硬件而有所不同。

    ```python
    import shutil

    from tqdm import tqdm

    for image, label in tqdm(zip(images, labels), total=len(images), desc="Copying files"):
        for split, k_split in folds_df.loc[image.stem].items():
            # 目标目录
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # 将图像和标签文件复制到新目录（如果文件已存在则 SamefileError）
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)
    ```

## 保存记录（可选）

可选地，您可以将 K 折划分和标签分布 DataFrame 的记录保存为 CSV 文件以供将来参考。

```python
folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
```

## 使用 K 折数据划分训练 YOLO

1. 首先，加载 YOLO 模型。

    ```python
    from ultralytics import YOLO

    weights_path = "path/to/weights.pt"  # 使用 yolo11n.pt 作为小型模型
    model = YOLO(weights_path, task="detect")
    ```

2. 接下来，遍历数据集 YAML 文件以运行训练。结果将保存到由 `project` 和 `name` 参数指定的目录中。默认情况下，此目录为 'runs/detect/train#'，其中 # 是整数索引。

    ```python
    results = {}

    # 在此定义您的附加参数
    batch = 16
    project = "kfold_demo"
    epochs = 100

    for k, dataset_yaml in enumerate(ds_yamls):
        model = YOLO(weights_path, task="detect")
        results[k] = model.train(
            data=dataset_yaml, epochs=epochs, batch=batch, project=project, name=f"fold_{k + 1}"
        )  # 包含任何附加训练参数
    ```

3. 您还可以使用 [Ultralytics data.utils.autosplit](https://docs.ultralytics.com/reference/data/utils/) 函数进行自动数据集划分：

    ```python
    from ultralytics.data.split import autosplit

    # 自动将数据集划分为 train/val/test
    autosplit(path="path/to/images", weights=(0.8, 0.2, 0.0), annotated_only=True)
    ```

## 结论

在本指南中，我们探索了使用 K 折交叉验证训练 YOLO 目标检测模型的过程。我们学习了如何将数据集划分为 K 个分区，确保不同折之间的类别分布平衡。

我们还探索了创建报告 DataFrame 的过程，以可视化这些划分中的数据划分和标签分布，为我们提供了对训练集和验证集结构的清晰洞察。

可选地，我们保存了记录以供将来参考，这在大型项目或排查模型性能问题时特别有用。

最后，我们在循环中使用每个划分实现了实际的模型训练，保存训练结果以供进一步分析和比较。

K 折交叉验证技术是充分利用可用数据的稳健方法，它有助于确保模型性能在不同数据子集上可靠且一致。这产生了一个更具泛化能力和可靠性的模型，不太可能[过拟合](https://www.ultralytics.com/glossary/overfitting)到特定的数据模式。

请记住，虽然我们在本指南中使用了 YOLO，但这些步骤大多可以转移到其他机器学习模型。理解这些步骤使您能够在自己的机器学习项目中有效地应用交叉验证。

## 常见问题

### 什么是 K 折交叉验证，为什么它在目标检测中有用？

K 折交叉验证是一种将数据集划分为 'k' 个子集（折）以更可靠地评估模型性能的技术。每折既作为训练数据又作为[验证数据](https://www.ultralytics.com/glossary/validation-data)。在目标检测的背景下，使用 K 折交叉验证有助于确保您的 Ultralytics YOLO 模型的性能在不同数据划分上是稳健和可泛化的，从而增强其可靠性。有关使用 Ultralytics YOLO 设置 K 折交叉验证的详细说明，请参阅[使用 Ultralytics 进行 K 折交叉验证](#简介)。

### 如何使用 Ultralytics YOLO 实现 K 折交叉验证？

要使用 Ultralytics YOLO 实现 K 折交叉验证，您需要按照以下步骤操作：

1. 验证标注是否采用 [YOLO 检测格式](../datasets/detect/index.md)。
2. 使用 Python 库如 `sklearn`、`pandas` 和 `pyyaml`。
3. 从数据集创建特征向量。
4. 使用 `sklearn.model_selection` 中的 `KFold` 划分数据集。
5. 在每个划分上训练 YOLO 模型。

有关全面指南，请参阅我们文档中的 [K 折数据集划分](#k-折数据集划分)部分。

### 为什么应该使用 Ultralytics YOLO 进行目标检测？

Ultralytics YOLO 提供最先进的实时目标检测，具有高[精度](https://www.ultralytics.com/glossary/accuracy)和效率。它功能多样，支持多种[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务，如检测、分割和分类。此外，它与 [Ultralytics HUB](https://docs.ultralytics.com/hub/) 等工具无缝集成，实现无代码模型训练和部署。有关更多详情，请在我们的 [Ultralytics YOLO 页面](https://www.ultralytics.com/yolo)上探索优势和功能。

### 如何确保我的标注采用正确的 Ultralytics YOLO 格式？

您的标注应遵循 YOLO 检测格式。每个标注文件必须列出对象类别，以及其在图像中的[边界框](https://www.ultralytics.com/glossary/bounding-box)坐标。YOLO 格式确保为训练目标检测模型提供简化和标准化的数据处理。有关正确标注格式的更多信息，请访问 [YOLO 检测格式指南](../datasets/detect/index.md)。

### 我可以将 K 折交叉验证用于水果检测以外的自定义数据集吗？

是的，只要标注采用 YOLO 检测格式，您就可以将 K 折交叉验证用于任何自定义数据集。将数据集路径和类别标签替换为您自定义数据集特定的路径和标签。这种灵活性确保任何目标检测项目都可以从使用 K 折交叉验证进行稳健模型评估中受益。有关实际示例，请查看我们的[为目标检测数据集生成特征向量](#为目标检测数据集生成特征向量)部分。
