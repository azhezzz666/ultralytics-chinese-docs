---
comments: true
description: 探索 Ultralytics 用于检测、分割、分类等视觉任务的多样化数据集。使用高质量标注数据增强您的项目。
keywords: Ultralytics, 数据集, 计算机视觉, 目标检测, 实例分割, 姿态估计, 图像分类, 多目标跟踪
---

# 数据集概述

Ultralytics 提供对各种数据集的支持，以促进检测、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)、姿态估计、分类和多目标跟踪等计算机视觉任务。以下是主要 Ultralytics 数据集的列表，以及每个计算机视觉任务及其相应数据集的摘要。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/YDXKa1EljmU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics 数据集概述
</p>

## [目标检测](detect/index.md)

[边界框](https://www.ultralytics.com/glossary/bounding-box)目标检测是一种计算机视觉技术，涉及通过在每个目标周围绘制边界框来检测和定位图像中的目标。

- [African-wildlife](detect/african-wildlife.md)：包含非洲野生动物图像的数据集，包括水牛、大象、犀牛和斑马。
- [Argoverse](detect/argoverse.md)：包含来自城市环境的 3D 跟踪和运动预测数据的数据集，具有丰富的标注。
- [Brain-tumor](detect/brain-tumor.md)：用于检测脑肿瘤的数据集，包括 MRI 或 CT 扫描图像，包含肿瘤存在、位置和特征的详细信息。
- [COCO](detect/coco.md)：通用目标上下文（COCO）是一个大规模目标检测、分割和字幕数据集，包含 80 个目标类别。
- [COCO8](detect/coco8.md)：COCO 训练集和验证集前 4 张图像的较小子集，适合快速测试。
- [COCO8-Grayscale](detect/coco8-grayscale.md)：通过将 RGB 转换为灰度创建的 COCO8 灰度版本，适用于单通道模型评估。
- [COCO8-Multispectral](detect/coco8-multispectral.md)：通过插值 RGB 波长创建的 COCO8 10 通道多光谱版本，适用于光谱感知模型评估。
- [COCO128](detect/coco128.md)：COCO 训练集和验证集前 128 张图像的较小子集，适合测试。
- [Construction-PPE](detect/construction-ppe.md)：建筑工地图像数据集，标注了关键安全装备如安全帽、背心、手套、靴子和护目镜，以及缺失装备的标签，支持开发用于合规性和工人保护的 AI 模型。
- [Global Wheat 2020](detect/globalwheat2020.md)：包含 2020 年全球小麦挑战赛小麦穗图像的数据集。
- [HomeObjects-3K](detect/homeobjects-3k.md)：标注室内场景数据集，包含 12 种常见家居物品，非常适合开发和测试智能家居系统、机器人和增强现实中的计算机视觉模型。
- [KITTI](detect/kitti.md) 新：著名的自动驾驶数据集，包含立体、激光雷达和 GPS/IMU 输入，用于各种道路场景中的 2D 目标检测。
- [LVIS](detect/lvis.md)：大规模目标检测、分割和字幕数据集，包含 1203 个目标类别。
- [Medical-pills](detect/medical-pills.md)：包含标注医药药片图像的数据集，旨在帮助制药质量控制、分类和确保符合行业标准等任务。
- [Objects365](detect/objects365.md)：高质量、大规模目标检测数据集，包含 365 个目标类别和超过 60 万张标注图像。
- [OpenImagesV7](detect/open-images-v7.md)：Google 的综合数据集，包含 170 万张训练图像和 4.2 万张验证图像。
- [RF100](detect/roboflow-100.md)：多样化的目标检测基准，包含 100 个数据集，涵盖七个图像领域，用于全面的模型评估。
- [Signature](detect/signature.md)：包含各种文档图像的数据集，带有标注签名，支持文档验证和欺诈检测研究。
- [SKU-110K](detect/sku-110k.md)：零售环境中密集目标检测数据集，包含超过 1.1 万张图像和 170 万个边界框。
- [VisDrone](detect/visdrone.md)：包含无人机拍摄图像的目标检测和多目标跟踪数据的数据集，包含超过 1 万张图像和视频序列。
- [VOC](detect/voc.md)：Pascal 视觉目标类别（VOC）目标检测和分割数据集，包含 20 个目标类别和超过 1.1 万张图像。
- [xView](detect/xview.md)：航拍图像中的目标检测数据集，包含 60 个目标类别和超过 100 万个标注目标。

## [实例分割](segment/index.md)

实例分割是一种计算机视觉技术，涉及在像素级别识别和定位图像中的目标。与仅对每个像素进行分类的语义分割不同，[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)区分同一类别的不同实例。

- [Carparts-seg](segment/carparts-seg.md)：专门用于识别车辆零件的数据集，满足设计、制造和研究需求。它同时用于目标检测和分割任务。
- [COCO](segment/coco.md)：为目标检测、分割和字幕任务设计的大规模数据集，包含超过 20 万张标注图像。
- [COCO8-seg](segment/coco8-seg.md)：用于实例分割任务的较小数据集，包含 8 张带有分割标注的 COCO 图像子集。
- [COCO128-seg](segment/coco128-seg.md)：用于实例分割任务的较小数据集，包含 128 张带有分割标注的 COCO 图像子集。
- [Crack-seg](segment/crack-seg.md)：专门用于检测道路和墙壁裂缝的数据集，适用于目标检测和分割任务。
- [Package-seg](segment/package-seg.md)：专门用于识别仓库或工业环境中包裹的数据集，适用于目标检测和分割应用。

## [姿态估计](pose/index.md)

姿态估计是一种用于确定目标相对于相机或世界坐标系姿态的技术。这涉及识别目标上的关键点或关节，特别是人类或动物。

- [COCO](pose/coco.md)：带有人体姿态标注的大规模数据集，专为姿态估计任务设计。
- [COCO8-pose](pose/coco8-pose.md)：用于姿态估计任务的较小数据集，包含 8 张带有人体姿态标注的 COCO 图像子集。
- [Dog-pose](pose/dog-pose.md)：综合数据集，包含约 6,000 张以狗为主题的图像，每只狗标注 24 个关键点，专为姿态估计任务定制。
- [Hand-Keypoints](pose/hand-keypoints.md)：简洁的数据集，包含超过 26,000 张以人手为中心的图像，每只手标注 21 个关键点，专为姿态估计任务设计。
- [Tiger-pose](pose/tiger-pose.md)：紧凑的数据集，包含 263 张以老虎为主题的图像，每只老虎标注 12 个关键点，用于姿态估计任务。

## [分类](classify/index.md)

[图像分类](https://www.ultralytics.com/glossary/image-classification)是一种计算机视觉任务，涉及根据图像的视觉内容将其分类到一个或多个预定义的类别或类别中。

- [Caltech 101](classify/caltech101.md)：包含 101 个目标类别图像的数据集，用于图像分类任务。
- [Caltech 256](classify/caltech256.md)：Caltech 101 的扩展版本，包含 256 个目标类别和更具挑战性的图像。
- [CIFAR-10](classify/cifar10.md)：包含 6 万张 32x32 彩色图像的数据集，分为 10 个类别，每个类别 6000 张图像。
- [CIFAR-100](classify/cifar100.md)：CIFAR-10 的扩展版本，包含 100 个目标类别，每个类别 600 张图像。
- [Fashion-MNIST](classify/fashion-mnist.md)：包含 70,000 张 10 个时尚类别灰度图像的数据集，用于图像分类任务。
- [ImageNet](classify/imagenet.md)：用于目标检测和图像分类的大规模数据集，包含超过 1400 万张图像和 20,000 个类别。
- [ImageNet-10](classify/imagenet10.md)：ImageNet 的较小子集，包含 10 个类别，用于更快的实验和测试。
- [Imagenette](classify/imagenette.md)：ImageNet 的较小子集，包含 10 个易于区分的类别，用于更快的训练和测试。
- [Imagewoof](classify/imagewoof.md)：ImageNet 更具挑战性的子集，包含 10 个狗品种类别，用于图像分类任务。
- [MNIST](classify/mnist.md)：包含 70,000 张手写数字灰度图像的数据集，用于图像分类任务。
- [MNIST160](classify/mnist.md)：MNIST 数据集中每个类别的前 8 张图像。数据集共包含 160 张图像。

## [定向边界框 (OBB)](obb/index.md)

定向边界框（OBB）是计算机视觉中使用旋转边界框检测图像中倾斜目标的方法，通常应用于航拍和卫星图像。与传统边界框不同，OBB 可以更好地适应各种方向的目标。

- [DOTA-v2](obb/dota-v2.md)：流行的 OBB 航拍图像数据集，包含 170 万个实例和 11,268 张图像。
- [DOTA8](obb/dota8.md)：DOTAv1 分割集前 8 张图像的较小子集，4 张用于训练，4 张用于验证，适合快速测试。

## [多目标跟踪](track/index.md)

多目标跟踪是一种计算机视觉技术，涉及在视频序列中随时间检测和跟踪多个目标。此任务通过在帧之间保持目标的一致身份来扩展目标检测。

- [Argoverse](detect/argoverse.md)：包含来自城市环境的 3D 跟踪和运动预测数据的数据集，具有丰富的多目标跟踪任务标注。
- [VisDrone](detect/visdrone.md)：包含无人机拍摄图像的目标检测和多目标跟踪数据的数据集，包含超过 1 万张图像和视频序列。

## 贡献新数据集

贡献新数据集涉及几个步骤，以确保它与现有基础设施良好对齐。以下是必要的步骤：

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yMR7BgwHQ3g?start=427"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何为 Ultralytics 数据集做贡献
</p>

### 贡献新数据集的步骤

1. **收集图像**：收集属于数据集的图像。这些可以从各种来源收集，如公共数据库或您自己的收藏。
2. **标注图像**：根据任务使用边界框、分割或关键点标注这些图像。
3. **导出标注**：将这些标注转换为 Ultralytics 支持的 YOLO `*.txt` 文件格式。
4. **组织数据集**：将数据集排列成正确的文件夹结构。您应该有 `images/` 和 `labels/` 顶级目录，每个目录中都有 `train/` 和 `val/` 子目录。

    ```
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    ```

5. **创建 `data.yaml` 文件**：在数据集的根目录中，创建一个描述数据集、类别和其他必要信息的 `data.yaml` 文件。
6. **优化图像（可选）**：如果您想减小数据集大小以提高处理效率，可以使用下面的代码优化图像。这不是必需的，但建议用于较小的数据集大小和更快的下载速度。
7. **压缩数据集**：将整个数据集文件夹压缩成 zip 文件。
8. **文档和 PR**：创建描述您的数据集以及它如何适应现有框架的文档页面。之后，提交拉取请求（PR）。有关如何提交 PR 的更多详细信息，请参阅 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing/)。

### 优化和压缩数据集的示例代码

!!! example "优化和压缩数据集"

    === "Python"

       ```python
       from pathlib import Path

       from ultralytics.data.utils import compress_one_image
       from ultralytics.utils.downloads import zip_directory

       # 定义数据集目录
       path = Path("path/to/dataset")

       # 优化数据集中的图像（可选）
       for f in path.rglob("*.jpg"):
           compress_one_image(f)

       # 将数据集压缩到 'path/to/dataset.zip'
       zip_directory(path)
       ```

通过遵循这些步骤，您可以贡献一个与 Ultralytics 现有结构良好集成的新数据集。

## 常见问题

### Ultralytics 支持哪些目标检测数据集？

Ultralytics 支持多种[目标检测](https://www.ultralytics.com/glossary/object-detection)数据集，包括：

- [COCO](detect/coco.md)：大规模目标检测、分割和字幕数据集，包含 80 个目标类别。
- [LVIS](detect/lvis.md)：包含 1203 个目标类别的广泛数据集，专为更细粒度的目标检测和分割设计。
- [Argoverse](detect/argoverse.md)：包含来自城市环境的 3D 跟踪和运动预测数据的数据集，具有丰富的标注。
- [VisDrone](detect/visdrone.md)：包含无人机拍摄图像的目标检测和多目标跟踪数据的数据集。
- [SKU-110K](detect/sku-110k.md)：零售环境中的密集目标检测，包含超过 1.1 万张图像。

这些数据集有助于为各种目标检测应用训练强大的 [Ultralytics YOLO](https://docs.ultralytics.com/models/) 模型。

### 如何向 Ultralytics 贡献新数据集？

贡献新数据集涉及几个步骤：

1. **收集图像**：从公共数据库或个人收藏中收集图像。
2. **标注图像**：根据任务应用边界框、分割或关键点。
3. **导出标注**：将标注转换为 YOLO `*.txt` 格式。
4. **组织数据集**：使用包含 `train/` 和 `val/` 目录的文件夹结构，每个目录包含 `images/` 和 `labels/` 子目录。
5. **创建 `data.yaml` 文件**：包含数据集描述、类别和其他相关信息。
6. **优化图像（可选）**：减小数据集大小以提高效率。
7. **压缩数据集**：将数据集压缩成 zip 文件。
8. **文档和 PR**：描述您的数据集并按照 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing/)提交拉取请求。

访问[贡献新数据集](#贡献新数据集)获取全面指南。

### 为什么应该使用 Ultralytics HUB 管理我的数据集？

[Ultralytics HUB](https://hub.ultralytics.com/) 为数据集管理和分析提供强大功能，包括：

- **无缝数据集管理**：在一个地方上传、组织和管理您的数据集。
- **即时训练集成**：直接使用上传的数据集进行模型训练，无需额外设置。
- **可视化工具**：探索和可视化您的数据集图像和标注。
- **数据集分析**：深入了解您的数据集分布和特征。

该平台简化了从数据集管理到模型训练的过渡，使整个过程更加高效。了解更多关于 [Ultralytics HUB 数据集](https://docs.ultralytics.com/hub/datasets/)的信息。

### Ultralytics YOLO 模型在计算机视觉方面有哪些独特功能？

Ultralytics YOLO 模型为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务提供了几个独特功能：

- **实时性能**：适用于时间敏感应用的高速推理和训练能力。
- **多功能性**：在统一框架中支持检测、分割、分类和姿态估计任务。
- **预训练模型**：访问各种应用的高性能预训练模型，减少训练时间。
- **广泛的社区支持**：活跃的社区和全面的文档，用于故障排除和开发。
- **易于集成**：简单的 API，可与现有项目和工作流程集成。

在 [Ultralytics 模型](https://docs.ultralytics.com/models/)页面了解更多关于 YOLO 模型的信息。

### 如何使用 Ultralytics 工具优化和压缩数据集？

要使用 Ultralytics 工具优化和压缩数据集，请按照以下示例代码操作：

!!! example "优化和压缩数据集"

    === "Python"

        ```python
        from pathlib import Path

        from ultralytics.data.utils import compress_one_image
        from ultralytics.utils.downloads import zip_directory

        # 定义数据集目录
        path = Path("path/to/dataset")

        # 优化数据集中的图像（可选）
        for f in path.rglob("*.jpg"):
            compress_one_image(f)

        # 将数据集压缩到 'path/to/dataset.zip'
        zip_directory(path)
        ```

此过程有助于减小数据集大小，以实现更高效的存储和更快的下载速度。了解更多关于如何[优化和压缩数据集](#优化和压缩数据集的示例代码)的信息。
