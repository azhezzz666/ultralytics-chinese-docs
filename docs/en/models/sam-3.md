---
comments: true
description: 探索 SAM 3，Meta 的 Segment Anything Model 的下一代演进，引入可提示概念分割，使用文本和图像示例提示检测图像和视频中视觉概念的所有实例。
keywords: SAM 3, Segment Anything 3, SAM3, SAM-3, 视频分割, 图像分割, 概念分割, 可提示 AI, SA-Co 数据集, Meta, Ultralytics, 计算机视觉, AI, 机器学习, 开放词汇
---

# SAM 3: 使用概念进行 Segment Anything

!!! success "现已在 Ultralytics 中可用"

    SAM 3 已从 **8.3.237 版本**起完全集成到 Ultralytics 包中（[PR #22897](https://github.com/ultralytics/ultralytics/pull/22897)）。使用 `pip install -U ultralytics` 安装或升级，即可访问所有 SAM 3 功能，包括基于文本的概念分割、图像示例提示和视频跟踪。

![SAM 3 概述](https://github.com/ultralytics/docs/releases/download/0/sam-3-overview.webp)

**SAM 3**（Segment Anything Model 3）是 Meta 发布的用于**可提示概念分割 (PCS)** 的基础模型。在 [SAM 2](sam-2.md) 的基础上，SAM 3 引入了一项全新功能：检测、分割和跟踪由文本提示、图像示例或两者指定的视觉概念的**所有实例**。与之前仅对每个提示分割单个对象的 SAM 版本不同，SAM 3 可以找到并分割图像或视频中任何位置出现的概念的每个实例，与现代[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)中的开放词汇目标保持一致。

SAM 3 现已完全集成到 `ultralytics` 包中，提供对文本提示、图像示例提示和视频跟踪功能的概念分割的原生支持。

## 概述

SAM 3 在可提示概念分割方面实现了比现有系统 **2 倍的性能提升**，同时保持和改进了 SAM 2 的交互式[视觉分割](../tasks/segment.md)功能。该模型擅长开放词汇分割，允许用户使用简单的名词短语（例如"黄色校车"、"条纹猫"）或提供目标对象的示例图像来指定概念。这些功能补充了依赖简化的[预测](../modes/predict.md)和[跟踪](../modes/track.md)工作流的生产就绪管道。

![SAM 3 分割](https://github.com/ultralytics/docs/releases/download/0/sam-3-segmentation.webp)

### 什么是可提示概念分割 (PCS)？

PCS 任务接受**概念提示**作为输入，并返回具有唯一标识的**所有匹配对象实例**的分割掩码。概念提示可以是：

- **文本**：简单的名词短语，如"红苹果"或"戴帽子的人"，类似于[零样本学习](https://www.ultralytics.com/glossary/zero-shot-learning)
- **图像示例**：围绕示例对象的边界框（正面或负面），用于快速泛化
- **组合**：文本和图像示例一起使用，以实现精确控制

这与传统的视觉提示（点、框、掩码）不同，后者仅分割单个特定对象实例，如原始 [SAM 系列](../models/sam.md)所普及的那样。

### 关键性能指标

| 指标                           | SAM 3 成就                                               |
| ------------------------------ | -------------------------------------------------------- |
| **LVIS 零样本掩码 AP**         | **47.0**（对比之前最佳 38.5，提升 22%）                  |
| **SA-Co 基准**                 | 比现有系统**好 2 倍**                                    |
| **推理速度 (H200 GPU)**        | 每张图像 **30 ms**，检测 100+ 个对象                     |
| **视频性能**                   | 约 5 个并发对象接近实时                                  |
| **MOSEv2 VOS 基准**            | **60.1 J&F**（比 SAM 2.1 提升 25.5%，比之前 SOTA 提升 17%）|
| **交互式细化**                 | 3 个示例提示后 **+18.6 CGF1** 改进                       |
| **人类性能差距**               | 在 SA-Co/Gold 上达到估计下限的 **88%**                   |

有关生产中模型指标和权衡的背景，请参阅[模型评估洞察](../guides/model-evaluation-insights.md)和 [YOLO 性能指标](../guides/yolo-performance-metrics.md)。

## 架构

SAM 3 由共享感知编码器 (PE) 视觉骨干的**检测器**和**跟踪器**组成。这种解耦设计避免了任务冲突，同时支持图像级检测和视频级跟踪，其接口与 Ultralytics [Python 用法](../usage/python.md)和 [CLI 用法](../usage/cli.md)兼容。

### 核心组件

- **检测器**：用于图像级概念检测的[基于 DETR 的架构](rtdetr.md)
    - 用于名词短语提示的文本编码器
    - 用于基于图像提示的示例编码器
    - 用于根据提示调节图像特征的融合编码器
    - 新颖的**存在头**，将识别（"什么"）与定位（"哪里"）解耦
    - 用于生成实例分割掩码的掩码头

- **跟踪器**：继承自 [SAM 2](sam-2.md) 的基于记忆的视频分割
    - 提示编码器、掩码解码器、记忆编码器
    - 用于存储跨帧对象外观的记忆库
    - 在多对象设置中由[卡尔曼滤波器](../reference/trackers/utils/kalman_filter.md)等技术辅助的时间消歧

- **存在令牌**：一个学习的全局令牌，预测目标概念是否存在于图像/帧中，通过将识别与定位分离来改进检测。

![SAM 3 架构](https://github.com/ultralytics/docs/releases/download/0/sam-3-architecture.webp)

### 关键创新

1. **解耦识别和定位**：存在头全局预测概念存在，而提议查询仅专注于定位，避免冲突目标。
2. **统一概念和视觉提示**：在单一模型中支持 PCS（概念提示）和 PVS（如 SAM 2 的点击/框等视觉提示）。
3. **交互式示例细化**：用户可以添加正面或负面图像示例来迭代细化结果，模型泛化到相似对象而不仅仅是纠正单个实例。
4. **时间消歧**：使用 masklet 检测分数和定期重新提示来处理视频中的遮挡、拥挤场景和跟踪失败，与[实例分割和跟踪](../guides/instance-segmentation-and-tracking.md)最佳实践保持一致。


## SA-Co 数据集

SAM 3 在 **Segment Anything with Concepts (SA-Co)** 上训练，这是 Meta 迄今为止最大和最多样化的分割数据集，超越了 [COCO](../datasets/detect/coco.md) 和 [LVIS](../datasets/detect/lvis.md) 等常见基准。

### 训练数据

| 数据集组件      | 描述                                                     | 规模                                    |
| --------------- | -------------------------------------------------------- | --------------------------------------- |
| **SA-Co/HQ**    | 来自 4 阶段数据引擎的高质量人工标注图像数据              | 520 万张图像，400 万个唯一名词短语      |
| **SA-Co/SYN**   | 由 AI 标注的合成数据集，无人工参与                       | 3800 万个名词短语，14 亿个掩码          |
| **SA-Co/EXT**   | 15 个外部数据集，增加了困难负样本                        | 因来源而异                              |
| **SA-Co/VIDEO** | 带有时间跟踪的视频标注                                   | 52,500 个视频，24,800 个唯一名词短语    |

### 基准数据

**SA-Co 评估基准**包含 **126,000 张图像和视频**中的 **214,000 个唯一短语**，提供比现有基准**多 50 倍以上的概念**。它包括：

- **SA-Co/Gold**：7 个领域，三重标注用于测量人类性能边界
- **SA-Co/Silver**：10 个领域，单人标注
- **SA-Co/Bronze** 和 **SA-Co/Bio**：9 个现有数据集，适配用于概念分割
- **SA-Co/VEval**：视频基准，包含 3 个领域（SA-V、YT-Temporal-1B、SmartGlasses）

### 数据引擎创新

SAM 3 的可扩展人机协作数据引擎通过以下方式实现了 **2 倍的标注吞吐量**：

1. **AI 标注器**：基于 [Llama](https://arxiv.org/abs/2302.13971) 的模型提出多样化的名词短语，包括困难负样本
2. **AI 验证器**：微调的[多模态 LLM](https://ai.google.dev/gemini-api/docs) 以接近人类的性能验证掩码质量和完整性
3. **主动挖掘**：将人工精力集中在 AI 难以处理的挑战性失败案例上
4. **本体驱动**：利用基于 [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) 的大型本体进行概念覆盖

## 安装

SAM 3 在 Ultralytics **8.3.237 版本**及更高版本中可用。使用以下命令安装或升级：

```bash
pip install -U ultralytics
```

!!! warning "需要 SAM 3 模型权重"

    与其他 Ultralytics 模型不同，SAM 3 权重（`sam3.pt`）**不会自动下载**。您必须首先在 [Hugging Face 上的 SAM 3 模型页面](https://huggingface.co/facebook/sam3)请求访问模型权重，然后在获得批准后下载 [`sam3.pt` 文件](https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true)。将下载的 `sam3.pt` 文件放在您的工作目录中，或在加载模型时指定完整路径。

## 如何使用 SAM 3：概念分割的多功能性

SAM 3 通过不同的预测器接口支持可提示概念分割 (PCS) 和可提示视觉分割 (PVS) 任务：

### 支持的任务和模型

| 任务类型                       | 提示类型                                   | 输出                                        |
| ------------------------------ | ------------------------------------------ | ------------------------------------------- |
| **概念分割 (PCS)**             | 文本（名词短语）、图像示例                 | 匹配概念的所有实例                          |
| **视觉分割 (PVS)**             | 点、框、掩码                               | 单个对象实例（SAM 2 风格）                  |
| **交互式细化**                 | 迭代添加/删除示例或点击                    | 精度提高的细化分割                          |

### 概念分割示例

#### 使用文本提示进行分割

!!! example "基于文本的概念分割"

    使用文本描述查找并分割概念的所有实例。文本提示需要 `SAM3SemanticPredictor` 接口。

    === "Python"

        ```python
        from ultralytics.models.sam import SAM3SemanticPredictor

        # 使用配置初始化预测器
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model="sam3.pt",
            half=True,  # 使用 FP16 加速推理
            save=True,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)

        # 设置一次图像用于多次查询
        predictor.set_image("path/to/image.jpg")

        # 使用多个文本提示查询
        results = predictor(text=["person", "bus", "glasses"])

        # 适用于描述性短语
        results = predictor(text=["person with red cloth", "person with blue cloth"])

        # 使用单个概念查询
        results = predictor(text=["a person"])
        ```

#### 使用图像示例进行分割

!!! example "基于图像示例的分割"

    使用边界框作为视觉提示来查找所有相似实例。这也需要 `SAM3SemanticPredictor` 进行基于概念的匹配。

    === "Python"

        ```python
        from ultralytics.models.sam import SAM3SemanticPredictor

        # 初始化预测器
        overrides = dict(conf=0.25, task="segment", mode="predict", model="sam3.pt", half=True, save=True)
        predictor = SAM3SemanticPredictor(overrides=overrides)

        # 设置图像
        predictor.set_image("path/to/image.jpg")

        # 提供边界框示例以分割相似对象
        results = predictor(bboxes=[[480.0, 290.0, 590.0, 650.0]])

        # 多个边界框用于不同概念
        results = predictor(bboxes=[[539, 599, 589, 639], [343, 267, 499, 662]])
        ```

#### 基于特征的推理以提高效率

!!! example "重用图像特征进行多次查询"

    提取一次图像特征并重用于多个分割查询以提高效率。

    === "Python"

        ```python
        import cv2

        from ultralytics.models.sam import SAM3SemanticPredictor
        from ultralytics.utils.plotting import Annotator, colors

        # 初始化预测器
        overrides = dict(conf=0.50, task="segment", mode="predict", model="sam3.pt", verbose=False)
        predictor = SAM3SemanticPredictor(overrides=overrides)
        predictor2 = SAM3SemanticPredictor(overrides=overrides)

        # 从第一个预测器提取特征
        source = "path/to/image.jpg"
        predictor.set_image(source)
        src_shape = cv2.imread(source).shape[:2]

        # 设置第二个预测器并重用特征
        predictor2.setup_model()

        # 使用共享特征和文本提示执行推理
        masks, boxes = predictor2.inference_features(predictor.features, src_shape=src_shape, text=["person"])

        # 使用共享特征和边界框提示执行推理
        masks, boxes = predictor2.inference_features(predictor.features, src_shape=src_shape, bboxes=[[439, 437, 524, 709]])

        # 可视化结果
        if masks is not None:
            masks, boxes = masks.cpu().numpy(), boxes.cpu().numpy()
            im = cv2.imread(source)
            annotator = Annotator(im, pil=False)
            annotator.masks(masks, [colors(x, True) for x in range(len(masks))])

            cv2.imshow("result", annotator.result())
            cv2.waitKey(0)
        ```


### 视频概念分割

#### 使用边界框跨视频跟踪概念

!!! example "使用视觉提示进行视频跟踪"

    使用边界框提示跨视频帧检测和跟踪对象实例。

    === "Python"

        ```python
        from ultralytics.models.sam import SAM3VideoPredictor

        # 创建视频预测器
        overrides = dict(conf=0.25, task="segment", mode="predict", model="sam3.pt", half=True)
        predictor = SAM3VideoPredictor(overrides=overrides)

        # 使用边界框提示跟踪对象
        results = predictor(source="path/to/video.mp4", bboxes=[[706.5, 442.5, 905.25, 555], [598, 635, 725, 750]], stream=True)

        # 处理和显示结果
        for r in results:
            r.show()  # 显示带有分割掩码的帧
        ```

#### 使用文本提示跟踪概念

!!! example "使用语义查询进行视频跟踪"

    跨视频帧跟踪由文本指定的概念的所有实例。

    === "Python"

        ```python
        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        # 初始化语义视频预测器
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=640, model="sam3.pt", half=True, save=True)
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)

        # 使用文本提示跟踪概念
        results = predictor(source="path/to/video.mp4", text=["person", "bicycle"], stream=True)

        # 处理结果
        for r in results:
            r.show()  # 显示带有跟踪对象的帧

        # 替代方案：使用边界框提示跟踪
        results = predictor(
            source="path/to/video.mp4",
            bboxes=[[864, 383, 975, 620], [705, 229, 782, 402]],
            labels=[1, 1],  # 正标签
            stream=True,
        )
        ```

### 视觉提示（SAM 2 兼容性）

SAM 3 保持与 SAM 2 视觉提示的完全向后兼容性，用于单对象分割：

!!! example "SAM 2 风格的视觉提示"

    基本的 `SAM` 接口行为与 SAM 2 完全相同，仅分割由视觉提示（点、框或掩码）指示的特定区域。

    === "Python"

        ```python
        from ultralytics import SAM

        model = SAM("sam3.pt")

        # 单点提示 - 分割特定位置的对象
        results = model.predict(source="path/to/image.jpg", points=[900, 370], labels=[1])
        results[0].show()

        # 多点 - 使用多个点提示分割单个对象
        results = model.predict(source="path/to/image.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

        # 框提示 - 分割边界框内的对象
        results = model.predict(source="path/to/image.jpg", bboxes=[100, 150, 300, 400])
        results[0].show()
        ```

    !!! warning "视觉提示与概念分割"

        使用带有视觉提示（点/框/掩码）的 `SAM("sam3.pt")` 将**仅分割该位置的特定对象**，就像 SAM 2 一样。要分割**概念的所有实例**，请使用带有文本或示例提示的 `SAM3SemanticPredictor`，如上所示。

## 性能基准

### 图像分割

SAM 3 在多个基准测试中取得了最先进的结果，包括 [LVIS](../datasets/detect/lvis.md) 和 [COCO 分割](../datasets/segment/coco.md)等真实世界数据集：

| 基准                          | 指标    | SAM 3    | 之前最佳      | 改进        |
| ----------------------------- | ------- | -------- | ------------- | ----------- |
| **LVIS（零样本）**            | Mask AP | **47.0** | 38.5          | +22.1%      |
| **SA-Co/Gold**                | CGF1    | **65.0** | 34.3 (OWLv2)  | +89.5%      |
| **COCO（零样本）**            | Box AP  | **53.5** | 52.2 (T-Rex2) | +2.5%       |
| **ADE-847（语义分割）**       | mIoU    | **14.7** | 9.2 (APE-D)   | +59.8%      |
| **PascalConcept-59**          | mIoU    | **59.4** | 58.5 (APE-D)  | +1.5%       |
| **Cityscapes（语义分割）**    | mIoU    | **65.1** | 44.2 (APE-D)  | +47.3%      |

在 [Ultralytics 数据集](../datasets/index.md)中探索快速实验的数据集选项。

### 视频分割性能

SAM 3 在 [DAVIS 2017](https://davischallenge.org/) 和 [YouTube-VOS](https://youtube-vos.org/) 等视频基准测试中相对于 SAM 2 和之前的最先进技术显示出显著改进：

| 基准           | 指标   | SAM 3    | SAM 2.1 L | 改进    |
| -------------- | ------ | -------- | --------- | ------- |
| **MOSEv2**     | J&F    | **60.1** | 47.9      | +25.5%  |
| **DAVIS 2017** | J&F    | **92.0** | 90.7      | +1.4%   |
| **LVOSv2**     | J&F    | **88.2** | 79.6      | +10.8%  |
| **SA-V**       | J&F    | **84.6** | 78.4      | +7.9%   |
| **YTVOS19**    | J&F    | **89.6** | 89.3      | +0.3%   |

### 少样本适应

SAM 3 擅长以最少的示例适应新领域，与[数据中心 AI](https://www.ultralytics.com/glossary/data-centric-ai) 工作流相关：

| 基准         | 0-shot AP | 10-shot AP | 之前最佳 (10-shot)    |
| ------------ | --------- | ---------- | --------------------- |
| **ODinW13**  | 59.9      | **71.6**   | 67.9 (gDino1.5-Pro)   |
| **RF100-VL** | 14.3      | **35.7**   | 33.7 (gDino-T)        |

### 交互式细化效果

SAM 3 基于概念的示例提示比视觉提示收敛更快：

| 添加的提示   | CGF1 分数  | 相对纯文本增益 | 相对 PVS 基线增益 |
| ------------ | ---------- | -------------- | ----------------- |
| 仅文本       | 46.4       | 基线           | 基线              |
| +1 个示例    | 57.6       | +11.2          | +6.7              |
| +2 个示例    | 62.2       | +15.8          | +9.7              |
| +3 个示例    | **65.0**   | **+18.6**      | **+11.2**         |
| +4 个示例    | 65.7       | +19.3          | +11.5（平台期）   |

### 对象计数精度

SAM 3 通过分割所有实例提供准确的计数，这是[对象计数](../guides/object-counting.md)中的常见需求：

| 基准            | 精度      | MAE  | 对比最佳 MLLM      |
| --------------- | --------- | ---- | ------------------ |
| **CountBench**  | **95.6%** | 0.11 | 92.4% (Gemini 2.5) |
| **PixMo-Count** | **87.3%** | 0.22 | 88.8% (Molmo-72B)  |


## SAM 3 与 SAM 2 与 YOLO 的比较

这里我们比较 SAM 3 与 SAM 2 和 [YOLO11](../models/yolo11.md) 模型的功能：

| 功能                         | SAM 3                                 | SAM 2                | YOLO11n-seg        |
| ---------------------------- | ------------------------------------- | -------------------- | ------------------ |
| **概念分割**                 | ✅ 从文本/示例获取所有实例            | ❌ 不支持            | ❌ 不支持          |
| **视觉分割**                 | ✅ 单实例（SAM 2 兼容）               | ✅ 单实例            | ✅ 所有实例        |
| **零样本能力**               | ✅ 开放词汇                           | ✅ 几何提示          | ❌ 封闭集          |
| **交互式细化**               | ✅ 示例 + 点击                        | ✅ 仅点击            | ❌ 不支持          |
| **视频跟踪**                 | ✅ 带标识的多对象                     | ✅ 多对象            | ✅ 多对象          |
| **LVIS Mask AP（零样本）**   | **47.0**                              | N/A                  | N/A                |
| **MOSEv2 J&F**               | **60.1**                              | 47.9                 | N/A                |
| **推理速度 (H200)**          | **30 ms**（100+ 对象）                | ~23 ms（每对象）     | **2-3 ms**（图像） |
| **模型大小**                 | 3.4GB                                 | 162 MB（base）       | **5.9 MB**         |

**关键要点**：

- **SAM 3**：最适合开放词汇概念分割，使用文本或示例提示查找概念的所有实例
- **SAM 2**：最适合使用几何提示在图像和视频中进行交互式单对象分割
- **YOLO11**：最适合在资源受限的部署中进行高速、实时分割，使用高效的[导出管道](../modes/export.md)如 [ONNX](../integrations/onnx.md) 和 [TensorRT](../integrations/tensorrt.md)

## 评估指标

SAM 3 引入了为 PCS 任务设计的新指标，补充了熟悉的度量如 [F1 分数](https://www.ultralytics.com/glossary/f1-score)、[精确率](https://www.ultralytics.com/glossary/precision)和[召回率](https://www.ultralytics.com/glossary/recall)。

### 分类门控 F1 (CGF1)

结合定位和分类的主要指标：

**CGF1 = 100 × pmF1 × IL_MCC**

其中：

- **pmF1**（正样本宏 F1）：测量正样本的定位质量
- **IL_MCC**（图像级马修斯相关系数）：测量二元分类精度（"概念是否存在？"）

### 为什么使用这些指标？

传统的 AP 指标不考虑校准，使模型在实践中难以使用。通过仅评估置信度高于 0.5 的预测，SAM 3 的指标强制执行良好的校准，并模拟交互式[预测](../modes/predict.md)和[跟踪](../modes/track.md)循环中的真实世界使用模式。

## 关键消融和洞察

### 存在头的影响

存在头将识别与定位解耦，提供显著改进：

| 配置           | CGF1     | IL_MCC   | pmF1     |
| -------------- | -------- | -------- | -------- |
| 无存在头       | 57.6     | 0.77     | 74.7     |
| **有存在头**   | **63.3** | **0.82** | **77.1** |

存在头提供 **+5.7 CGF1 提升**（+9.9%），主要改进识别能力（IL_MCC +6.5%）。

### 困难负样本的效果

| 每张图像困难负样本数 | CGF1     | IL_MCC   | pmF1     |
| -------------------- | -------- | -------- | -------- |
| 0                    | 31.8     | 0.44     | 70.2     |
| 5                    | 44.8     | 0.62     | 71.9     |
| **30**               | **49.2** | **0.68** | **72.3** |

困难负样本对开放词汇识别至关重要，将 IL_MCC 提高了 **54.5%**（0.44 → 0.68）。

### 训练数据扩展

| 数据来源             | CGF1     | IL_MCC   | pmF1     |
| -------------------- | -------- | -------- | -------- |
| 仅外部               | 30.9     | 0.46     | 66.3     |
| 外部 + 合成          | 39.7     | 0.57     | 70.6     |
| 外部 + HQ            | 51.8     | 0.71     | 73.2     |
| **全部三者**         | **54.3** | **0.74** | **73.5** |

高质量人工标注相比仅合成或外部数据提供了巨大收益。有关数据质量实践的背景，请参阅[数据收集和标注](../guides/data-collection-and-annotation.md)。

## 应用

SAM 3 的概念分割功能支持新的用例：

- **内容审核**：在媒体库中查找特定内容类型的所有实例
- **电子商务**：在目录图像中分割某类产品的所有实例，支持[自动标注](../guides/preprocessing_annotated_data.md)
- **医学成像**：识别特定组织类型或异常的所有出现
- **自主系统**：按类别跟踪交通标志、行人或车辆的所有实例
- **视频分析**：计数和跟踪穿着特定服装或执行动作的所有人
- **数据集标注**：快速标注稀有对象类别的所有实例
- **科学研究**：量化和分析匹配特定标准的所有样本

## SAM 3 Agent：扩展语言推理

SAM 3 可以与多模态大语言模型 (MLLM) 结合处理需要推理的复杂查询，类似于 [OWLv2](https://arxiv.org/abs/2306.09683) 和 [T-Rex](https://arxiv.org/abs/2401.03533) 等开放词汇系统。

### 推理任务性能

| 基准                       | 指标   | SAM 3 Agent (Gemini 2.5 Pro) | 之前最佳      |
| -------------------------- | ------ | ---------------------------- | ------------- |
| **ReasonSeg（验证）**      | gIoU   | **76.0**                     | 65.0 (SoTA)   |
| **ReasonSeg（测试）**      | gIoU   | **73.8**                     | 61.3 (SoTA)   |
| **OmniLabel（验证）**      | AP     | **46.7**                     | 36.5 (REAL)   |
| **RefCOCO+**               | Acc    | **91.2**                     | 89.3 (LISA)   |

### 复杂查询示例

SAM 3 Agent 可以处理需要推理的查询：

- "坐着但手中没有拿礼盒的人"
- "离相机最近且没有戴项圈的狗"
- "比人手大的红色物体"

MLLM 向 SAM 3 提出简单的名词短语查询，分析返回的掩码，并迭代直到满意。

## 局限性

虽然 SAM 3 代表了重大进步，但它有一些局限性：

- **短语复杂性**：最适合简单的名词短语；长的指代表达或复杂推理可能需要 MLLM 集成
- **歧义处理**：某些概念本质上是模糊的（例如"小窗户"、"舒适的房间"）
- **计算需求**：比 [YOLO](../models/yolo26.md) 等专用检测模型更大更慢
- **词汇范围**：专注于原子视觉概念；没有 MLLM 辅助，组合推理受限
- **稀有概念**：在训练数据中未充分表示的极稀有或细粒度概念上性能可能下降

## 引用

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{sam3_2025,
          title     = {SAM 3: Segment Anything with Concepts},
          author    = {Anonymous authors},
          booktitle = {Submitted to ICLR 2026},
          year      = {2025},
          url       = {https://openreview.net/forum?id=r35clVtGzw},
          note      = {Paper ID: 4183, under double-blind review}
        }
        ```

---


## 常见问题

### SAM 3 何时发布？

SAM 3 由 Meta 于 **2025 年 11 月 20 日**发布，并从 **8.3.237 版本**起完全集成到 Ultralytics 中（[PR #22897](https://github.com/ultralytics/ultralytics/pull/22897)）。完全支持[预测模式](../modes/predict.md)和[跟踪模式](../modes/track.md)。

### SAM 3 是否集成到 Ultralytics 中？

是的！SAM 3 已完全集成到 Ultralytics Python 包中，包括概念分割、SAM 2 风格的视觉提示和多对象视频跟踪。您可以[导出](../modes/export.md)到 [ONNX](../integrations/onnx.md) 和 [TensorRT](../integrations/tensorrt.md) 等格式进行部署，具有简化的 [Python](../usage/python.md) 和 [CLI](../usage/cli.md) 工作流。

### 什么是可提示概念分割 (PCS)？

PCS 是 SAM 3 引入的新任务，分割图像或视频中视觉概念的**所有实例**。与针对特定对象实例的传统分割不同，PCS 查找类别的每个出现。例如：

- **文本提示**："黄色校车" → 分割场景中所有黄色校车
- **图像示例**：围绕一只狗的框 → 分割图像中所有狗
- **组合**："条纹猫" + 示例框 → 分割所有匹配示例的条纹猫

请参阅[对象检测](https://www.ultralytics.com/glossary/object-detection)和[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)的相关背景。

### SAM 3 与 SAM 2 有何不同？

| 特性                     | SAM 2                         | SAM 3                                 |
| ------------------------ | ----------------------------- | ------------------------------------- |
| **任务**                 | 每个提示单个对象              | 概念的所有实例                        |
| **提示类型**             | 点、框、掩码                  | + 文本短语、图像示例                  |
| **检测能力**             | 需要外部检测器                | 内置开放词汇检测器                    |
| **识别**                 | 仅基于几何                    | 文本和视觉识别                        |
| **架构**                 | 仅跟踪器                      | 检测器 + 带存在头的跟踪器             |
| **零样本性能**           | N/A（需要视觉提示）           | LVIS 上 47.0 AP，SA-Co 上好 2 倍      |
| **交互式细化**           | 仅点击                        | 点击 + 示例泛化                       |

SAM 3 保持与 [SAM 2](sam-2.md) 视觉提示的向后兼容性，同时添加基于概念的功能。

### SAM 3 使用哪些数据集进行训练？

SAM 3 在 **Segment Anything with Concepts (SA-Co)** 数据集上训练：

**训练数据**：

- **520 万张图像**，**400 万个唯一名词短语**（SA-Co/HQ）- 高质量人工标注
- **52,500 个视频**，**24,800 个唯一名词短语**（SA-Co/VIDEO）
- **14 亿个合成掩码**，跨 **3800 万个名词短语**（SA-Co/SYN）
- **15 个外部数据集**，增加了困难负样本（SA-Co/EXT）

**基准数据**：

- **214,000 个唯一概念**，跨 **126,000 张图像/视频**
- 比现有基准**多 50 倍以上的概念**（例如 LVIS 有约 4,000 个概念）
- SA-Co/Gold 上的三重标注用于测量人类性能边界

这种大规模和多样性使 SAM 3 在开放词汇概念上具有卓越的零样本泛化能力。

### SAM 3 与 YOLO11 在分割方面如何比较？

SAM 3 和 YOLO11 服务于不同的用例：

**SAM 3 优势**：

- **开放词汇**：通过文本提示分割任何概念，无需训练
- **零样本**：立即适用于新类别
- **交互式**：基于示例的细化泛化到相似对象
- **基于概念**：自动查找类别的所有实例
- **精度**：LVIS 零样本实例分割上 47.0 AP

**YOLO11 优势**：

- **速度**：推理速度快 10-15 倍（每张图像 2-3ms 对比 30ms）
- **效率**：模型小 576 倍（5.9MB 对比 3.4GB）
- **资源友好**：在边缘设备和移动端运行
- **实时**：针对生产部署优化

**建议**：

- 使用 **SAM 3** 进行灵活的开放词汇分割，需要通过文本或示例描述查找概念的所有实例
- 使用 **YOLO11** 进行高速生产部署，类别预先已知
- 使用 **SAM 2** 使用几何提示进行交互式单对象分割

### SAM 3 能处理复杂的语言查询吗？

SAM 3 设计用于简单的名词短语（例如"红苹果"、"戴帽子的人"）。对于需要推理的复杂查询，将 SAM 3 与 MLLM 结合作为 **SAM 3 Agent**：

**简单查询（原生 SAM 3）**：

- "黄色校车"
- "条纹猫"
- "戴红帽子的人"

**复杂查询（SAM 3 Agent 与 MLLM）**：

- "坐着但没有拿礼盒的人"
- "离相机最近且没有项圈的狗"
- "比人手大的红色物体"

SAM 3 Agent 在 ReasonSeg 验证上达到 **76.0 gIoU**（对比之前最佳 65.0，提升 16.9%），通过结合 SAM 3 的分割与 MLLM 推理能力。

### SAM 3 与人类性能相比有多准确？

在具有三重人工标注的 SA-Co/Gold 基准上：

- **人类下限**：74.2 CGF1（最保守的标注者）
- **SAM 3 性能**：65.0 CGF1
- **达成**：估计人类下限的 **88%**
- **人类上限**：81.4 CGF1（最宽松的标注者）

SAM 3 在开放词汇概念分割上取得了接近人类水平的强劲性能，差距主要在模糊或主观概念上（例如"小窗户"、"舒适的房间"）。
