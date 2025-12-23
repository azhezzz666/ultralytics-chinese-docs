---
comments: true
description: 学习 Ultralytics YOLO 中的基本数据增强技术。探索各种变换、它们的影响以及如何有效实现它们以提高模型性能。
keywords: YOLO 数据增强, 计算机视觉, 深度学习, 图像变换, 模型训练, Ultralytics YOLO, HSV 调整, 几何变换, 马赛克增强
---

# 使用 Ultralytics YOLO 进行数据增强

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/albumentations-augmentation.avif" alt="图像增强示例">
</p>

## 简介

[数据增强](https://www.ultralytics.com/glossary/data-augmentation)是计算机视觉中的一项关键技术，通过对现有图像应用各种变换来人工扩展训练数据集。在训练像 Ultralytics YOLO 这样的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型时，数据增强有助于提高模型鲁棒性、减少过拟合并增强对真实场景的泛化能力。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/e-TwqFtay90"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Mosaic、MixUp 等数据增强帮助 Ultralytics YOLO 模型更好地泛化 🚀
</p>

### 为什么数据增强很重要

数据增强在训练计算机视觉模型中有多个关键作用：

- **扩展数据集**：通过创建现有图像的变体，您可以有效地增加训练数据集大小而无需收集新数据。
- **改善泛化**：模型学会在各种条件下识别对象，使其在实际应用中更加鲁棒。
- **减少过拟合**：通过在训练数据中引入变化，模型不太可能记住特定的图像特征。
- **增强性能**：使用适当增强训练的模型通常在验证和测试集上获得更好的[准确率](https://www.ultralytics.com/glossary/accuracy)。

Ultralytics YOLO 的实现提供了一套全面的增强技术，每种技术都有特定的用途，并以不同的方式对模型性能做出贡献。本指南将详细探讨每个增强参数，帮助您了解何时以及如何在项目中有效使用它们。

### 配置示例

您可以使用 Python API、命令行界面 (CLI) 或配置文件自定义每个参数。以下是如何在每种方法中设置数据增强的示例。

!!! example "配置示例"

    === "Python"

        ```python
        import albumentations as A

        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")

        # 使用自定义增强参数训练
        model.train(data="coco.yaml", epochs=100, hsv_h=0.03, hsv_s=0.6, hsv_v=0.5)

        # 不使用任何增强训练（为清晰起见省略禁用值）
        model.train(
            data="coco.yaml",
            epochs=100,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.0,
            mosaic=0.0,
            erasing=0.0,
            auto_augment=None,
        )

        # 使用自定义 Albumentations 变换训练（仅 Python API）
        custom_transforms = [
            A.Blur(blur_limit=7, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
        ]
        model.train(data="coco.yaml", epochs=100, augmentations=custom_transforms)
        ```

    === "CLI"

        ```bash
        # 使用自定义增强参数训练
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 hsv_h=0.03 hsv_s=0.6 hsv_v=0.5
        ```

#### 使用配置文件

您可以在 YAML 配置文件（例如 `train_custom.yaml`）中定义所有训练参数，包括增强。`mode` 参数仅在使用 CLI 时需要。这个新的 YAML 文件将覆盖位于 `ultralytics` 包中的[默认配置](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)。

```yaml
# train_custom.yaml
# 'mode' 仅在 CLI 使用时需要
mode: train
data: coco8.yaml
model: yolo11n.pt
epochs: 100
hsv_h: 0.03
hsv_s: 0.6
hsv_v: 0.5
```

然后使用 Python API 启动训练：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLO11n 模型
        model = YOLO("yolo11n.pt")

        # 使用自定义配置训练模型
        model.train(cfg="train_custom.yaml")
        ```

    === "CLI"

        ```bash
        # 使用自定义配置训练模型
        yolo detect train model="yolo11n.pt" cfg=train_custom.yaml
        ```

## 色彩空间增强

### 色相调整 (`hsv_h`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ hsv_h }}`
- **用法**：在保持颜色关系的同时移动图像颜色。`hsv_h` 超参数定义移动幅度，最终调整在 `-hsv_h` 和 `hsv_h` 之间随机选择。例如，`hsv_h=0.3` 时，移动在 `-0.3` 到 `0.3` 范围内随机选择。对于大于 `0.5` 的值，色相移动会环绕色轮，这就是为什么 `0.5` 和 `-0.5` 之间的增强看起来相同。
- **目的**：特别适用于光照条件可能显著影响对象外观的户外场景。例如，香蕉在明亮阳光下可能看起来更黄，但在室内可能更偏绿。
- **Ultralytics 实现**：[RandomHSV](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomHSV)

|                                                           **`-0.5`**                                                            |                                                            **`-0.25`**                                                            |                                                          **`0.0`**                                                          |                                                           **`0.25`**                                                            |                                                           **`0.5`**                                                            |
| :-----------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_h_-0.5.avif" alt="hsv_h_-0.5_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_h_-0.25.avif" alt="hsv_h_-0.25_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_h_0.25.avif" alt="hsv_h_0.25_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_h_0.5.avif" alt="hsv_h_-0.5_augmentation"/> |

### 饱和度调整 (`hsv_s`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ hsv_s }}`
- **用法**：修改图像中颜色的强度。`hsv_s` 超参数定义移动幅度，最终调整在 `-hsv_s` 和 `hsv_s` 之间随机选择。例如，`hsv_s=0.7` 时，强度在 `-0.7` 到 `0.7` 范围内随机选择。
- **目的**：帮助模型处理不同的天气条件和相机设置。例如，红色交通标志在晴天可能看起来非常鲜艳，但在雾天可能看起来暗淡褪色。
- **Ultralytics 实现**：[RandomHSV](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomHSV)

|                                                         **`-1.0`**                                                          |                                                           **`-0.5`**                                                            |                                                          **`0.0`**                                                          |                                                           **`0.5`**                                                           |                                                         **`1.0`**                                                         |
| :-------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_s_-1.avif" alt="hsv_s_-1_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_s_-0.5.avif" alt="hsv_s_-0.5_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_s_0.5.avif" alt="hsv_s_0.5_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_s_1.avif" alt="hsv_s_1_augmentation"/> |

### 亮度调整 (`hsv_v`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ hsv_v }}`
- **用法**：改变图像的亮度。`hsv_v` 超参数定义移动幅度，最终调整在 `-hsv_v` 和 `hsv_v` 之间随机选择。例如，`hsv_v=0.4` 时，强度在 `-0.4` 到 `0.4` 范围内随机选择。
- **目的**：对于需要在不同光照条件下执行的模型训练至关重要。例如，红苹果在阳光下可能看起来很亮，但在阴影中会暗得多。
- **Ultralytics 实现**：[RandomHSV](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomHSV)

|                                                         **`-1.0`**                                                          |                                                           **`-0.5`**                                                            |                                                          **`0.0`**                                                          |                                                           **`0.5`**                                                           |                                                         **`1.0`**                                                         |
| :-------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_v_-1.avif" alt="hsv_v_-1_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_v_-0.5.avif" alt="hsv_v_-0.5_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_v_0.5.avif" alt="hsv_v_0.5_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_hsv_v_1.avif" alt="hsv_v_1_augmentation"/> |


## 几何变换

### 旋转 (`degrees`)

- **范围**：`0.0` 到 `180`
- **默认值**：`{{ degrees }}`
- **用法**：在指定范围内随机旋转图像。`degrees` 超参数定义旋转角度，最终调整在 `-degrees` 和 `degrees` 之间随机选择。例如，`degrees=10.0` 时，旋转在 `-10.0` 到 `10.0` 范围内随机选择。
- **目的**：对于对象可能以不同方向出现的应用至关重要。例如，在航拍无人机图像中，车辆可以朝向任何方向，需要模型无论旋转如何都能识别对象。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomPerspective)

|                                                                  **`-180`**                                                                   |                                                                  **`-90`**                                                                  |                                                          **`0.0`**                                                          |                                                                 **`90`**                                                                  |                                                                  **`180`**                                                                  |
| :-------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_degrees_-180.avif" alt="degrees_-180_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_degrees_-90.avif" alt="degrees_-90_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_degrees_90.avif" alt="degrees_90_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_degrees_180.avif" alt="degrees_180_augmentation"/> |

### 平移 (`translate`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ translate }}`
- **用法**：按图像大小的随机比例水平和垂直移动图像。`translate` 超参数定义移动幅度，最终调整在 `-translate` 和 `translate` 范围内随机选择两次（每个轴一次）。例如，`translate=0.5` 时，x 轴上的平移在 `-0.5` 到 `0.5` 范围内随机选择，y 轴上另一个独立的随机值在相同范围内选择。
- **目的**：帮助模型学习检测部分可见的对象并提高对对象位置的鲁棒性。例如，在车辆损伤评估应用中，汽车部件可能根据摄影师的位置和距离完全或部分出现在画面中，平移增强将教会模型无论其完整性或位置如何都能识别这些特征。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomPerspective)
- **注意**：为简单起见，下面应用的平移每次在 `x` 和 `y` 轴上都相同。值 `-1.0` 和 `1.0` 未显示，因为它们会将图像完全移出画面。

|                                                                      `-0.5`                                                                       |                                                                     **`-0.25`**                                                                     |                                                          **`0.0`**                                                          |                                                                    **`0.25`**                                                                     |                                                                    **`0.5`**                                                                    |
| :-----------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_translate_-0.5.avif" alt="translate_-0.5_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_translate_-0.25.avif" alt="translate_-0.25_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_translate_0.25.avif" alt="translate_0.25_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_translate_0.5.avif" alt="translate_0.5_augmentation"/> |

### 缩放 (`scale`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ scale }}`
- **用法**：在指定范围内按随机因子调整图像大小。`scale` 超参数定义缩放因子，最终调整在 `1-scale` 和 `1+scale` 之间随机选择。例如，`scale=0.5` 时，缩放在 `0.5` 到 `1.5` 范围内随机选择。
- **目的**：使模型能够处理不同距离和大小的对象。例如，在自动驾驶应用中，车辆可能出现在距离相机不同距离的位置，需要模型无论大小如何都能识别它们。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomPerspective)
- **注意**：
    - 值 `-1.0` 未显示，因为它会使图像消失，而 `1.0` 只是产生 2 倍缩放。
    - 下表中显示的值是通过超参数 `scale` 应用的值，而不是最终缩放因子。
    - 如果 `scale` 大于 `1.0`，图像可能会非常小或翻转，因为缩放因子在 `1-scale` 和 `1+scale` 之间随机选择。例如，`scale=3.0` 时，缩放在 `-2.0` 到 `4.0` 范围内随机选择。如果选择负值，图像会翻转。

|                                                                **`-0.5`**                                                                 |                                                                 **`-0.25`**                                                                 |                                                          **`0.0`**                                                          |                                                                **`0.25`**                                                                 |                                                                **`0.5`**                                                                |
| :---------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_scale_-0.5.avif" alt="scale_-0.5_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_scale_-0.25.avif" alt="scale_-0.25_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_scale_0.25.avif" alt="scale_0.25_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_scale_0.5.avif" alt="scale_0.5_augmentation"/> |

### 剪切 (`shear`)

- **范围**：`-180` 到 `+180`
- **默认值**：`{{ shear }}`
- **用法**：引入沿 x 轴和 y 轴倾斜图像的几何变换，有效地将图像的部分向一个方向移动，同时保持平行线。`shear` 超参数定义剪切角度，最终调整在 `-shear` 和 `shear` 之间随机选择。例如，`shear=10.0` 时，x 轴上的剪切在 `-10` 到 `10` 范围内随机选择，y 轴上另一个独立的随机值在相同范围内选择。
- **目的**：帮助模型泛化到由轻微倾斜或斜视角引起的视角变化。例如，在交通监控中，由于非垂直的相机放置，汽车和路标等对象可能看起来倾斜。应用剪切增强确保模型学会识别对象，尽管存在这种倾斜变形。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomPerspective)
- **注意**：
    - `shear` 值可能会快速扭曲图像，因此建议从小值开始并逐渐增加。
    - 与透视变换不同，剪切不会引入深度或消失点，而是通过改变角度来扭曲对象的形状，同时保持对边平行。

|                                                                **`-10`**                                                                |                                                               **`-5`**                                                                |                                                          **`0.0`**                                                          |                                                               **`5`**                                                               |                                                               **`10`**                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_shear_-10.avif" alt="shear_-10_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_shear_-5.avif" alt="shear_-5_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_shear_5.avif" alt="shear_5_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_shear_10.avif" alt="shear_10_augmentation"/> |

### 透视 (`perspective`)

- **范围**：`0.0` - `0.001`
- **默认值**：`{{ perspective }}`
- **用法**：沿 x 轴和 y 轴应用完整的透视变换，模拟从不同深度或角度观看对象时的外观。`perspective` 超参数定义透视幅度，最终调整在 `-perspective` 和 `perspective` 之间随机选择。例如，`perspective=0.001` 时，x 轴上的透视在 `-0.001` 到 `0.001` 范围内随机选择，y 轴上另一个独立的随机值在相同范围内选择。
- **目的**：透视增强对于处理极端视角变化至关重要，特别是在由于透视变化而导致对象出现缩短或扭曲的场景中。例如，在基于无人机的对象检测中，建筑物、道路和车辆可能根据无人机的倾斜和高度而出现拉伸或压缩。通过应用透视变换，模型学会识别对象，尽管存在这些透视引起的变形，从而提高其在实际部署中的鲁棒性。
- **Ultralytics 实现**：[RandomPerspective](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomPerspective)

|                                                                       **`-0.001`**                                                                        |                                                                        **`-0.0005`**                                                                        |                                                          **`0.0`**                                                          |                                                                       **`0.0005`**                                                                        |                                                                       **`0.001`**                                                                       |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_perspective_-0.001.avif" alt="perspective_-0.001_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_perspective_-0.0005.avif" alt="perspective_-0.0005_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_perspective_0.0005.avif" alt="perspective_0.0005_augmentation"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_geometric_perspective_0.001.avif" alt="perspective_0.001_augmentation"/> |

### 上下翻转 (`flipud`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ flipud }}`
- **用法**：通过沿 y 轴反转图像执行垂直翻转。此变换将整个图像上下颠倒，但保留对象之间的所有空间关系。flipud 超参数定义应用变换的概率，`flipud=1.0` 确保所有图像都被翻转，`flipud=0.0` 完全禁用变换。例如，`flipud=0.5` 时，每张图像有 50% 的概率被上下翻转。
- **目的**：适用于对象可能上下颠倒出现的场景。例如，在机器人视觉系统中，传送带或机械臂上的对象可能以各种方向被拾取和放置。垂直翻转帮助模型无论其上下定位如何都能识别对象。
- **Ultralytics 实现**：[RandomFlip](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomFlip)

|                                                            **`flipud` 关闭**                                                             |                                                                 **`flipud` 开启**                                                                 |
| :-------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity" width="38%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_flip_vertical_1.avif" alt="flipud_on_augmentation" width="38%"/> |

### 左右翻转 (`fliplr`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ fliplr }}`
- **用法**：通过沿 x 轴镜像图像执行水平翻转。此变换交换左右两侧，同时保持空间一致性，帮助模型泛化到以镜像方向出现的对象。`fliplr` 超参数定义应用变换的概率，`fliplr=1.0` 确保所有图像都被翻转，`fliplr=0.0` 完全禁用变换。例如，`fliplr=0.5` 时，每张图像有 50% 的概率被左右翻转。
- **目的**：水平翻转广泛用于对象检测、姿态估计和面部识别，以提高对左右变化的鲁棒性。例如，在自动驾驶中，车辆和行人可能出现在道路的任一侧，水平翻转帮助模型在两个方向上同样好地识别它们。
- **Ultralytics 实现**：[RandomFlip](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomFlip)

|                                                            **`fliplr` 关闭**                                                             |                                                                  **`fliplr` 开启**                                                                  |
| :-------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity" width="38%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_flip_horizontal_1.avif" alt="fliplr_on_augmentation" width="38%"/> |

### BGR 通道交换 (`bgr`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ bgr }}`
- **用法**：将图像的颜色通道从 RGB 交换为 BGR，改变颜色表示的顺序。`bgr` 超参数定义应用变换的概率，`bgr=1.0` 确保所有图像都进行通道交换，`bgr=0.0` 禁用它。例如，`bgr=0.5` 时，每张图像有 50% 的概率从 RGB 转换为 BGR。
- **目的**：增加对不同颜色通道顺序的鲁棒性。例如，当训练必须跨各种相机系统和成像库工作的模型时，RGB 和 BGR 格式可能被不一致地使用，或者当将模型部署到输入颜色格式可能与训练数据不同的环境时。
- **Ultralytics 实现**：[Format](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.Format)

|                                                              **`bgr` 关闭**                                                              |                                                                  **`bgr` 开启**                                                                   |
| :-------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity" width="38%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_bgr_channel_swap_1.avif" alt="bgr_on_augmentation" width="38%"/> |

### 马赛克 (`mosaic`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ mosaic }}`
- **用法**：将四张训练图像组合成一张。`mosaic` 超参数定义应用变换的概率，`mosaic=1.0` 确保所有图像都被组合，`mosaic=0.0` 禁用变换。例如，`mosaic=0.5` 时，每张图像有 50% 的概率与其他三张图像组合。
- **目的**：对于改善小对象检测和上下文理解非常有效。例如，在野生动物保护项目中，动物可能以各种距离和尺度出现，马赛克增强通过从有限数据中人工创建多样化的训练样本，帮助模型学习在不同大小、部分遮挡和环境背景下识别同一物种。
- **Ultralytics 实现**：[Mosaic](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.Mosaic)
- **注意**：
    - 即使 `mosaic` 增强使模型更加鲁棒，它也可能使训练过程更具挑战性。
    - 可以通过将 `close_mosaic` 设置为完成前应关闭的轮次数来在训练结束时禁用 `mosaic` 增强。例如，如果 `epochs` 设置为 `200`，`close_mosaic` 设置为 `20`，则 `mosaic` 增强将在 `180` 轮后禁用。如果 `close_mosaic` 设置为 `0`，则 `mosaic` 增强将在整个训练过程中启用。
    - 生成的马赛克中心使用随机值确定，可以在图像内部或外部。
    - 当前 `mosaic` 增强的实现组合从数据集中随机选择的 4 张图像。如果数据集很小，同一张图像可能在同一马赛克中多次使用。

|                                                            **`mosaic` 关闭**                                                             |                                                              **`mosaic` 开启**                                                              |
| :-------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_identity" width="38%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_mosaic_on.avif" alt="mosaic_on_augmentation" width="55%"/> |

### 混合 (`mixup`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ mixup }}`
- **用法**：以给定概率混合两张图像及其标签。`mixup` 超参数定义应用变换的概率，`mixup=1.0` 确保所有图像都被混合，`mixup=0.0` 禁用变换。例如，`mixup=0.5` 时，每张图像有 50% 的概率与另一张图像混合。
- **目的**：提高模型鲁棒性并减少过拟合。例如，在零售产品识别系统中，mixup 通过混合不同产品的图像帮助模型学习更鲁棒的特征，教会它即使在拥挤的商店货架上产品部分可见或被其他产品遮挡时也能识别物品。
- **Ultralytics 实现**：[Mixup](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.MixUp)
- **注意**：
    - `mixup` 比率是从 `np.random.beta(32.0, 32.0)` beta 分布中选择的随机值，意味着每张图像贡献大约 50%，略有变化。

|                                                          **第一张图像，`mixup` 关闭**                                                           |                                                              **第二张图像，`mixup` 关闭**                                                              |                                                             **`mixup` 开启**                                                              |
| :---------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_identity.avif" alt="augmentation_mixup_identity_1" width="60%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_mixup_identity_2.avif" alt="augmentation_mixup_identity_2" width="60%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_mixup_on.avif" alt="mixup_on_augmentation" width="85%"/> |

### 剪切混合 (`cutmix`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ cutmix }}`
- **用法**：以给定概率从一张图像中剪切一个矩形区域并粘贴到另一张图像上。`cutmix` 超参数定义应用变换的概率，`cutmix=1.0` 确保所有图像都进行此变换，`cutmix=0.0` 完全禁用它。例如，`cutmix=0.5` 时，每张图像有 50% 的概率用另一张图像的补丁替换一个区域。
- **目的**：通过创建真实的遮挡场景同时保持局部特征完整性来增强模型性能。例如，在自动驾驶系统中，cutmix 帮助模型学习识别即使被其他对象部分遮挡的车辆或行人，提高在具有重叠对象的复杂真实环境中的检测准确性。
- **Ultralytics 实现**：[CutMix](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.CutMix)
- **注意**：
    - 剪切区域的大小和位置对于每次应用都是随机确定的。
    - 与全局混合像素值的 mixup 不同，`cutmix` 在剪切区域内保持原始像素强度，保留局部特征。
    - 只有当区域不与任何现有边界框重叠时，才会将区域粘贴到目标图像中。此外，只有在粘贴区域内保留至少 `0.1`（10%）原始面积的边界框才会被保留。
    - 此最小边界框面积阈值在当前实现中无法更改，默认设置为 `0.1`。

|                                                               **第一张图像，`cutmix` 关闭**                                                               |                                                              **第二张图像，`cutmix` 关闭**                                                               |                                                              **`cutmix` 开启**                                                              |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_cutmix_identity_1.avif" alt="augmentation_cutmix_identity_1" width="85%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_cutmix_identity_2.avif" alt="augmentation_cutmix_identity_2" width="85%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_cutmix_on.avif" alt="cutmix_on_augmentation" width="85%"/> |


## 分割特定增强

### 复制粘贴 (`copy_paste`)

- **范围**：`0.0` - `1.0`
- **默认值**：`{{ copy_paste }}`
- **用法**：仅适用于分割任务，此增强根据指定概率在图像内或图像之间复制对象，由 [`copy_paste_mode`](#复制粘贴模式-copy_paste_mode) 控制。`copy_paste` 超参数定义应用变换的概率，`copy_paste=1.0` 确保所有图像都被复制，`copy_paste=0.0` 禁用变换。例如，`copy_paste=0.5` 时，每张图像有 50% 的概率从另一张图像复制对象。
- **目的**：特别适用于实例分割任务和稀有对象类别。例如，在工业缺陷检测中，某些类型的缺陷出现频率较低，复制粘贴增强可以通过将这些稀有缺陷从一张图像复制到另一张图像来人工增加其出现频率，帮助模型更好地学习这些代表性不足的情况，而无需额外的缺陷样本。
- **Ultralytics 实现**：[CopyPaste](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.CopyPaste)
- **注意**：
    - 如下面的 gif 所示，`copy_paste` 增强可用于将对象从一张图像复制到另一张图像。
    - 一旦对象被复制，无论 `copy_paste_mode` 如何，都会计算其与源图像所有对象的交集面积比 (IoA)。如果所有 IoA 都低于 `0.3`（30%），则对象被粘贴到目标图像中。如果只有一个 IoA 高于 `0.3`，则对象不会被粘贴到目标图像中。
    - IoA 阈值在当前实现中无法更改，默认设置为 `0.3`。

|                                                             **`copy_paste` 关闭**                                                              |                                                  **`copy_paste` 开启，`copy_paste_mode=flip`**                                                  |                                                            可视化 `copy_paste` 过程                                                             |
| :-------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_copy_paste_off.avif" alt="augmentation_identity" width="80%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_copy_paste_on.avif" alt="copy_paste_on_augmentation" width="80%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_copy_paste_demo.avif" alt="copy_paste_augmentation_gif_demo" width="97%"/> |

### 复制粘贴模式 (`copy_paste_mode`)

- **选项**：`'flip'`、`'mixup'`
- **默认值**：`'{{ copy_paste_mode }}'`
- **用法**：确定用于[复制粘贴](#复制粘贴-copy_paste)增强的方法。如果设置为 `'flip'`，对象来自同一张图像，而 `'mixup'` 允许从不同图像复制对象。
- **目的**：允许灵活地将复制的对象集成到目标图像中。
- **Ultralytics 实现**：[CopyPaste](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.CopyPaste)
- **注意**：
    - 两种 `copy_paste_mode` 的 IoA 原则相同，但复制对象的方式不同。
    - 根据图像大小，对象有时可能被部分或完全复制到画面外。
    - 根据多边形标注的质量，复制的对象可能与原始对象相比有轻微的形状变化。

|                                                                   **参考图像**                                                                   |                                                       **为 `copy_paste` 选择的图像**                                                       |                                                       **`copy_paste` 开启，`copy_paste_mode=mixup`**                                                       |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_mixup_identity_2.avif" alt="augmentation_mixup_identity_2" width="77%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_copy_paste_off.avif" alt="augmentation_identity" width="80%"/> | <img src="https://github.com/ultralytics/docs/releases/download/0/augmentation_copy_paste_mixup.avif" alt="copy_paste_mode_mixup_augmentation" width="77%"/> |

## 分类特定增强

### 自动增强 (`auto_augment`)

- **选项**：`'randaugment'`、`'autoaugment'`、`'augmix'`、`None`
- **默认值**：`'{{ auto_augment }}'`
- **用法**：为分类应用自动增强策略。`'randaugment'` 选项使用 RandAugment，`'autoaugment'` 使用 AutoAugment，`'augmix'` 使用 AugMix。设置为 `None` 禁用自动增强。
- **目的**：自动优化分类任务的增强策略。区别如下：
    - **AutoAugment**：此模式应用从 ImageNet、CIFAR10 和 SVHN 等数据集学习的预定义增强策略。用户可以选择这些现有策略，但无法在 Torchvision 中训练新策略。要为特定数据集发现最佳增强策略，需要外部库或自定义实现。参考 [AutoAugment 论文](https://arxiv.org/abs/1805.09501)。
    - **RandAugment**：应用具有统一幅度的随机变换选择。这种方法减少了对广泛搜索阶段的需求，使其在计算上更高效，同时仍能增强模型性能。参考 [RandAugment 论文](https://arxiv.org/abs/1909.13719)。
    - **AugMix**：混合多个增强链以创建多样化的训练样本，同时保持图像一致性。它还包括 Jensen-Shannon 散度一致性损失以提高鲁棒性。参考 [AugMix 论文](https://arxiv.org/abs/1912.02781)。
- **Ultralytics 实现**：[classify_augmentations](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.classify_augmentations)

## 高级增强功能

### 随机擦除 (`erasing`)

- **范围**：`0.0` - `0.9`
- **默认值**：`{{ erasing }}`
- **用法**：在分类训练期间随机擦除图像的一部分。`erasing` 超参数定义应用变换的概率，`erasing=0.9` 确保 90% 的图像有一部分被擦除，`erasing=0.0` 禁用变换。
- **目的**：通过模拟遮挡来提高模型鲁棒性。例如，在面部识别系统中，面部可能被太阳镜、口罩或其他物体部分遮挡。随机擦除帮助模型学习即使在部分可见时也能识别对象。
- **Ultralytics 实现**：[RandomErasing](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.RandomErasing)

### 裁剪比例 (`crop_fraction`)

- **范围**：`0.1` - `1.0`
- **默认值**：`{{ crop_fraction }}`
- **用法**：将分类图像裁剪到其大小的一部分以突出中心特征并丢弃背景噪声。`crop_fraction` 超参数定义裁剪比例，`crop_fraction=1.0` 保持原始图像大小，`crop_fraction=0.1` 将图像裁剪到其原始大小的 10%。
- **目的**：帮助模型专注于对象的中心特征并减少背景噪声的影响。例如，在产品分类中，产品通常位于图像中心，裁剪可以帮助模型专注于产品本身而不是背景。
- **Ultralytics 实现**：[CenterCrop](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.CenterCrop)

## 常见问题

### 我应该使用哪些增强？

简而言之：保持简单。从一小组增强开始，根据需要逐渐添加更多。目标是提高模型的泛化能力和鲁棒性，而不是使训练过程过于复杂。此外，确保您应用的增强反映模型在生产中将遇到的相同数据分布。

### 当开始训练时，我看到 `albumentations: Blur[...]` 引用。这是否意味着 Ultralytics YOLO 运行了额外的增强如模糊？

如果安装了 `albumentations` 包，Ultralytics 会自动使用它应用一组额外的图像增强。这些增强在内部处理，不需要额外配置。

您可以在我们的[技术文档](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.Albumentations)以及我们的 [Albumentations 集成指南](https://docs.ultralytics.com/integrations/albumentations/)中找到应用变换的完整列表。请注意，只有概率 `p` 大于 `0` 的增强才是活动的。这些增强故意以低频率应用，以模拟真实世界的视觉伪影，如模糊或灰度效果。

您还可以使用 Python API 提供自己的自定义 Albumentations 变换。有关更多详细信息，请参阅[高级增强功能](#高级增强功能)部分。

### 当开始训练时，我没有看到任何 albumentations 引用。为什么？

检查是否安装了 `albumentations` 包。如果没有，您可以通过运行 `pip install albumentations` 安装它。安装后，该包应该会被 Ultralytics 自动检测和使用。

### 如何自定义我的增强？

您可以通过创建自定义数据集类和训练器来自定义增强。例如，您可以用 PyTorch 的 [torchvision.transforms.Resize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html) 或其他变换替换默认的 Ultralytics 分类增强。有关实现详细信息，请参阅分类文档中的[自定义训练示例](../tasks/classify.md#train)。
