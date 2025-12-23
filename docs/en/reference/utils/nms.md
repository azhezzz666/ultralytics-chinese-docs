---
description: Ultralytics YOLO 的自定义 NMS 实现，包含用于无 torchvision 推理的 TorchNMS 类和用于旋转边界框的 fast-nms。针对速度和准确性进行了优化。
keywords: NMS, 非极大值抑制, TorchNMS, YOLO, 无 torchvision, 旋转 NMS, 目标检测, 边界框, IoU 阈值, 自定义实现
---

# `ultralytics/utils/nms.py` 参考

!!! success "改进建议"

    本页面源自 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/nms.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/nms.py)。有改进建议或示例要添加？请提交 [Pull Request](https://docs.ultralytics.com/help/contributing/) — 感谢！🙏

<br>

## ::: ultralytics.utils.nms.TorchNMS

<br><br><hr><br>

## ::: ultralytics.utils.nms.non_max_suppression

<br><br>
