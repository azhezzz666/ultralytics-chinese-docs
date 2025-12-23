# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
from typing import Any

import cv2.dnn
import numpy as np

from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml

CLASSES = YAML.load(check_yaml("coco8.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(
    img: np.ndarray, class_id: int, confidence: float, x: int, y: int, x_plus_w: int, y_plus_h: int
) -> None:
    """根据提供的参数在输入图像上绘制边界框。

    Args:
        img (np.ndarray): 要绘制边界框的输入图像。
        class_id (int): 检测到的目标的类别 ID。
        confidence (float): 检测到的目标的置信度分数。
        x (int): 边界框左上角的 X 坐标。
        y (int): 边界框左上角的 Y 坐标。
        x_plus_w (int): 边界框右下角的 X 坐标。
        y_plus_h (int): 边界框右下角的 Y 坐标。
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model: str, input_image: str) -> list[dict[str, Any]]:
    """加载 ONNX 模型，执行推理，绘制边界框，并显示输出图像。

    Args:
        onnx_model (str): ONNX 模型的路径。
        input_image (str): 输入图像的路径。

    Returns:
        (list[dict[str, Any]]): 包含检测信息的字典列表，包括类别 ID、类别名称、
            置信度、边界框坐标和缩放因子。
    """
    # 加载 ONNX 模型
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # 读取输入图像
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape

    # 准备用于推理的正方形图像
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # 计算缩放因子
    scale = length / 640

    # 预处理图像并为模型准备 blob
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # 执行推理
    outputs = model.forward()

    # 准备输出数组
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # 遍历输出以收集边界框、置信度分数和类别 ID
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (_minScore, maxScore, _minClassLoc, (_x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # x 中心 - 宽度/2 = 左边 x
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # y 中心 - 高度/2 = 顶部 y
                outputs[0][i][2],  # 宽度
                outputs[0][i][3],  # 高度
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # 应用 NMS（非极大值抑制）
    result_boxes = np.array(cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)).flatten()

    detections = []

    # 遍历 NMS 结果以绘制边界框和标签
    for index in result_boxes:
        index = int(index)
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # 显示带边界框的图像
    cv2.imshow("image", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    args = parser.parse_args()
    main(args.model, args.img)
