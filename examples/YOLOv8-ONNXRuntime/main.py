# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_requirements, check_yaml


class YOLOv8:
    """YOLOv8 目标检测模型类，用于处理 ONNX 推理和可视化。

    该类提供加载 YOLOv8 ONNX 模型、对图像执行推理以及使用边界框和标签可视化检测结果的功能。

    Attributes:
        onnx_model (str): ONNX 模型文件路径。
        input_image (str): 输入图像文件路径。
        confidence_thres (float): 过滤检测的置信度阈值。
        iou_thres (float): 非极大值抑制的 IoU 阈值。
        classes (list[str]): COCO 数据集的类别名称列表。
        color_palette (np.ndarray): 用于可视化不同类别的随机颜色调色板。
        input_width (int): 模型输入的宽度维度。
        input_height (int): 模型输入的高度维度。
        img (np.ndarray): 加载的输入图像。
        img_height (int): 输入图像的高度。
        img_width (int): 输入图像的宽度。

    Methods:
        letterbox: 在保持宽高比的同时通过添加填充来调整和重塑图像。
        draw_detections: 根据检测到的目标在输入图像上绘制边界框和标签。
        preprocess: 在执行推理之前预处理输入图像。
        postprocess: 对模型输出进行后处理以提取和可视化检测结果。
        main: 使用 ONNX 模型执行推理并返回带有绘制检测结果的输出图像。

    Examples:
        初始化 YOLOv8 检测器并运行推理
        >>> detector = YOLOv8("yolov8n.onnx", "image.jpg", 0.5, 0.5)
        >>> output_image = detector.main()
    """

    def __init__(self, onnx_model: str, input_image: str, confidence_thres: float, iou_thres: float):
        """初始化 YOLOv8 类的实例。

        Args:
            onnx_model (str): ONNX 模型的路径。
            input_image (str): 输入图像的路径。
            confidence_thres (float): 过滤检测的置信度阈值。
            iou_thres (float): 非极大值抑制的 IoU 阈值。
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 从 COCO 数据集加载类别名称
        self.classes = YAML.load(check_yaml("coco8.yaml"))["names"]

        # 为类别生成颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def letterbox(self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
        """在保持宽高比的同时通过添加填充来调整和重塑图像。

        Args:
            img (np.ndarray): 要调整大小的输入图像。
            new_shape (tuple[int, int]): 图像的目标形状（高度，宽度）。

        Returns:
            img (np.ndarray): 调整大小并填充后的图像。
            pad (tuple[int, int]): 应用于图像的填充值（顶部，左侧）。
        """
        shape = img.shape[:2]  # 当前形状 [高度, 宽度]

        # 缩放比例（新 / 旧）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # 计算填充
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # 宽高填充

        if shape[::-1] != new_unpad:  # 调整大小
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: list[float], score: float, class_id: int) -> None:
        """根据检测到的目标在输入图像上绘制边界框和标签。"""
        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别 ID 对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类别名称和分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充矩形作为标签文本的背景
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self) -> tuple[np.ndarray, tuple[int, int]]:
        """在执行推理之前预处理输入图像。

        该方法读取输入图像，转换颜色空间，应用 letterbox 保持宽高比，
        归一化像素值，并为模型输入准备图像数据。

        Returns:
            image_data (np.ndarray): 准备好进行推理的预处理图像数据，形状为 (1, 3, height, width)。
            pad (tuple[int, int]): letterbox 过程中应用的填充值（顶部，左侧）。
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(self.input_image)

        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, (self.input_width, self.input_height))

        # 通过除以 255.0 归一化图像数据
        image_data = np.array(img) / 255.0

        # 转置图像，使通道维度成为第一个维度
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 扩展图像数据的维度以匹配预期的输入形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # 返回预处理后的图像数据
        return image_data, pad

    def postprocess(self, input_image: np.ndarray, output: list[np.ndarray], pad: tuple[int, int]) -> np.ndarray:
        """对模型输出进行后处理以提取和可视化检测结果。

        该方法处理原始模型输出以提取边界框、分数和类别 ID。它应用非极大值抑制
        来过滤重叠的检测，并在输入图像上绘制结果。

        Args:
            input_image (np.ndarray): 输入图像。
            output (list[np.ndarray]): 模型的输出数组。
            pad (tuple[int, int]): letterbox 过程中使用的填充值（顶部，左侧）。

        Returns:
            (np.ndarray): 绘制了检测结果的输入图像。
        """
        # 转置并压缩输出以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))

        # 获取输出数组的行数
        rows = outputs.shape[0]

        # 用于存储检测的边界框、分数和类别 ID 的列表
        boxes = []
        scores = []
        class_ids = []

        # 计算边界框坐标的缩放因子
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # 遍历输出数组中的每一行
        for i in range(rows):
            # 从当前行提取类别分数
            classes_scores = outputs[i][4:]

            # 找到类别分数中的最大值
            max_score = np.amax(classes_scores)

            # 如果最大分数高于置信度阈值
            if max_score >= self.confidence_thres:
                # 获取分数最高的类别 ID
                class_id = np.argmax(classes_scores)

                # 从当前行提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算边界框的缩放坐标
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # 将类别 ID、分数和边界框坐标添加到相应的列表
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 应用非极大值抑制来过滤重叠的边界框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # 遍历非极大值抑制后选择的索引
        for i in np.array(indices).flatten():
            # 获取对应索引的边界框、分数和类别 ID
            box = boxes[int(i)]
            score = scores[int(i)]
            class_id = class_ids[int(i)]

            # 在输入图像上绘制检测结果
            self.draw_detections(input_image, box, score, class_id)

        # 返回修改后的输入图像
        return input_image

    def main(self) -> np.ndarray:
        """使用 ONNX 模型执行推理并返回带有绘制检测结果的输出图像。

        Returns:
            (np.ndarray): 带有绘制检测结果的输出图像。
        """
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        session = ort.InferenceSession(self.onnx_model, providers=providers or available)

        # 获取模型输入
        model_inputs = session.get_inputs()

        # 存储输入形状以供后续使用
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # 预处理图像数据
        img_data, pad = self.preprocess()

        # 使用预处理后的图像数据运行推理
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # 对输出进行后处理以获得输出图像
        return self.postprocess(self.img, outputs, pad)


if __name__ == "__main__":
    # 创建参数解析器来处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="输入您的 ONNX 模型。")
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="输入图像的路径。")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU 阈值")
    args = parser.parse_args()

    # 检查依赖并选择适当的后端（CPU 或 GPU）
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # 使用指定参数创建 YOLOv8 类的实例
    detection = YOLOv8(args.model, args.img, args.conf_thres, args.iou_thres)

    # 执行目标检测并获取输出图像
    output_image = detection.main()

    # 在窗口中显示输出图像
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)

    # 等待按键退出
    cv2.waitKey(0)
