# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse

import cv2
import numpy as np
import yaml

from ultralytics.utils import ASSETS

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


class YOLOv8TFLite:
    """使用 TensorFlow Lite 进行高效推理的 YOLOv8 目标检测类。

    该类处理模型加载、预处理、推理以及 YOLOv8 模型转换为 TensorFlow Lite 格式后的检测结果可视化。

    Attributes:
        model (Interpreter): YOLOv8 模型的 TensorFlow Lite 解释器。
        conf (float): 过滤检测的置信度阈值。
        iou (float): 非极大值抑制的交并比阈值。
        classes (dict): 将类别 ID 映射到类别名称的字典。
        color_palette (np.ndarray): 用于可视化的随机颜色调色板，形状为 (num_classes, 3)。
        in_width (int): 模型所需的输入宽度。
        in_height (int): 模型所需的输入高度。
        in_index (int): 模型中的输入张量索引。
        in_scale (float): 输入量化缩放因子。
        in_zero_point (int): 输入量化零点。
        int8 (bool): 模型是否使用 int8 量化。
        out_index (int): 模型中的输出张量索引。
        out_scale (float): 输出量化缩放因子。
        out_zero_point (int): 输出量化零点。

    Methods:
        letterbox: 在保持宽高比的同时调整和填充图像。
        draw_detections: 在输入图像上绘制边界框和标签。
        preprocess: 在推理之前预处理输入图像。
        postprocess: 处理模型输出以提取和可视化检测结果。
        detect: 对输入图像执行目标检测。

    Examples:
        初始化检测器并运行推理
        >>> detector = YOLOv8TFLite("yolov8n.tflite", conf=0.25, iou=0.45)
        >>> result = detector.detect("image.jpg")
        >>> cv2.imshow("Result", result)
    """

    def __init__(self, model: str, conf: float = 0.25, iou: float = 0.45, metadata: str | None = None):
        """初始化 YOLOv8TFLite 检测器。

        Args:
            model (str): TFLite 模型文件的路径。
            conf (float): 过滤检测的置信度阈值。
            iou (float): 非极大值抑制的 IoU 阈值。
            metadata (str | None): 包含类别名称的元数据文件路径。
        """
        self.conf = conf
        self.iou = iou
        if metadata is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(metadata) as f:
                self.classes = yaml.safe_load(f)["names"]
        np.random.seed(42)  # 设置种子以获得可重复的颜色
        self.color_palette = np.random.uniform(128, 255, size=(len(self.classes), 3))

        # 初始化 TFLite 解释器
        self.model = Interpreter(model_path=model)
        self.model.allocate_tensors()

        # 获取输入详情
        input_details = self.model.get_input_details()[0]
        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]
        self.in_scale, self.in_zero_point = input_details["quantization"]
        self.int8 = input_details["dtype"] == np.int8

        # 获取输出详情
        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]
        self.out_scale, self.out_zero_point = output_details["quantization"]

    def letterbox(
        self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """在保持宽高比的同时调整和填充图像。

        Args:
            img (np.ndarray): 形状为 (H, W, C) 的输入图像。
            new_shape (tuple[int, int]): 目标形状（高度，宽度）。

        Returns:
            (np.ndarray): 调整大小并填充后的图像。
            (tuple[float, float]): 用于坐标调整的填充比例（顶部/高度，左侧/宽度）。
        """
        shape = img.shape[:2]  # 当前形状 [高度, 宽度]

        # 缩放比例（新 / 旧）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # 计算填充
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # 宽高填充

        if shape[::-1] != new_unpad:  # 如果需要则调整大小
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top / img.shape[0], left / img.shape[1])

    def draw_detections(self, img: np.ndarray, box: np.ndarray, score: np.float32, class_id: int) -> None:
        """根据检测到的目标在输入图像上绘制边界框和标签。

        Args:
            img (np.ndarray): 要绘制检测结果的输入图像。
            box (np.ndarray): 检测到的边界框，格式为 [x1, y1, width, height]。
            score (np.float32): 检测的置信度分数。
            class_id (int): 检测到的目标的类别 ID。
        """
        x1, y1, w, h = box
        color = self.color_palette[class_id]

        # 绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类别名称和分数的标签
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 获取背景矩形的文本尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 根据空间将标签放置在边界框上方或下方
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制标签背景
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # 绘制文本
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        """在执行推理之前预处理输入图像。

        Args:
            img (np.ndarray): 形状为 (H, W, C) 的要预处理的输入图像。

        Returns:
            (np.ndarray): 准备好进行模型输入的预处理图像。
            (tuple[float, float]): 用于坐标调整的填充比例。
        """
        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[..., ::-1][None]  # BGR 转 RGB 并添加批次维度 (N, H, W, C) 用于 TFLite
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        return img / 255, pad  # 归一化到 [0, 1]

    def postprocess(self, img: np.ndarray, outputs: np.ndarray, pad: tuple[float, float]) -> np.ndarray:
        """处理模型输出以提取和可视化检测结果。

        Args:
            img (np.ndarray): 原始输入图像。
            outputs (np.ndarray): 原始模型输出。
            pad (tuple[float, float]): 预处理时的填充比例。

        Returns:
            (np.ndarray): 绘制了检测结果的输入图像。
        """
        # 根据填充调整坐标并缩放到原始图像尺寸
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(img.shape)

        # 将输出转换为 [x, y, w, h] 格式
        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0] -= outputs[..., 2] / 2  # x 中心转左上角 x
        outputs[..., 1] -= outputs[..., 3] / 2  # y 中心转左上角 y

        for out in outputs:
            # 获取分数并应用置信度阈值
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)

            # 应用非极大值抑制
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou).flatten()

            # 绘制通过 NMS 的检测结果
            [self.draw_detections(img, boxes[i], scores[i], class_ids[i]) for i in indices]

        return img

    def detect(self, img_path: str) -> np.ndarray:
        """对输入图像执行目标检测。

        Args:
            img_path (str): 输入图像文件的路径。

        Returns:
            (np.ndarray): 绘制了检测结果的输出图像。
        """
        # 加载并预处理图像
        img = cv2.imread(img_path)
        x, pad = self.preprocess(img)

        # 如果模型是 int8 则应用量化
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # 设置输入张量并运行推理
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()

        # 获取输出并在必要时反量化
        y = self.model.get_tensor(self.out_index)
        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

        # 处理检测结果并返回
        return self.postprocess(img, y, pad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n_saved_model/yolov8n_full_integer_quant.tflite",
        help="TFLite 模型的路径。",
    )
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="输入图像的路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--metadata", type=str, default="yolov8n_saved_model/metadata.yaml", help="元数据 yaml 文件")
    args = parser.parse_args()

    detector = YOLOv8TFLite(args.model, args.conf, args.iou, args.metadata)
    result = detector.detect(args.img)

    cv2.imshow("Output", result)
    cv2.waitKey(0)
