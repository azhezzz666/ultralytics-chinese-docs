# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import requests
import yaml


def download_file(url: str, local_path: str) -> str:
    """从 URL 下载文件到本地路径。

    Args:
        url (str): 要下载的文件 URL。
        local_path (str): 文件保存的本地路径。
    """
    # 检查本地路径是否已存在
    if os.path.exists(local_path):
        print(f"文件已存在于 {local_path}。跳过下载。")
        return local_path
    # 从 URL 下载文件
    print(f"正在从 {url} 下载到 {local_path}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return local_path


class RTDETR:
    """RT-DETR（实时检测 Transformer）目标检测模型，用于 ONNX 推理和可视化。

    该类实现了用于目标检测任务的 RT-DETR 模型，支持 ONNX 模型推理和使用边界框及类别标签可视化检测结果。

    Attributes:
        model_path (str): ONNX 模型文件路径。
        img_path (str): 输入图像路径。
        conf_thres (float): 过滤检测的置信度阈值。
        iou_thres (float): 非极大值抑制的 IoU 阈值。
        session (ort.InferenceSession): ONNX 运行时推理会话。
        model_input (list): 模型输入元数据。
        input_width (int): 模型所需的宽度维度。
        input_height (int): 模型所需的高度维度。
        classes (list[str]): COCO 数据集的类别名称列表。
        color_palette (np.ndarray): 用于可视化的随机颜色调色板。
        img (np.ndarray): 加载的输入图像。
        img_height (int): 输入图像的高度。
        img_width (int): 输入图像的宽度。

    Methods:
        draw_detections: 在输入图像上绘制边界框和标签。
        preprocess: 为模型推理预处理输入图像。
        bbox_cxcywh_to_xyxy: 将边界框从中心格式转换为角点格式。
        postprocess: 后处理模型输出以提取和可视化检测结果。
        main: 执行完整的目标检测流程。

    Examples:
        初始化 RT-DETR 检测器并运行推理
        >>> detector = RTDETR("rtdetr-l.onnx", "image.jpg", conf_thres=0.5)
        >>> output_image = detector.main()
        >>> cv2.imshow("Detections", output_image)
    """

    def __init__(
        self,
        model_path: str,
        img_path: str,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        class_names: str | None = None,
    ):
        """初始化 RT-DETR 目标检测模型。

        Args:
            model_path (str): ONNX 模型文件路径。
            img_path (str): 输入图像路径。
            conf_thres (float, optional): 过滤检测的置信度阈值。
            iou_thres (float, optional): 非极大值抑制的 IoU 阈值。
            class_names (Optional[str], optional): 包含类别名称的 YAML 文件路径。如果为 None，则使用 COCO 数据集类别。
        """
        self.model_path = model_path
        self.img_path = img_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = class_names

        # 使用可用的执行提供程序设置 ONNX 运行时会话
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(model_path, providers=providers or available)

        self.model_input = self.session.get_inputs()
        self.input_width = self.model_input[0].shape[2]
        self.input_height = self.model_input[0].shape[3]

        if self.classes is None:
            # 从 COCO 数据集 YAML 文件加载类别名称
            self.classes = download_file(
                "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco8.yaml",
                "coco8.yaml",
            )

        # 解析 YAML 文件以获取类别名称
        with open(self.classes) as f:
            class_data = yaml.safe_load(f)
            self.classes = list(class_data["names"].values())

        # 确保类别是列表
        if not isinstance(self.classes, list):
            raise ValueError("类别应该是类别名称的列表。")

        # 生成用于绘制边界框的颜色调色板
        self.color_palette: np.ndarray = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, box: np.ndarray, score: float, class_id: int) -> None:
        """在输入图像上为检测到的目标绘制边界框和标签。"""
        # 提取边界框的坐标
        x1, y1, x2, y2 = box

        # 获取类别 ID 对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 创建包含类别名称和分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充矩形作为标签文本的背景
        cv2.rectangle(
            self.img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # 在图像上绘制标签文本
        cv2.putText(
            self.img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    def preprocess(self) -> np.ndarray:
        """为模型推理预处理输入图像。

        加载图像，将颜色空间从 BGR 转换为 RGB，调整大小以匹配模型输入维度，并将像素值归一化到 [0, 1] 范围。

        Returns:
            (np.ndarray): 形状为 (1, 3, H, W) 的预处理图像数据，准备用于推理。
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            raise FileNotFoundError(f"找不到或无法读取图像: '{self.img_path}'")

        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 调整图像大小以匹配输入形状
        img = cv2.resize(img, (self.input_width, self.input_height))

        # 通过除以 255.0 归一化图像数据
        image_data = np.array(img) / 255.0

        # 转置图像，使通道维度成为第一个维度
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 扩展图像数据的维度以匹配预期的输入形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def bbox_cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """将边界框从中心格式转换为角点格式。

        Args:
            boxes (np.ndarray): 形状为 (N, 4) 的数组，每行表示一个边界框，格式为 (center_x, center_y, width, height)。

        Returns:
            (np.ndarray): 形状为 (N, 4) 的数组，边界框格式为 (x_min, y_min, x_max, y_max)。
        """
        # 计算边界框的半宽和半高
        half_width = boxes[:, 2] / 2
        half_height = boxes[:, 3] / 2

        # 计算边界框的坐标
        x_min = boxes[:, 0] - half_width
        y_min = boxes[:, 1] - half_height
        x_max = boxes[:, 0] + half_width
        y_max = boxes[:, 1] + half_height

        # 返回 (x_min, y_min, x_max, y_max) 格式的边界框
        return np.column_stack((x_min, y_min, x_max, y_max))

    def postprocess(self, model_output: list[np.ndarray]) -> np.ndarray:
        """后处理模型输出以提取和可视化检测结果。

        应用置信度阈值过滤，转换边界框格式，将坐标缩放到原始图像尺寸，并绘制检测标注。

        Args:
            model_output (list[np.ndarray]): 模型推理的输出张量。

        Returns:
            (np.ndarray): 带有检测边界框和标签的标注图像。
        """
        # 压缩模型输出以移除不必要的维度
        outputs = np.squeeze(model_output[0])

        # 从模型输出中提取边界框和分数
        boxes = outputs[:, :4]
        scores = outputs[:, 4:]

        # 获取每个检测的类别标签和分数
        labels = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        # 应用置信度阈值过滤低置信度检测
        mask = scores > self.conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        # 将边界框转换为 (x_min, y_min, x_max, y_max) 格式
        boxes = self.bbox_cxcywh_to_xyxy(boxes)

        # 缩放边界框以匹配原始图像尺寸
        boxes[:, 0::2] *= self.img_width
        boxes[:, 1::2] *= self.img_height

        # 应用非极大值抑制（对于 RT-DETR 是可选的，但对于过滤重叠很有用）
        xywh_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(xywh_boxes, scores.tolist(), self.conf_thres, self.iou_thres)
        indices = indices.flatten().tolist() if len(indices) else []

        # 在图像上绘制检测结果
        for i in indices:
            self.draw_detections(boxes[i], float(scores[i]), int(labels[i]))

        return self.img

    def main(self) -> np.ndarray:
        """在输入图像上执行完整的目标检测流程。

        执行预处理、ONNX 模型推理和后处理，生成带标注的检测结果。

        Returns:
            (np.ndarray): 带有检测标注（包括边界框和类别标签）的输出图像。
        """
        # 为模型输入预处理图像
        image_data = self.preprocess()

        # 运行模型推理
        model_output = self.session.run(None, {self.model_input[0].name: image_data})

        # 处理并返回模型输出
        return self.postprocess(model_output)


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rtdetr-l.onnx", help="ONNX 模型文件路径。")
    parser.add_argument("--img", type=str, default="bus.jpg", help="输入图像路径。")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="目标检测的置信度阈值。")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="非极大值抑制的 IoU 阈值。")
    args = parser.parse_args()

    # 使用指定参数创建检测器实例
    detection = RTDETR(args.model, args.img, args.conf_thres, args.iou_thres)

    # 执行检测并获取输出图像
    output_image = detection.main()

    # 显示标注后的输出图像
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
