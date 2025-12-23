# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML, nms, ops
from ultralytics.utils.checks import check_yaml


class YOLOv8Seg:
    """使用 ONNX Runtime 执行实例分割的 YOLOv8 分割模型。

    该类使用 ONNX Runtime 实现 YOLOv8 实例分割模型进行推理。它处理输入图像的预处理、
    使用 ONNX 模型运行推理，以及后处理结果以生成边界框和分割掩码。

    Attributes:
        session (ort.InferenceSession): 用于模型执行的 ONNX Runtime 推理会话。
        imgsz (tuple[int, int]): 模型的输入图像尺寸，格式为（高度，宽度）。
        classes (dict): 将类别索引映射到数据集类别名称的字典。
        conf (float): 过滤检测的置信度阈值。
        iou (float): 非极大值抑制使用的 IoU 阈值。

    Methods:
        letterbox: 在保持宽高比的同时调整和填充图像。
        preprocess: 在输入模型之前预处理输入图像。
        postprocess: 后处理模型预测以提取有意义的结果。
        process_mask: 使用预测的掩码系数处理原型掩码以生成实例分割掩码。

    Examples:
        >>> model = YOLOv8Seg("yolov8n-seg.onnx", conf=0.25, iou=0.7)
        >>> img = cv2.imread("image.jpg")
        >>> results = model(img)
        >>> cv2.imshow("Segmentation", results[0].plot())
    """

    def __init__(self, onnx_model: str, conf: float = 0.25, iou: float = 0.7, imgsz: int | tuple[int, int] = 640):
        """使用 ONNX 模型初始化实例分割模型。

        Args:
            onnx_model (str): ONNX 模型文件的路径。
            conf (float, optional): 过滤检测的置信度阈值。
            iou (float, optional): 非极大值抑制的 IoU 阈值。
            imgsz (int | tuple[int, int], optional): 模型的输入图像尺寸。可以是整数（正方形输入）
                或元组（矩形输入）。
        """
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)

        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.classes = YAML.load(check_yaml("coco8.yaml"))["names"]
        self.conf = conf
        self.iou = iou

    def __call__(self, img: np.ndarray) -> list[Results]:
        """使用 ONNX 模型对输入图像运行推理。

        Args:
            img (np.ndarray): BGR 格式的原始输入图像。

        Returns:
            (list[Results]): 后处理后的检测结果，包含边界框和分割掩码。
        """
        prep_img = self.preprocess(img, self.imgsz)
        outs = self.session.run(None, {self.session.get_inputs()[0].name: prep_img})
        return self.postprocess(img, prep_img, outs)

    def letterbox(self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> np.ndarray:
        """在保持宽高比的同时调整和填充图像。

        Args:
            img (np.ndarray): BGR 格式的输入图像。
            new_shape (tuple[int, int], optional): 目标形状，格式为（高度，宽度）。

        Returns:
            (np.ndarray): 调整大小并填充后的图像。
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

        return img

    def preprocess(self, img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
        """在输入模型之前预处理输入图像。

        Args:
            img (np.ndarray): BGR 格式的输入图像。
            new_shape (tuple[int, int]): 调整大小的目标形状，格式为（高度，宽度）。

        Returns:
            (np.ndarray): 准备好进行模型推理的预处理图像，形状为 (1, 3, height, width)，
                归一化到 [0, 1]。
        """
        img = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]  # BGR 转 RGB，BHWC 转 BCHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255  # 归一化到 [0, 1]
        return img

    def postprocess(self, img: np.ndarray, prep_img: np.ndarray, outs: list) -> list[Results]:
        """后处理模型预测以提取有意义的结果。

        Args:
            img (np.ndarray): 原始输入图像。
            prep_img (np.ndarray): 用于推理的预处理图像。
            outs (list): 包含预测和原型掩码的模型输出。

        Returns:
            (list[Results]): 包含边界框和分割掩码的处理后检测结果。
        """
        preds, protos = (torch.from_numpy(p) for p in outs)
        preds = nms.non_max_suppression(preds, self.conf, self.iou, nc=len(self.classes))

        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = ops.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append(Results(img, path="", names=self.classes, boxes=pred[:, :6], masks=masks))

        return results

    def process_mask(
        self, protos: torch.Tensor, masks_in: torch.Tensor, bboxes: torch.Tensor, shape: tuple[int, int]
    ) -> torch.Tensor:
        """使用预测的掩码系数处理原型掩码以生成实例分割掩码。

        Args:
            protos (torch.Tensor): 形状为 (mask_dim, mask_h, mask_w) 的原型掩码。
            masks_in (torch.Tensor): 形状为 (N, mask_dim) 的预测掩码系数，其中 N 是检测数量。
            bboxes (torch.Tensor): 形状为 (N, 4) 的边界框，其中 N 是检测数量。
            shape (tuple[int, int]): 输入图像的尺寸，格式为（高度，宽度）。

        Returns:
            (torch.Tensor): 形状为 (N, height, width) 的二值分割掩码。
        """
        c, mh, mw = protos.shape  # CHW
        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # 矩阵乘法
        masks = ops.scale_masks(masks[None], shape)[0]  # 将掩码缩放到原始图像尺寸
        masks = ops.crop_mask(masks, bboxes)  # 将掩码裁剪到边界框
        return masks.gt_(0.0)  # 转换为二值掩码


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型的路径")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="输入图像的路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    args = parser.parse_args()

    model = YOLOv8Seg(args.model, args.conf, args.iou)
    img = cv2.imread(args.source)
    results = model(img)

    cv2.imshow("Segmented Image", results[0].plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
