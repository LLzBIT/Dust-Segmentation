from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationMetrics:
    iou_dust: float
    iou_background: float
    mean_iou: float
    accuracy: float
    precision: float
    recall: float


def compute_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> SegmentationMetrics:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = np.logical_and(y_pred, y_true).sum()
    fp = np.logical_and(y_pred, np.logical_not(y_true)).sum()
    fn = np.logical_and(np.logical_not(y_pred), y_true).sum()
    tn = np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)).sum()

    iou_dust = tp / (tp + fp + fn + eps)
    iou_background = tn / (tn + fp + fn + eps)
    mean_iou = (iou_dust + iou_background) / 2.0

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return SegmentationMetrics(
        iou_dust=float(iou_dust),
        iou_background=float(iou_background),
        mean_iou=float(mean_iou),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
    )