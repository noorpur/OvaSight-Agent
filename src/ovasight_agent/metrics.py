from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = np.logical_and(yt, yp).sum()
    fp = np.logical_and(~yt, yp).sum()
    fn = np.logical_and(yt, ~yp).sum()
    tn = np.logical_and(~yt, ~yp).sum()
    return tp, fp, fn, tn


def dice_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    return float((tp + eps) / (tp + fp + fn + eps))


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    tp, _, fn, _ = confusion_counts(y_true, y_pred)
    return float((tp + eps) / (tp + fn + eps))


def specificity(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    _, fp, _, tn = confusion_counts(y_true, y_pred)
    return float((tn + eps) / (tn + fp + eps))


def precision(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    return float((tp + eps) / (tp + fp + eps))


def area_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    true_area = y_true.astype(bool).sum()
    pred_area = y_pred.astype(bool).sum()
    return float(abs(pred_area - true_area) / (true_area + eps))


def boundary_distance_proxy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = y_true.astype(bool)
    pred = y_pred.astype(bool)
    if true.sum() == 0 or pred.sum() == 0:
        return float("inf")
    true_edge = true ^ ndi.binary_erosion(true)
    pred_edge = pred ^ ndi.binary_erosion(pred)
    dist_true = ndi.distance_transform_edt(~true_edge)
    dist_pred = ndi.distance_transform_edt(~pred_edge)
    d1 = dist_true[pred_edge].mean() if pred_edge.any() else np.inf
    d2 = dist_pred[true_edge].mean() if true_edge.any() else np.inf
    return float(np.mean([d1, d2]))


def summarize(values: list[float]) -> dict:
    clean = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if len(clean) == 0:
        return {"mean": None, "std": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "median": float(np.median(clean)),
        "p10": float(np.percentile(clean, 10)),
        "p90": float(np.percentile(clean, 90)),
    }
