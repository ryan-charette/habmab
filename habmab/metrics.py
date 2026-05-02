"""Scientific evaluation metrics for adaptive monitoring."""

from __future__ import annotations

import numpy as np


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def mae(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - truth)))


def hotspot_recall(selected_unique: np.ndarray, hotspot_mask: np.ndarray) -> float:
    total_hotspots = int(np.sum(hotspot_mask))
    if total_hotspots == 0:
        return 0.0
    return float(np.sum(hotspot_mask[selected_unique]) / total_hotspots)


def precision_at_selected(selected_unique: np.ndarray, hotspot_mask: np.ndarray) -> float:
    if len(selected_unique) == 0:
        return 0.0
    return float(np.mean(hotspot_mask[selected_unique]))


def top_risk_precision(prediction: np.ndarray, hotspot_mask: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    k = min(k, len(prediction))
    top = np.argsort(-prediction)[:k]
    return float(np.mean(hotspot_mask[top]))
