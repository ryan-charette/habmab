"""Small robust regression helper for noisy HAB observations."""

from __future__ import annotations

import numpy as np


def huber_irls(
    design: np.ndarray,
    target: np.ndarray,
    *,
    delta: float = 1.0,
    ridge: float = 1e-3,
    max_iter: int = 40,
) -> np.ndarray:
    """Huber regression via iteratively reweighted least squares."""

    weights = np.zeros(design.shape[1], dtype=float)
    eye = np.eye(design.shape[1])
    for _ in range(max_iter):
        residual = target - design @ weights
        scale = np.maximum(1.0, np.abs(residual) / max(delta, 1e-8))
        sample_weights = 1.0 / scale
        weighted_design = design * sample_weights[:, None]
        lhs = design.T @ weighted_design + ridge * eye
        rhs = weighted_design.T @ target
        next_weights = np.linalg.solve(lhs, rhs)
        if np.linalg.norm(next_weights - weights) <= 1e-6 * max(1.0, np.linalg.norm(weights)):
            weights = next_weights
            break
        weights = next_weights
    return weights
