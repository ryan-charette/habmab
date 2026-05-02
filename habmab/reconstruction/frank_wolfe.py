"""Frank-Wolfe solver for an L1-constrained least-squares reconstruction."""

from __future__ import annotations

import numpy as np


def frank_wolfe_lasso(
    design: np.ndarray,
    target: np.ndarray,
    *,
    radius: float = 1.0,
    max_iter: int = 200,
) -> np.ndarray:
    """Solve min 0.5 ||A w - y||^2 subject to ||w||_1 <= radius."""

    weights = np.zeros(design.shape[1], dtype=float)
    n = max(1, len(target))
    for iteration in range(max_iter):
        gradient = design.T @ (design @ weights - target) / n
        vertex = np.zeros_like(weights)
        coordinate = int(np.argmax(np.abs(gradient)))
        vertex[coordinate] = -radius * np.sign(gradient[coordinate])
        step = 2.0 / (iteration + 2.0)
        weights = (1.0 - step) * weights + step * vertex
    return weights
