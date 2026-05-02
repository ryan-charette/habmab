"""Sparse radial-basis reconstruction solved with FISTA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


@dataclass
class SparseRBFReconstructor:
    """LASSO-style bloom-field reconstruction over spatial basis functions."""

    n_centers: int = 36
    length_scale: float = 0.45
    l1: float = 0.015
    ridge: float = 0.002
    max_iter: int = 250
    tolerance: float = 1e-5

    def _choose_centers(self, coordinates: np.ndarray) -> np.ndarray:
        n = len(coordinates)
        if n <= self.n_centers:
            return coordinates.copy()
        order = np.linspace(0, n - 1, self.n_centers, dtype=int)
        sorted_index = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
        return coordinates[sorted_index[order]]

    def _basis(self, coordinates: np.ndarray, centers: np.ndarray) -> np.ndarray:
        deltas = coordinates[:, None, :] - centers[None, :, :]
        dist2 = np.sum(deltas * deltas, axis=2)
        return np.exp(-dist2 / (2.0 * self.length_scale**2))

    def fit_predict(
        self,
        coordinates: np.ndarray,
        observed_mask: np.ndarray,
        observed_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit sparse basis weights and predict all candidate sites.

        Returns predicted log-cell-count intensity and an uncertainty proxy based
        on distance to the nearest observed sample.
        """

        if not np.any(observed_mask):
            return np.zeros(len(coordinates), dtype=float), np.ones(len(coordinates), dtype=float)

        centers = self._choose_centers(coordinates)
        design_all = self._basis(coordinates, centers)
        design = design_all[observed_mask]
        target = observed_values[observed_mask]

        lipschitz = float(np.linalg.norm(design, ord=2) ** 2 / max(1, len(target)) + self.ridge)
        step = 1.0 / max(lipschitz, 1e-8)
        weights = np.zeros(design.shape[1], dtype=float)
        momentum_weights = weights.copy()
        momentum = 1.0

        for _ in range(self.max_iter):
            residual = design @ momentum_weights - target
            gradient = design.T @ residual / max(1, len(target)) + self.ridge * momentum_weights
            next_weights = soft_threshold(momentum_weights - step * gradient, self.l1 * step)
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            momentum_weights = next_weights + ((momentum - 1.0) / next_momentum) * (next_weights - weights)
            if np.linalg.norm(next_weights - weights) <= self.tolerance * max(1.0, np.linalg.norm(weights)):
                weights = next_weights
                break
            weights = next_weights
            momentum = next_momentum

        prediction = np.maximum(0.0, design_all @ weights)
        uncertainty = self.uncertainty(coordinates, observed_mask)
        return prediction, uncertainty

    def uncertainty(self, coordinates: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        if not np.any(observed_mask):
            return np.ones(len(coordinates), dtype=float)
        observed_coords = coordinates[observed_mask]
        deltas = coordinates[:, None, :] - observed_coords[None, :, :]
        distance = np.sqrt(np.sum(deltas * deltas, axis=2)).min(axis=1)
        uncertainty = 1.0 - np.exp(-distance / max(self.length_scale, 1e-8))
        uncertainty[observed_mask] *= 0.35
        return uncertainty
