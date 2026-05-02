"""Linear contextual UCB policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinUCBPolicy:
    alpha: float = 0.9
    ridge: float = 1.0
    use_reconstruction: bool = False
    policy_label: str | None = None

    def reset(
        self,
        *,
        n_sites: int,
        feature_dim: int,
        rng: np.random.Generator,
        coordinates: np.ndarray,
        prior_scores: np.ndarray,
        oracle_values: np.ndarray | None = None,
    ) -> None:
        del n_sites, rng, coordinates, prior_scores, oracle_values
        self.feature_dim = feature_dim if self.use_reconstruction else max(1, feature_dim - 2)
        self.A = self.ridge * np.eye(self.feature_dim)
        self.b = np.zeros(self.feature_dim, dtype=float)
        self._name = self.policy_label or ("linucb_reconstruction" if self.use_reconstruction else "linucb_context")

    def _features(self, features: np.ndarray) -> np.ndarray:
        if self.use_reconstruction:
            return features
        return features[:, : self.feature_dim]

    def select(self, features: np.ndarray, sample_counts: np.ndarray) -> int:
        del sample_counts
        x = self._features(features)
        inv_a = np.linalg.inv(self.A)
        theta = inv_a @ self.b
        means = x @ theta
        uncertainty = np.sqrt(np.sum((x @ inv_a) * x, axis=1))
        return int(np.argmax(means + self.alpha * uncertainty))

    def update(self, site_index: int, reward: float, features: np.ndarray) -> None:
        x = self._features(features)[site_index]
        self.A += np.outer(x, x)
        self.b += reward * x


    @property
    def name(self) -> str:
        return self._name if hasattr(self, "_name") else ("linucb_reconstruction" if self.use_reconstruction else "linucb_context")


def make_linucb_with_reconstruction(alpha: float = 0.9) -> LinUCBPolicy:
    return LinUCBPolicy(alpha=alpha, use_reconstruction=True)
