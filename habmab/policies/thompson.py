"""Bounded-reward Thompson Sampling policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThompsonSamplingPolicy:
    prior_strength: float = 1.0
    name: str = "thompson"

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
        del feature_dim, coordinates, prior_scores, oracle_values
        self.rng = rng
        self.alpha = np.full(n_sites, self.prior_strength, dtype=float)
        self.beta = np.full(n_sites, self.prior_strength, dtype=float)

    def select(self, features: np.ndarray, sample_counts: np.ndarray) -> int:
        del features, sample_counts
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, site_index: int, reward: float, features: np.ndarray) -> None:
        del features
        bounded = float(np.clip(reward, 0.0, 1.0))
        self.alpha[site_index] += bounded
        self.beta[site_index] += 1.0 - bounded
