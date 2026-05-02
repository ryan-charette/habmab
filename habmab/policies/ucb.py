"""Non-contextual UCB policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UCBPolicy:
    alpha: float = 1.5
    name: str = "ucb"

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
        del feature_dim, rng, coordinates, prior_scores, oracle_values
        self.n_sites = n_sites
        self.counts = np.zeros(n_sites, dtype=int)
        self.values = np.zeros(n_sites, dtype=float)
        self.t = 0

    def select(self, features: np.ndarray, sample_counts: np.ndarray) -> int:
        del features, sample_counts
        unplayed = np.flatnonzero(self.counts == 0)
        if unplayed.size:
            return int(unplayed[0])
        bonus = np.sqrt(self.alpha * np.log(max(2, self.t + 1)) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, site_index: int, reward: float, features: np.ndarray) -> None:
        del features
        self.t += 1
        self.counts[site_index] += 1
        step = 1.0 / self.counts[site_index]
        self.values[site_index] += step * (reward - self.values[site_index])
