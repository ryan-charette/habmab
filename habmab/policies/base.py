"""Policy interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SamplingPolicy(Protocol):
    name: str

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
        ...

    def select(self, features: np.ndarray, sample_counts: np.ndarray) -> int:
        ...

    def update(self, site_index: int, reward: float, features: np.ndarray) -> None:
        ...


def unsampled_mask(sample_counts: np.ndarray) -> np.ndarray:
    return sample_counts == 0


def first_unsampled(sample_counts: np.ndarray) -> int:
    candidates = np.flatnonzero(unsampled_mask(sample_counts))
    if candidates.size:
        return int(candidates[0])
    return int(np.argmin(sample_counts))
