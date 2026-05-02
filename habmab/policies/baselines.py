"""Non-adaptive and oracle sampling baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RandomPolicy:
    name: str = "random"

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
        self.n_sites = n_sites
        self.rng = rng

    def select(self, features: np.ndarray, sample_counts: np.ndarray) -> int:
        del features
        candidates = np.flatnonzero(sample_counts == 0)
        if not candidates.size:
            candidates = np.arange(self.n_sites)
        return int(self.rng.choice(candidates))

    def update(self, site_index: int, reward: float, features: np.ndarray) -> None:
        del site_index, reward, features


@dataclass
class GridPolicy:
    """Farthest-first spatial coverage baseline."""

    name: str = "grid"

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
        del feature_dim, rng, prior_scores, oracle_values
        self.n_sites = n_sites
        self.coordinates = coordinates
        self.selected: list[int] = []

    def select(self, features: np.ndarray, sample_counts: np.ndarray) -> int:
        del features
        candidates = np.flatnonzero(sample_counts == 0)
        if not candidates.size:
            return int(np.argmin(sample_counts))
        if not self.selected:
            center = self.coordinates.mean(axis=0)
            distance = np.sum((self.coordinates[candidates] - center) ** 2, axis=1)
            return int(candidates[np.argmin(distance)])
        selected_coords = self.coordinates[np.array(self.selected)]
        deltas = self.coordinates[candidates, None, :] - selected_coords[None, :, :]
        nearest_distance = np.sqrt(np.sum(deltas * deltas, axis=2)).min(axis=1)
        return int(candidates[np.argmax(nearest_distance)])

    def update(self, site_index: int, reward: float, features: np.ndarray) -> None:
        del reward, features
        if site_index not in self.selected:
            self.selected.append(site_index)


@dataclass
class HistoricalFrequencyPolicy:
    """Samples sites that historically receive more monitoring attention."""

    name: str = "historical_frequency"

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
        del n_sites, feature_dim, rng, coordinates, oracle_values
        self.order = list(np.argsort(-prior_scores))

    def select(self, features: np.ndarray, sample_counts: np.ndarray) -> int:
        del features
        for site_index in self.order:
            if sample_counts[site_index] == 0:
                return int(site_index)
        return int(self.order[0])

    def update(self, site_index: int, reward: float, features: np.ndarray) -> None:
        del site_index, reward, features


@dataclass
class GreedyHotspotOraclePolicy:
    """Upper-bound baseline that knows the hidden field."""

    name: str = "greedy_oracle"

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
        del n_sites, feature_dim, rng, coordinates, prior_scores
        if oracle_values is None:
            raise ValueError("GreedyHotspotOraclePolicy requires oracle_values.")
        self.order = list(np.argsort(-oracle_values))

    def select(self, features: np.ndarray, sample_counts: np.ndarray) -> int:
        del features
        for site_index in self.order:
            if sample_counts[site_index] == 0:
                return int(site_index)
        return int(self.order[0])

    def update(self, site_index: int, reward: float, features: np.ndarray) -> None:
        del site_index, reward, features
