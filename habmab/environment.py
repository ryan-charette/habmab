"""Historical replay environment for adaptive HAB monitoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayWindow:
    """Candidate sampling sites and hidden bloom intensity for one replay task."""

    site_id: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    month: int
    log_cell_count: np.ndarray
    water_temperature: np.ndarray
    salinity: np.ndarray
    wind_speed: np.ndarray
    wind_direction: np.ndarray
    sampling_density: np.ndarray
    species: str = "Karenia brevis"
    region: str = "Florida Gulf Coast"

    def __post_init__(self) -> None:
        n = len(self.site_id)
        arrays = [
            self.latitude,
            self.longitude,
            self.log_cell_count,
            self.water_temperature,
            self.salinity,
            self.wind_speed,
            self.wind_direction,
            self.sampling_density,
        ]
        if any(len(values) != n for values in arrays):
            raise ValueError("All replay window arrays must have the same length.")
        if n < 2:
            raise ValueError("Replay window requires at least two candidate sites.")

    @property
    def n_sites(self) -> int:
        return int(len(self.site_id))

    @property
    def coordinates(self) -> np.ndarray:
        return np.column_stack([self.latitude, self.longitude])

    @property
    def hotspot_threshold(self) -> float:
        return float(np.quantile(self.log_cell_count, 0.9))

    @property
    def hotspot_mask(self) -> np.ndarray:
        return self.log_cell_count >= self.hotspot_threshold

    @property
    def normalized_target(self) -> np.ndarray:
        values = self.log_cell_count
        span = float(values.max() - values.min())
        if span <= 1e-12:
            return np.zeros_like(values)
        return (values - values.min()) / span

    def sample(self, site_index: int, rng: np.random.Generator, noise_scale: float = 0.08) -> float:
        """Reveal a noisy log-cell-count observation at a candidate site."""

        noise = rng.normal(0.0, noise_scale)
        return float(max(0.0, self.log_cell_count[site_index] + noise))


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(np.nanmax(values) - np.nanmin(values))
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - np.nanmin(values)) / span


def make_context_features(
    window: ReplayWindow,
    sample_counts: np.ndarray,
    sample_sums: np.ndarray,
    reconstruction_mean: np.ndarray,
    reconstruction_uncertainty: np.ndarray,
) -> np.ndarray:
    """Build site-level context features for contextual policies."""

    observed = sample_counts > 0
    average_observed = np.divide(
        sample_sums,
        np.maximum(1, sample_counts),
        out=np.zeros_like(sample_sums, dtype=float),
        where=np.maximum(1, sample_counts) > 0,
    )
    if np.any(observed):
        observed_values = average_observed[observed]
        high_cutoff = np.quantile(observed_values, 0.75)
        high_points = window.coordinates[observed][observed_values >= high_cutoff]
    else:
        high_points = np.empty((0, 2), dtype=float)

    if len(high_points):
        deltas = window.coordinates[:, None, :] - high_points[None, :, :]
        distance_to_recent_high = np.sqrt(np.sum(deltas * deltas, axis=2)).min(axis=1)
    else:
        distance_to_recent_high = np.full(window.n_sites, np.sqrt(2.0), dtype=float)

    month_angle = 2.0 * np.pi * (window.month - 1) / 12.0
    wind_angle = np.deg2rad(window.wind_direction)

    features = np.column_stack(
        [
            _normalize(window.latitude),
            _normalize(window.longitude),
            np.full(window.n_sites, np.sin(month_angle)),
            np.full(window.n_sites, np.cos(month_angle)),
            _normalize(window.water_temperature),
            _normalize(window.salinity),
            _normalize(window.wind_speed),
            np.sin(wind_angle),
            np.cos(wind_angle),
            1.0 - _normalize(distance_to_recent_high),
            _normalize(window.sampling_density),
            np.log1p(sample_counts) / np.log1p(max(1, int(sample_counts.max()))),
            _normalize(reconstruction_mean),
            _normalize(reconstruction_uncertainty),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def make_synthetic_window(
    *,
    n_sites: int = 140,
    month: int = 8,
    seed: int = 7,
    region: str = "Florida Gulf Coast",
) -> ReplayWindow:
    """Create a realistic HABSOS-like replay task for offline demos and tests."""

    rng = np.random.default_rng(seed)
    # Approximate Florida Gulf Coast bounding box.
    latitude = rng.uniform(25.0, 30.4, size=n_sites)
    longitude = rng.uniform(-85.8, -81.0, size=n_sites)
    coast_curve = 27.8 + 0.6 * np.sin((longitude + 84.0) * 1.5)
    latitude = 0.72 * latitude + 0.28 * coast_curve + rng.normal(0.0, 0.12, size=n_sites)

    centers = np.array(
        [
            [26.55, -82.85],
            [27.65, -82.65],
            [29.15, -83.45],
        ]
    )
    amplitudes = np.array([7.8, 5.4, 4.2])
    length_scales = np.array([0.42, 0.55, 0.70])
    coords = np.column_stack([latitude, longitude])
    field = np.zeros(n_sites, dtype=float)
    for center, amplitude, length_scale in zip(centers, amplitudes, length_scales):
        dist2 = np.sum((coords - center) ** 2, axis=1)
        field += amplitude * np.exp(-dist2 / (2.0 * length_scale**2))
    seasonal_boost = 1.0 + 0.25 * np.cos(2.0 * np.pi * (month - 9) / 12.0)
    log_cell_count = np.maximum(0.0, seasonal_boost * field + rng.gamma(0.8, 0.25, size=n_sites))

    water_temperature = 24.0 + 5.0 * seasonal_boost + 0.18 * log_cell_count + rng.normal(0.0, 0.7, size=n_sites)
    salinity = 33.0 + 1.2 * np.cos((longitude + 83.0) * 2.0) - 0.08 * log_cell_count + rng.normal(0.0, 0.5, size=n_sites)
    wind_speed = 4.0 + rng.gamma(2.0, 1.1, size=n_sites)
    wind_direction = (210.0 + 35.0 * np.sin(latitude) + rng.normal(0.0, 25.0, size=n_sites)) % 360.0
    sampling_density = 0.2 + 0.8 * _normalize(
        np.exp(-((latitude - 27.8) ** 2 + (longitude + 82.8) ** 2) / 2.0) + 0.25 * rng.random(n_sites)
    )

    return ReplayWindow(
        site_id=np.arange(n_sites),
        latitude=latitude,
        longitude=longitude,
        month=month,
        log_cell_count=log_cell_count,
        water_temperature=water_temperature,
        salinity=salinity,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        sampling_density=sampling_density,
        region=region,
    )
