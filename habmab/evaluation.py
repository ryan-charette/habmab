"""Experiment runner for adaptive HAB sampling replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from habmab.environment import ReplayWindow, make_context_features, make_synthetic_window
from habmab.metrics import hotspot_recall, mae, precision_at_selected, rmse, top_risk_precision
from habmab.policy_factory import make_policy
from habmab.reconstruction import SparseRBFReconstructor


@dataclass(frozen=True)
class ExperimentConfig:
    budget: int = 50
    runs: int = 20
    n_sites: int = 140
    month: int = 8
    seed: int = 7
    policies: tuple[str, ...] = (
        "random",
        "grid",
        "historical_frequency",
        "ucb",
        "thompson",
        "linucb_context",
        "linucb_reconstruction",
    )
    hotspot_weight: float = 0.65
    reconstruction_weight: float = 0.45
    redundancy_penalty: float = 0.18
    observation_noise: float = 0.08

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("Sampling budget must be positive.")
        if self.runs < 1:
            raise ValueError("Runs must be positive.")
        if self.n_sites < 2:
            raise ValueError("At least two candidate sites are required.")


@dataclass
class PolicyTrace:
    policy: str
    selected_sites: np.ndarray
    rewards: np.ndarray
    rmse: np.ndarray
    mae: np.ndarray
    hotspot_recall: np.ndarray
    precision_selected: np.ndarray
    top_risk_precision: np.ndarray
    redundancy_rate: np.ndarray
    final_prediction: np.ndarray


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    window: ReplayWindow
    traces: dict[str, list[PolicyTrace]]

    def aggregate_curves(self) -> dict[str, dict[str, np.ndarray]]:
        output: dict[str, dict[str, np.ndarray]] = {}
        metric_names = ("rmse", "mae", "hotspot_recall", "precision_selected", "top_risk_precision", "redundancy_rate")
        for policy, traces in self.traces.items():
            output[policy] = {}
            for metric in metric_names:
                values = np.vstack([getattr(trace, metric) for trace in traces])
                output[policy][f"{metric}_mean"] = values.mean(axis=0)
                if len(traces) > 1:
                    half_width = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(traces))
                else:
                    half_width = np.zeros(values.shape[1], dtype=float)
                output[policy][f"{metric}_low"] = output[policy][f"{metric}_mean"] - half_width
                output[policy][f"{metric}_high"] = output[policy][f"{metric}_mean"] + half_width
        return output

    def summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        curves = self.aggregate_curves()
        for policy, metrics in curves.items():
            rows.append(
                {
                    "policy": policy,
                    "final_rmse": float(metrics["rmse_mean"][-1]),
                    "final_mae": float(metrics["mae_mean"][-1]),
                    "final_hotspot_recall": float(metrics["hotspot_recall_mean"][-1]),
                    "final_precision_selected": float(metrics["precision_selected_mean"][-1]),
                    "final_top_risk_precision": float(metrics["top_risk_precision_mean"][-1]),
                    "final_redundancy_rate": float(metrics["redundancy_rate_mean"][-1]),
                }
            )
        return sorted(rows, key=lambda row: (row["final_rmse"], -row["final_hotspot_recall"]))

    def representative_trace(self, policy: str | None = None) -> PolicyTrace:
        rows = self.summary_rows()
        chosen = policy or rows[0]["policy"]
        return self.traces[chosen][0]


def _observed_values(sample_counts: np.ndarray, sample_sums: np.ndarray) -> np.ndarray:
    return np.divide(sample_sums, np.maximum(1, sample_counts), out=np.zeros_like(sample_sums), where=sample_counts >= 0)


def run_policy_trace(
    window: ReplayWindow,
    policy_name: str,
    config: ExperimentConfig,
    *,
    seed: int,
) -> PolicyTrace:
    rng = np.random.default_rng(seed)
    policy = make_policy(policy_name)
    reconstructor = SparseRBFReconstructor()

    sample_counts = np.zeros(window.n_sites, dtype=int)
    sample_sums = np.zeros(window.n_sites, dtype=float)
    prediction = np.zeros(window.n_sites, dtype=float)
    uncertainty = np.ones(window.n_sites, dtype=float)
    features = make_context_features(window, sample_counts, sample_sums, prediction, uncertainty)
    policy.reset(
        n_sites=window.n_sites,
        feature_dim=features.shape[1],
        rng=rng,
        coordinates=window.coordinates,
        prior_scores=window.sampling_density,
        oracle_values=window.log_cell_count,
    )

    selected_sites = np.empty(config.budget, dtype=int)
    rewards = np.empty(config.budget, dtype=float)
    rmse_curve = np.empty(config.budget, dtype=float)
    mae_curve = np.empty(config.budget, dtype=float)
    recall_curve = np.empty(config.budget, dtype=float)
    precision_curve = np.empty(config.budget, dtype=float)
    top_precision_curve = np.empty(config.budget, dtype=float)
    redundancy_curve = np.empty(config.budget, dtype=float)

    previous_rmse = rmse(prediction, window.log_cell_count)
    for step in range(config.budget):
        features = make_context_features(window, sample_counts, sample_sums, prediction, uncertainty)
        site_index = policy.select(features, sample_counts)
        duplicate = sample_counts[site_index] > 0
        observation = window.sample(site_index, rng, noise_scale=config.observation_noise)
        sample_counts[site_index] += 1
        sample_sums[site_index] += observation

        observed_mask = sample_counts > 0
        average_values = _observed_values(sample_counts, sample_sums)
        prediction, uncertainty = reconstructor.fit_predict(window.coordinates, observed_mask, average_values)
        current_rmse = rmse(prediction, window.log_cell_count)
        improvement = max(0.0, previous_rmse - current_rmse)
        previous_rmse = current_rmse

        normalized_hotspot_value = window.normalized_target[site_index]
        reward = (
            config.hotspot_weight * normalized_hotspot_value
            + config.reconstruction_weight * improvement
            - (config.redundancy_penalty if duplicate else 0.0)
        )
        reward = float(np.clip(reward, 0.0, 1.0))
        policy.update(site_index, reward, features)

        unique_selected = np.flatnonzero(sample_counts > 0)
        selected_sites[step] = site_index
        rewards[step] = reward
        rmse_curve[step] = current_rmse
        mae_curve[step] = mae(prediction, window.log_cell_count)
        recall_curve[step] = hotspot_recall(unique_selected, window.hotspot_mask)
        precision_curve[step] = precision_at_selected(unique_selected, window.hotspot_mask)
        top_precision_curve[step] = top_risk_precision(prediction, window.hotspot_mask, k=max(1, int(0.1 * window.n_sites)))
        redundancy_curve[step] = 1.0 - (len(unique_selected) / float(step + 1))

    return PolicyTrace(
        policy=policy.name,
        selected_sites=selected_sites,
        rewards=rewards,
        rmse=rmse_curve,
        mae=mae_curve,
        hotspot_recall=recall_curve,
        precision_selected=precision_curve,
        top_risk_precision=top_precision_curve,
        redundancy_rate=redundancy_curve,
        final_prediction=prediction,
    )


def run_experiment(config: ExperimentConfig, window: ReplayWindow | None = None) -> ExperimentResult:
    traces: dict[str, list[PolicyTrace]] = {policy: [] for policy in config.policies}
    first_window = window
    for run in range(config.runs):
        run_window = window or make_synthetic_window(n_sites=config.n_sites, month=config.month, seed=config.seed + run)
        if first_window is None:
            first_window = run_window
        for policy in config.policies:
            trace = run_policy_trace(run_window, policy, config, seed=config.seed + 10_000 * run + len(policy))
            traces[policy].append(trace)
    assert first_window is not None
    return ExperimentResult(config=config, window=first_window, traces=traces)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
