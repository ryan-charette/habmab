"""Factory for named sampling policies."""

from __future__ import annotations

from habmab.policies.baselines import GridPolicy, GreedyHotspotOraclePolicy, HistoricalFrequencyPolicy, RandomPolicy
from habmab.policies.linucb import LinUCBPolicy
from habmab.policies.thompson import ThompsonSamplingPolicy
from habmab.policies.ucb import UCBPolicy


def available_policies() -> tuple[str, ...]:
    return (
        "random",
        "grid",
        "historical_frequency",
        "ucb",
        "thompson",
        "linucb_context",
        "linucb_reconstruction",
        "greedy_oracle",
    )


def make_policy(name: str):
    key = name.strip().lower().replace("-", "_")
    if key == "random":
        return RandomPolicy()
    if key in {"grid", "uniform_grid"}:
        return GridPolicy()
    if key in {"historical", "historical_frequency", "frequency"}:
        return HistoricalFrequencyPolicy()
    if key == "ucb":
        return UCBPolicy()
    if key in {"thompson", "thompson_sampling"}:
        return ThompsonSamplingPolicy()
    if key in {"linucb", "linucb_context"}:
        return LinUCBPolicy(use_reconstruction=False)
    if key in {"linucb_reconstruction", "linucb_convex", "full_system"}:
        return LinUCBPolicy(use_reconstruction=True)
    if key in {"greedy", "greedy_oracle", "oracle"}:
        return GreedyHotspotOraclePolicy()
    raise ValueError(f"Unknown policy: {name}")


def parse_policy_list(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return (
            "random",
            "grid",
            "historical_frequency",
            "ucb",
            "thompson",
            "linucb_context",
            "linucb_reconstruction",
            "greedy_oracle",
        )
    policies = tuple(part.strip().lower().replace("-", "_") for part in raw.split(",") if part.strip())
    if not policies:
        raise ValueError("At least one policy is required.")
    return policies
