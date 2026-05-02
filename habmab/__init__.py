"""HABMAB.

Research-style prototype for adaptive harmful algal bloom monitoring with
contextual bandits and sparse convex reconstruction.
"""

from habmab.environment import ReplayWindow, make_synthetic_window
from habmab.evaluation import ExperimentConfig, run_experiment

__all__ = [
    "ExperimentConfig",
    "ReplayWindow",
    "make_synthetic_window",
    "run_experiment",
]
