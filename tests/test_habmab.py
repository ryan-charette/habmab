from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import numpy as np

from habmab.environment import make_context_features, make_synthetic_window
from habmab.evaluation import ExperimentConfig, run_experiment
from habmab.policy_factory import parse_policy_list
from habmab.reconstruction import SparseRBFReconstructor, frank_wolfe_lasso, huber_irls
from habmab.visualization import write_report


class AdaptivehabmabTests(unittest.TestCase):
    def test_synthetic_window_and_features(self) -> None:
        window = make_synthetic_window(n_sites=40, seed=3)
        counts = np.zeros(window.n_sites, dtype=int)
        sums = np.zeros(window.n_sites, dtype=float)
        prediction = np.zeros(window.n_sites, dtype=float)
        uncertainty = np.ones(window.n_sites, dtype=float)
        features = make_context_features(window, counts, sums, prediction, uncertainty)

        self.assertEqual(window.n_sites, 40)
        self.assertEqual(features.shape, (40, 14))
        self.assertTrue(np.any(window.hotspot_mask))

    def test_sparse_reconstruction_improves_after_observations(self) -> None:
        window = make_synthetic_window(n_sites=60, seed=4)
        observed = np.zeros(window.n_sites, dtype=bool)
        observed[np.argsort(-window.log_cell_count)[:12]] = True
        reconstructor = SparseRBFReconstructor(n_centers=20, max_iter=80)
        prediction, uncertainty = reconstructor.fit_predict(window.coordinates, observed, window.log_cell_count)

        self.assertEqual(prediction.shape, (60,))
        self.assertEqual(uncertainty.shape, (60,))
        self.assertLess(np.sqrt(np.mean((prediction - window.log_cell_count) ** 2)), np.sqrt(np.mean(window.log_cell_count**2)))

    def test_convex_helpers_return_coefficients(self) -> None:
        rng = np.random.default_rng(5)
        design = rng.normal(size=(30, 6))
        target = design[:, 2] * 0.7 + rng.normal(scale=0.01, size=30)
        fw = frank_wolfe_lasso(design, target, radius=1.0, max_iter=20)
        huber = huber_irls(design, target, max_iter=8)

        self.assertEqual(fw.shape, (6,))
        self.assertEqual(huber.shape, (6,))

    def test_experiment_runs_all_core_policies(self) -> None:
        config = ExperimentConfig(
            budget=12,
            runs=2,
            n_sites=45,
            policies=("random", "grid", "ucb", "thompson", "linucb_context", "linucb_reconstruction"),
            seed=9,
        )
        result = run_experiment(config)
        rows = result.summary_rows()

        self.assertEqual(len(rows), 6)
        self.assertEqual(set(result.traces), set(config.policies))
        for policy in config.policies:
            self.assertEqual(result.traces[policy][0].rmse.shape, (12,))

    def test_all_policy_parser_and_report(self) -> None:
        policies = parse_policy_list("all")
        self.assertIn("linucb_reconstruction", policies)
        output = Path.cwd() / ".habmab-test-report"
        if output.exists():
            shutil.rmtree(output)
        config = ExperimentConfig(budget=8, runs=1, n_sites=35, policies=("random", "linucb_reconstruction"), seed=10)
        result = run_experiment(config)
        paths = write_report(result, output)

        self.assertTrue(paths["html"].exists())
        self.assertTrue(paths["summary"].exists())
        self.assertTrue(paths["curves"].exists())
        self.assertTrue(paths["sample_sequence"].exists())
        shutil.rmtree(output)


if __name__ == "__main__":
    unittest.main()
