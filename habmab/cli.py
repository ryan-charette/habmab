"""Command-line entry point for HABMAB experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from habmab.evaluation import ExperimentConfig, run_experiment
from habmab.io import load_replay_window
from habmab.policy_factory import available_policies, parse_policy_list
from habmab.visualization import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HABMAB replay experiments.")
    parser.add_argument("--data", default=None, help="Optional cleaned HABSOS CSV. If omitted, a synthetic HABSOS-like replay is used.")
    parser.add_argument("--budget", type=int, default=50, help="Number of samples the monitoring team can collect.")
    parser.add_argument("--runs", type=int, default=20, help="Monte Carlo replay runs.")
    parser.add_argument("--sites", type=int, default=140, help="Candidate sites for synthetic replay or max loaded sites.")
    parser.add_argument("--month", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policies", default="all", help=f"Comma-separated policies or all. Available: {', '.join(available_policies())}.")
    parser.add_argument("--output", default="reports/latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policies = parse_policy_list(args.policies)
    config = ExperimentConfig(
        budget=args.budget,
        runs=args.runs,
        n_sites=args.sites,
        month=args.month,
        seed=args.seed,
        policies=policies,
    )
    window = load_replay_window(args.data, max_sites=args.sites) if args.data else None
    result = run_experiment(config, window=window)
    paths = write_report(result, Path(args.output))

    print("HABMAB experiment complete.")
    print(f"Samples: {config.budget}; runs: {config.runs}; candidate sites: {result.window.n_sites}")
    print("Final policy ranking by reconstruction RMSE:")
    for rank, row in enumerate(result.summary_rows(), start=1):
        print(
            f"  {rank}. {row['policy']}: "
            f"RMSE={row['final_rmse']:.3f}, "
            f"hotspot_recall={row['final_hotspot_recall']:.3f}, "
            f"top_risk_precision={row['final_top_risk_precision']:.3f}"
        )
    print(f"HTML report: {paths['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
