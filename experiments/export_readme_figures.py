"""Export README-friendly SVG figures from the latest HABMAB report CSVs."""

from __future__ import annotations

import csv
from pathlib import Path


PALETTE = {
    "greedy_oracle": "#111827",
    "grid": "#30638e",
    "linucb_reconstruction": "#d1495b",
    "linucb_context": "#6a4c93",
    "historical_frequency": "#2a9d8f",
    "ucb": "#edae49",
    "random": "#7f4f24",
    "thompson": "#4d908e",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def scale(value: float, source_min: float, source_max: float, out_min: float, out_max: float) -> float:
    if abs(source_max - source_min) < 1e-12:
        return 0.5 * (out_min + out_max)
    return out_min + (value - source_min) * (out_max - out_min) / (source_max - source_min)


def line_chart(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    y_label: str,
    output: Path,
    *,
    lower_is_better: bool,
) -> None:
    width, height = 920, 340
    left, right, top, bottom = 58, 24, 36, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    policies = []
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        policy = row["policy"]
        if policy not in grouped:
            policies.append(policy)
            grouped[policy] = []
        grouped[policy].append((int(row["sample"]), float(row[metric])))

    values = [value for points in grouped.values() for _, value in points]
    x_values = [sample for points in grouped.values() for sample, _ in points]
    y_min, y_max = min(values), max(values)
    padding = 0.08 * (y_max - y_min if y_max > y_min else abs(y_max) + 1.0)
    y_min -= padding
    y_max += padding
    x_min, x_max = min(x_values), max(x_values)
    axis_y = top + plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
        "<style>.axis{stroke:#667085}.grid{stroke:#e5eaee}.line{fill:none;stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round}.title{font:700 16px system-ui}.label,.tick,.legend{font:12px system-ui;fill:#475467}</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="23" class="title">{title}</text>',
        f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" class="axis"/>',
    ]
    for fraction in (0.25, 0.5, 0.75):
        gy = top + plot_h * fraction
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" class="grid"/>')
    parts.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" class="label">samples used</text>',
            f'<text x="16" y="{top + plot_h / 2:.1f}" transform="rotate(-90 16 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">{y_label}</text>',
            f'<text x="{left - 8}" y="{axis_y + 4}" text-anchor="end" class="tick">{y_min:.3f}</text>',
            f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" class="tick">{y_max:.3f}</text>',
        ]
    )
    for index, policy in enumerate(policies):
        color = PALETTE.get(policy, "#344054")
        points = sorted(grouped[policy])
        path_points = []
        for sample, value in points:
            x = scale(sample, x_min, x_max, left, left + plot_w)
            y = scale(value, y_min, y_max, axis_y, top)
            path_points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(path_points)}" class="line" stroke="{color}"/>')
        legend_x = left + 10 + (index % 3) * 260
        legend_y = height - 30 - (index // 3) * 18
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" class="line" stroke="{color}"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}" class="legend">{policy}</text>')
    direction = "lower is better" if lower_is_better else "higher is better"
    parts.append(f'<text x="{width - 24}" y="23" text-anchor="end" class="label">{direction}</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def summary_chart(rows: list[dict[str, str]], output: Path) -> None:
    width, height = 920, 380
    left, top = 190, 42
    row_h = 33
    rmse_max = max(float(row["final_rmse"]) for row in rows)
    recall_max = max(float(row["final_hotspot_recall"]) for row in rows)
    bar_w = 250
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Final policy summary">',
        "<style>.title{font:700 16px system-ui}.text,.tick{font:12px system-ui;fill:#475467}.policy{font:600 12px system-ui;fill:#1d2939}</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="24" y="24" class="title">Final policy summary after 40 samples</text>',
        f'<text x="{left}" y="24" class="text">RMSE</text>',
        f'<text x="{left + bar_w + 105}" y="24" class="text">Hotspot recall</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_h
        policy = row["policy"]
        color = PALETTE.get(policy, "#344054")
        rmse = float(row["final_rmse"])
        recall = float(row["final_hotspot_recall"])
        rmse_w = scale(rmse, 0.0, rmse_max, 0.0, bar_w)
        recall_w = scale(recall, 0.0, max(1.0, recall_max), 0.0, bar_w)
        parts.append(f'<text x="24" y="{y + 14}" class="policy">{policy}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{rmse_w:.1f}" height="18" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{left + rmse_w + 6:.1f}" y="{y + 14}" class="tick">{rmse:.3f}</text>')
        rx = left + bar_w + 105
        parts.append(f'<rect x="{rx}" y="{y}" width="{recall_w:.1f}" height="18" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{rx + recall_w + 6:.1f}" y="{y + 14}" class="tick">{recall:.3f}</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    report_dir = Path("reports/latest")
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    curves = read_rows(report_dir / "curves.csv")
    summary = read_rows(report_dir / "summary.csv")
    line_chart(curves, "rmse", "Reconstruction error versus sampling budget", "RMSE", figure_dir / "rmse_curve.svg", lower_is_better=True)
    line_chart(curves, "hotspot_recall", "Hotspot recall versus sampling budget", "hotspot recall", figure_dir / "hotspot_recall_curve.svg", lower_is_better=False)
    summary_chart(summary, figure_dir / "final_policy_summary.svg")
    print(f"Wrote README figures to {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
