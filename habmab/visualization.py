"""Static HTML visualizations for HABMAB."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np

from habmab.evaluation import ExperimentResult, PolicyTrace, write_csv


PALETTE = ("#176f7a", "#d1495b", "#edae49", "#30638e", "#6a4c93", "#2a9d8f", "#7f4f24", "#4d908e")


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _scale(values: np.ndarray, low: float, high: float, out_low: float, out_high: float) -> np.ndarray:
    if np.isclose(low, high):
        return np.full_like(values, (out_low + out_high) / 2.0, dtype=float)
    return out_low + (values - low) * (out_high - out_low) / (high - low)


def _color(value: float, low: float, high: float) -> str:
    fraction = 0.0 if np.isclose(low, high) else float(np.clip((value - low) / (high - low), 0.0, 1.0))
    r = int(34 + 210 * fraction)
    g = int(72 + 92 * (1.0 - abs(fraction - 0.45)))
    b = int(112 - 72 * fraction)
    return f"rgb({r},{g},{max(32, b)})"


def _map_panel(
    title: str,
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    *,
    selected: np.ndarray | None = None,
    width: int = 300,
    height: int = 270,
) -> str:
    pad = 24
    x = _scale(lon, float(lon.min()), float(lon.max()), pad, width - pad)
    y = _scale(lat, float(lat.min()), float(lat.max()), height - pad, pad + 20)
    low, high = float(values.min()), float(values.max())
    selected_set = set(int(index) for index in selected) if selected is not None else set()

    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{pad}" y="20" class="chart-title">{html.escape(title)}</text>',
        f'<rect x="{pad}" y="{pad + 8}" width="{width - 2 * pad}" height="{height - 2 * pad - 8}" rx="4" class="map-frame"/>',
    ]
    for index in np.argsort(values):
        radius = 3.0 + 5.0 * (values[index] - low) / max(1e-9, high - low)
        elements.append(
            f'<circle cx="{x[index]:.2f}" cy="{y[index]:.2f}" r="{radius:.2f}" fill="{_color(float(values[index]), low, high)}" opacity="0.78"/>'
        )
    if selected is not None:
        for rank, index in enumerate(selected[: min(25, len(selected))], start=1):
            elements.append(
                f'<circle cx="{x[index]:.2f}" cy="{y[index]:.2f}" r="7" fill="none" stroke="#111827" stroke-width="1.6"/>'
            )
            if rank <= 10:
                elements.append(f'<text x="{x[index] + 7:.2f}" y="{y[index] - 5:.2f}" class="rank">{rank}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def _curve_chart(
    title: str,
    curves: dict[str, np.ndarray],
    *,
    y_label: str,
    width: int = 920,
    height: int = 340,
) -> str:
    margin_left, margin_right, margin_top, margin_bottom = 58, 18, 34, 46
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_all = np.concatenate(list(curves.values()))
    y_min, y_max = float(y_all.min()), float(y_all.max())
    padding = 0.08 * (y_max - y_min if y_max > y_min else abs(y_max) + 1.0)
    y_min -= padding
    y_max += padding
    axis_y = margin_top + plot_h
    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{margin_left}" y="22" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{axis_y}" x2="{margin_left + plot_w}" y2="{axis_y}" class="axis"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_y}" class="axis"/>',
        f'<text x="14" y="{margin_top + plot_h / 2:.0f}" class="axis-label rotated">{html.escape(y_label)}</text>',
        f'<text x="{margin_left + plot_w / 2:.0f}" y="{height - 10}" class="axis-label">samples used</text>',
        f'<text x="{margin_left - 8}" y="{axis_y + 4}" text-anchor="end" class="tick">{_fmt(y_min)}</text>',
        f'<text x="{margin_left - 8}" y="{margin_top + 4}" text-anchor="end" class="tick">{_fmt(y_max)}</text>',
    ]
    for index, (name, values) in enumerate(curves.items()):
        color = PALETTE[index % len(PALETTE)]
        x_values = np.arange(1, len(values) + 1, dtype=float)
        x = _scale(x_values, 1.0, float(len(values)), margin_left, margin_left + plot_w)
        y = _scale(values, y_min, y_max, axis_y, margin_top)
        points = " ".join(f"{px:.2f},{py:.2f}" for px, py in zip(x, y))
        elements.append(f'<polyline points="{points}" class="line" stroke="{color}"/>')
        legend_x = margin_left + 12 + (index % 3) * 245
        legend_y = height - 28 - (index // 3) * 18
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" class="line"/>')
        elements.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}" class="legend">{html.escape(name)}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def render_html_report(result: ExperimentResult) -> str:
    curves = result.aggregate_curves()
    summary = result.summary_rows()
    hero_policy = "linucb_reconstruction" if "linucb_reconstruction" in result.traces else summary[0]["policy"]
    trace = result.representative_trace(hero_policy)
    window = result.window
    rmse_curves = {policy: metrics["rmse_mean"] for policy, metrics in curves.items()}
    recall_curves = {policy: metrics["hotspot_recall_mean"] for policy, metrics in curves.items()}
    hero = "\n".join(
        [
            '<div class="hero-grid">',
            _map_panel("A. Hidden HABSOS-like observations", window.latitude, window.longitude, window.log_cell_count),
            _map_panel("B. Adaptive sampling sequence", window.latitude, window.longitude, window.log_cell_count, selected=trace.selected_sites),
            _map_panel("C. Sparse reconstructed field", window.latitude, window.longitude, trace.final_prediction),
            "</div>",
        ]
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['policy'])}</td>"
        f"<td>{_fmt(row['final_rmse'])}</td>"
        f"<td>{_fmt(row['final_hotspot_recall'])}</td>"
        f"<td>{_fmt(row['final_precision_selected'])}</td>"
        f"<td>{_fmt(row['final_top_risk_precision'])}</td>"
        f"<td>{_fmt(row['final_redundancy_rate'])}</td>"
        "</tr>"
        for row in summary
    )
    notes = [
        "This is a research prototype for sampling-strategy evaluation, not an operational public-health tool.",
        "The default demo uses a synthetic HABSOS-like replay window so the project runs offline; the data scripts can fetch and preprocess NOAA HABSOS records when network access is available.",
        "LinUCB with reconstruction feedback uses uncertainty from the convex sparse field model as part of its context.",
        "Hotspots are defined by the top 10% of log cell-count values within the replay window, avoiding overclaiming a public-health threshold.",
    ]
    note_items = "\n".join(f"<li>{html.escape(note)}</li>" for note in notes)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HABMAB Report</title>
<style>
body {{ margin: 32px; background: #f5f7f8; color: #172026; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
main {{ max-width: 1080px; margin: 0 auto; }}
h1 {{ margin: 0 0 6px; font-size: 34px; }}
h2 {{ font-size: 20px; margin: 28px 0 12px; }}
p {{ line-height: 1.5; }}
.subtle {{ color: #5f6f7a; }}
section {{ background: white; border: 1px solid #dbe4e8; border-radius: 8px; padding: 18px; margin: 18px 0; }}
.hero-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
.map-frame {{ fill: #eef4f6; stroke: #bdcbd2; }}
.chart-title {{ font-size: 14px; font-weight: 700; fill: #172026; }}
.rank {{ font-size: 10px; font-weight: 700; fill: #111827; }}
.axis {{ stroke: #7b8a93; stroke-width: 1; }}
.line {{ fill: none; stroke-width: 2.3; stroke-linecap: round; stroke-linejoin: round; }}
.axis-label, .legend, .tick {{ font-size: 12px; fill: #53626c; }}
.rotated {{ transform: rotate(-90deg); transform-origin: 14px center; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #e3eaee; padding: 9px 10px; text-align: left; font-size: 14px; }}
th {{ background: #f0f4f6; color: #2c3a42; }}
li {{ margin: 7px 0; line-height: 1.45; }}
@media (max-width: 900px) {{ .hero-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<h1>HABMAB</h1>
<p class="subtle">Harmful Algal Bloom Multi-Armed Bandit: adaptive sampling with contextual bandits and sparse convex reconstruction.</p>
<section>
<h2>Portfolio Hero Figure</h2>
{hero}
<p class="subtle">Panel B shows <strong>{html.escape(hero_policy)}</strong>, the proposed adaptive sampler. It concentrates measurements near high-cell-count regions while maintaining enough exploration to reconstruct the broader bloom field.</p>
</section>
<section>
<h2>Policy Summary</h2>
<table>
<thead><tr><th>Policy</th><th>RMSE</th><th>Hotspot recall</th><th>Selected precision</th><th>Top-risk precision</th><th>Redundancy</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</section>
<section>{_curve_chart("Reconstruction quality versus sampling budget", rmse_curves, y_label="RMSE")}</section>
<section>{_curve_chart("Hotspot discovery versus sampling budget", recall_curves, y_label="hotspot recall")}</section>
<section>
<h2>Responsible Framing</h2>
<ul>{note_items}</ul>
</section>
</main>
</body>
</html>
"""


def write_report(result: ExperimentResult, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.csv"
    write_csv(summary_path, result.summary_rows())

    curve_rows: list[dict[str, Any]] = []
    for policy, metrics in result.aggregate_curves().items():
        for step in range(result.config.budget):
            curve_rows.append(
                {
                    "policy": policy,
                    "sample": step + 1,
                    "rmse": float(metrics["rmse_mean"][step]),
                    "hotspot_recall": float(metrics["hotspot_recall_mean"][step]),
                    "top_risk_precision": float(metrics["top_risk_precision_mean"][step]),
                    "redundancy_rate": float(metrics["redundancy_rate_mean"][step]),
                }
            )
    curves_path = output / "curves.csv"
    write_csv(curves_path, curve_rows)

    best_trace = result.representative_trace()
    sequence_rows = [
        {
            "rank": rank,
            "site_index": int(site),
            "latitude": float(result.window.latitude[site]),
            "longitude": float(result.window.longitude[site]),
            "log_cell_count": float(result.window.log_cell_count[site]),
        }
        for rank, site in enumerate(best_trace.selected_sites, start=1)
    ]
    sequence_path = output / "sample_sequence.csv"
    write_csv(sequence_path, sequence_rows)

    html_path = output / "habmab_report.html"
    html_path.write_text(render_html_report(result), encoding="utf-8")
    return {"summary": summary_path, "curves": curves_path, "sample_sequence": sequence_path, "html": html_path}
