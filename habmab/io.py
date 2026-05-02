"""Load cleaned HABSOS observations into replay windows."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from habmab.environment import ReplayWindow


def _read_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_replay_window(path: str | Path, *, max_sites: int | None = None, region: str = "HABSOS replay") -> ReplayWindow:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_sites is not None:
        rows = rows[:max_sites]
    if len(rows) < 2:
        raise ValueError("Need at least two cleaned HABSOS rows to create a replay window.")

    n = len(rows)
    month_values = np.array([int(_read_float(row, "month", 1)) for row in rows], dtype=int)
    sampling_density = np.ones(n, dtype=float)
    return ReplayWindow(
        site_id=np.array([int(_read_float(row, "site_id", index)) for index, row in enumerate(rows)], dtype=int),
        latitude=np.array([_read_float(row, "latitude") for row in rows], dtype=float),
        longitude=np.array([_read_float(row, "longitude") for row in rows], dtype=float),
        month=int(np.bincount(np.clip(month_values, 1, 12)).argmax()),
        log_cell_count=np.array([_read_float(row, "log_cell_count") for row in rows], dtype=float),
        water_temperature=np.array([_read_float(row, "water_temperature") for row in rows], dtype=float),
        salinity=np.array([_read_float(row, "salinity") for row in rows], dtype=float),
        wind_speed=np.array([_read_float(row, "wind_speed") for row in rows], dtype=float),
        wind_direction=np.array([_read_float(row, "wind_direction") for row in rows], dtype=float),
        sampling_density=sampling_density,
        region=region,
    )
