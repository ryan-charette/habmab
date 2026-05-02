"""Prepare downloaded HABSOS observations for replay experiments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FIELD_ALIASES = {
    "latitude": ("LATITUDE", "Latitude", "latitude", "geometry_y", "Y"),
    "longitude": ("LONGITUDE", "Longitude", "longitude", "geometry_x", "X"),
    "cell_count": ("CELLCOUNT", "CELL_COUNT", "CellCount", "cell_count", "COUNT_"),
    "species": ("SPECIES", "Species", "species"),
    "sample_date": ("SAMPLE_DATE", "SAMPLEDATE", "Date", "sample_date", "COLLECT_DATE"),
    "water_temperature": ("WATER_TEMP", "WATERTEMP", "TEMP", "Temperature", "water_temperature"),
    "salinity": ("SALINITY", "Salinity", "salinity"),
    "wind_speed": ("WIND_SPEED", "WindSpeed", "wind_speed"),
    "wind_direction": ("WIND_DIR", "WIND_DIRECTION", "WindDirection", "wind_direction"),
}


def _first(row: dict[str, Any], aliases: tuple[str, ...], default: str = "") -> str:
    for alias in aliases:
        value = row.get(alias)
        if value not in (None, ""):
            return str(value)
    return default


def _float(value: str, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _month(value: str) -> int:
    if not value:
        return 1
    try:
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        return datetime.fromtimestamp(number, tz=timezone.utc).month
    except (TypeError, ValueError, OSError):
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value[:10], fmt).month
        except ValueError:
            continue
    return 1


def preprocess(input_path: Path, output_path: Path, *, species_filter: str = "karenia brevis") -> int:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        species = _first(row, FIELD_ALIASES["species"]).lower()
        if species_filter and species_filter.lower() not in species:
            continue
        lat = _float(_first(row, FIELD_ALIASES["latitude"]))
        lon = _float(_first(row, FIELD_ALIASES["longitude"]))
        cell_count = max(0.0, _float(_first(row, FIELD_ALIASES["cell_count"]), 0.0))
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        clean_rows.append(
            {
                "site_id": len(clean_rows),
                "latitude": lat,
                "longitude": lon,
                "month": _month(_first(row, FIELD_ALIASES["sample_date"])),
                "cell_count": cell_count,
                "log_cell_count": float(np.log1p(cell_count)),
                "water_temperature": _float(_first(row, FIELD_ALIASES["water_temperature"]), np.nan),
                "salinity": _float(_first(row, FIELD_ALIASES["salinity"]), np.nan),
                "wind_speed": _float(_first(row, FIELD_ALIASES["wind_speed"]), np.nan),
                "wind_direction": _float(_first(row, FIELD_ALIASES["wind_direction"]), 0.0),
                "species": species or species_filter,
            }
        )

    for field in ("water_temperature", "salinity", "wind_speed"):
        values = np.array([row[field] for row in clean_rows], dtype=float)
        fill = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else 0.0
        for row in clean_rows:
            if not np.isfinite(row[field]):
                row[field] = fill

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if clean_rows:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(clean_rows[0].keys()))
            writer.writeheader()
            writer.writerows(clean_rows)
    return len(clean_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean HABSOS CSV records for HABMAB.")
    parser.add_argument("input")
    parser.add_argument("--output", default="data/processed/habsos_clean.csv")
    parser.add_argument("--species", default="karenia brevis")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    count = preprocess(Path(args.input), Path(args.output), species_filter=args.species)
    print(f"Wrote {count} cleaned observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
