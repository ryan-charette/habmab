"""Download HABSOS cell-count observations from the NOAA ArcGIS REST layer.

The script intentionally uses the Python standard library so it can run in a
fresh environment. Network access is optional for the project because the demo
pipeline can also run with the bundled synthetic replay window.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


HABSOS_LAYER_URL = "https://gis.ncdc.noaa.gov/arcgis/rest/services/ms/HABSOS_CellCounts/MapServer/0/query"


def fetch_features(
    *,
    where: str,
    out_fields: str = "*",
    page_size: int = 2000,
    max_records: int | None = None,
) -> list[dict]:
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "f": "json",
            "resultRecordCount": page_size,
            "resultOffset": offset,
        }
        with urlopen(f"{HABSOS_LAYER_URL}?{urlencode(params)}", timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        batch = payload.get("features", [])
        if not batch:
            break
        features.extend(batch)
        offset += len(batch)
        if max_records is not None and len(features) >= max_records:
            return features[:max_records]
        if len(batch) < page_size:
            break
    return features


def write_csv(features: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for feature in features for key in feature.get("attributes", {}).keys()})
    fieldnames.extend(["geometry_x", "geometry_y"])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature in features:
            row = dict(feature.get("attributes", {}))
            geometry = feature.get("geometry", {})
            row["geometry_x"] = geometry.get("x")
            row["geometry_y"] = geometry.get("y")
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch NOAA HABSOS ArcGIS cell-count observations.")
    parser.add_argument("--where", default="UPPER(SPECIES) LIKE '%KARENIA BREVIS%'", help="ArcGIS SQL where clause.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output", default="data/raw/habsos_cell_counts.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features = fetch_features(where=args.where, max_records=args.max_records)
    write_csv(features, Path(args.output))
    print(f"Wrote {len(features)} HABSOS records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
