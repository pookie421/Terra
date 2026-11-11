#!/usr/bin/env python3
"""Apply current cited manual COOLR point labels to matching feature rows only."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.build_real_labeled_dataset import event_candidates, fetch_nasa_coolr_events, fetch_nasa_coolr_polygons, _write_event_group_holdout


async def refresh(path: Path, max_events: int, timeout_s: float) -> Dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not path.exists():
        raise FileNotFoundError(path)
    old_table = pq.read_table(path)
    rows = old_table.to_pylist()
    by_id = {str(row.get("event_id")): index for index, row in enumerate(rows) if row.get("event_id")}
    headers = {"User-Agent": "TERRA-SHIELD/0.3 (open-data research prototype)", "Accept": "application/json"}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s), headers=headers,
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=8), follow_redirects=True,
    ) as client:
        point_rows, polygon_rows = await asyncio.gather(
            fetch_nasa_coolr_events(client, max_events),
            fetch_nasa_coolr_polygons(client, max_events),
        )

    candidates = event_candidates([], point_rows, polygon_rows, [], [], max_events)
    matched_positive = matched_negative = 0
    matched_groups = set()
    for event in candidates:
        if event.event_record_type == "cited_manual_point_buffer_cell" and event.label == 1:
            index = by_id.get(event.event_id)
            if index is not None:
                row = rows[index]
                row.update({
                    "label": 1,
                    "label_source": event.label_source,
                    "label_quality": event.label_quality,
                    "label_note": "Existing event-time features matched to a cited manual NASA COOLR point within 300 m; point is an approximate initiation-zone proxy, not a crown polygon.",
                    "event_record_type": event.event_record_type,
                    "event_group_id": event.event_group_id,
                    "mapping_method": event.mapping_method,
                    "citation": event.citation,
                    "source_url": event.source_url,
                })
                matched_positive += 1
                matched_groups.add(str(event.event_group_id))
            control_id = event.event_id + ":tminus60"
            control_index = by_id.get(control_id)
            if control_index is not None:
                control = rows[control_index]
                control.update({
                    "label": 0,
                    "label_source": "Same source-event cell 60 days before reported date; inventory absence assumed, not field-verified",
                    "label_quality": "temporal_inventory_absence_pseudo_negative_unverified",
                    "label_note": "Matched event-cell 60-day inventory-absence pseudo-negative; not surveyed intact terrain.",
                    "event_record_type": "temporal_absence_sample",
                    "event_group_id": event.event_group_id,
                    "mapping_method": event.mapping_method,
                    "citation": event.citation,
                    "source_url": event.source_url,
                })
                matched_negative += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(old_table.schema.metadata or {})
    metadata[b"label_policy"] = b"y=1: H3 centers in cited manual COOLR polygons or within 300 m of cited manual COOLR points (approximate initiation-zone proxy, not a crown polygon). y=0: matched 60-day inventory-absence pseudo-negatives, not surveyed intact terrain. Other event points remain unlabeled."
    metadata[b"dataset_scope"] = b"Existing event-time feature observations with source-linked manual point/polygon inventory labels and explicitly marked temporal controls."
    table = pa.Table.from_pylist(rows).replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")

    label_counts = Counter(row.get("label") for row in rows)
    positives = [row for row in rows if row.get("label") == 1]
    summary_path = path.with_suffix(".metadata.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update({
        "output": str(path),
        "row_count": len(rows),
        "labeled_positive_rows": label_counts.get(1, 0),
        "labeled_negative_rows": label_counts.get(0, 0),
        "unlabeled_event_candidate_rows": label_counts.get(None, 0),
        "distinct_positive_source_events": len({row.get("source_event_id") for row in positives if row.get("source_event_id")}),
        "positive_event_dates": sorted({row.get("event_date") for row in positives if row.get("event_date")}),
        "positive_regimes": sorted({row.get("regime") for row in positives if row.get("regime")}),
        "positive_geographic_extent": {
            "latitude_min": min((float(row["latitude"]) for row in positives if row.get("latitude") is not None), default=None),
            "latitude_max": max((float(row["latitude"]) for row in positives if row.get("latitude") is not None), default=None),
            "longitude_min": min((float(row["longitude"]) for row in positives if row.get("longitude") is not None), default=None),
            "longitude_max": max((float(row["longitude"]) for row in positives if row.get("longitude") is not None), default=None),
        },
        "label_policy": "Cited manual NASA COOLR polygon cells and existing H3 feature rows within 300 m of cited manual COOLR points are positive inventory proxies; point-buffer rows are not mapped crowns. Matched 60-day pre-event rows are inventory-absence pseudo-negatives, not surveyed or verified intact terrain. All other candidates remain unlabeled.",
        "warning": "Current positives span the 2021 Haiti polygon inventory and matched 2024 Brazil manual-point inventory. Only the trigger regimes present in the source catalog are represented; global or five-regime calibration is not established.",
        "point_label_refresh": {
            "status": "updated_existing_matches",
            "matched_manual_point_positive_rows": matched_positive,
            "matched_temporal_controls": matched_negative,
            "matched_inventory_groups": sorted(matched_groups),
            "catalog_manual_point_positive_candidates": sum(1 for row in candidates if row.event_record_type == "cited_manual_point_buffer_cell" and row.label == 1),
            "limitation": "Only source points matching existing feature rows were updated; no feature rows or event labels were fabricated.",
        },
    })
    summary["event_group_holdout"] = _write_event_group_holdout(path, rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "data" / "real_global_landslide_observations.parquet"))
    parser.add_argument("--max-events", type=int, default=450)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    summary = asyncio.run(refresh(Path(args.dataset), args.max_events, args.timeout))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
