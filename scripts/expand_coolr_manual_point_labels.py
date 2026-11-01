#!/usr/bin/env python3
"""Enrich source-backed manual COOLR points and matched controls missing from the Parquet."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.build_real_labeled_dataset import (
    EventCandidate,
    _write_event_group_holdout,
    enrich_one,
    enrich_temporal_control,
    event_candidates,
    fetch_nasa_coolr_events,
    fetch_nasa_coolr_polygons,
)
from backend.ml.global_xgb import FEATURE_NAMES
from backend.realtime.dem_processor import DEMProcessor


def _set_label_fields(row: Dict[str, Any], event: EventCandidate, control: bool = False) -> Dict[str, Any]:
    if control:
        row.update({
            "event_id": event.event_id,
            "name": event.name,
            "event_date": event.event_date,
            "label": 0,
            "source_name": "NASA COOLR event-cell inventory-absence sample",
            "event_record_type": "temporal_absence_sample",
            "label_source": event.label_source,
            "label_quality": "temporal_inventory_absence_pseudo_negative_unverified",
            "label_note": "Matched event-cell 60-day inventory-absence pseudo-negative; not surveyed intact terrain.",
        })
    else:
        row.update({
            "event_id": event.event_id,
            "name": event.name,
            "event_date": event.event_date,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "regime": event.regime,
            "source_name": event.source_name,
            "source_url": event.source_url,
            "event_record_type": event.event_record_type,
            "label": 1,
            "label_source": event.label_source,
            "label_quality": event.label_quality,
            "label_note": "Cited manual NASA COOLR point within 300 m of the existing H3 feature cell; approximate source-location proxy, not a mapped crown polygon.",
        })
    row.update({
        "source_event_id": event.source_event_id,
        "event_group_id": event.event_group_id,
        "mapping_method": event.mapping_method,
        "citation": event.citation,
    })
    if event.h3_cell_id and not row.get("h3_cell_id"):
        row["h3_cell_id"] = event.h3_cell_id
    return row


def _ensure_inventory_groups(rows: List[Dict[str, Any]]) -> None:
    """Persist the source inventory group on COOLR rows and their controls."""
    for row in rows:
        source = row.get("source_name")
        if source not in ("NASA COOLR", "NASA COOLR manually mapped polygon"):
            continue
        name, day, citation = row.get("name"), row.get("event_date"), row.get("citation")
        if name and day and citation:
            seed = "|".join((str(name).strip(), str(day), str(citation)))
            row["event_group_id"] = "nasa_coolr_inventory:{}".format(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12])

    by_event = {str(row.get("event_id")): row for row in rows if row.get("event_id")}
    for row in rows:
        if row.get("event_record_type") != "temporal_absence_sample":
            continue
        event_id = str(row.get("event_id") or "")
        parent = by_event.get(event_id.removesuffix(":tminus60"))
        if parent and parent.get("event_group_id"):
            row["event_group_id"] = parent["event_group_id"]


def _refresh_summary(path: Path, rows: List[Dict[str, Any]], schema_metadata: Dict[bytes, bytes]) -> Dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _ensure_inventory_groups(rows)
    temp_path = path.with_name(path.name + ".tmp")
    table = pa.Table.from_pylist(rows)
    group_array = pa.array([row.get("event_group_id") for row in rows], type=pa.string())
    if "event_group_id" in table.column_names:
        table = table.set_column(table.schema.get_field_index("event_group_id"), "event_group_id", group_array)
    else:
        table = table.append_column("event_group_id", group_array)
    pq.write_table(table.replace_schema_metadata(schema_metadata), temp_path, compression="zstd")
    os.replace(str(temp_path), str(path))

    summary_path = path.with_suffix(".metadata.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    labels = Counter(row.get("label") for row in rows)
    positives = [row for row in rows if row.get("label") == 1]
    candidate_rows = [row for row in rows if row.get("event_record_type") != "temporal_absence_sample"]
    point_rows = [row for row in positives if row.get("event_record_type") == "cited_manual_point_buffer_cell"]
    polygon_rows = [row for row in positives if row.get("event_record_type") == "manually_mapped_event_polygon_cell"]
    positive_events = {row.get("source_event_id") for row in positives if row.get("source_event_id")}
    dated_external = {
        row.get("source_event_id") for row in candidate_rows
        if row.get("event_date") and row.get("source_event_id")
        and row.get("source_name") not in (None, "user-provided project candidate list")
    }
    dated_coolr = {
        row.get("source_event_id") for row in candidate_rows
        if row.get("event_date") and row.get("source_event_id")
        and row.get("source_name") in ("NASA COOLR", "NASA COOLR manually mapped polygon")
    }
    summary.update({
        "output": str(path),
        "row_count": len(rows),
        "catalog_candidate_count_initial_build": summary.get("catalog_candidate_count_initial_build", summary.get("catalog_candidate_count")),
        "catalog_candidate_count": len(candidate_rows),
        "expanded_dataset_candidate_rows": len(candidate_rows),
        "nasa_coolr_candidate_count": sum(row.get("source_name") == "NASA COOLR" for row in candidate_rows),
        "labeled_positive_rows": labels.get(1, 0),
        "labeled_negative_rows": labels.get(0, 0),
        "unlabeled_event_candidate_rows": labels.get(None, 0),
        "complete_14_feature_rows": sum(all(row.get(name) is not None and math.isfinite(float(row[name])) for name in FEATURE_NAMES) for row in rows),
        "distinct_positive_source_events": len(positive_events),
        "distinct_dated_external_source_events": len(dated_external),
        "distinct_dated_nasa_coolr_events": len(dated_coolr),
        "positive_event_dates": sorted({row.get("event_date") for row in positives if row.get("event_date")}),
        "positive_regimes": sorted({row.get("regime") for row in positives if row.get("regime")}),
        "regimes": sorted({row.get("regime") for row in rows if row.get("regime")}),
        "nasa_coolr_manual_polygon_cell_count": len(polygon_rows),
        "nasa_coolr_manual_point_cell_count": len(point_rows),
        "temporal_absence_controls": labels.get(0, 0),
        "positive_geographic_extent": {
            "latitude_min": min((float(row["latitude"]) for row in positives if row.get("latitude") is not None), default=None),
            "latitude_max": max((float(row["latitude"]) for row in positives if row.get("latitude") is not None), default=None),
            "longitude_min": min((float(row["longitude"]) for row in positives if row.get("longitude") is not None), default=None),
            "longitude_max": max((float(row["longitude"]) for row in positives if row.get("longitude") is not None), default=None),
        },
        "label_policy": "Cited manual COOLR polygon H3 centers and H3 cells within 300 m of cited manual COOLR points are positive inventory proxies; point buffers are not mapped crowns. Matched 60-day pre-event rows are inventory-absence pseudo-negatives, not surveyed or verified intact terrain.",
        "warning": "The current cited manual point candidates span rainfall-triggered and trigger-unclassified inventories; the layer does not provide source-backed seismic, cryosphere, wildfire, and anthropogenic positive labels for five-regime calibration.",
    })
    summary.pop("point_label_refresh", None)
    expansion = dict(summary.get("manual_point_label_expansion") or {})
    expansion.update({
        "status": "complete_for_selected_manual_candidates",
        "selected_candidates": len(point_rows),
        "inventory_groups": len({row.get("event_group_id") for row in point_rows if row.get("event_group_id")}),
        "positive_regimes_in_candidates": dict(Counter(row.get("regime") for row in point_rows)),
        "negative_controls_added_or_refreshed": labels.get(0, 0),
        "note": "Features were fetched or reused per cited manual source point; no user-supplied candidate location was labeled.",
    })
    summary["manual_point_label_expansion"] = expansion
    summary["event_group_holdout"] = _write_event_group_holdout(path, rows)
    summary["event_group_split_audit"] = {
        "positive_inventory_groups": len({row.get("event_group_id") for row in positives if row.get("event_group_id")}),
        "training_and_holdout_groups_disjoint": summary["event_group_holdout"].get("status") == "ready",
        "heldout_group_id": summary["event_group_holdout"].get("holdout_event_group_id"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


async def expand(path: Path, max_events: int, timeout_s: float, concurrency: int) -> Dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not path.exists():
        raise FileNotFoundError(path)
    old_table = pq.read_table(path)
    old_rows = old_table.to_pylist()
    by_id = {str(row.get("event_id")): row for row in old_rows if row.get("event_id")}
    headers = {"User-Agent": "TERRA-SHIELD/0.3 (open-data research prototype)", "Accept": "application/json"}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s),
        headers=headers,
        limits=httpx.Limits(max_connections=max(24, concurrency * 4), max_keepalive_connections=max(12, concurrency * 2)),
        follow_redirects=True,
    ) as client:
        point_catalog, polygon_catalog = await asyncio.gather(
            fetch_nasa_coolr_events(client, max_events),
            fetch_nasa_coolr_polygons(client, max_events),
        )
        candidates = event_candidates([], point_catalog, polygon_catalog, [], [], max_events)
        positives = [
            event for event in candidates
            if event.event_record_type == "cited_manual_point_buffer_cell" and event.label == 1
        ]
        if not positives:
            raise RuntimeError("NASA COOLR returned no cited manual point candidates; dataset was not changed")

        dem = DEMProcessor(zoom=13)
        semaphore = asyncio.Semaphore(concurrency)

        async def build_positive(event: EventCandidate) -> Dict[str, Any]:
            existing = by_id.get(event.event_id)
            if existing is not None and all(existing.get(name) is not None for name in FEATURE_NAMES):
                return _set_label_fields(dict(existing), event)
            row = await enrich_one(event, client, dem, semaphore)
            return _set_label_fields(row, event)

        positive_rows = await asyncio.gather(*(build_positive(event) for event in positives))
        positive_by_event = {str(row["event_id"]): row for row in positive_rows}

        async def build_control(event: EventCandidate) -> Dict[str, Any]:
            control_day = date.fromisoformat(str(event.event_date)) - timedelta(days=60)
            control = replace(
                event,
                event_id=event.event_id + ":tminus60",
                name=event.name + " (60-day pre-event sample)",
                event_date=control_day.isoformat(),
                source_name="NASA COOLR event-cell inventory-absence sample",
                event_record_type="temporal_absence_sample",
                label=0,
                label_source="Same source-event cell 60 days before reported date; inventory absence assumed, not field-verified",
                label_quality="temporal_inventory_absence_pseudo_negative_unverified",
                failure_geometry=None,
            )
            existing = by_id.get(control.event_id)
            if existing is not None and all(existing.get(name) is not None for name in FEATURE_NAMES):
                return _set_label_fields(dict(existing), control, control=True)
            row = await enrich_temporal_control(control, positive_by_event[event.event_id], client, semaphore)
            return _set_label_fields(row, control, control=True)

        controls = await asyncio.gather(*(build_control(event) for event in positives))

    merged: Dict[str, Dict[str, Any]] = {str(row.get("event_id")): dict(row) for row in old_rows if row.get("event_id")}
    for row in positive_rows + controls:
        merged[str(row["event_id"])] = row
    rows = list(merged.values())
    metadata = dict(old_table.schema.metadata or {})
    metadata[b"label_policy"] = b"y=1: H3 centers in cited manual COOLR polygons or within 300 m of cited manual COOLR points (approximate initiation-zone proxy, not crown). y=0: matched 60-day inventory-absence pseudo-negatives, not surveyed intact terrain. Other event candidates remain unlabeled."
    metadata[b"dataset_scope"] = b"Event-time observations with cited inventory proxy labels and explicit temporal inventory-absence controls."
    return _refresh_summary(path, rows, metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "data" / "real_global_landslide_observations.parquet"))
    parser.add_argument("--max-events", type=int, default=450)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--refresh-existing", action="store_true", help="repair inventory-group fields and refresh split/metadata without network fetches")
    args = parser.parse_args()
    if args.max_events < 1 or args.concurrency < 1:
        parser.error("max-events and concurrency must be positive")
    dataset = Path(args.dataset)
    if args.refresh_existing:
        import pyarrow.parquet as pq
        table = pq.read_table(dataset)
        summary = _refresh_summary(dataset, table.to_pylist(), dict(table.schema.metadata or {}))
    else:
        summary = asyncio.run(expand(dataset, args.max_events, args.timeout, args.concurrency))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

