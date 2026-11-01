#!/usr/bin/env python3
"""Build sourced observations, fit the event-group training split, then scan five regions."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.build_real_labeled_dataset import build_dataset
from backend.release import portable_payload
from backend.ml.global_xgb import FEATURE_NAMES, GlobalMonotonicXGB, blend_failure_probability
from backend.ml.validation_suite import (
    build_validation_report,
    load_labeled_training_data,
    run_kinematic_branch_controls,
)
from scripts.verify_5_global_causes import print_environment_table, scan_environments


def _fit_real_model(path: Path) -> Optional[GlobalMonotonicXGB]:
    try:
        X, y, groups = load_labeled_training_data(str(path))
        positive_blocks = len(set(groups[y == 1]))
        n_splits = min(5, positive_blocks)
        if n_splits < 3:
            print("Model not fitted: only {} spatial blocks contain positive labels; at least three are required for grouped calibration.".format(positive_blocks))
            return None
        model = GlobalMonotonicXGB()
        model.fit(X, y, groups, n_splits=n_splits)
        from backend.ml.conformal_calibration import fit_reserved_conformal
        conformal=fit_reserved_conformal(model,ROOT/'data',groups)
        import joblib
        import pyarrow.parquet as pq
        artifact = ROOT / "artifacts" / "global_multi_causal_xgb.pkl"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifact)
        rows = pq.read_table(path).to_pylist()
        positive_rows = [row for row in rows if row.get("label") == 1]
        print("Fitted and saved calibrated XGBoost using {} labeled rows (cited polygon/point-buffer inventory positives plus marked absence pseudo-negatives).".format(len(y)))
        metadata = {
            "training_rows": int(len(y)),
            "positive_rows": int((y == 1).sum()),
            "negative_rows": int((y == 0).sum()),
            "synthetic_training_only_rows": sum(bool(row.get("is_synthetic")) for row in rows),
            "physics_model_version": "boundary_conformal_v5", "conformal":conformal,
            "probability_policy": "mechanical gate times regime-weighted calibrated XGBoost and physical union; deployed score is not separately calibrated",
            "label_quality": "NASA COOLR polygon/point inventory and report-location proxies; temporal inventory-absence pseudo-negatives plus explicitly simulated training-only mechanical negatives",
            "calibration": "isotonic with {} grouped spatial folds".format(n_splits),
            "calibration_folds": int(n_splits),
            "positive_bearing_spatial_blocks": int(positive_blocks),
            "positive_source_events": len({row.get("source_event_id") for row in positive_rows if row.get("source_event_id")}),
            "positive_event_dates": sorted({row.get("event_date") for row in positive_rows if row.get("event_date")}),
            "positive_regimes": sorted({row.get("regime") for row in positive_rows if row.get("regime")}),
            "scope_warning": "Positive rows include polygon footprints and manual point-buffer proxies. Spatial and temporal sampling assumptions limit calibration; event-group holdout scores do not establish global or five-regime generalization.",
            "feature_names": list(__import__("backend.ml.global_xgb", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES),
        }
        (artifact.parent / "global_multi_causal_xgb.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return model
    except Exception as exc:
        print("Model not fitted: {}: {}".format(type(exc).__name__, exc))
        return None


def _write_hindcast_scores(candidates_path: Path, model: GlobalMonotonicXGB, destination: Path) -> Dict[str, Any]:
    """Score a whole cited inventory group that was excluded from model fitting."""
    import pyarrow.parquet as pq

    rows = pq.read_table(candidates_path).to_pylist()
    rows = [
        row for row in rows
        if row.get("label") in (0, 1) and all(row.get(name) is not None for name in FEATURE_NAMES)
    ]
    if not rows:
        raise ValueError("held-out hindcast inventory has no complete binary-labeled feature rows")
    X = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    p_xgb = model.predict_proba(X)
    p_physical = np.asarray([float(row.get("p_failure_physical", row["p_f_fosm"])) for row in rows], dtype=np.float64)
    probability = p_xgb  # Location-uncertain event-buffer evaluation; live scores remain mechanically gated.
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case_id", "p_failure", "observed_failure", "event_date", "source_event_id",
        "event_group_id", "event_id", "source_url", "label_quality", "label_source",
        "split", "probability_source",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row, score in zip(rows, probability):
            writer.writerow({
                "case_id": row.get("event_group_id") or "heldout_inventory",
                "p_failure": float(score),
                "observed_failure": int(row["label"]),
                "event_date": row.get("event_date"),
                "source_event_id": row.get("source_event_id"),
                "event_group_id": row.get("event_group_id"),
                "event_id": row.get("event_id"),
                "source_url": row.get("source_url"),
                "label_quality": row.get("label_quality"),
                "label_source": row.get("label_source"),
                "split": "inventory_group_holdout_excluded_from_training",
                "probability_source": "isotonic XGBoost event-buffer score; different estimand from live mechanically gated score",
            })
    return {
        "path": str(destination.relative_to(ROOT)),
        "rows": len(rows),
        "positive_rows": int(sum(row["label"] == 1 for row in rows)),
        "negative_rows": int(sum(row["label"] == 0 for row in rows)),
        "heldout_event_group_ids": sorted({row.get("event_group_id") for row in rows}),
        "probability_source": "isotonic XGBoost event-buffer score; different estimand from live mechanically gated score",
    }


async def run(args: argparse.Namespace) -> int:
    from backend.release import invalidate_readiness
    invalidate_readiness(ROOT / "artifacts", "Mac rebuild started; rerun both audits for the resulting artifacts")
    data_path = ROOT / args.dataset
    dataset_summary: Dict[str, Any]
    if args.skip_dataset_build and data_path.exists():
        metadata_path = data_path.with_suffix(".metadata.json")
        dataset_summary = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {
            "output": str(data_path), "status": "existing_parquet_metadata_unavailable"
        }
        print("\nUSING EXISTING OBSERVATION DATASET\n{}".format(json.dumps(dataset_summary, indent=2)))
    else:
        try:
            dataset_summary = await build_dataset(data_path, args.max_events, args.fetch_timeout)
            print("\nOBSERVATION DATASET\n{}".format(json.dumps(dataset_summary, indent=2)))
        except Exception as exc:
            dataset_summary = {"status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)}
            print("\nObservation dataset build failed: {}".format(dataset_summary["error"]))

    if data_path.exists():
        try:
            import pyarrow.parquet as pq
            observed_rows = pq.read_table(data_path).to_pylist()
            positive_rows = [row for row in observed_rows if row.get("label") == 1]
            dataset_summary.update({
                "distinct_positive_source_events": len({row.get("source_event_id") for row in positive_rows if row.get("source_event_id")}),
                "positive_event_dates": sorted({row.get("event_date") for row in positive_rows if row.get("event_date")}),
                "positive_regimes": sorted({row.get("regime") for row in positive_rows if row.get("regime")}),
                "positive_geographic_extent": {
                    "latitude_min": min((row["latitude"] for row in positive_rows), default=None),
                    "latitude_max": max((row["latitude"] for row in positive_rows), default=None),
                    "longitude_min": min((row["longitude"] for row in positive_rows), default=None),
                    "longitude_max": max((row["longitude"] for row in positive_rows), default=None),
                },
            })
        except Exception as exc:
            dataset_summary["positive_scope_summary_error"] = "{}: {}".format(type(exc).__name__, exc)

    holdout_split = dataset_summary.get("event_group_holdout", {})
    if holdout_split.get("status") != "ready":
        failure = {"pretraining_ready": False, "training_model_fitted": False,
                   "audit_status": "blocked: source enrichment or frozen holdout split incomplete",
                   "dataset_build": dataset_summary}
        target = ROOT / "artifacts" / "scientific_validation_report.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(portable_payload(failure), indent=2))
        print("Training stopped: finish source enrichment and the frozen event holdout first.")
        return 2
    split_training_path = ROOT / str(holdout_split.get("training_rows", "")) if holdout_split.get("status") == "ready" else None
    if split_training_path is None or not split_training_path.exists():
        raise FileNotFoundError("Required embargoed model_training.parquet is missing; rebuild the frozen split")
    training_path = split_training_path or (data_path if data_path.exists() else ROOT / "data" / "landslide_training.csv")
    model = _fit_real_model(training_path) if training_path.exists() else None

    hindcast_path = ROOT / "data" / "historical_hindcast.csv"
    hindcast_scores = None
    if model is not None and holdout_split.get("status") == "ready":
        candidate_path = ROOT / str(holdout_split.get("hindcast_candidates", ""))
        if candidate_path.exists():
            try:
                hindcast_scores = _write_hindcast_scores(candidate_path, model, hindcast_path)
            except Exception as exc:
                hindcast_scores = {"status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)}
                print("Historical hindcast score export failed: {}".format(hindcast_scores["error"]))

    print("\nFetching live five-region verification data…")
    try:
        scans = await scan_environments(args.provider_timeout, args.radius_km)
        live_physics = print_environment_table(scans, model)
    except Exception as exc:
        scans = []
        live_physics = {"status": "not_available", "reason": "{}: {}".format(type(exc).__name__, exc)}
        print("Live region physics unavailable: {}".format(live_physics["reason"]))

    report = build_validation_report(str(training_path), str(hindcast_path), model)
    branches = run_kinematic_branch_controls()
    if model is not None:
        report.stage1.details.append("Training includes temporal inventory-absence pseudo-negatives and simulated mechanical controls. Simulated rows are excluded from calibration and validation holdouts; negative observations are not field-verified.")
        try:
            import pyarrow.parquet as pq
            model_rows = pq.read_table(training_path).to_pylist()
            model_positives = [row for row in model_rows if row.get("label") == 1]
            training_dates = sorted({row.get("event_date") for row in model_positives if row.get("event_date")})
            training_regimes = sorted({row.get("regime") for row in model_positives if row.get("regime")})
            dataset_summary["training_positive_source_events"] = len({row.get("source_event_id") for row in model_positives if row.get("source_event_id")})
            dataset_summary["training_positive_event_dates"] = training_dates
            dataset_summary["training_positive_regimes"] = training_regimes
            report.stage1.details.append("Training positives: {} source records on {} in {} regime(s); point-buffer labels are approximate initiation-zone proxies.".format(
                dataset_summary["training_positive_source_events"],
                ", ".join(training_dates) or "unknown dates",
                len(training_regimes),
            ))
        except Exception as exc:
            report.stage1.details.append("Training-positive scope unavailable: {}: {}".format(type(exc).__name__, exc))
    payload = report.to_dict()
    if model is not None:
        pinls_path=ROOT/'data/historical_hindcast_pinls.parquet'
        pinls_csv=ROOT/'data/historical_hindcast_pinls.csv'
        _write_hindcast_scores(pinls_path,model,pinls_csv)
        from backend.ml.validation_suite import load_hindcast_csv, historical_hindcast_metrics
        p,y,c=load_hindcast_csv(str(pinls_csv))
        payload['stage2_pinls_filtered']=historical_hindcast_metrics(p,y,c).to_dict()
        payload['stage2_pinls_filtered']['scope']='Selection-biased stability-filtered cohort; not comparable to the unchanged original holdout'
        payload['conformal_calibration']=model.conformal_
        from backend.ml.conformal_calibration import heldout_conformal_diagnostics
        payload['conformal_holdout']=heldout_conformal_diagnostics(model,pq.read_table(ROOT/'data/historical_hindcast_candidates.parquet').to_pylist())

    payload["stage4_kinematics"] = [item.__dict__ for item in branches]
    payload["dataset_build"] = dataset_summary
    payload["historical_hindcast_export"] = hindcast_scores or {"status": "not_available", "reason": "no source-backed inventory-group holdout was produced"}
    payload["training_model_fitted"] = model is not None
    payload["training_label_quality"] = (
        "Cited NASA COOLR mapped-polygon positives and manual point-buffer initiation-zone proxies plus 60-day inventory-absence pseudo-negatives. Negative cells are not field-surveyed intact terrain."
        if model is not None else "No model fitted."
    )
    payload["model_scope_warning"] = "The model is exploratory. Polygon and point-buffer labels have different spatial precision; the held-out inventory group and pseudo-negative assumptions do not establish global or five-regime calibration."
    payload["live_physics"] = live_physics
    payload["live_environment_readiness"] = [
        {
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "regime": regime,
            "h3_cells": scan.terrain.h3_cell_count if scan.terrain else None,
            "elevation_min_m": scan.terrain.min_elevation_m if scan.terrain else None,
            "elevation_max_m": scan.terrain.max_elevation_m if scan.terrain else None,
            "slope_p95_deg": scan.terrain.slope_p95_deg if scan.terrain else None,
            "max_twi": scan.terrain.max_twi if scan.terrain else None,
            "rain_24h_mm": scan.weather.rain_24h_mm if scan.weather else None,
            "ari_7d_mm": scan.weather.ari_7d_mm if scan.weather else None,
            "max_pga_g": scan.seismic.max_pga_g if scan.seismic else None,
            "soil_property_source": scan.hydrology.soil_property_source if scan.hydrology else None,
            "soil_property_quality": scan.hydrology.soil_property_quality if scan.hydrology else None,
            "satellite_cell_count": len(scan.satellite_cells),
            "sources": {key: value.available for key, value in scan.sources.items()},
        }
        for (name, lat, lon, regime), scan in zip(
            __import__("scripts.verify_5_global_causes", fromlist=["ENVIRONMENTS"]).ENVIRONMENTS, scans
        )
    ]
    payload["pretraining_ready"] = False
    payload["audit_status"] = "pending fresh full-system audit"
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["label_policy"] = (
        "Cited manual NASA COOLR polygon cells and H3 cells within 300 m of cited manual COOLR points are y=1 inventory proxies. Matched 60-day pre-event samples are y=0 inventory-absence pseudo-negatives, not surveyed or verified intact terrain. Stage 1 reflects these assumptions. Stage 2 scores balanced source groups with complete spatial blocks excluded from model fitting."
    )
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "scientific_validation_report.json"
    report_path.write_text(json.dumps(portable_payload(payload), indent=2, allow_nan=False), encoding="utf-8")

    print("\nSCIENTIFIC VALIDATION")
    print("Stage | Status | Metrics | Details")
    print("---|---|---|---")
    for stage in (report.stage1, report.stage2, report.stage3_monotonicity):
        metrics = ", ".join("{}={}".format(key, value) for key, value in stage.metrics.items()) or "N/A"
        print("{} | {} | {} | {}".format(stage.name, stage.status.upper(), metrics, "; ".join(stage.details)))
    print("Stage 3 counterfactual physics controls: {}".format(
        ", ".join("{}={}".format(item.name, "PASS" if item.passed else "FAIL") for item in report.stage3_controls)
    ))
    print("Stage 3 causal attribution controls: {}".format(
        ", ".join("{}={}".format(item["name"], "PASS" if item["passed"] else "FAIL") for item in report.stage3_causal_controls)
    ))
    print("Stage 4 kinematics branch controls: {}".format(
        ", ".join("{}={}".format(item.name, "PASS" if item.passed else "FAIL") for item in branches)
    ))
    print("Report: {}".format(report_path))
    if model is None:
        print("No model artifact was written; the available dataset has insufficient binary labeled rows or spatial blocks with complete feature rows.")
    return 1 if (
        any(not item.passed for item in report.stage3_controls + branches)
        or any(not item["passed"] for item in report.stage3_causal_controls)
        or report.stage3_monotonicity.status != "passed"
    ) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/real_global_landslide_observations.parquet")
    parser.add_argument("--max-events", type=int, default=450)
    parser.add_argument("--skip-dataset-build", action="store_true", help="reuse existing Parquet and only refresh model/region verification")
    parser.add_argument("--fetch-timeout", type=float, default=15.0)
    parser.add_argument("--provider-timeout", type=float, default=4.0)
    parser.add_argument("--radius-km", type=float, default=3.5)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
