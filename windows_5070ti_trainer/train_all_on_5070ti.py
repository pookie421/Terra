"""Portable CUDA XGBoost and guarded QLoRA training entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.ml.global_xgb import FEATURE_NAMES, MONOTONE_VECTOR, GlobalMonotonicXGB, make_base_estimator, spatial_group_splits
from backend.release import portable_payload
from backend.ml.validation_suite import (
    build_validation_report,
    load_labeled_training_data,
    run_kinematic_branch_controls,
)


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _write_block_csv_if_needed(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """No synthetic records are emitted; used only for the existing CSV hindcast contract."""
    del path, rows


def tune_cuda_xgb(X: np.ndarray, y: np.ndarray, groups: np.ndarray, trials: int = 20) -> Dict[str, Any]:
    import optuna
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    positive_blocks = len(np.unique(groups[y == 1]))
    n_splits = min(5, positive_blocks)
    if n_splits < 3:
        raise ValueError("at least three positive-bearing spatial blocks are required for grouped tuning")
    folds = spatial_group_splits(y, groups, n_splits)
    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[valid_idx])) < 2:
            raise ValueError("spatial fold {} lacks both labeled classes; tuning stopped".format(fold))

    def objective(trial: Any) -> float:
        depth = trial.suggest_int("max_depth", 3, 9)
        n_estimators = trial.suggest_int("n_estimators", 200, 900, step=100)
        learning_rate = trial.suggest_float("learning_rate", 0.015, 0.12, log=True)
        min_child_weight = trial.suggest_float("min_child_weight", 1.0, 12.0)
        scores = []
        for train_idx, valid_idx in folds:
            model = XGBClassifier(
                n_estimators=n_estimators, max_depth=depth, learning_rate=learning_rate,
                min_child_weight=min_child_weight, objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", device="cuda", monotone_constraints=MONOTONE_VECTOR,
                n_jobs=-1, verbosity=0,
            )
            model.fit(X[train_idx], y[train_idx], sample_weight=np.asarray([0.25 if str(g).startswith("synthetic:") else 1.0 for g in groups[train_idx]]))
            scores.append(float(roc_auc_score(y[valid_idx], model.predict_proba(X[valid_idx])[:, 1])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name="terra_shield_spatial_cuda_xgb")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {"best_value_spatial_auc": study.best_value, "best_params": study.best_params, "trials": trials, "spatial_folds": n_splits}


def verify_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    training_path = ROOT / "data" / "model_training.parquet"
    if not training_path.exists():
        raise FileNotFoundError("Required embargoed model_training.parquet is missing; rebuild the frozen split")
    metadata=json.loads((ROOT/'data/real_global_landslide_observations.metadata.json').read_text())
    if metadata.get('event_group_holdout',{}).get('status')!='ready':
        raise ValueError("Source enrichment or frozen holdout split incomplete; refusing full-dataset training")
    X, y, groups = load_labeled_training_data(str(training_path))
    labeled = [row for row in rows if row.get("label") in (0, 1) and all(row.get(name) is not None for name in FEATURE_NAMES)]
    if len(np.unique(y)) != 2:
        raise ValueError("training requires both source-backed inventory positives and temporal inventory-absence controls")
    if any(row.get("label") == 0 and not str(row.get("label_quality", "")).startswith(("temporal_inventory_absence_pseudo_negative", "simulated_mechanical_negative")) for row in labeled):
        raise ValueError("negative labels must carry the explicit temporal inventory-absence quality marker")
    positive_blocks = len(np.unique(groups[y == 1]))
    if len(np.unique(groups)) < 5 or positive_blocks < 3:
        raise ValueError("training requires at least five coarse blocks and three positive-bearing blocks")
    if len(y) < 450:
        raise ValueError("training requires at least 450 complete labeled rows; found {}".format(len(y)))
    if any(row.get("label_quality", "").startswith("user_supplied") for row in rows if row.get("label") in (0, 1)):
        raise ValueError("user-supplied candidate points are not accepted as labeled ground truth")
    return X, y, groups


def export_chatml_from_observations(rows: Sequence[Mapping[str, Any]], destination: Path) -> int:
    """Convert one real event observation per record; never clone rows to reach 600."""
    candidates = [
        row for row in rows
        if row.get("event_candidate") == 1
        and row.get("source_name") not in (None, "user-provided project candidate list")
        and row.get("event_id") and row.get("event_date") and row.get("h3_cell_id")
    ]
    unique = {}
    for row in candidates:
        trajectory_id = "{}|{}|{}".format(row.get("source_event_id") or row["event_id"], row["h3_cell_id"], row["event_date"])
        unique[trajectory_id] = row
    if len(unique) < 600:
        raise ValueError("600 distinct source-linked cell/date trajectories are required for QLoRA; found {} (no duplicates will be generated)".format(len(unique)))
    pools = {}
    for row in unique.values():
        pools.setdefault(str(row.get("regime") or "unknown"), []).append(row)
    selected = []
    while len(selected) < 600:
        for key in sorted(pools):
            if pools[key] and len(selected) < 600:
                selected.append(pools[key].pop(0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            obs = {
                key: row.get(key) for key in (
                    "event_id", "name", "event_date", "latitude", "longitude", "regime", "source_name",
                    "weather_source_status", "rain_event_day_mm", "rain_antecedent_7d_mm", "snowfall_event_day_mm",
                    "temperature_event_day_mean_c", "soil_moisture_event_day_mean_m3m3", "terrain_source_status",
                    "terrain_slope_deg", "terrain_twi", "terrain_elevation_m", "usgs_source_status",
                    "usgs_max_magnitude", "pga_attenuation_proxy_g", "label", "label_quality",
                    *FEATURE_NAMES, "physics_feature_provenance", "physics_feature_quality", "source_event_id", "h3_cell_id", "source_url", "citation",
                )
            }
            messages = [
                {"role": "user", "content": "Summarize the sourced observations for this landslide event candidate. Distinguish measurements, estimates, and missing evidence; do not infer a cell failure probability."},
                {"role": "assistant", "content": "I will inspect the event-linked measurements and their provenance."},
                {"role": "assistant", "tool_calls": [{"id": "obs1", "type": "function", "function": {"name": "get_event_observations", "arguments": json.dumps({"event_id": row["event_id"]})}}]},
                {"role": "tool", "tool_call_id": "obs1", "name": "get_event_observations", "content": json.dumps(obs, sort_keys=True)},
                {"role": "assistant", "content": "These are event-linked cell/date observations. Use the label only with its stated quality: polygon positives are inventory observations, point/report labels are location proxies, and 60-day pre-event negatives are inventory-absence assumptions. Synthetic mechanical controls are excluded from these source trajectories. Feature provenance distinguishes observations, estimates, and explicit defaults; no calibrated probability is supplied here."},
            ]
            handle.write(json.dumps({"messages": messages, "source_event_id": row["event_id"]}, ensure_ascii=False) + "\n")
    return 600


def finetune_qwen(dataset_path: Path, output_dir: Path, gguf_path: Path) -> Dict[str, Any]:
    from train_qwen25_agent import train_qwen25
    return train_qwen25(dataset_path, output_dir, gguf_path)


def write_hindcast_scores(candidates_path: Path, model: GlobalMonotonicXGB, destination: Path) -> Dict[str, Any]:
    import csv
    import pyarrow.parquet as pq
    from backend.ml.global_xgb import blend_failure_probability

    rows = [row for row in pq.read_table(candidates_path).to_pylist()
            if row.get("label") in (0, 1) and all(row.get(name) is not None for name in FEATURE_NAMES)]
    if not rows or {row["label"] for row in rows} != {0, 1}:
        raise ValueError("held-out inventory must contain complete features for both source labels")
    X = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    probability = model.predict_proba(X)  # Ungated event-buffer evaluation only.
    columns = ["case_id", "p_failure", "observed_failure", "event_date", "source_event_id", "event_group_id", "source_url", "label_quality", "split", "probability_source"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row, score in zip(rows, probability):
            writer.writerow({
                "case_id": row.get("event_group_id"), "p_failure": float(score),
                "observed_failure": int(row["label"]), "event_date": row.get("event_date"),
                "source_event_id": row.get("source_event_id"), "event_group_id": row.get("event_group_id"),
                "source_url": row.get("source_url"), "label_quality": row.get("label_quality"),
                "split": "inventory_group_holdout_excluded_from_training",
                "probability_source": "isotonic XGBoost event-buffer score; live scores remain gated",
            })
    return {"status": "written", "path": str(destination.relative_to(ROOT)), "rows": len(rows), "positive_rows": int(sum(row["label"] == 1 for row in rows)), "negative_rows": int(sum(row["label"] == 0 for row in rows))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dataset-build", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true", help="fetch a fresh source catalog even when the bundled Parquet already exists")
    parser.add_argument("--train-llm", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--max-events", type=int, default=450)
    args = parser.parse_args()
    data_path = ROOT / "data" / "real_global_landslide_observations.parquet"
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    from backend.release import invalidate_readiness
    invalidate_readiness(artifacts, "GPU training started; copied audit results no longer grant readiness")
    report: Dict[str, Any] = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": "cuda", "engineering_ready": False, "pretraining_ready": False, "audit_status": "GPU retraining requires a fresh full-system audit", "label_policy": "COOLR polygon/manual/report-point proxies, temporal absence pseudo-negatives and explicit simulated training-only mechanical negatives."}
    if not args.skip_dataset_build and (args.rebuild_dataset or not data_path.exists()):
        result = subprocess.run([sys.executable, str(ROOT / "backend" / "ml" / "build_real_labeled_dataset.py"), "--output", str(data_path), "--max-events", str(args.max_events)], cwd=str(ROOT), check=False)
        report["dataset_build_exit_code"] = result.returncode
    else:
        report["dataset_build"] = "using bundled existing observation Parquet"
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is not available")
        if not data_path.exists():
            raise FileNotFoundError("dataset not found: {}".format(data_path))
        rows = _load_rows(data_path)
        X, y, groups = verify_rows(rows)
        training_path = ROOT / "data" / "model_training.parquet"
        if not training_path.exists():
            training_path = data_path
        report["labeled_rows"] = len(y)
        report["positive_rows"] = int((y == 1).sum())
        report["negative_rows"] = int((y == 0).sum())
        positive_rows = [row for row in _load_rows(training_path) if row.get("label") == 1]
        report["positive_source_events"] = len({row.get("source_event_id") for row in positive_rows if row.get("source_event_id")})
        report["positive_event_dates"] = sorted({row.get("event_date") for row in positive_rows if row.get("event_date")})
        report["positive_regimes"] = sorted({row.get("regime") for row in positive_rows if row.get("regime")})
        report["model_scope_warning"] = "Inventory polygon and point-buffer proxies differ in spatial precision; temporal-absence controls are not surveyed intact slopes; validation does not establish global or five-regime generalization."
        report["spatial_blocks"] = int(len(np.unique(groups)))
        report["cuda_device"] = torch.cuda.get_device_name(0)
        tuning = tune_cuda_xgb(X, y, groups, args.optuna_trials)
        report["optuna"] = tuning
        model = GlobalMonotonicXGB(
            n_estimators=int(tuning["best_params"]["n_estimators"]),
            max_depth=int(tuning["best_params"]["max_depth"]),
            learning_rate=float(tuning["best_params"]["learning_rate"]),
            min_child_weight=float(tuning["best_params"]["min_child_weight"]),
        )
        model.device = "cuda"
        model.fit(X, y, groups, n_splits=int(tuning["spatial_folds"]))
        from backend.ml.conformal_calibration import fit_reserved_conformal
        report['conformal_calibration']=fit_reserved_conformal(model,ROOT/'data',groups)
        import joblib
        joblib.dump(model, artifacts / "global_multi_causal_xgb.pkl")
        (artifacts / "global_multi_causal_xgb.metadata.json").write_text(json.dumps({
            "training_rows":int(len(y)), "positive_rows":int((y==1).sum()), "negative_rows":int((y==0).sum()),
            "synthetic_training_only_rows":int(sum(str(g).startswith("synthetic:") for g in groups)),
            "physics_model_version":"mechanical_fs_v6", "conformal":model.conformal_, "feature_names":list(FEATURE_NAMES),
            "calibration":"isotonic on source/absence holdouts; simulated controls enter fitting only",
            "probability_policy":"mechanically gated blend; deployed score not separately calibrated",
        }, indent=2), encoding="utf-8")
        report["model"] = {"status": "trained", "calibration": "spatial GroupKFold isotonic", "feature_names": list(FEATURE_NAMES), "monotonic_constraints": list(MONOTONE_VECTOR)}
        try:
            shap_values = model.shap_log_odds_contributions(X)
            report["tree_shap_mean_absolute_log_odds"] = {
                name: float(value) for name, value in zip(FEATURE_NAMES, np.mean(shap_values, axis=0))
            }
        except Exception as exc:
            report["tree_shap_status"] = "unavailable: {}: {}".format(type(exc).__name__, exc)
        hindcast_path = ROOT / "data" / "historical_hindcast.csv"
        metadata_path = data_path.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        holdout = metadata.get("event_group_holdout", {})
        if holdout.get("status") == "ready":
            candidate_path = ROOT / "data" / "historical_hindcast_candidates.parquet"
            if candidate_path.exists():
                try:
                    report["historical_hindcast_export"] = write_hindcast_scores(candidate_path, model, hindcast_path)
                except Exception as exc:
                    report["historical_hindcast_export"] = {"status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)}
        validation = build_validation_report(str(training_path), str(hindcast_path), model)
        validation.stage1.details.append("Training labels include 60-day inventory-absence pseudo-negatives; scores are not verified non-failure metrics.")
        validation.stage1.details.append("Spatial CV and the event-group holdout have different scopes; point-buffer labels are approximate source-location proxies.")
        report["validation"] = validation.to_dict()
        from backend.ml.conformal_calibration import heldout_conformal_diagnostics
        report['conformal_calibration']=model.conformal_
        report['conformal_holdout']=heldout_conformal_diagnostics(model,_load_rows(ROOT/'data/historical_hindcast_candidates.parquet'))
        filtered_csv=ROOT/'data/historical_hindcast_pinls.csv'
        write_hindcast_scores(ROOT/'data/historical_hindcast_pinls.parquet',model,filtered_csv)
        from backend.ml.validation_suite import load_hindcast_csv,historical_hindcast_metrics
        hp,hy,hc=load_hindcast_csv(str(filtered_csv))
        report['stage2_pinls_filtered']=historical_hindcast_metrics(hp,hy,hc).to_dict()
        report['stage2_pinls_filtered']['scope']='Secondary stability-filtered cohort; original holdout retained separately'
        report["stage4_kinematics"] = [item.__dict__ for item in run_kinematic_branch_controls()]
        report["model_status"] = "trained"
    except Exception as exc:
        report["model_status"] = "not_run"
        report["model_error"] = "{}: {}".format(type(exc).__name__, exc)
        report["validation"] = build_validation_report(str(data_path), str(ROOT / "data" / "historical_hindcast.csv"), None).to_dict()
        report["stage4_kinematics"] = [item.__dict__ for item in run_kinematic_branch_controls()]
        report["stage3_controls_status"] = "These physics controls are synthetic checks, not observed scores."
        report_path = artifacts / "scientific_validation_report.json"
        report_path.write_text(json.dumps(portable_payload(report), indent=2, allow_nan=False), encoding="utf-8")
        print("Training stopped safely: {}".format(report["model_error"]))
        print("Report: {}".format(report_path))
        return 2

    if args.train_llm:
        try:
            chatml = ROOT / "data" / "chatml_trajectories.jsonl"
            if chatml.exists():
                from train_qwen25_agent import validate_chatml
                count = validate_chatml(chatml)
            else:
                count = export_chatml_from_observations(rows, chatml)
            report["chatml_trajectory_count"] = count
            report["qwen_exports"] = finetune_qwen(chatml, artifacts / "qwen2_5_7b_lora", artifacts / "terra-shield-qwen-q4_k_m.gguf")
            report["qwen_status"] = "trained; see qwen_exports for conversion status"
        except Exception as exc:
            report["qwen_status"] = "not_run: {}: {}".format(type(exc).__name__, exc)
            print("QLoRA not run: {}".format(report["qwen_status"]))
    report_path = artifacts / "scientific_validation_report.json"
    report_path.write_text(json.dumps(portable_payload(report), indent=2, allow_nan=False), encoding="utf-8")
    print("Training report written to {}".format(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
