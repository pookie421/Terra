"""Spatial, historical-hindcast, counterfactual, and monotonicity validation."""

from __future__ import annotations
from backend.ml.global_xgb import row_component_groups

import csv
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from backend.ml.global_xgb import training_sample_weights
import numpy as np
from backend.ml.global_xgb import make_base_estimator
import h3

from backend.physics.multi_causal_engine import MultiCausalEngine, PhysicsInputs, PhysicsState
from backend.physics.kinematics import solve_kinematics
from backend.physics.causal_attribution import CAUSES, explain_causes, select_numbers_that_matter
from .global_xgb import FEATURE_NAMES, GlobalMonotonicXGB, _validate_X, spatial_group_splits


@dataclass
class StageResult:
    name: str
    status: str
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    details: List[str] = field(default_factory=list)
    confidence_intervals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualResult:
    name: str
    fs_static: float
    fs_dynamic: float
    p_failure_physical: float
    passed: bool
    assertion: str
    peak_velocity_m_s: float
    velocity_class: int
    failure_mode: str
    cause_percentages: Dict[str, float]
    numbers_that_matter: List[Dict[str, Any]]


@dataclass
class KinematicBranchResult:
    name: str
    expected_mode: str
    actual_mode: str
    peak_velocity_m_s: float
    velocity_class: int
    passed: bool


@dataclass
class ValidationReport:
    stage1: StageResult
    stage2: StageResult
    stage3_controls: List[CounterfactualResult]
    stage3_causal_controls: List[Dict[str, Any]]
    stage3_monotonicity: StageResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage1": self.stage1.to_dict(),
            "stage2": self.stage2.to_dict(),
            "stage3_controls": [asdict(result) for result in self.stage3_controls],
            "stage3_causal_controls": self.stage3_causal_controls,
            "stage3_monotonicity": self.stage3_monotonicity.to_dict(),
        }


def load_labeled_training_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read explicitly labeled, spatially blocked rows; never generate training labels."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = set(FEATURE_NAMES) | {"label", "region_block", "event_group_id"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError("training CSV is missing columns: {}".format(", ".join(missing)))
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("training CSV contains no labeled rows")
    X = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    groups = row_component_groups(rows)
    if not np.isfinite(X).all():
        raise ValueError("training CSV contains missing/non-finite feature values")
    return X, y, groups


def load_labeled_training_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load explicitly labeled rows from CSV or Parquet; silently discard no nulls."""
    if path.lower().endswith((".parquet", ".pq")):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet validation requires pyarrow") from exc
        table = pq.read_table(path)
        required = set(FEATURE_NAMES) | {"label", "region_block", "event_group_id"}
        missing = required - set(table.column_names)
        if missing:
            raise ValueError("training Parquet is missing columns: {}".format(", ".join(sorted(missing))))
        rows = table.to_pylist()
        rows = [row for row in rows if row.get("label") in (0, 1) and all(row.get(name) is not None for name in FEATURE_NAMES)]
        if not rows:
            raise ValueError("Parquet has no rows with verified binary labels and complete model features")
        X = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
        y = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
        groups = row_component_groups(rows)
        if not np.isfinite(X).all():
            raise ValueError("training Parquet contains non-finite feature values")
        return X, y, groups
    return load_labeled_training_csv(path)


def coarse_spatial_blocks(
    latitudes: Sequence[float], longitudes: Sequence[float], resolution: int = 4
) -> np.ndarray:
    """Map samples to coarse H3 region IDs for leakage-resistant GroupKFold."""
    lat = np.asarray(latitudes, dtype=np.float64)
    lon = np.asarray(longitudes, dtype=np.float64)
    if lat.ndim != 1 or lon.shape != lat.shape or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("latitude and longitude must be matching finite 1D arrays")
    if resolution < 0 or resolution > 15:
        raise ValueError("H3 coarse resolution must be between 0 and 15")
    cell_fn = getattr(h3, "latlng_to_cell", getattr(h3, "geo_to_h3", None))
    if cell_fn is None:
        raise RuntimeError("installed h3-py version has no coordinate-to-cell API")
    return np.asarray([str(cell_fn(float(a), float(b), resolution)) for a, b in zip(lat, lon)])


def load_hindcast_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read independently held-out event-time probability rows."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    predictions, labels, case_ids = [], [], []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"case_id", "p_failure", "observed_failure"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError("hindcast CSV is missing columns: {}".format(", ".join(sorted(missing))))
        for row in reader:
            case_ids.append(row["case_id"])
            predictions.append(float(row["p_failure"]))
            labels.append(int(row["observed_failure"]))
    if len(set(case_ids)) < 1:
        raise ValueError("hindcast CSV contains no held-out case identifiers")
    return np.asarray(predictions), np.asarray(labels, dtype=np.int8), np.asarray(case_ids)


def spatial_block_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    regions: Sequence[Any],
    n_splits: int = 5,
    model_factory: Optional[Any] = None,
) -> StageResult:
    """Report spatially held-out ROC-AUC, PR-AUC, and Brier Skill Score."""
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        from .global_xgb import make_base_estimator
    except ImportError as exc:
        return StageResult("Stage 1: spatial block CV", "not_run", details=["scikit-learn/XGBoost unavailable: {}".format(exc)])
    matrix = _validate_X(X)
    labels = np.asarray(y, dtype=np.int8)
    groups = np.asarray(regions)
    if labels.shape != (matrix.shape[0],) or groups.shape != labels.shape:
        raise ValueError("X, y, and regions must have matching row counts")
    if np.unique(labels).size != 2:
        return StageResult(
            "Stage 1: spatial block CV",
            "not_run",
            details=["both positive and negative labels are required; found class(es) {}".format(np.unique(labels).tolist())],
        )
    if np.unique(groups).size < n_splits:
        return StageResult(
            "Stage 1: spatial block CV",
            "not_run",
            details=["{} spatial blocks; {} required".format(np.unique(groups).size, n_splits)],
        )
    fold_metrics = []
    oof=np.full(len(labels),np.nan)
    try:
        folds = spatial_group_splits(labels, groups, n_splits)
    except ValueError as exc:
        return StageResult("Stage 1: spatial block CV", "not_run", details=[str(exc)])
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        if np.unique(labels[train_idx]).size != 2 or np.unique(labels[test_idx]).size != 2:
            return StageResult(
                "Stage 1: spatial block CV",
                "not_run",
                details=["fold {} lacks both classes; no random split fallback was used".format(fold)],
            )
        try:
            estimator = model_factory() if model_factory else make_base_estimator()
        except RuntimeError as exc:
            return StageResult("Stage 1: spatial block CV", "not_run", details=[str(exc)])
        try:
            weights = training_sample_weights(groups[train_idx])
            estimator.fit(matrix[train_idx], labels[train_idx], sample_weight=weights)
            probability = np.asarray(estimator.predict_proba(matrix[test_idx])[:, 1], dtype=np.float64)
        except Exception as exc:
            return StageResult(
                "Stage 1: spatial block CV",
                "failed",
                details=["fold {} model error: {}: {}".format(fold, type(exc).__name__, exc)],
            )
        oof[test_idx]=probability
        observed_train = train_idx[[not str(g).startswith("synthetic:") for g in groups[train_idx]]]
        prevalence = float(labels[observed_train].mean())
        brier_model = float(np.mean((probability - labels[test_idx]) ** 2))
        brier_reference = float(np.mean((prevalence - labels[test_idx]) ** 2))
        brier_skill = 1.0 - brier_model / brier_reference if brier_reference > 0 else float("nan")
        fold_metrics.append(
            (
                float(roc_auc_score(labels[test_idx], probability)),
                float(average_precision_score(labels[test_idx], probability)),
                brier_model,
                brier_skill,
            )
        )
    scores = np.asarray(fold_metrics, dtype=np.float64)
    from .uncertainty import cluster_intervals
    seen=np.isfinite(oof)
    intervals=cluster_intervals(labels[seen],oof[seen],groups[seen])
    return StageResult(
        "Stage 1: spatial block CV",
        "scored",
        {
            "roc_auc": float(roc_auc_score(labels[seen],oof[seen])),
            "pr_auc": float(average_precision_score(labels[seen],oof[seen])),
            "brier_score": float(np.mean((labels[seen]-oof[seen])**2)),
            "brier_skill_score": float(np.nanmean(scores[:, 3])),
            **{k:v for k,v in run_lhasa_benchmark_suite(labels[seen],oof[seen]).items() if k in ("tss","csi","fnr","tpr","fpr")},
        },
        ["Pooled out-of-fold predictions from {} spatial holdout folds; simulated regime anchors enter training only and are excluded from evaluation and calibration holdouts. Metrics here describe the raw constrained XGBoost score; Stage 2 describes the ungated prior-adjusted sigmoid event-buffer score, a different estimand from live gated scores.".format(n_splits)],
        confidence_intervals=intervals,
    )


def historical_hindcast_metrics(
    predictions: np.ndarray, observed: np.ndarray, case_ids: Sequence[str]
) -> StageResult:
    """Score probability hindcasts only when event-time labels are supplied."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from .uncertainty import cluster_intervals

    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(observed, dtype=np.int8)
    cases = np.asarray(case_ids).astype(str)
    if p.shape != y.shape or p.shape != cases.shape or p.ndim != 1:
        raise ValueError("hindcast predictions, labels, and case IDs must be matching vectors")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
        raise ValueError("hindcast probabilities must be in [0,1] and labels binary")
    if p.size < 2 or np.unique(y).size != 2:
        return StageResult(
            "Stage 2: historical catastrophe hindcast",
            "not_run",
            details=["independent hindcast requires at least one observed event and one observed non-event row"],
        )
    brier = float(np.mean((p - y) ** 2))
    hit_rate = float(np.mean((p >= 0.5) == y))
    k = max(1, int(np.ceil(0.10 * len(y))))
    top_k = np.argsort(-p, kind="stable")[:k]
    recall_at_k = float(np.sum(y[top_k] == 1) / np.sum(y == 1))
    unique_cases = sorted(set(cases.tolist()))
    return StageResult(
        "Stage 2: historical catastrophe hindcast",
        "scored",
        {
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y,p)),
            "brier_score": brier,
            "threshold_0_5_accuracy": hit_rate,
            **{k:v for k,v in run_lhasa_benchmark_suite(y,p).items() if k in ("tss","csi","fnr","tpr","fpr")},
            "recall_at_k": recall_at_k,
            "recall_at_40": float(np.sum(y[np.argsort(-p, kind="stable")[:min(40,len(y))]] == 1) / np.sum(y == 1)),
            "recall_at_40_k": float(min(40,len(y))),
            "rows": float(len(y)),
            "positive_rows": float(np.sum(y==1)),
            "heldout_events": float(len(unique_cases)),
            "recall_at_k_k": float(k),
            "recall_at_k_fraction": float(k / len(y)),
        },
        ["{} event-time rows across {} held-out case(s); Recall@K uses the top 10% by predicted probability (K={}); Recall@40 also reported with its actual K. Stable input order breaks tied scores. On small samples Recall@40 can be uninformative.".format(len(y), len(unique_cases), k)],
        confidence_intervals=cluster_intervals(y,p,cases),
    )


def run_counterfactual_controls(engine: Optional[MultiCausalEngine] = None) -> List[CounterfactualResult]:
    """Run the two explicit counterfactual controls from the phase prompt."""
    solver = engine or MultiCausalEngine()
    scenarios = [
        (
            "Flat Floodplain Control",
            PhysicsInputs(
                sand_fraction=0.40,
                clay_fraction=0.25,
                organic_matter_fraction=0.02,
                bulk_density_g_cm3=1.35,
                slope_deg=1.5,
                twi=8.0,
                soil_saturation_fraction=0.50,
                rain_24h_mm=350.0,
                storm_duration_hr=24.0,
                snow_depth_m=0.0,
                air_temperature_c=20.0,
                soil_temperature_c=18.0,
                freeze_thaw_cycles_7d=0.0,
                burn_severity=0.0,
                ndvi_norm=0.80,
                pga_g=0.0,
                road_cut_distance_m=1000.0,
                river_distance_m=1000.0,
            ),
            lambda s: s.fs_dynamic[0] > 3.0 and s.p_failure_physical[0] < 0.02,
            "FS_dynamic > 3.0 and FOSM P(failure) < 0.02",
        ),
        (
            "Dry Cliff Control",
            PhysicsInputs(
                sand_fraction=0.40,
                clay_fraction=0.25,
                organic_matter_fraction=0.02,
                bulk_density_g_cm3=1.35,
                slope_deg=40.0,
                twi=7.0,
                soil_saturation_fraction=0.15,
                rain_24h_mm=0.0,
                storm_duration_hr=24.0,
                snow_depth_m=0.0,
                air_temperature_c=15.0,
                soil_temperature_c=14.0,
                freeze_thaw_cycles_7d=0.0,
                burn_severity=0.0,
                ndvi_norm=0.80,
                pga_g=0.0,
                road_cut_distance_m=1000.0,
                river_distance_m=1000.0,
            ),
            lambda s: s.fs_dynamic[0] > 1.25 and s.p_failure_physical[0] < 0.05,
            "FS_dynamic > 1.25 and FOSM P(failure) < 0.05",
        ),
    ]
    results = []
    for name, inputs, assertion, text in scenarios:
        state = solver.compute(inputs)
        passed = bool(assertion(state))
        kinematics = solve_kinematics(inputs, state)
        breakdown = explain_causes(inputs, state)
        numbers = select_numbers_that_matter(0, inputs, state, breakdown)
        results.append(
            CounterfactualResult(
                name,
                float(state.fs_static[0]),
                float(state.fs_dynamic[0]),
                float(state.p_failure_physical[0]),
                passed,
                text,
                float(kinematics.peak_velocity_m_s[0]),
                int(kinematics.velocity_class[0]),
                str(kinematics.failure_mode[0]),
                {cause: float(breakdown.combined_percentages[cause][0]) for cause in breakdown.combined_percentages},
                [item.to_dict() for item in numbers],
            )
        )
    return results


def run_causal_regime_controls(engine: Optional[MultiCausalEngine] = None) -> List[Dict[str, Any]]:
    """Check trigger attribution with explicit, synthetic physical scenarios."""
    solver = engine or MultiCausalEngine()
    shared = dict(
        sand_fraction=0.40,
        clay_fraction=0.25,
        organic_matter_fraction=0.02,
        bulk_density_g_cm3=1.35,
        slope_deg=35.0,
        twi=12.0,
        soil_saturation_fraction=0.25,
        rain_24h_mm=0.0,
        storm_duration_hr=24.0,
        snow_depth_m=0.0,
        air_temperature_c=15.0,
        soil_temperature_c=14.0,
        freeze_thaw_cycles_7d=0.0,
        burn_severity=0.0,
        ndvi_norm=0.65,
        pga_g=0.0,
        road_cut_distance_m=9999.0,
        river_distance_m=1000.0,
    )
    scenarios = [
        ("Dry seismic trigger", "seismic_tectonic", {"pga_g": 0.35}, "Dry slope, explicit +0.35 g PGA."),
        ("Snowmelt trigger", "cryospheric", {"snow_depth_m": 0.50, "air_temperature_c": 8.0, "soil_temperature_c": 4.0}, "Explicit 0.50 m snow cover and +8 °C warming; this is a synthetic snow-present scenario."),
        ("Post-wildfire rain trigger", "post_wildfire_biotic", {"burn_severity": 0.85, "rain_24h_mm": 65.0}, "Explicit 0.85 burn severity and 65 mm rainfall."),
    ]
    results = []
    for name, expected, changes, assumption in scenarios:
        values = dict(shared)
        values.update(changes)
        inputs = PhysicsInputs(**values)
        state = solver.compute(inputs)
        breakdown = explain_causes(inputs, state)
        percentages = {cause: float(breakdown.combined_percentages[cause][0]) for cause in CAUSES}
        actual = max(percentages, key=percentages.get)
        results.append({
            "name": name,
            "expected_dominant_cause": expected,
            "actual_dominant_cause": actual,
            "cause_percentages": percentages,
            "passed": actual == expected,
            "scenario_assumption": assumption,
        })
    return results


def run_kinematic_branch_controls(engine: Optional[MultiCausalEngine] = None) -> List[KinematicBranchResult]:
    """Exercise each ordered velocity branch with explicit scenario inputs."""
    solver = engine or MultiCausalEngine()
    shared = dict(
        sand_fraction=0.05,
        clay_fraction=0.05,
        organic_matter_fraction=0.005,
        bulk_density_g_cm3=1.70,
        slope_deg=40.0,
        twi=7.0,
        soil_saturation_fraction=0.40,
        rain_24h_mm=0.0,
        storm_duration_hr=24.0,
        snow_depth_m=0.0,
        air_temperature_c=15.0,
        soil_temperature_c=14.0,
        freeze_thaw_cycles_7d=0.0,
        burn_severity=0.0,
        ndvi_norm=0.20,
        pga_g=0.0,
        road_cut_distance_m=1000.0,
        river_distance_m=1000.0,
    )
    # Saturated-yield branch fixture; separate forecast tests cover runoff over stable slabs.
    scenarios = [
        ("Branch A rapid flow", "Rapid fluidized debris flow / avalanche", {"soil_saturation_fraction": 0.99}),
        ("Branch B coseismic", "Coseismic rock avalanche / block slide", {"pga_g": 1.0}),
        # Transitional rain fixture exercises the cohesive slump branch.
        ("Branch C slump", "Cohesive slump / thaw slump / toe-cut wedge", {"rain_24h_mm": 25.0}),
        (
            "Branch D stable",
            "Stable / sub-perception creep",
            {
                "sand_fraction": 0.40,
                "clay_fraction": 0.25,
                "organic_matter_fraction": 0.02,
                "bulk_density_g_cm3": 1.35,
                "soil_saturation_fraction": 0.15,
                "ndvi_norm": 0.80,
            },
        ),
    ]
    results = []
    for name, expected, changes in scenarios:
        scenario = dict(shared)
        scenario.update(changes)
        inputs = PhysicsInputs(**scenario)
        state = solver.compute(inputs)
        kinematics = solve_kinematics(inputs, state)
        actual = str(kinematics.failure_mode[0])
        results.append(
            KinematicBranchResult(
                name,
                expected,
                actual,
                float(kinematics.peak_velocity_m_s[0]),
                int(kinematics.velocity_class[0]),
                actual == expected,
            )
        )
    return results


def monotonicity_sweep(
    model: Optional[GlobalMonotonicXGB],
    baseline_features: Optional[np.ndarray] = None,
    steps: int = 21,
) -> StageResult:
    """Check that model probability respects the four defined stress sweeps."""
    if model is None or model.model is None:
        return StageResult(
            "Stage 3: strict monotonicity sweep",
            "not_run",
            details=["no labeled, fitted monotonic classifier is available"],
        )
    baseline = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32) if baseline_features is None else _validate_X(baseline_features)
    if baseline.shape[0] != 1 or steps < 2:
        raise ValueError("baseline_features must have one row and steps must be at least 2")
    columns = {
        "rain_24h_mm": (7, np.linspace(0.0, 500.0, steps), 1),
        "pga_g": (9, np.linspace(0.0, 1.0, steps), 1),
        "snowmelt_mm_day": (8, np.linspace(0.0, 200.0, steps), 1),
        "burn_severity": (11, np.linspace(0.0, 1.0, steps), 1),
    }
    total = violations = 0
    details = []
    for name, (column, values, direction) in columns.items():
        sweep = np.repeat(baseline, steps, axis=0)
        sweep[:, column] = values
        probabilities = model.predict_proba(sweep)
        differences = np.diff(probabilities) * direction
        count = int(np.sum(differences < -1e-8))
        total += max(0, len(differences))
        violations += count
        details.append("{}: {}/{} adjacent comparisons violated".format(name, count, len(differences)))
    rate = violations / total if total else 0.0
    return StageResult(
        "Stage 3: strict monotonicity sweep",
        "passed" if violations == 0 else "failed",
        {"violation_rate": rate, "violations": float(violations), "comparisons": float(total)},
        details,
    )


def build_validation_report(
    training_csv: str = "data/landslide_training.csv",
    hindcast_csv: str = "data/historical_hindcast.csv",
    model: Optional[GlobalMonotonicXGB] = None,
) -> ValidationReport:
    try:
        X, y, regions = load_labeled_training_data(training_csv)
        positives_by_region = len(np.unique(np.asarray(regions)[np.asarray(y) == 1]))
        folds = min(5, positives_by_region)
        if folds < 3:
            stage1 = StageResult(
                "Stage 1: spatial block CV", "not_run",
                details=["only {} positive-bearing spatial blocks; at least 3 are required".format(positives_by_region)],
            )
        else:
            factory = None
            if model is not None:
                factory = lambda: make_base_estimator(n_estimators=model.n_estimators,
                    max_depth=model.max_depth, learning_rate=model.learning_rate,
                    min_child_weight=getattr(model, "min_child_weight", 2.5), device=model.device)
            stage1 = spatial_block_cross_validation(X, y, regions, n_splits=folds, model_factory=factory)
    except (FileNotFoundError, ValueError) as exc:
        stage1 = StageResult(
            "Stage 1: spatial block CV",
            "not_run",
            details=["{}; expected columns: 14 model features, label, region_block".format(exc)],
        )
    try:
        p, y, cases = load_hindcast_csv(hindcast_csv)
        stage2 = historical_hindcast_metrics(p, y, cases)
    except (FileNotFoundError, ValueError) as exc:
        stage2 = StageResult("Stage 2: historical catastrophe hindcast", "not_run", details=[str(exc)])
    controls = run_counterfactual_controls()
    causal_controls = run_causal_regime_controls()
    monotonic = monotonicity_sweep(model)
    return ValidationReport(stage1, stage2, controls, causal_controls, monotonic)


def evaluate_holdout_score_variants(model, rows):
    """Score identical frozen cases using calibrated ML and the deployed blend."""
    from .global_xgb import blend_failure_probability
    matrix=np.asarray([[r[k] for k in FEATURE_NAMES] for r in rows],dtype=np.float32)
    labels=np.asarray([r['label'] for r in rows])
    groups=[r['event_group_id'] for r in rows]
    calibrated=model.predict_proba(matrix)
    physical=np.asarray([r['p_failure_physical'] for r in rows])
    blended=blend_failure_probability(calibrated,physical,matrix[:,4],matrix[:,0])
    raw=model.predict_calibration_population_proba(matrix)
    out={'threshold':.12,'assumed_field_prevalence':getattr(model,'field_prevalence',None),
         'scope':'Same cases, no threshold tuning; calibrated means prior-adjusted sigmoid ensemble; .15 prevalence is assumed, not operationally measured'}
    for name,p in [('calibrated',calibrated),('live_gated_blend',blended),('uncorrected_calibration_population',raw)]:
        result=historical_hindcast_metrics(p,labels,groups).to_dict()
        m=result['metrics']
        result['targets']={'auc_gte_0_80':m['roc_auc']>=.80,'fpr_lte_0_25':m['fpr']<=.25,'tss_gte_0_55':m['tss']>=.55}
        out[name]=result
    out['all_targets_passed']=all(all(out[k]['targets'].values()) for k in ('calibrated','live_gated_blend'))
    return out


def holdout_pair_diagnostics(original, revised):
    from .pinls import quiescent_control
    old={r['event_id']:r for r in original};new={r['event_id']:r for r in revised};pairs=[]
    for positive in original:
        if positive['label']!=1:continue
        key=positive['event_id']+':tminus60';negative=old[key];replacement=new[key]
        pairs.append({'positive_id':positive['event_id'],'event_date':positive['event_date'],
            'representative_terrain_cell':positive.get('representative_terrain_cell_id'),
            'positive_fs':positive['fs_dynamic'],'positive_slope':positive['slope_max_deg'],
            'original_control_date':negative['event_date'],'original_control_rain':negative['rain_24h_mm'],
            'original_control_fs':negative['fs_dynamic'],'original_control_quiescent':quiescent_control(negative),
            'revised_control_date':replacement['event_date'],'revised_control_fs':replacement['fs_dynamic'],
            'revised_control_rain':replacement['rain_24h_mm'],'revised_control_quiescent':quiescent_control(replacement),
            'date_changed':negative['event_date']!=replacement['event_date'],
            'same_representative_cell':positive.get('representative_terrain_cell_id')==negative.get('representative_terrain_cell_id')==replacement.get('representative_terrain_cell_id')})
    return {'pairs':pairs,'changed_controls':sum(r['date_changed'] for r in pairs),
            'all_pairs_share_terrain':all(r['same_representative_cell'] for r in pairs),
            'all_revised_controls_quiescent':all(r['revised_control_quiescent'] for r in pairs),
            'scope':'Revised controls change the evaluation population; original benchmark retained separately'}


def ranking_metrics(labels, probabilities):
    """Attainable empirical operating points; no test-set deployment tuning.

    AP depends on background prevalence. Ties remain tied; no ROC interpolation.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    y, p = np.asarray(labels), np.asarray(probabilities, float)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isin(y, [0,1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected binary labels and finite probabilities')
    result = {'rows':int(len(y)), 'positive_rows':int(np.count_nonzero(y == 1)), 'negative_rows':int(np.count_nonzero(y == 0)),
              'sample_prevalence':float(np.mean(y)), 'brier_score':float(np.mean((p-y)**2))}
    if len(np.unique(y)) != 2:
        return {**result, 'status':'unavailable_single_class', 'roc_auc':None, 'average_precision':None}
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    result.update(status='scored', roc_auc=float(roc_auc_score(y,p)), average_precision=float(average_precision_score(y,p)),
                  empirical_fpr_step=1./sum(y == 0),
                  zero_false_positive_one_sided_95_upper=1.-.05**(1./sum(y == 0)))
    result['operating_points'] = {}
    for target in (.01, .05, .10):
        candidates = np.flatnonzero(fpr <= target)
        best = candidates[np.argmax(tpr[candidates])]
        result['operating_points'][str(target)] = {'tpr':float(tpr[best]), 'achieved_fpr':float(fpr[best]),
            'threshold':float(thresholds[best]) if np.isfinite(thresholds[best]) else None}
    result['scope'] = 'Descriptive empirical ROC points; thresholds are not approved deployment thresholds. AP and Brier describe this sampled population.'
    return result


def run_lhasa_benchmark_suite(labels=None, probabilities=None, scenarios=None, threshold=.12):
    """LHASA-inspired metrics on TERRA data, not a reproduction of NASA validation.

    The 0.12 threshold is fixed before scoring; no holdout threshold optimization.
    Scenarios are deliberately constructed tests, not observed catchment outcomes.
    """
    result={'threshold':threshold,'scope':'TERRA descriptive evaluation; no scenario acceptance targets or NASA certification'}
    if labels is not None or probabilities is not None:
        y=np.asarray(labels);p=np.asarray(probabilities,float)
        if y.ndim!=1 or y.shape!=p.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Expected matching binary labels and finite probability vectors')
        if len(np.unique(y))!=2:raise ValueError('Both classes required for TSS/FNR')
        hit=p>=threshold
        tp=int(np.sum(hit&(y==1)));fn=int(np.sum(~hit&(y==1)))
        fp=int(np.sum(hit&(y==0)));tn=int(np.sum(~hit&(y==0)))
        tpr=tp/(tp+fn);fpr=fp/(fp+tn)
        result.update(tp=tp,fn=fn,fp=fp,tn=tn,tpr=tpr,fpr=fpr,fnr=1-tpr,tss=tpr-fpr,csi=tp/(tp+fn+fp))
        result['ranking_metrics']=ranking_metrics(y,p)
    if scenarios is not None:
        active=[r for r in scenarios if r['expected_active']];calm=[r for r in scenarios if not r['expected_active']]
        if not active or not calm:raise ValueError('Both active and calm scenarios required')
        scores=np.asarray([r['score'] for r in scenarios],float)
        if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):raise ValueError('Invalid scenario score')
        result['scenarios']=scenarios
        result['scenario_tss']=sum(r['score']>=.85 for r in active)/len(active)-sum(r['score']>=.85 for r in calm)/len(calm)
        result['scenario_scope']='Constructed diagnostics only; no predictive acceptance gate'
    return result
