"""Monotonic XGBoost with spatially grouped isotonic calibration."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend.realtime.device import detect_xgboost_device

FEATURE_CONSTRAINTS: Tuple[Tuple[str, int], ...] = (
    ("fs_dynamic", -1),
    ("p_f_fosm", 1),
    ("psi_head_m", 1),
    ("newmark_dn_cm", 1),
    ("slope_max_deg", 1),
    ("twi", 1),
    ("soil_saturation", 1),
    ("rain_24h_mm", 1),
    ("snowmelt_mm_day", 1),
    ("pga_g", 1),
    ("thaw_degradation", 1),
    ("burn_severity", 1),
    ("toe_undercut_mult", 1),
    ("profile_curvature", 0),
)
FEATURE_NAMES = tuple(name for name, _ in FEATURE_CONSTRAINTS)
MONOTONE_VECTOR = tuple(direction for _, direction in FEATURE_CONSTRAINTS)


class TrainingDataError(ValueError):
    """Raised when labeled spatial training data is insufficient or malformed."""


def spatial_group_splits(y: Sequence[int], groups: Sequence[Any], n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build deterministic grouped folds with a positive-bearing block in each holdout."""
    labels = np.asarray(y, dtype=np.int8)
    group_values = np.asarray(groups)
    simulated = np.asarray([str(group).startswith("synthetic:") for group in group_values])
    unique = np.unique(group_values[~simulated])
    positive_groups = [group for group in unique if np.any((group_values == group) & (labels == 1))]
    if n_splits < 2 or len(positive_groups) < n_splits:
        raise TrainingDataError("{} positive-bearing spatial groups cannot support {} holdout folds".format(len(positive_groups), n_splits))
    assignments: Dict[Any, int] = {}
    for index, group in enumerate(positive_groups):
        key = group.item() if hasattr(group, "item") else group
        assignments[key] = index % n_splits
    negative_only = [group for group in unique if group not in positive_groups]
    for index, group in enumerate(negative_only):
        key = group.item() if hasattr(group, "item") else group
        assignments[key] = index % n_splits
    splits = []
    for fold in range(n_splits):
        held_out = [group for group in unique if assignments[group.item() if hasattr(group, "item") else group] == fold]
        test = np.flatnonzero(np.isin(group_values, held_out))
        train = np.flatnonzero(~np.isin(group_values, held_out))
        splits.append((train, test))
    return splits


class GlobalMonotonicXGB:
    """XGB binary classifier; construction and training require optional ML packages."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 6, learning_rate: float = 0.04, min_child_weight: float = 1.0) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.device = detect_xgboost_device()
        self.model: Optional[Any] = None
        self.feature_names_in_ = FEATURE_NAMES
        self.conformal_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, spatial_groups: Sequence[Any], n_splits: int = 5) -> "GlobalMonotonicXGB":
        matrix, labels = _validate_xy(X, y)
        groups = np.asarray(spatial_groups)
        if groups.ndim != 1 or groups.size != labels.size:
            raise TrainingDataError("spatial_groups must provide one coarse-region ID per row")
        unique_groups = np.unique(groups)
        if unique_groups.size < n_splits:
            raise TrainingDataError(
                "{} spatial blocks are available; at least {} are required for grouped isotonic calibration".format(
                    unique_groups.size, n_splits
                )
            )
        try:
            from sklearn.calibration import CalibratedClassifierCV
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt to train the CUDA/CPU XGBoost model") from exc

        splits = spatial_group_splits(labels, groups, n_splits)
        for train_idx, test_idx in splits:
            if np.unique(labels[train_idx]).size < 2 or np.unique(labels[test_idx]).size < 2:
                raise TrainingDataError(
                    "each spatial calibration fold must contain both event and non-event labels; revise the blocks or data"
                )
        estimator = make_base_estimator(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=getattr(self, "min_child_weight", 1.0),
            device=self.device,
        )
        self.model = CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=splits, ensemble=True)
        weights = np.asarray([0.25 if str(g).startswith("synthetic:") else 1.0 for g in groups])
        self.model.fit(matrix, labels, sample_weight=weights)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        model = self._require_fitted()
        matrix = _validate_X(X)
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)

    def calibrate_conformal(self, probabilities, labels, groups, training_groups, alpha=.10, scales=None):
        """Ensemble-scaled cell residuals; spatial dependence prevents a coverage guarantee."""
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('Conformal alpha must be between zero and one')
        p=np.asarray(probabilities,float); y=np.asarray(labels,float); g=np.asarray(groups).astype(str)
        if p.ndim!=1 or p.shape!=y.shape or p.shape!=g.shape or not len(p):
            raise ValueError('Conformal vectors must be nonempty and matching')
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isin(y,[0,1]).all():
            raise ValueError('Conformal probabilities and binary labels are invalid')
        if set(g)&set(map(str,training_groups)):
            raise TrainingDataError('Conformal blocks overlap training/isotonic folds')
        scale=np.ones_like(p) if scales is None else np.maximum(np.asarray(scales,float),1e-6)
        if scale.shape!=p.shape or not np.isfinite(scale).all():raise ValueError('Invalid ensemble scales')
        scores=np.abs(y-p)/scale
        rank=int(np.ceil((len(scores)+1)*(1-alpha)))
        q=float(np.sort(scores)[rank-1]) if rank<=len(scores) else 1.0
        self.conformal_={'q':q,'alpha':alpha,'blocks':len(np.unique(g)),'rows':len(y),
                         'target':'binary_event_label_not_true_probability', 'scale':'ensemble_std' if scales is not None else 'unit',
                         'assumption':'exchangeable calibration/test cells; spatial dependence and global shift invalidate nominal coverage guarantees'}
        return self.conformal_

    def ensemble_std(self, features):
        x=_validate_X(features)
        predictions=[fold.predict_proba(x)[:,1] for fold in self._require_fitted().calibrated_classifiers_]
        return np.std(predictions,axis=0)

    def conformal_intervals(self, probabilities, features=None):
        p=np.asarray(probabilities,float)
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Invalid conformal prediction score')
        calibration=getattr(self,'conformal_',None)
        q=calibration['q'] if calibration else 1.0
        if calibration and calibration.get('scale')=='ensemble_std' and features is not None:
            q=q*np.maximum(self.ensemble_std(features),1e-6)
        lower=np.maximum(0.,p-q); upper=np.minimum(1.,p+q)
        width=upper-lower
        classes=np.char.add(np.where(p>=.5,'HIGH_SUSCEPTIBILITY_','LOW_SUSCEPTIBILITY_'),
                            np.where(width>=.25,'HIGH_UNCERTAINTY','LOW_UNCERTAINTY'))
        return np.column_stack((lower,upper)),width,classes

    def epistemic_probability_intervals(self, feature_rows, blended_prob, fosm_sigma_p):
        """Normal spread proxy; not calibrated probability coverage.

        Fold disagreement and a bounded FOSM spread heuristic are treated as independent, although
        both use the same inputs. This does not measure all epistemic uncertainty.
        """
        p = np.asarray(blended_prob, dtype=float)
        physical = np.asarray(fosm_sigma_p, dtype=float)
        fold = self.ensemble_std(feature_rows)
        if p.shape != fold.shape or physical.shape != p.shape:
            raise ValueError('Probability spread arrays must have matching shapes')
        if not np.isfinite(p).all() or not np.isfinite(physical).all() or np.any(physical < 0) or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid probability/spread')
        sigma = np.sqrt(fold**2 + physical**2)
        return np.column_stack((np.clip(p-1.645*sigma,0,1),np.clip(p+1.645*sigma,0,1)))

    def shap_log_odds_contributions(self, X: np.ndarray, absolute: bool = True) -> np.ndarray:
        """Average absolute TreeSHAP contributions from calibrated fold models.

        XGBoost's ``pred_contribs`` are exact TreeSHAP values in raw-margin
        (log-odds) space. The bias column is intentionally omitted.
        """
        model = self._require_fitted()
        matrix = _validate_X(X)
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt for TreeSHAP extraction") from exc
        contributions = []
        for calibrated in getattr(model, "calibrated_classifiers_", []):
            estimator = getattr(calibrated, "estimator", None)
            if estimator is None:
                estimator = getattr(calibrated, "base_estimator", None)
            if estimator is None or not hasattr(estimator, "get_booster"):
                continue
            # Training uses NumPy matrices without column metadata; retain that same
            # positional feature contract for booster contribution predictions.
            dmatrix = xgb.DMatrix(matrix)
            raw_contrib = estimator.get_booster().predict(dmatrix, pred_contribs=True)
            contributions.append(np.asarray(raw_contrib[:, :-1], dtype=np.float64))
        if not contributions:
            raise RuntimeError("calibrated model does not expose XGBoost fold estimators for TreeSHAP")
        return np.mean(np.stack([np.abs(value) if absolute else value for value in contributions], axis=0), axis=0)

    def feature_importance(self) -> Dict[str, float]:
        model = self._require_fitted()
        importances = []
        for calibrated in getattr(model, "calibrated_classifiers_", []):
            estimator = getattr(calibrated, "estimator", None)
            if estimator is not None and hasattr(estimator, "feature_importances_"):
                importances.append(np.asarray(estimator.feature_importances_, dtype=np.float64))
        if not importances:
            return {name: 0.0 for name in FEATURE_NAMES}
        mean = np.mean(np.stack(importances, axis=0), axis=0)
        return {name: float(value) for name, value in zip(FEATURE_NAMES, mean)}

    def _require_fitted(self) -> Any:
        if self.model is None:
            raise RuntimeError("GlobalMonotonicXGB has no fitted model; labeled training data is required")
        return self.model


def mechanical_stability_gate(slope_deg: np.ndarray, fs_dynamic: np.ndarray, p_physical: Optional[np.ndarray] = None) -> np.ndarray:
    """Project-defined smooth mechanical screen; not a calibrated probability."""
    slope, fs = np.broadcast_arrays(np.asarray(slope_deg, dtype=float), np.asarray(fs_dynamic, dtype=float))
    if not np.isfinite(slope).all() or not np.isfinite(fs).all():
        raise ValueError("slope and factor of safety must be finite")
    effective_fs = fs
    if p_physical is not None:
        phys = np.broadcast_to(np.asarray(p_physical, dtype=float), fs.shape)
        if not np.isfinite(phys).all():
            raise ValueError("physical probabilities must be finite")
        effective_fs = np.minimum(fs, 1.40 - 0.65 * np.clip(phys, 0.0, 1.0))
    slope_screen = 1.0 / (1.0 + np.exp(np.clip(-(slope - 8.5) / 0.65, -700, 700)))
    dynamic_screen = 1.0 / (1.0 + np.exp(np.clip((effective_fs - 1.35) / 0.12, -700, 700)))
    return slope_screen * dynamic_screen


def blend_failure_probability(p_xgb: np.ndarray, p_physical: np.ndarray,
                              slope_deg: np.ndarray, fs_dynamic: np.ndarray) -> np.ndarray:
    ml, physical, slope, fs = np.broadcast_arrays(
        np.asarray(p_xgb, dtype=float), np.asarray(p_physical, dtype=float),
        np.asarray(slope_deg, dtype=float), np.asarray(fs_dynamic, dtype=float))
    if not np.isfinite(ml).all() or not np.isfinite(physical).all():
        raise ValueError("probabilities must be finite")
    base_blend = .65*ml + .35*physical
    yield_weight = .35 + .50/(1.+np.exp(np.clip((fs-1.05)/.06,-700,700)))
    yield_blend = (1.-yield_weight)*ml + yield_weight*physical
    combined = np.where(fs<=1.05,np.maximum(base_blend,yield_blend),base_blend)
    return mechanical_stability_gate(slope,fs,physical)*np.clip(combined,0.,1.)


def make_base_estimator(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.04,
    device: Optional[str] = None,
    min_child_weight: float = 1.0,
) -> Any:
    """Construct the constrained XGB estimator, preferring CUDA when available."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("Install requirements-ml.txt for XGBoost") from exc
    selected_device = device or detect_xgboost_device()
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=selected_device,
        monotone_constraints=MONOTONE_VECTOR,
        n_jobs=-1,
    )


def feature_matrix(features: Mapping[str, Any]) -> np.ndarray:
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError("missing model features: {}".format(", ".join(missing)))
    arrays = [np.atleast_1d(np.asarray(features[name], dtype=np.float64)) for name in FEATURE_NAMES]
    matrix = np.column_stack(np.broadcast_arrays(*arrays))
    return _validate_X(matrix)


def _validate_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix = _validate_X(X)
    labels = np.asarray(y, dtype=np.int8)
    if labels.ndim != 1 or labels.size != matrix.shape[0]:
        raise TrainingDataError("labels must be a one-dimensional vector matching X rows")
    if not np.isin(labels, (0, 1)).all() or np.unique(labels).size != 2:
        raise TrainingDataError("binary training labels must contain both 0 (no failure) and 1 (failure)")
    return matrix, labels


def _validate_X(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("X must have shape (n_cells, 14) in FEATURE_NAMES order")
    if not np.isfinite(matrix).all():
        raise ValueError("X contains missing or non-finite values; imputation must be explicit and trained")
    return matrix
