"""Monotonic XGBoost with spatially grouped sigmoid calibration."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import hashlib

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
PHYSICS_MODEL_VERSION = 'uncapped_joint_fosm_v15'
MONOTONE_VECTOR = tuple(direction for _, direction in FEATURE_CONSTRAINTS)
# Multi-trigger adaptation of LHASA's rainfall/precondition interaction design.
# Encodes static/dynamic groups; cannot force every tree to contain a trigger.
_STATIC = (4, 5, 13)
_DYNAMIC = tuple(i for i in range(len(FEATURE_NAMES)) if i not in _STATIC)
INTERACTION_CONSTRAINTS = [list(_DYNAMIC)] + [[s, d] for s in _STATIC for d in _DYNAMIC]

TRIGGER_FEATURE_GROUPS = {
    'hydro': ('rain_24h_mm',), 'seismic': ('newmark_dn_cm','pga_g'),
    'cryospheric': ('snowmelt_mm_day','thaw_degradation'),
    'post_fire': ('burn_severity',), 'fluvial_road': ('toe_undercut_mult',),
}

def training_sample_weights(groups):
    """Shared weighting contract for base fits, tuning and out-of-fold scoring."""
    return np.asarray([.60 if str(g).startswith('synthetic:regime:') else
                       .25 if str(g).startswith('synthetic:') else 1. for g in groups],dtype=np.float32)

def trigger_importance_audit(model):
    importance=model.feature_importance()
    groups={name:sum(importance[k] for k in keys) for name,keys in TRIGGER_FEATURE_GROUPS.items()}
    individual={k:importance[k]>0 for k in ('rain_24h_mm','burn_severity','thaw_degradation','snowmelt_mm_day','newmark_dn_cm')}
    return {'feature_importance':importance,'group_importance':groups,
            'all_five_groups_positive':all(v>0 for v in groups.values()),
            'requested_individual_features_positive':individual,
            'all_14_features_positive':all(v>0 for v in importance.values()),
            'passed':all(v>0 for v in importance.values()),
            'scope':'Booster feature usage only; not observed trigger validation or causal identification'}


class TrainingDataError(ValueError):
    """Raised when labeled spatial training data is insufficient or malformed."""


def connected_component_groups(region_blocks, event_groups):
    """Transitive closure: a shared spatial block OR event links two rows.

    Synthetic groups remain training-only and do not connect observational rows.
    Missing event identifiers must be repaired in the source manifest, not guessed.
    """
    regions = np.asarray(region_blocks).astype(str)
    events = np.asarray(event_groups).astype(str)
    if regions.ndim != 1 or regions.shape != events.shape:
        raise ValueError('Region and event vectors must match')
    parent = {}
    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for region, event in zip(regions, events):
        if region.startswith('synthetic:'):
            continue
        if event in ('', 'None', 'nan') or region in ('', 'None', 'nan'):
            raise TrainingDataError('Observational rows require region_block and event_group_id')
        a, b = find('region:'+region), find('event:'+event)
        parent[max(a, b)] = min(a, b)
    members = {}
    for node in sorted(parent):
        members.setdefault(find(node), []).append(node)
    identifiers = {root: 'component:'+hashlib.sha256(json.dumps(nodes).encode()).hexdigest()[:24]
                   for root, nodes in members.items()}
    return np.asarray([region if region.startswith('synthetic:') else identifiers[find('region:'+region)]
                       for region in regions])


def row_component_groups(rows):
    return connected_component_groups([r['region_block'] for r in rows],
                                      [r.get('event_group_id') for r in rows])


def spatial_group_splits(y: Sequence[int], groups: Sequence[Any], n_splits: int, event_groups=None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Allocate intact components by class counts, never by model scores.

    Mixed-label components seed the folds; single-label components then fill
    class deficits. Every fold has both classes when enough components support
    each class. Synthetic rows remain exclusively in the training partitions.
    """
    labels = np.asarray(y, dtype=np.int8)
    group_values = connected_component_groups(groups, event_groups) if event_groups is not None else np.asarray(groups)
    simulated = np.asarray([str(group).startswith("synthetic:") for group in group_values])
    unique = np.unique(group_values[~simulated])
    positive_groups = [group for group in unique if np.any((group_values == group) & (labels == 1))]
    if n_splits < 2 or len(positive_groups) < n_splits:
        raise TrainingDataError("{} positive-bearing spatial groups cannot support {} holdout folds".format(len(positive_groups), n_splits))
    counts = {g: np.bincount(labels[group_values == g], minlength=2) for g in unique}
    mixed = [g for g in unique if np.all(counts[g] > 0)]
    positive_only = [g for g in unique if counts[g][1] and not counts[g][0]]
    negative_only = [g for g in unique if counts[g][0] and not counts[g][1]]
    assignments: Dict[Any, int] = {}
    totals = np.zeros((n_splits, 2), dtype=int)
    for cohort, label in ((mixed, None), (positive_only, 1), (negative_only, 0)):
        for group in sorted(cohort, key=lambda g: (-int(counts[g].sum()), str(g))):
            fold = min(range(n_splits), key=lambda k:
                       (int(totals[k].sum()), k) if label is None else
                       (int(totals[k, label]), int(totals[k].sum()), k))
            assignments[group] = fold
            totals[fold] += counts[group]
    splits = []
    for fold in range(n_splits):
        held_out = [group for group in unique if assignments[group.item() if hasattr(group, "item") else group] == fold]
        test = np.flatnonzero(np.isin(group_values, held_out))
        train = np.flatnonzero(~np.isin(group_values, held_out))
        splits.append((train, test))
    return splits


class GlobalMonotonicXGB:
    """XGB binary classifier; construction and training require optional ML packages."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 4, learning_rate: float = 0.04, min_child_weight: float = 2.5, field_prevalence: Optional[float] = None) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        if field_prevalence is not None and not 0 < field_prevalence < 1:
            raise ValueError('field_prevalence must lie strictly between zero and one')
        self.field_prevalence = field_prevalence
        self.device = detect_xgboost_device()
        self.model: Optional[Any] = None
        self.feature_names_in_ = FEATURE_NAMES
        self.conformal_ = None
        self.physics_model_version_ = PHYSICS_MODEL_VERSION

    def fit(self, X: np.ndarray, y: np.ndarray, spatial_groups: Sequence[Any], n_splits: int = 5) -> "GlobalMonotonicXGB":
        matrix, labels = _validate_xy(X, y)
        groups = np.asarray(spatial_groups)
        if groups.ndim != 1 or groups.size != labels.size:
            raise TrainingDataError("spatial_groups must provide one coarse-region ID per row")
        unique_groups = np.unique(groups)
        if unique_groups.size < n_splits:
            raise TrainingDataError(
                "{} spatial blocks are available; at least {} are required for grouped sigmoid calibration".format(
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
            min_child_weight=getattr(self, "min_child_weight", 2.5),
            device=self.device,
        )
        self.model = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=splits, ensemble=True)
        # Match XGBoost's float32 predictions for sklearn's sigmoid loss buffers.
        weights = training_sample_weights(groups)
        self.model.fit(matrix, labels, sample_weight=weights)
        self.calibration_fold_prevalences_ = [float(np.mean(labels[test])) for _,test in splits]
        observed = ~np.asarray([str(g).startswith('synthetic:') for g in groups])
        self.calibration_prevalence_ = float(np.mean(labels[observed]))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean(self.fold_probabilities(X, corrected=True), axis=0)

    def predict_calibration_population_proba(self, X):
        return np.mean(self.fold_probabilities(X, corrected=False), axis=0)

    def fold_probabilities(self, X, corrected=True):
        model=self._require_fitted();matrix=_validate_X(X)
        predictions=np.asarray([fold.predict_proba(matrix)[:,1] for fold in model.calibrated_classifiers_],dtype=float)
        priors=getattr(self,'calibration_fold_prevalences_',None)
        if corrected and priors is not None and self.field_prevalence is not None:
            if len(priors)!=len(predictions):raise ValueError('Calibration fold prevalence metadata mismatch')
            predictions=np.asarray([prior_adjust_probability(p,prior,self.field_prevalence) for p,prior in zip(predictions,priors)])
        return predictions

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
            raise TrainingDataError('Conformal blocks overlap training/sigmoid folds')
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
        return np.std(self.fold_probabilities(features,corrected=True),axis=0)

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


def prior_adjust_probability(probability, calibration_prevalence, field_prevalence):
    """Label-shift odds adjustment; field prevalence is assumed, not estimated."""
    if not 0<calibration_prevalence<1 or not 0<field_prevalence<1:
        raise ValueError('Prevalences must lie strictly between zero and one')
    p=np.asarray(probability,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probability')
    odds_ratio=(field_prevalence/(1-field_prevalence))/(calibration_prevalence/(1-calibration_prevalence))
    return p*odds_ratio/(1-p+p*odds_ratio)


def mechanical_stability_gate(slope_deg: np.ndarray, fs_dynamic: np.ndarray, p_physical: Optional[np.ndarray] = None) -> np.ndarray:
    """Deprecated compatibility function: no uncalibrated mechanical gate."""
    slope, fs = np.broadcast_arrays(np.asarray(slope_deg, float), np.asarray(fs_dynamic, float))
    if not np.isfinite(slope).all() or not np.isfinite(fs).all():
        raise ValueError('Slope and FS must be finite')
    if p_physical is not None and not np.isfinite(p_physical).all():
        raise ValueError('Physical score must be finite')
    return np.ones_like(fs)


def blend_failure_probability(p_xgb, p_physical, slope_deg, fs_dynamic):
    """Legacy name: return calibrated ML without an unvalidated score mixture.

    Physical mechanism scores are reported separately. No combined probability
    is fitted or claimed by this compatibility adapter.
    """
    ml, physical, slope, fs = np.broadcast_arrays(*[np.asarray(v, float) for v in
        (p_xgb, p_physical, slope_deg, fs_dynamic)])
    if not all(np.isfinite(v).all() for v in (ml, physical, slope, fs)) or np.any((ml < 0) | (ml > 1)):
        raise ValueError('Invalid probability or physical inputs')
    return ml.copy()


def make_base_estimator(
    n_estimators: int = 500,
    max_depth: int = 4,
    learning_rate: float = 0.04,
    device: Optional[str] = None,
    min_child_weight: float = 2.5,
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
        interaction_constraints=json.dumps(INTERACTION_CONSTRAINTS),
        subsample=0.85, colsample_bytree=0.75, colsample_bynode=0.45,
        reg_lambda=2.0, reg_alpha=0.25, random_state=20261103,
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
