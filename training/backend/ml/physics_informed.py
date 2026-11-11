"""Feature-based monotonic fusion; no per-row FOSM training offset.

The calibrated learned score and a conditional gravity-release score are exposed
separately. The latter couples the learned trigger score to a necessary mechanical
condition (driving shear exceeds cohesion), under explicit lognormal uncertainty.
It is not a calibrated field probability or an empirical non-failure label.
"""
import numpy as np
from scipy.special import expit, ndtr
from scipy.optimize import minimize_scalar
from .rainfall_features import FEATURES
from .rainfall_baseline import grouped_folds
from backend.physics.reference_context import CONTEXT_FEATURES, CONSTRAINTS

VERSION='physics_informed_initiation_v2_features'
EXTRA=('fosm_log10_p','log_fs_mid','critical_rain_deficit_mm','pressure_head_m',
       'newmark_displacement_cm','newmark_available','rain_to_critical_deficit_ratio',
       'effective_shear_stress_surge','gravity_release_admissibility','critical_rain_attainable')
NAMES=(*FEATURES,*CONTEXT_FEATURES,'fosm_reference_score',*EXTRA)
KINDS=('Rainfall-Only','Physics-Only','Weather+Terrain ML','Physics-Informed ML')
MONOTONIC=(1,1,1,1,0,0,*CONSTRAINTS,1,1,-1,-1,1,1,0,1,1,1,0)
assert len(MONOTONIC)==len(NAMES)


def release_admissibility(physical):
    """P(tau_driving > cohesion), a necessary condition after friction loss.

    Log-moment uncertainty avoids negative cohesion tails and flat-slope ratio
    singularities. This is a reference-model upper bound, not a release forecast.
    No location, slope cutoff, rainfall cutoff or desired example score enters it.
    """
    p=physical['input_values'];s=physical['standard_deviations']
    slope=np.radians(p['slope_deg']);c=p['cohesion_kpa']
    shear=p['unit_weight_kn_m3']*p['depth_m']*np.sin(slope)*np.cos(slope)
    if shear<=0:return 0.
    if c<=0:return 1.
    relative_slope=abs(np.cos(2*slope)/max(np.sin(slope)*np.cos(slope),1e-12))*np.radians(s['slope_deg'])
    log_var=np.log1p((s['cohesion_kpa']/c)**2)+np.log1p((s['unit_weight_kn_m3']/p['unit_weight_kn_m3'])**2)+np.log1p((s['depth_m']/p['depth_m'])**2)+np.log1p(relative_slope**2)
    return float(ndtr(np.log(shear/c)/max(np.sqrt(log_var),1e-12)))


def matrix(observable,physical):
    o=np.asarray(observable,float);p=physical;f=p['mechanistic_features']
    deficit=p['critical_rainfall_deficit_mm'];fs=p['fs_mid'];prob=p['p_failure_physical']
    ratio=0. if deficit is None else o[0]/max(deficit,1.)
    # Rain-induced frictional resistance reduction, in kPa. This is an
    # effective-stress perturbation, not an observed external shear load.
    pressure_reduction=9.81*max(p['pressure_head_m']-p['initial_pressure_head_m'],0.)*np.tan(np.radians(p['input_values']['friction_angle_deg']))
    extra=[np.log10(max(prob,1e-300)),np.log(max(fs,1e-12)) if fs is not None else None,
           deficit,p['pressure_head_m'],p.get('newmark_displacement_cm'),
           float(p.get('newmark_displacement_cm') is not None),ratio,
           pressure_reduction,release_admissibility(p),
           float(p['rainfall_threshold_attainable_at_duration'])]
    return np.asarray([*o,*(f[k] for k in CONTEXT_FEATURES),prob,*extra],float)


class ContextModel:
    def __init__(self,kind='Physics-Informed ML',device='cpu'):
        if kind not in KINDS:raise ValueError('Unknown ablation')
        self.kind=kind;self.device=device;self.version=VERSION;self.folds=[];self.tuning=[]

    def indices(self):
        return list(range(2)) if self.kind=='Rainfall-Only' else list(range(6)) if self.kind=='Weather+Terrain ML' else list(range(len(NAMES)))

    def _learner(self,y,regularization=5.,depth=2):
        from xgboost import XGBClassifier
        indices=self.indices()
        return XGBClassifier(n_estimators=120,max_depth=depth,learning_rate=.04,min_child_weight=3.,
            reg_lambda=regularization,max_delta_step=1.,tree_method='hist',device=self.device,n_jobs=2,
            random_state=42,base_score=float(np.mean(y)),
            monotone_constraints=tuple(MONOTONIC[i] for i in indices),objective='binary:logistic',
            colsample_bynode=.85 if self.kind=='Physics-Informed ML' else 1.)

    def _fit_tree(self,x,y,regularization,depth):
        indices=self.indices();tree=self._learner(y,regularization,depth)
        # Weight static-prior groups during column sampling, rather than scaling
        # their values (tree splits are invariant to such value rescaling).
        weights=np.array([1. if i in (0,1,2) or NAMES[i] in ('rain_to_critical_deficit_ratio','effective_shear_stress_surge','pressure_head_m') else .25 for i in indices])
        tree.fit(x[:,indices],y,feature_weights=weights)
        if self.device=='cuda':
            import json
            selected=json.loads(tree.get_booster().save_config())['learner']['generic_param']['device']
            if not selected.startswith('cuda'):raise RuntimeError('Requested CUDA context model fell back to CPU')
        return tree

    def fit(self,x,y,groups):
        x=np.asarray(x,float);y=np.asarray(y);groups=np.asarray(groups);indices=self.indices()
        if self.kind=='Physics-Only':return self
        self.folds=[];self.tuning=[]
        for train,cal in grouped_folds(y,groups,maximum=3):
            candidates=((5.,2),(20.,1),(20.,2)) if self.kind=='Physics-Informed ML' else ((5.,2),)
            losses=[]
            if len(candidates)>1:
                for lam,depth in candidates:
                    scores=[]
                    for a,b in grouped_folds(y[train],groups[train],maximum=2):
                        tree=self._fit_tree(x[train][a],y[train][a],lam,depth)
                        scores.extend((tree.predict_proba(x[train][b][:,indices])[:,1]-y[train][b])**2)
                    losses.append(float(np.mean(scores)))
                chosen=int(np.argmin(losses))
            else:chosen=0
            lam,depth=candidates[chosen]
            tree=self._fit_tree(x[train],y[train],lam,depth)
            # Only the disjoint calibration components fit this intercept.
            margin=tree.predict(x[cal][:,indices],output_margin=True)
            offset=minimize_scalar(lambda b:float(np.mean(np.logaddexp(0,margin+b)-y[cal]*(margin+b))),bounds=(-12.,12.),method='bounded').x
            self.folds.append((tree,float(offset)))
            self.tuning.append({'fit_components':sorted(map(str,set(groups[train]))),
                'calibration_components':sorted(map(str,set(groups[cal]))),'candidates':candidates,
                'inner_brier':losses,'selected_lambda':lam,'selected_depth':depth,
                'initial_prevalence':float(np.mean(y[train])),'initial_hessian':float(np.mean(y[train])*(1-np.mean(y[train])))})
        return self

    def predict_learned_proba(self,x):
        x=np.asarray(x,float)
        if self.kind=='Physics-Only':return x[:,NAMES.index('fosm_reference_score')].copy()
        if not self.folds:raise RuntimeError('Unfitted context model')
        return np.mean([expit(tree.predict(x[:,self.indices()],output_margin=True)+offset) for tree,offset in self.folds],axis=0)

    def predict_proba(self,x):
        score=self.predict_learned_proba(x)
        if self.kind=='Physics-Informed ML':score=score*np.asarray(x,float)[:,NAMES.index('gravity_release_admissibility')]
        return score

    def explain(self,x):
        import xgboost as xgb
        x=np.asarray(x,float);indices=self.indices();results=[]
        for tree,offset in self.folds:
            terms=tree.get_booster().predict(xgb.DMatrix(x[:,indices]),pred_contribs=True)
            results.append((terms,offset))
        gate=x[:,NAMES.index('gravity_release_admissibility')] if self.kind=='Physics-Informed ML' else np.ones(len(x))
        return [{'feature_log_odds_terms':{NAMES[k]:float(np.mean([t[j,i] for t,_ in results])) for i,k in enumerate(indices)},
                 'folds':[{'terms':dict(zip([NAMES[k] for k in indices],map(float,t[j,:-1]))),
                           'bias':float(t[j,-1]),'calibration_offset':offset} for t,offset in results],
                 'learned_score':float(self.predict_learned_proba(x[j:j+1])[0]),'mechanical_admissibility':float(gate[j]),
                 'scope':'Native tree SHAP on learned log odds; calibrated learned score times exposed necessary-condition mechanical factor. Neither is an identified causal effect or calibrated global probability.'} for j in range(len(x))]
