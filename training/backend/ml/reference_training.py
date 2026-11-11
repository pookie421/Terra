"""Offline baseline fitting and event/spatial component evaluation."""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .global_xgb import (FEATURE_NAMES, PHYSICS_MODEL_VERSION, GlobalMonotonicXGB,
                        spatial_group_splits, make_base_estimator, row_component_groups)
from .validation_suite import load_labeled_training_data, ranking_metrics


def tune_parameters(x, y, groups, device, trials):
    if trials <= 0:
        return {'n_estimators':500,'max_depth':4,'learning_rate':.04,'min_child_weight':2.5}
    import optuna
    from sklearn.metrics import roc_auc_score
    n=min(5,len(set(groups[y == 1])))
    folds=spatial_group_splits(y,groups,n)
    def objective(trial):
        params={'n_estimators':trial.suggest_int('n_estimators',200,600,step=100),
                'learning_rate':trial.suggest_float('learning_rate',.015,.12,log=True),
                'max_depth':4,'min_child_weight':2.5}
        scores=[]
        for train,test in folds:
            model=make_base_estimator(device=device,**params)
            model.fit(x[train],y[train])
            scores.append(roc_auc_score(y[test],model.predict_proba(x[test])[:,1]))
        return float(np.mean(scores))
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=trials)
    return {**study.best_params,'max_depth':4,'min_child_weight':2.5}


def complete(rows):
    return [r for r in rows if r.get('label') in (0,1) and all(r.get(k) is not None and np.isfinite(float(r[k])) for k in FEATURE_NAMES)]


def score_rows(model, rows):
    eligible=complete(rows)
    coverage={'total_rows':len(rows),'scored_rows':len(eligible),
              'omitted_ids':[r['event_id'] for r in rows if r not in eligible],
              'omitted_reasons':{r['event_id']:r.get('physics_feature_error') for r in rows if r not in eligible}}
    if not eligible:return {**coverage,'status':'unavailable_no_compatible_inputs'}
    x=np.asarray([[r[k] for k in FEATURE_NAMES] for r in eligible],np.float32)
    y=np.asarray([r['label'] for r in eligible]);p=model.predict_proba(x)
    return {**coverage,'p_xgb_calibrated':ranking_metrics(y,p),
        'p_failure_physical':ranking_metrics(y,[r['p_failure_physical'] for r in eligible]),
        'combined_score':None,
        'predictions':[{'event_id':r['event_id'],'event_group_id':r.get('event_group_id'),'label':int(r['label']),
                        'p_xgb_calibrated':float(v),'p_failure_physical':float(r['p_failure_physical'])} for r,v in zip(eligible,p)]}


def fit_reference_baseline(root, device='cpu', trials=0):
    import joblib
    import pyarrow.parquet as pq
    from backend.release import invalidate_readiness
    root=Path(root);data=root/'data';artifacts=root/'artifacts';artifacts.mkdir(exist_ok=True)
    invalidate_readiness(artifacts,'Reference baseline rebuild requires verification of the resulting artifacts')
    training=data/'model_training.parquet';rows=complete(pq.read_table(training).to_pylist())
    if any(r.get('physics_model_version') != PHYSICS_MODEL_VERSION for r in rows):
        raise ValueError('Stale features: refresh the frozen dataset before training')
    x,y,groups=load_labeled_training_data(str(training))
    folds=spatial_group_splits(y,groups,min(5,len(set(groups[y == 1]))))
    oof=np.full(len(y),np.nan);constant_oof=np.full(len(y),np.nan);fold_audit=[]
    for index,(train,test) in enumerate(folds):
        event_train={rows[i]['event_group_id'] for i in train};event_test={rows[i]['event_group_id'] for i in test}
        region_train={rows[i]['region_block'] for i in train};region_test={rows[i]['region_block'] for i in test}
        assert not event_train & event_test and not region_train & region_test
        inner=min(5,len(set(groups[train][y[train] == 1])))
        params=tune_parameters(x[train],y[train],groups[train],device,trials)
        candidate=GlobalMonotonicXGB(**params);candidate.device=device
        candidate.fit(x[train],y[train],groups[train],n_splits=inner)
        oof[test]=candidate.predict_proba(x[test])
        constant_oof[test]=float(np.mean(y[train]))
        fold_audit.append({'fold':index,'train_rows':len(train),'test_rows':len(test),
            'shared_event_groups':0,'shared_region_blocks':0,'inner_calibration_folds':inner,
            'train_components':sorted(set(groups[train])),'test_components':sorted(set(groups[test]))})
    assert np.isfinite(oof).all()
    params=tune_parameters(x,y,groups,device,trials)
    model=GlobalMonotonicXGB(**params);model.device=device
    model.fit(x,y,groups,n_splits=len(folds))
    # Independent reserved components, never an FS-filtered calibration sample.
    calibration=complete(pq.read_table(data/'conformal_calibration.parquet').to_pylist())
    if calibration:
        assert not {r['event_group_id'] for r in rows} & {r['event_group_id'] for r in calibration}
        assert not {r['region_block'] for r in rows} & {r['region_block'] for r in calibration}
        from .conformal_calibration import fit_reserved_conformal
        fit_reserved_conformal(model,data,groups)
    model_path=artifacts/'global_multi_causal_xgb.pkl';joblib.dump(model,model_path)
    restored=joblib.load(model_path)
    np.testing.assert_allclose(restored.predict_proba(x[:16]),model.predict_proba(x[:16]),rtol=0,atol=0)
    hold=pq.read_table(data/'historical_hindcast_candidates.parquet').to_pylist()
    assert not {r['event_group_id'] for r in rows} & {r['event_group_id'] for r in hold}
    assert not {r['region_block'] for r in rows} & {r['region_block'] for r in hold}
    from .uncertainty import cluster_intervals
    report={'physics_model_version':PHYSICS_MODEL_VERSION,'device':device,'training_rows':len(y),
        'positive_rows':int(np.count_nonzero(y == 1)), 'negative_rows':int(np.count_nonzero(y == 0)), 'connected_components':len(set(groups)),
        'parameters':params,'optuna_trials':trials,'optuna_sampler_seed':42,'assumed_field_prevalence':None,
        'calibration_fold_prevalences':model.calibration_fold_prevalences_,
        'cross_validation':{'p_xgb_calibrated':ranking_metrics(y,oof),
            'training_prevalence_only_baseline':ranking_metrics(y,constant_oof),
            'p_failure_physical':ranking_metrics(y,[r['p_failure_physical'] for r in rows]),
            'predictions':[{'event_id':r['event_id'],'component':str(g),'label':int(label),
                            'p_xgb_calibrated':float(p),'training_prevalence_baseline':float(q)}
                           for r,g,label,p,q in zip(rows,groups,y,oof,constant_oof)],
            'fold_audit':fold_audit,'confidence_intervals':cluster_intervals(y,oof,groups),
            'scope':'Outer event/spatial components; sigmoid calibration and optional parameter tuning fitted only inside each outer training split'},
        'original_development_holdout':score_rows(model,hold),
        'independent_external_validation':'not_run; equation verification does not establish observed warning skill',
        'probability_policy':'Unshifted sigmoid on the sampled inventory/absence population; physics separate; no combined score',
        'feature_names':list(FEATURE_NAMES),'feature_importance':model.feature_importance(),
        'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'training_sha256':hashlib.sha256(training.read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),**{p:version(p) for p in ('numpy','scipy','scikit-learn','xgboost','pyarrow','joblib')}},
        'limitations':['Retained v12 external-weather/date controls are not random global backgrounds or surveyed stable slopes',
            'Original holdout has been inspected during development; it is not new independent validation',
            'Missing or incompatible moisture excludes rows; coverage and omissions are reported',
            'Mapped footprints and report coordinates do not establish initiation crowns',
            'Runout, ice/root strength and erosion lack observational validation'],
        'engineering_ready':False,'pretraining_ready':False}
    for name in ('scientific_validation_report.json','reference_baseline_report.json'):
        (artifacts/name).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    metadata={k:report[k] for k in ('physics_model_version','training_rows','positive_rows','negative_rows','assumed_field_prevalence',
              'calibration_fold_prevalences','probability_policy','feature_names','model_sha256','training_sha256','environment')}
    (artifacts/'global_multi_causal_xgb.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (artifacts/'lhasa_benchmark_suite.json').write_text(json.dumps({'scope':'LHASA-inspired metrics, not NASA benchmark reproduction',
        'spatial_event_cv':report['cross_validation'],'original_holdout':report['original_development_holdout']},indent=2)+'\n')
    return report
