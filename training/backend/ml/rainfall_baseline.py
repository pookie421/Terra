"""Small experimental rainfall classifiers independent of estimated slope strength."""
import numpy as np
from .rainfall_features import FEATURES,VERSION
from .global_xgb import spatial_group_splits

PRIMARY='source_logistic'  # predeclared, never selected from holdout scores
CANDIDATES=('rainfall_only','source_logistic','source_shallow_tree')


def grouped_folds(y,groups,maximum=5):
    y=np.asarray(y);groups=np.asarray(groups)
    n=min(maximum,*(len(set(groups[y==label])) for label in (0,1)))
    folds=spatial_group_splits(y,groups,n)
    for train,test in folds:
        if len(set(y[train]))<2 or len(set(y[test]))<2:
            raise ValueError('Insufficient independent components containing both classes')
    return folds


def estimator(kind):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if kind not in CANDIDATES:raise ValueError('Unknown rainfall baseline')
    if kind=='source_shallow_tree':
        from xgboost import XGBClassifier
        learner=XGBClassifier(n_estimators=100,max_depth=2,learning_rate=.05,min_child_weight=5,
            tree_method='hist',device='cpu',n_jobs=2,random_state=42)
    else:learner=LogisticRegression(C=1.,max_iter=1000,random_state=42)
    return Pipeline([('missing',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                     ('scale',StandardScaler()),('classifier',learner)])


class RainfallBaseline:
    def __init__(self,kind=PRIMARY):
        self.kind=kind;self.feature_version=VERSION;self.feature_names=FEATURES
        self.model=None;self.training_range=None

    def matrix(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(FEATURES) or np.isinf(x).any():
            raise ValueError('Invalid rainfall source feature matrix')
        if not np.isfinite(x[:,[0,1,3,4,5]]).all():raise ValueError('Required features cannot be imputed')
        return x[:,:2] if self.kind=='rainfall_only' else x

    def fit(self,x,y,groups):
        from sklearn.calibration import CalibratedClassifierCV
        x=np.asarray(x,float)
        self.model=CalibratedClassifierCV(estimator=estimator(self.kind),method='sigmoid',
            cv=grouped_folds(y,groups,maximum=3),ensemble=True)
        # Imputation/scaling are inside the pipeline, fitted anew in each fold.
        self.model.fit(self.matrix(x),y)
        self.training_range=[{'min':float(np.nanmin(x[:,i])),'max':float(np.nanmax(x[:,i]))}
                             if np.isfinite(x[:,i]).any() else {'min':None,'max':None} for i in range(len(FEATURES))]
        return self

    def predict_proba(self,x):
        if self.model is None:raise RuntimeError('Rainfall baseline has not been fitted')
        return self.model.predict_proba(self.matrix(x))[:,1]

    def out_of_range(self,x):
        return [[FEATURES[i] for i,(value,bounds) in enumerate(zip(row,self.training_range))
                 if np.isfinite(value) and bounds['min'] is not None and not bounds['min']<=value<=bounds['max']]
                for row in np.asarray(x,float)]

    def explain(self,x):
        """Exact per-fold calibrated logistic terms, not causal effects or SHAP.

        The served score is the mean of fold probabilities. Mean log-odds
        contributions describe model behavior but are not additive probabilities.
        """
        if self.kind not in ('source_logistic','rainfall_only'):return None
        x=self.matrix(x);folds=[]
        for fitted in self.model.calibrated_classifiers_:
            pipeline=fitted.estimator
            transformed=pipeline[:-1].transform(x)
            classifier=pipeline.named_steps['classifier'];calibrator=fitted.calibrators[0]
            names=list(pipeline.named_steps['missing'].get_feature_names_out(list(FEATURES[:x.shape[1]])))
            terms=-calibrator.a_*transformed*classifier.coef_[0]
            intercept=float(-calibrator.a_*classifier.intercept_[0]-calibrator.b_)
            folds.append((names,terms,intercept,fitted.predict_proba(x)[:,1]))
        output=[]
        for i in range(len(x)):
            contributions={name:float(np.mean([terms[i,names.index(name)] if name in names else 0.
                for names,terms,_,_ in folds])) for name in sorted({name for names,_,_,_ in folds for name in names})}
            output.append({'mean_calibrated_log_odds_terms':contributions,
                'leading_upward_feature':max(contributions,key=contributions.get) if max(contributions.values())>0 else None,
                'folds':[{'terms':dict(zip(names,map(float,terms[i]))),'intercept':intercept,'probability':float(p[i])}
                         for names,terms,intercept,p in folds],
                'scope':'Exact logistic/calibration decomposition per fold; mean terms explain ranking, not physical causes or contributions to failure probability'})
        return output
