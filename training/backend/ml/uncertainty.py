"""Cluster bootstrap uncertainty conditional on fixed fitted predictions."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def cluster_intervals(y,p,groups,repetitions=500,seed=20261031):
    y=np.asarray(y);p=np.asarray(p);groups=np.asarray(groups)
    unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(seed);scores=[]
    for _ in range(repetitions):
        idx=np.concatenate([members[k] for k in rng.integers(0,len(unique),len(unique))])
        yy=y[idx];pp=p[idx]
        if len(np.unique(yy))<2:continue
        k=min(40,len(yy));top=np.argsort(-pp,kind='stable')[:k]
        top10=np.argsort(-pp,kind='stable')[:max(1,int(np.ceil(.10*len(yy))))]
        scores.append([roc_auc_score(yy,pp),average_precision_score(yy,pp),brier_score_loss(yy,pp),yy[top].sum()/yy.sum(),yy[top10].sum()/yy.sum()])
    names=['roc_auc','pr_auc','brier_score','recall_at_40','recall_at_k']
    return {'method':'95% percentile cluster bootstrap of fixed predictions; conditional on fitted models; not retraining uncertainty',
            'cluster_count':len(unique),'valid_replicates':len(scores),'seed':seed,
            'intervals':{k:np.percentile(np.asarray(scores)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(names)} if scores else {}}
