"""Predeclared class/regime-balanced event holdout, with coarse spatial embargo.

The older Brazil benchmark remains reserved. This new protocol is not comparable
to its scores; matched pre-event negatives remain unverified absence assumptions.
"""
import hashlib
import json
from pathlib import Path
from collections import Counter
from backend.ml.global_xgb import FEATURE_NAMES

REGIMES=('monsoon_rain','coseismic','cryosphere')

def write_holdout(output,rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    by_id={r['event_id']:r for r in rows}
    path=output.parent/'event_holdout_manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
    else:
        # Identify the previous benchmark from metadata, never from its predictions.
        meta=output.with_suffix('.metadata.json')
        old=json.loads(meta.read_text()).get('event_group_holdout',{}) if meta.exists() else {}
        legacy=old.get('legacy_reserved_groups') or [old.get('holdout_event_group_id')]
        legacy=[g for g in legacy if g]
        pools={regime:{} for regime in REGIMES}
        for row in rows:
            g=row.get('event_group_id');regime=row.get('regime');control=by_id.get(row['event_id']+':tminus60')
            if regime not in pools or row.get('label')!=1 or not control or control.get('label')!=0 or g in legacy:continue
            if not all(row.get(k) is not None and control.get(k) is not None for k in FEATURE_NAMES):continue
            pools[regime].setdefault(g,[]).append(row)
        n=min(8,*(max(0,len(pools[r])-2) for r in REGIMES))
        if n<2:raise ValueError('Insufficient paired source events for a multi-regime holdout')
        selected=[];groups=[]
        for regime in REGIMES:
            # Fixed hash ordering avoids choosing events by outcome or performance.
            ordered=sorted(pools[regime],key=lambda g:hashlib.sha256(('terra-holdout-v3:'+g).encode()).hexdigest())[:n]
            for group in ordered:
                positive=min(pools[regime][group],key=lambda r:r['event_id'])
                selected.extend([positive['event_id'],positive['event_id']+':tminus60']);groups.append(group)
        manifest={'version':'balanced_events_v3','selection':'fixed SHA-256 group order; one source/control pair per group',
                  'selected_ids':selected,'heldout_groups':groups,'legacy_reserved_groups':legacy,'pairs_per_regime':n}
        path.write_text(json.dumps(manifest,indent=2))
    missing=set(manifest['selected_ids'])-set(by_id)
    if missing:raise ValueError('Frozen holdout source rows missing: '+str(sorted(missing)))
    hold=[by_id[eid] for eid in manifest['selected_ids']]
    excluded=set(manifest['heldout_groups'])|set(manifest['legacy_reserved_groups'])
    blocks={r['region_block'] for r in rows if r.get('event_group_id') in excluded}
    train=[r for r in rows if r.get('event_group_id') not in excluded and r.get('region_block') not in blocks]
    from .pinls import pinls_filter, reserve_conformal_blocks
    unfiltered_train=list(train)
    train,conformal,reserved=reserve_conformal_blocks(train)
    train,training_filter=pinls_filter(train)
    filtered_hold,holdout_filter=pinls_filter(hold)
    labeled=[r for r in train if r.get('label') in (0,1)]
    if len(labeled)<450 or len({r['region_block'] for r in labeled if r['label']==1})<3:
        raise ValueError('Spatially embargoed holdout leaves insufficient training rows: '+str(len(labeled)))
    for filename,records in [('model_training_unfiltered.parquet',unfiltered_train),('model_training.parquet',train),('historical_hindcast_candidates.parquet',hold),('historical_hindcast_pinls.parquet',filtered_hold),('conformal_calibration.parquet',conformal)]:
        keys=sorted(set().union(*(r.keys() for r in records)))
        pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in keys} for r in records]),output.with_name(filename),compression='zstd')
    return {'status':'ready','protocol':'balanced_events_v3_with_declared_pinls_v5','holdout_event_group_ids':manifest['heldout_groups'],
            'legacy_reserved_groups':manifest['legacy_reserved_groups'],'holdout_rows':len(hold),
            'holdout_positive_rows':len(hold)//2,'holdout_inventory_absence_controls':len(hold)//2,
            'holdout_positive_regimes':dict(Counter(r['regime'] for r in hold if r['label']==1)),
            'training_rows':'data/model_training.parquet','hindcast_candidates':'data/historical_hindcast_candidates.parquet',
            'pinls_training':training_filter,'pinls_holdout':holdout_filter,'conformal_blocks':reserved,'conformal_rows':len(conformal),'training_labeled_rows':len(labeled),'spatial_embargo_blocks':sorted(blocks),
            'label_quality':'Balanced source-location event proxies and matched 60-day inventory absences, not verified intact slopes.',
            'scope_warning':'Small balanced multi-event benchmark; changed population, not comparable with prior Brazil-only metrics.'}
