"""Radar context and explicitly calibrated site constraints, never proxy strength.

Coherence is affected by vegetation, geometry and temporal decorrelation. One
interferogram does not identify a velocity time series or a soil-strength law.
Missing radar retains the complete existing joint-FOSM reference distribution.
"""
import math
from datetime import datetime, timezone

FEATURE_VERSION = 'nisar_context_features_v1'
FEATURE_NAMES = ('nisar_coherence', 'nisar_los_velocity_mm_yr',
                 'nisar_polarimetric_moisture_index', 'nisar_granule_count',
                 'nisar_acquisition_age_days', 'nisar_coherence_available',
                 'nisar_velocity_available', 'nisar_moisture_available')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def radar_cell(payload, cell_id):
    payload = payload or {}
    record = (payload.get('by_h3') or {}).get(cell_id)
    if record is None:
        record = next((r for r in payload.get('cells', []) if r.get('h3_index') == cell_id), {})
    result = {key: record.get(key) for key in FEATURE_NAMES[:3]}
    for key in FEATURE_NAMES[:3]:
        if not finite(result[key]):result[key] = None
    if result['nisar_coherence'] is not None and not 0 <= result['nisar_coherence'] <= 1:
        result['nisar_coherence'] = None
    # Footprint-aware cell metadata may legitimately be zero or unknown, and a
    # cell's latest acquisition can predate the newest scan-region granule.
    # Keep that support intact instead of spreading a regional summary to it.
    result.update(nisar_granule_count=record.get('nisar_granule_count'),
                  nisar_latest_acquisition_utc=record.get('nisar_latest_acquisition_utc'),
                  nisar_region_granule_count=payload.get('granule_count'),
                  nisar_region_latest_acquisition_utc=payload.get('latest_acquisition_utc'),
                  nisar_measurement_provenance=record.get('measurement_provenance', record.get('provenance')),
                  nisar_mode=payload.get('mode', 'NOT_ACQUIRED'))
    return result


def feature_context(payload, cell_id, at_utc=None):
    record = radar_cell(payload, cell_id)
    age = None
    if record['nisar_latest_acquisition_utc']:
        try:
            at = datetime.fromisoformat((at_utc or datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
            acquired = datetime.fromisoformat(record['nisar_latest_acquisition_utc'].replace('Z', '+00:00'))
            delta = (at - acquired).total_seconds() / 86400
            if delta >= 0:age = delta
        except (TypeError, ValueError):pass
    values = {**record, 'nisar_acquisition_age_days': age,
              'nisar_coherence_available': int(record['nisar_coherence'] is not None),
              'nisar_velocity_available': int(record['nisar_los_velocity_mm_yr'] is not None),
              'nisar_moisture_available': int(record['nisar_polarimetric_moisture_index'] is not None)}
    return {'version': FEATURE_VERSION, 'names': list(FEATURE_NAMES),
            'values': {name: values[name] for name in FEATURE_NAMES},
            'trained_radar_weights_available': False,
            'scope': 'Versioned observation context; the active 25-feature model is not a SAR-trained classifier'}


def calibrated_ground_inputs(scan, supplied=None):
    """Use an absolute site-calibrated strength with a real measurement anchor.

    Local manifests may supply strength_calibration_by_h3 entries with a sourced
    cohesion_kpa observation, granule_id and relation_version. Their dates and
    units are checked through the same site-input contract as other overrides.
    Invalid radar constraints are disclosed and leave the reference column alive.
    Explicit ground inputs take precedence over radar-derived site calibrations.
    """
    from backend.api.ground_assessment import observation
    from backend.ml.rainfall_features import utc
    payload = getattr(scan, 'nisar', None) or {}
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in (supplied or {}).items()}
    statuses = {}
    known = {g.get('granule_id') for g in payload.get('granules', [])}
    calibration = payload.get('strength_calibration_by_h3') or {}
    for cell_id in (scan.terrain.cells if scan.terrain else {}):
        cell = radar_cell(payload, cell_id)
        measured = cell['nisar_coherence'] is not None or cell['nisar_los_velocity_mm_yr'] is not None
        status = {'status': 'NO_CALIBRATED_STRENGTH_RELATION' if measured else 'REFERENCE_PRIOR_UNCONSTRAINED_BY_RADAR',
                  'strain_mid_prior': 0., 'strain_interval': [None, None],
                  'uncertainty': 'Existing broad joint-FOSM material/state priors retained; radar supplies no invented variance',
                  'applied': False}
        entry = calibration.get(cell_id)
        if entry:
            try:
                if not measured:raise ValueError('A measured radar anchor is required')
                if not entry.get('relation_version') or entry.get('granule_id') not in known:
                    raise ValueError('A versioned site calibration and matching real granule are required')
                strength = entry.get('cohesion_kpa')
                if not isinstance(strength, dict) or strength.get('method') != 'site_calibration':
                    raise ValueError('Expected a sourced site_calibration cohesion value')
                value = observation({'cohesion_kpa': strength}, 'cohesion_kpa', utc(scan.generated_at_utc))
                if value < 0:raise ValueError('Calibrated cohesion must be nonnegative')
                if not isinstance(merged.get(cell_id, {}), dict):
                    raise ValueError('Explicit ground input has invalid shape')
                if 'cohesion_kpa' not in merged.get(cell_id, {}):
                    merged.setdefault(cell_id, {})['cohesion_kpa'] = dict(strength)
                    status.update(status='CALIBRATED_SITE_RADAR_CONSTRAINT', applied=True,
                                  relation_version=entry['relation_version'], granule_id=entry['granule_id'])
                else:status['status'] = 'EXPLICIT_GROUND_INPUT_TAKES_PRECEDENCE'
            except (ValueError, TypeError, KeyError) as exc:
                status.update(status='RADAR_CALIBRATION_REJECTED_REFERENCE_RETAINED', error=str(exc))
        statuses[cell_id] = status
    return merged or supplied, statuses
