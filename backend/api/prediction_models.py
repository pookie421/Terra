"""Comparable display contracts, without equating grid spacing and prediction skill."""
import copy
import h3
from backend.physics.radar_context import radar_cell, feature_context

MODEL_IDS = ('terra_physics_ml', 'nasa_lhasa_2', 'physics_only', 'weather_terrain_ml', 'rainfall_only')
KINDS = {'terra_physics_ml': 'Physics-Informed ML', 'physics_only': 'Physics-Only',
         'weather_terrain_ml': 'Weather+Terrain ML', 'rainfall_only': 'Rainfall-Only'}


def attach_prediction_models(response, scan):
    nasa = copy.deepcopy(getattr(scan, 'nasa_lhasa', None) or {})
    radar = copy.deepcopy(getattr(scan, 'nisar', None) or {})
    hour = response.get('forecast_view', {}).get('hour', 0)
    native = nasa.get('native_resolution_m_approx', 1000)
    rows = nasa.get('cells') or list((nasa.get('by_h3') or {}).values())
    normalized = []
    for row in rows:
        official = row.get('probability') if hour == 0 else row.get('tomorrow_probability') if hour == 24 else None
        reference = row.get('reference_proxy_score') if hour == 0 else None
        normalized.append({**row, 'probability': official, 'reference_proxy_score': reference,
            'tomorrow_probability': row.get('tomorrow_probability') if hour <= 24 else None,
            'official_score': official, 'reference_score': reference,
            'active_score': official if official is not None else reference,
            'score_kind': 'NASA_relative_hazard_probability' if official is not None else 'unvalidated_reference_index',
            'time_alignment': 'NASA_daily_product_not_exact_T_plus_24' if hour == 24 else 'NOW' if hour == 0 else 'NO_NASA_PRODUCT_FOR_HORIZON',
            'provenance': 'UNAVAILABLE_FOR_FORECAST_HORIZON' if hour > 24 else 'LIVE_NASA_IMAGESERVER' if official is not None else row.get('provenance', 'UNAVAILABLE')})
    nasa.update(cells=normalized, by_h3={r['h3_index']: r for r in normalized},
                native_resolution_m=native, forecast_hour=hour,
                scope='Native NASA daily product or explicitly separate unvalidated coarse reference index; no 30m NASA prediction')
    if hour > 24:
        nasa.update(mode='UNAVAILABLE_FOR_FORECAST_HORIZON', reason='NASA supplies no matching 48/72-hour product here')
    response.update(nasa_lhasa=nasa, nisar=radar)
    physical = response.get('physical_assessment', {}).get('cells', [])
    learned = {r['h3_index']: r for r in response.get('physics_informed_assessment', {}).get('cells', [])}
    patches = {}
    for row in physical:
        key = row['h3_index']; ml = learned.get(key, {})
        scores = {model: ml.get('ablation_scores', {}).get(kind) for model, kind in KINDS.items()}
        if scores['terra_physics_ml'] is None:scores['terra_physics_ml'] = ml.get('score')
        if scores['physics_only'] is None:scores['physics_only'] = row.get('p_failure_physical')
        comparison = nasa['by_h3'].get(key, {})
        scores['nasa_lhasa_2'] = comparison.get('active_score')
        patches[key] = {**radar_cell(radar, key), 'prediction_scores': scores,
            'lhasa_probability': comparison.get('official_score'),
            'lhasa_reference_score': comparison.get('reference_score'),
            'lhasa_susceptibility': comparison.get('susceptibility_class'),
            'lhasa_provenance': comparison.get('provenance'),
            'radar_feature_context': feature_context(radar, key, scan.generated_at_utc)}
    for name in ('physical_assessment', 'physics_informed_assessment', 'statistical_assessment'):
        for row in response.get(name, {}).get('cells', []):row.update(patches.get(row['h3_index'], {}))
    for row in response.get('cell_comparisons', []):row.update(patches.get(row['h3_index'], {}))
    for name in ('h3_geojson', 'terrain_geojson'):
        for feature in response.get(name, {}).get('features', []):
            feature['properties'].update(patches.get(feature['properties']['h3_index'], {}))
    response['prediction_scores'] = {key: value['prediction_scores'] for key, value in patches.items()}
    version = nasa.get('product_version')
    nasa_label = 'NASA LHASA' + (' ' + str(version) if version and version != 'NOT_DECLARED_BY_SERVICE' else '')
    labels = {'terra_physics_ml': 'Physics + ML', 'nasa_lhasa_2': nasa_label,
              'physics_only': 'Physics reference', 'weather_terrain_ml': 'Weather + terrain ML', 'rainfall_only': 'Rainfall only'}
    catalog = []
    for model in MODEL_IDS:
        available = [p['prediction_scores'][model] for p in patches.values() if p['prediction_scores'].get(model) is not None]
        official = [r['official_score'] for r in normalized if r.get('official_score') is not None] if model == 'nasa_lhasa_2' else []
        reference = [r['reference_score'] for r in normalized if r.get('reference_score') is not None] if model == 'nasa_lhasa_2' else []
        if model == 'nasa_lhasa_2':available = official or reference
        catalog.append({'id': model, 'key': model, 'label': labels[model],
            'available': bool(available), 'peak_score': max(available) if available else None,
            'official_peak': max(official) if official else None,
            'reference_peak': max(reference) if reference else None,
            'peak_scope': 'official_NASA_cells' if official else 'unvalidated_reference_cells' if reference else 'experimental_reference',
            'mode': nasa.get('mode', 'NOT_ACQUIRED') if model == 'nasa_lhasa_2' else 'EXPERIMENTAL_REFERENCE',
            'product_version': nasa.get('product_version') if model == 'nasa_lhasa_2' else response.get('physics_informed_assessment', {}).get('version'),
            'native_resolution_m': native if model == 'nasa_lhasa_2' else None,
            'valid_time_utc': nasa.get('valid_time_utc') if model == 'nasa_lhasa_2' else response.get('forecast_view', {}).get('valid_time_utc'),
            'scope': nasa['scope'] if model == 'nasa_lhasa_2' else 'Experimental sampled-population score; environmental forcing is coarser than DEM/H3 support'})
    response['prediction_models'] = catalog
    response['spatial_support'] = {'dem_nominal_pixel_size_m': 30, 'dem_native_support_varies': True,
        'h3_resolution': 9, 'h3_mean_cell_area_m2': h3.average_hexagon_area(9, unit='m^2'),
        'scope': 'H3 aggregation cells are not 30m pixels; finer geometry does not establish greater predictive skill'}
    response.setdefault('live_inputs', {})['nisar'] = {key: radar.get(key) for key in ('mode', 'granule_count', 'latest_acquisition_utc')}
    return response
