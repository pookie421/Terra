"""Source-availability badges, never operational validation guarantees."""
def reliability_report(scan):
    def status(key,available,missing='UNAVAILABLE'):
        value=scan.sources.get(key)
        return available if value and value.available else missing
    soil_quality=getattr(scan.hydrology,'soil_property_quality',None)
    soil_origin=getattr(scan.hydrology,'soil_property_source','')
    if soil_origin=='hybrid_soilgrids_and_pedotransfer' or soil_quality=='mixed_measured_prediction_and_estimate':
        soil_source='MIXED_ISRIC_AND_PEDOTRANSFER'
    elif soil_origin=='topographic_climatic_pedotransfer_estimate' or soil_quality in ('estimated','estimated_not_measured'):
        soil_source='PEDOTRANSFER_TOPOGRAPHIC_ESTIMATE'
    else:
        soil_source=status('soilgrids','LIVE_OR_CACHED_ISRIC_SOILGRIDS','UNAVAILABLE')
    return {
        'dem_source':status('terrarium_dem','LIVE_OR_CACHED_AWS_TERRARIUM_Z13'),
        'weather_source':status('open_meteo','LIVE_OR_CACHED_OPEN_METEO_BEST_MATCH'),
        'nasa_hydrology_source':status('nasa_power','LIVE_OR_CACHED_NASA_POWER_GWETROOT'),
        'seismic_source':status('usgs','LIVE_OR_CACHED_USGS_FDSN'),
        'soil_source':soil_source,
        'osm_exposure_source':status('overpass','LIVE_OSM_OVERPASS','UNIT_EXPOSURE_PROXY_NO_OSM'),
        'channel_source':'DEM_TWI_CHANNEL_PROXY' if scan.terrain else 'UNAVAILABLE',
        'operational_confidence_tier':'TIER_2_PARTIAL_PROXY',
        'tier_reason':'Strength, road-cut geometry, runout parameters and alpine snowpack may be assumed. Feed availability is not calibrated operational confidence.',
        'soil_property_quality':soil_quality,
        'terrain_resolution_note':'Zoom-13 Terrarium sampling; not a verified uniform 30 m NASA/SRTM product.',
        'weather_model_note':'Open-Meteo default model selection; ECMWF alone is not guaranteed.',
        'feeds':{name:status.__dict__ for name,status in scan.sources.items()},
        'coolr':scan.coolr_nearby,
    }
