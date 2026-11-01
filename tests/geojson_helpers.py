"""Shared live API geometry assertions."""
from typing import Any, Dict

REQUIRED_CELL_PROPERTIES = {
    "h3_index", "lat", "lon", "elevation_m", "slope_deg", "twi", "fs_dynamic",
    "p_failure", "hotspot_score", "extrusion_height_m", "cruden_varnes_class",
    "cruden_varnes_label", "peak_velocity_ms", "failure_mode", "dominant_cause",
    "road_name", "households", "causal_breakdown_pct", "numbers_that_matter",
}


def check_geojson(payload: Dict[str, Any]) -> None:
    collection = payload.get("h3_geojson")
    assert collection and collection.get("type") == "FeatureCollection"
    features = collection.get("features")
    assert isinstance(features, list) and features, "Live terrain unavailable; cannot validate the scientific scenario"
    for feature in features:
        assert feature.get("type") == "Feature"
        geometry = feature.get("geometry") or {}
        assert geometry.get("type") == "Polygon"
        rings = geometry.get("coordinates") or []
        assert rings and len(rings[0]) >= 4 and rings[0][0] == rings[0][-1]
        for point in rings[0]:
            assert len(point) == 2 and -180 <= float(point[0]) <= 180 and -90 <= float(point[1]) <= 90
        assert REQUIRED_CELL_PROPERTIES.issubset((feature.get("properties") or {}).keys())
    routes = payload.get("runout_geojson")
    assert routes and routes.get("type") == "FeatureCollection"
    for feature in routes.get("features", []):
        assert feature.get("geometry", {}).get("type") == "LineString"
        assert len(feature.get("geometry", {}).get("coordinates", [])) >= 2

