> Historical design notes. Implementation and current evidence are documented in the root README and latest audit.

# TERRA-SHIELD Phase 1 Architecture Plan

## Scope and constraints

Phase 1 provides an on-demand terrain and live-observation backend. It does not add a frontend, train a model, or claim a calibrated landslide probability. Its outputs are observations and physically derived terrain indicators that later hazard models can consume. Every unavailable source is reported as unavailable, with provenance and a reason; no fabricated weather, elevation, earthquake, or infrastructure values are substituted.

## Proposed layout

```text
backend/
  realtime/
    __init__.py
    cache.py                 # bounded async TTL cache and in-flight request coalescing
    device.py                # optional CUDA → MPS → CPU selection for tensor workloads
    dem_processor.py         # Terrarium mosaic, derivatives, H3 terrain and flow metrics
    natural_events.py        # NASA EONET geometry and proximity helpers
    satellite_processor.py   # Esri z13 RGB vegetation/burn proxies with explicit quality
    soilgrids.py             # SoilGrids WCS and coupled pedotransfer fallback
    global_ingestor.py       # place resolution and parallel live-source scan
  ml/
    build_real_labeled_dataset.py # catalog/event observations, provenance, cautious labels
scripts/
  test_phase1_ingestion.py   # three-location, live terminal proof
  run_mac_baseline_and_verify.py # dataset build, honest validation, five-region scan
windows_5070ti_trainer/      # portable CUDA/XGBoost and guarded Unsloth entry point
requirements.txt
ARCHITECTURE_PLAN.md
```

The modules expose typed dataclasses (`TerrainCell`, `TerrainScan`, `WeatherSnapshot`, `EarthquakeEvent`, `InfrastructureSummary`, `GlobalScan`) so API values, units, provenance, timestamps, and partial failures remain explicit. Cell-level values are keyed by H3 index at resolution 9. The top-level scan accepts either a place name or latitude/longitude and returns source-specific status and errors alongside whatever completed successfully.

Phase 1 terrain math stays in NumPy and runs on CPU on every supported machine. `detect_compute_device()` lazily chooses CUDA, then Apple MPS, then CPU when a later tensor/model workload needs a device; importing the live ingestor does not require PyTorch.

## Terrain pipeline

1. Resolve the requested center and a metric-radius bounding box, then build an H3 resolution-9 cover. Prefer H3's overlap containment when supported; retain a compatibility path for older `h3-py` versions and include edge cells.
2. Determine the intersecting XYZ tiles at zoom 13. Fetch distinct tiles concurrently through one reusable `httpx.AsyncClient`, decode Terrarium RGB elevations, and mosaic into a north-up array. Keep missing pixels as `NaN` until a bounded neighbor interpolation can use real surrounding terrain; if no surrounding observations exist, fail the terrain source rather than inventing a height.
3. Compute derivatives in metric pixel spacing with NumPy. Sample the raster around each H3 center for cell-level slope, aspect, curvature, and elevation. Build a downhill adjacency graph and route contributing area from high cells to lower neighbors using fourth-power multiple-flow weights; derive clipped TWI from that routed area.
4. Return the pixel spacing, tile list, missing/interpolated pixel count, cover size, and per-cell metrics so consumers can audit coverage and provenance.

## Live ingestion and caching

`GlobalIngestor.scan()` resolves a place name if necessary and starts terrain, Open-Meteo, NASA POWER, SoilGrids, USGS, Overpass, and NASA EONET queries concurrently. Esri World Imagery RGB z13 tiles are sampled at H3 cell centers. The returned vegetation value is a normalized visible-band VDVI proxy, not multispectral NDVI; burn is an EONET-proximity plus RGB bare/char proxy, and its quality states when no EONET wildfire geometry was available. An optional per-cell simulation override is kept distinguishable from imagery-derived values. SoilGrids REST is tried before WCS; when both fail, a slope/curvature/TWI/moisture pedotransfer estimate fills soil texture and bulk-density values with `estimated_not_measured` provenance. These estimates are not presented as SoilGrids observations. Weather is summarized into trailing 24-hour rain, peak 3-hour rain, 24/48-hour forecast rain, 7-day exponentially weighted ARI, soil moisture, snow, temperatures, and observed freeze/thaw transitions. NASA root-zone wetness and Open-Meteo moisture are blended only when values are present. USGS events are converted to PGA using the supplied attenuation equation and hypocentral distance. Overpass geometries are assigned to H3 cells and summarized as roads/cuts, waterways, quarries, and buildings.

The shared async TTL cache stores immutable provider payloads and DEM tiles, limits memory with LRU eviction, and coalesces identical in-flight requests. Suggested TTLs: geocoding and SoilGrids 30 days, DEM tiles 7 days, NASA POWER 1 day, weather 10 minutes, OSM 6 hours, and earthquakes 30 seconds. Cache keys include normalized coordinates/query, bounding box, resolution, and provider parameters. Unexpired cached observations are reused; an expired or missing entry that cannot be refreshed is marked unavailable rather than silently treated as current.

## Latency and recovery targets

Independent network calls run concurrently and tile downloads are bounded by semaphores. Short-lived providers retain a small deadline; EONET and SoilGrids receive longer provider-specific deadlines and cache public responses to avoid refetching them for each scan location. A warm scan uses cached observations; cold scans remain best-effort because public providers, tile count, rate limits, and network path are outside this service's control. A later slider/what-if layer should consume returned cell features in memory and run vectorized calculations without network or raster decoding. These are service goals, not guarantees for arbitrary client networks or provider outages.

Missing DEM pixels are interpolated only from neighboring observed elevations, with the interpolation count surfaced. If an entire unsupported region has no surrounding DEM data, terrain is unavailable. Weather gaps remain gaps; rainfall sums use available intervals and report coverage. No earthquake events yields a measured zero for the query window, while an unavailable earthquake API yields `None`. Empty OSM results are valid only after a successful query. Soil fractions may be filled by a marked pedotransfer estimate. An unavailable EONET fire feed does not mean a verified burn-free surface; the burn proxy reports that distinction. Request failures, HTTP status, parsing errors, and timestamps remain attached to each provider result.

## Phase 1 acceptance proof

`scripts/test_phase1_ingestion.py` performs live scans at Wayanad, Zermatt, and Kaikōura coordinates, prints H3 cover size and available terrain/weather/hydrology/seismic/OSM values, and exits nonzero if required live sources fail or the output cannot be derived. It is intentionally a terminal proof, not a substitute for later automated unit and integration tests.

## Phase 2 physics, ML, and validation contracts

`backend/physics/multi_causal_engine.py` accepts explicit one-dimensional per-cell `PhysicsInputs`; missing/non-finite values are rejected rather than silently replaced. The live adapter in `scripts/verify_5_global_causes.py` supplies numeric inputs with row-level provenance: PTF soil and visible-band VDVI proxies are accepted; burn uses zero when not assessed; absent road cuts use 9999 m; a DEM TWI >=14.5 cell is treated as a drainage proxy and river distances are measured to those cells. Weather and saturation gaps use named fallbacks. These outputs are computed scenario estimates, not all direct measurements. `dem_processor.py` removes H3 cells with mean elevation <=0.5 m before routing and statistics. `kinematics.py` applies the four ordered velocity branches and routes runout only when calibrated Voellmy parameters are supplied. `causal_attribution.py` keeps kPa physical stress shares separate from absolute TreeSHAP log-odds shares, then provides normalized regime weights and six threshold cards.

`backend/ml/global_xgb.py` owns the exact 14-feature order, monotonic constraints, XGBoost CPU/CUDA selection, spatially grouped isotonic calibration, 65/35 probability blend, and TreeSHAP extraction. `build_real_labeled_dataset.py` combines NASA COOLR cited event points/polygons, EONET, Overpass, Open-Meteo archive, USGS, Terrarium terrain, and soil provenance. A cited manually mapped polygon labels an H3 center inside its footprint as inventory presence, not an initiation crown. A cited manual point may label an existing H3 feature row within 300 m as an approximate source-location proxy; it is not a mapped crown. Matched 60-day prior samples from dated NASA-reported locations are inventory-absence pseudo-negatives, explicitly not surveyed or verified intact slopes. Other point events remain unlabeled. Feature fallbacks are numeric and marked per row; current RGB is not reused as historical imagery. The event-group holdout excludes a whole inventory from training before writing `data/historical_hindcast.csv`. Stage 1 and Stage 2 scores remain conditional on the label geometry and pseudo-negative assumptions. Counterfactual controls are specified physics checks and are reported separately from observed global environments.

`scripts/run_mac_baseline_and_verify.py` writes the model and validation report, then prints live all-land physics and active what-if results for the five regions. New Delhi also receives a live-input flat-ground counterfactual with slope set to zero. The refreshed dataset has 59 positive inventory proxies (47 polygon cells in Haiti and 12 manual point-buffer cells in Brazil), 430 temporal-absence pseudo-negatives, and 391 unlabeled candidates. The training split excludes the full 2024 Brazil inventory group and contains 465 labeled rows; the separate Stage 2 file contains 12 positives and 12 pseudo-negative controls from that one held-out group. Training positives remain the Haiti polygon inventory, so Stage 1 spatial scores do not establish cross-event generalization. The available positive regimes are `trigger_unclassified` and `monsoon_rain`; the catalog does not support the requested five-regime coverage or calibration. The Stage 2 score is an exploratory source-group holdout and is not a calibrated operational forecast. The Windows trainer bundle includes the refreshed full dataset, training partition, hindcast candidates/CSV, report, model, and 600 distinct source-linked QLoRA trajectories.

## Phase 3 API and local agent

`backend/api/main.py` exposes live scan, agent chat, validation-report, and CAP 1.2 XML endpoints. `backend/api/scan_service.py` serializes land H3 cells as closed `[longitude, latitude]` GeoJSON polygons, ranks hotspots by failure probability times mapped exposure, summarizes P05 factor of safety among steep land cells and P95 failure probability, and includes provisional runout lines. Runout parameters are marked uncalibrated. `backend/llm/qwen_agent.py` checks Ollama for the imported `terra-shield-qwen`, then any Qwen2.5 model, and uses deterministic parsing plus live tools when Ollama is offline. Its briefing is generated from scan telemetry and passes a numeric parity audit before it is returned. Named project coordinates are used only as location fallbacks when place-name geocoding fails; they do not create training labels. Install and run the service as documented in `API_USAGE.md`.
