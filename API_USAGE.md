# TERRA-SHIELD API

Install the project dependencies and start the local service from the repository root:

```bash
python -m pip install -r requirements.txt
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

The service exposes:

- `GET /api/v1/scan` — live H3 polygons, hotspot summary, source status, and runout lines. Pass `lat` and `lon` together to scan coordinates, or provide a place name for live geocoding.
- `POST /api/v1/agent/chat` — accepts `message`, `current_query`, `agency_mode`, and `sliders`. It checks Ollama for `terra-shield-qwen`, then a Qwen2.5 model, and falls back to the deterministic live tools when Ollama is unavailable.
- `GET /api/v1/validation/report` — returns the latest Stage 1–4 validation report.
- `POST /api/v1/export/cap-xml` — returns a CAP 1.2 XML `Test` message with `Private` scope.

Example scan:

```bash
curl 'http://127.0.0.1:8000/api/v1/scan?query=Wayanad&rain_delta_mm=180'
```

Example agent request:

```bash
curl -X POST 'http://127.0.0.1:8000/api/v1/agent/chat' \
  -H 'Content-Type: application/json' \
  -d '{"message":"Wayanad +180 mm rain","current_query":"Chooralmala, Wayanad","agency_mode":"LOCAL_EOC","sliders":{}}'
```

The API reports empty GeoJSON collections when live terrain sources return no land cells. Runout parameters are explicitly marked uncalibrated. The scientific report records the source label limits and held-out event group used for Stage 2.

Training and artifact-import instructions are in [training/README.md](training/README.md).

Current source telemetry is in `location.nasa_telemetry` and `location.reliability_report`, including live-query or fallback COOLR count provenance and independent cached 100 km counts. The source badge remains `TIER_2_PARTIAL_PROXY`. Agent replies expose `briefing_guard`; unverified prose is replaced or withheld. Measured reference results are documented in [evaluation](docs/current_status.md); generated reports remain local.

## Output contract

`fs_dynamic` is the mechanical Mohr-Coulomb FS. Post-fire `p_failure_m1` stays separate in the physical probability union; the old compatibility yield-index field does not govern FS. `lateral_resistance_kpa` is retained as zero for compatibility. Snowpack, joint-temperature, soil strength and runout inputs include disclosed scenario assumptions.

`location.bedrock_lithology` exposes the Macrostrat mapped unit and assumed strength adjustment. `location.river_discharge` exposes modeled GloFAS flow and the seven-day surge ratio. Neither is a local ground measurement. Source statuses distinguish `LIVE`, `CACHED`, `FALLBACK` and supplied-coordinate `PROVIDED` states. Provider deadlines are configurable with a 15-second minimum inside a 60-second ingestion budget.

Cells expose `conformal_interval_90`, `uncertainty_width`, `bivariate_class`, `conformal_target` and `uncertainty_status`. Ensemble-scaled intervals target binary event labels, not true-probability confidence bounds. They may be [0,1]; spatial dependence invalidates an unconditional nominal coverage guarantee.

Critical summaries use steep candidate cells where available. Whole-viewport percentiles and the selected hotspot are different statistics. The first two `numbers_that_matter` cards are mechanical FS and the blended score; four distinct mechanism cards follow. Cause percentages are heuristic stress allocations, not causal inference.

`summary.kinematics_top_hotspot` includes `cascading_dbi`, `impoundment_status` and optional dam/catchment geometry. Unknown geometry remains null. No observed dam is inferred from a channel intersection alone.

If terrain is unavailable, geometry is empty and the hazard verdict is `UNAVAILABLE`. Successfully fetched weather, NASA, river and bedrock telemetry is still returned. Compatibility metric placeholders are marked `available: false`; do not display their numeric zeros as observations.

Agent `briefing_guard` preserves valid numeric-grounded prose as `VERIFIED_LLM_PROSE`; offline prose uses `VERIFIED_TEMPLATE`. Rejected numeric prose is replaced or withheld. This guard does not independently verify names, semantics or recommendations.

`engineering_ready` and `pretraining_ready` require matching full-system, randomized, boundary and Phase 4 forecast engineering reports. Empirical score/latency targets remain separate. Local readiness reports, [evaluation](docs/current_status.md) and the [roadmap](ROADMAP.md) describe the scope. A Git clone does not include the ignored local datasets or trained model.


## Hourly forecast and uncertainty

`GET /api/v1/scan` accepts `inject_forecast_storm_mm`, `storm_arrival_hour` (default 18), and `storm_duration_hours` (default 12). Injection explicitly sets forecast T+0 rain, intensity and pressure head to zero and replaces future liquid rainfall. It returns `initial_condition_quality=SYNTHETIC_CALM_NOT_LIVE` and `observed_current_risk_t0_pct` separately; the live scan summary remains unchanged. Antecedent moisture and non-rain triggers remain, so a low T+0 score is not guaranteed. The interval must fit in 0..72 hours. The default case uses forecast weather. Five forecast days are requested and the first 72 future hours selected. Missing hourly soil temperature falls back to air temperature and is enumerated in `soil_temperature_fallback_hours`; missing air or liquid rainfall keeps ordinary forecasts unavailable.

`predictive_forecast_72h` contains current and peak P50/P90 scores, minimum FS, peak hour, threshold-crossing hour, lead hours, selected hourly rows, and both complete 73-state paths. Crossing uses the first regional FS<1 or score>=65%; an already-critical initial state reports zero lead. Missing forcing yields UNAVAILABLE. These are hourly scenario crossings, not observed or validated failure times. Pointwise P90 forcing is not a 90% failure probability or a worst-case bound; fallback spread is explicitly marked 1.5× deterministic forcing.

Per-cell `epistemic_interval_90` (fractions) and `epistemic_interval_90_pct` (percentages) describe a normal fold/FOSM spread proxy with unvalidated coverage. `conformal_binary_label_interval` remains separate. `physical_cause_pct` and `shap_cause_pct` disclose both attribution components. Shared SHAP features are split equally across the five display groups, not identified as causal effects.

Named scans query live Open-Meteo geocoding first. A failed lookup can use the explicit known-coordinate fallback, marked `available=false`, `mode=FALLBACK`. Source statuses include `cache_policy` and `cache_trace`; any cache hit makes the whole source `CACHED`. API ingestion is live-first, with disclosed unexpired memory-cache fallback on provider failure.
