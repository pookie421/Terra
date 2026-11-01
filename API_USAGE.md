# TERRA-SHIELD API

Install the project dependencies and start the local service from the repository root:

```bash
python -m pip install -r requirements.txt
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

The service exposes:

- `GET /api/v1/scan` — live H3 polygons, hotspot summary, source status, and runout lines. Pass `lat` and `lon` together to scan coordinates, or use a supported project place name such as Wayanad, Kaikōura, Zermatt, Malibu, or New Delhi.
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

On Windows, `windows_5070ti_trainer/RUN_TRAINING_5070TI.bat` runs the XGBoost and optional QLoRA training workflow. After copying the Windows artifact files into `windows_5070ti_trainer/artifacts/` on the Mac, run `./IMPORT_ARTIFACTS_MAC.sh` to import the model and register its GGUF with Ollama.

Pre-training audit: scores now include the mechanical stability gate. Alpine snowpack/cycle values are explicitly assumed per-cell scenarios; observed station snow remains in the meteorological metrics. See `PRETRAINING_AUDIT.md` and `artifacts/full_system_bug_audit.json` for requested target results.

Current source telemetry is in `location.nasa_telemetry` and `location.reliability_report`, including live-query or fallback COOLR count provenance and independent cached 100 km counts. The source badge remains `TIER_2_PARTIAL_PROXY`. Agent replies expose `briefing_guard`; unverified prose is replaced or withheld. Current results are in `artifacts/full_system_bug_audit.md` and `.json`.

Physics v5 exposes `fs_mohr_coulomb`, `fs_postfire_yield_index`, `p_failure_m1`, `debris_volume_m3`, `suction_cohesion_kpa`, `cryo_perched_head_m` and `equivalent_water_mm` per cell. The combined `fs_dynamic` can be governed by the post-fire probability-derived index; that index is not a force-balance FS. Volume and cryospheric inputs include explicit scenario proxies. See `docs/physics_sources.md`.

`engineering_ready` is computed separately from `empirical_benchmark_metrics`. Combined readiness requires three fingerprint-matched audits plus the latest mandatory targets; `artifacts/readiness.json` is authoritative. Bootstrap intervals describe fixed-prediction spatial/event resampling, not operational reliability.


V5 cell additions: `cryospheric_eligible`, `effective_burn_severity`, `prefire_fuel_index_proxy` (RGB, not measured NDVI), `lateral_resistance_kpa`, `conformal_interval_90`, `uncertainty_width`, `bivariate_class`, `conformal_target`, `uncertainty_status`. The intervals predict binary labels under an exchangeable-block assumption; they are not confidence bounds on probability. Missing calibration produces [0,1].

`summary.kinematics_top_hotspot` adds `cascading_dbi`, `impoundment_status`, `channel_h3_index`, `dam_height_m`, `dam_volume_m3`, `catchment_area_m2`. Unknown geometry remains null. A channel hit additionally carries a full-source-cell failure-volume scenario and viewport-only contributing area, both explicitly labeled; neither is measured dam geometry.
