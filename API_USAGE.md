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

## Current v6 output contract

`fs_dynamic` is the mechanical Mohr-Coulomb FS. Post-fire `p_failure_m1` stays separate in the physical probability union; the old compatibility yield-index field does not govern FS. `lateral_resistance_kpa` is retained as zero for compatibility. Snowpack, joint-temperature, soil strength and runout inputs include disclosed scenario assumptions.

`location.bedrock_lithology` exposes the Macrostrat mapped unit and assumed strength adjustment. `location.river_discharge` exposes modeled GloFAS flow and the seven-day surge ratio. Neither is a local ground measurement. Source statuses distinguish `LIVE`, `CACHED`, `FALLBACK` and supplied-coordinate `PROVIDED` states. Provider deadlines are 2.2 seconds each; the full scan can take longer than three seconds.

Cells expose `conformal_interval_90`, `uncertainty_width`, `bivariate_class`, `conformal_target` and `uncertainty_status`. Ensemble-scaled intervals target binary event labels, not true-probability confidence bounds. They may be [0,1]; spatial dependence invalidates an unconditional nominal coverage guarantee.

Critical summaries use steep candidate cells where available. Whole-viewport percentiles and the selected hotspot are different statistics. The first two `numbers_that_matter` cards are mechanical FS and the blended score; four distinct mechanism cards follow. Cause percentages are heuristic stress allocations, not causal inference.

`summary.kinematics_top_hotspot` includes `cascading_dbi`, `impoundment_status` and optional dam/catchment geometry. Unknown geometry remains null. No observed dam is inferred from a channel intersection alone.

If terrain is unavailable, geometry is empty and the hazard verdict is `UNAVAILABLE`. Successfully fetched weather, NASA, river and bedrock telemetry is still returned. Compatibility metric placeholders are marked `available: false`; do not display their numeric zeros as observations.

Agent `briefing_guard` preserves valid numeric-grounded prose as `VERIFIED_LLM_PROSE`; offline prose uses `VERIFIED_TEMPLATE`. Rejected numeric prose is replaced or withheld. This guard does not independently verify names, semantics or recommendations.

`engineering_ready` and `pretraining_ready` require matching full-system, randomized and boundary engineering reports. Empirical score/latency targets remain separate. [Readiness](artifacts/readiness.json), [current evidence](docs/current_status.md) and [roadmap](ROADMAP.md) describe the scope. A Git clone does not include the ignored local datasets or trained model.
