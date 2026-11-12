# TERRA-SHIELD live analysis API

Start FastAPI with `python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000`. In a second terminal run `npm ci` and `npm run dev` from `frontend/`. Vite proxies `/api` to that local service. The browser starts on the online world map without a preset scan, replay, or manual weather controls.

The current feature-fusion model is `physics_informed_initiation_v2_features`; the physical reference contract is `pedotransfer_reference_column_v1`. The older model's fixed FOSM `base_margin` is superseded. [Current methods and measured results](docs/feature_fusion_phase3.md) distinguish learned and mechanically coupled scores.

## Automatic coordinate or place scan

```sh
curl -X POST http://127.0.0.1:8000/api/v1/scan \
  -H 'Content-Type: application/json' \
  -d '{"latitude":45.901,"longitude":7.718}'
```

A place search submits `{"query":"place name"}` to the same route. Raw coordinates bypass named-place geocoding and retain an explicit coordinate label when no place name is known. Legacy `lat`/`lon` aliases are supported; conflicting aliases are rejected. `autonomous` defaults to true and rejects manual rainfall/duration perturbations. The browser sends only the requested location.

`GET /api/v1/health` checks the local engine and actual learned-artifact availability without acquiring environmental providers. A healthy physical service can report `model_ready=false`; the approved pretrained weights are included in a normal clone. Install with `-c constraints-runtime.txt`, preserve `artifacts/physics_informed_models.pkl`, and restart the backend after replacing it. See [setup](docs/SETUP.md).

A scan returns:

- `scan_id`, `scenario_id`, response schema/version, location and source timestamps.
- `source_status`: actual availability, live/cache/fallback modes, provider errors and recorded latency. Fourteen channels do not imply fourteen independent measurements or fourteen successful responses.
- `live_inputs`: the actual observed model window, including 24h liquid rain, preceding nonoverlapping 144h rain, 24h mean surface moisture, soil properties and provenance, catalog-based PGA proxy and modeled river discharge.
- `h3_geojson`, `terrain_geojson`, full physical and learned cell records, a coherent selected hotspot, conditional runout and automatic forecast metadata.

Requested display/H3 resolution does not improve native source accuracy. Polar terrain limits, provisional coastal masking and provider outages remain explicit domain/coverage limitations. Coordinates are accepted globally; a cell outside available terrain coverage cannot receive invented topography.

## Physical columns and learned scores

`physical_assessment.cells` exposes corrected slope/TWI, uncapped `fs_mid`, `fs_low`, `fs_high`, `fs_interval`, pressure initialization, material/hydraulic inputs, joint covariance, Jacobian, variance allocation, critical rainfall deficit and provenance. Missing boreholes use declared reference priors; they do not disable the physical column or conditional routing. Exactly zero driving shear has a specific status and undefined FS.

`physics_informed_assessment.cells` joins the same cells' mathematical and source features. It includes the conditional `score`, `learned_trigger_score`, `mechanical_admissibility`, four ablation scores and model explanations. The learned score and its mechanical coupling must not be confused with one another or with globally calibrated event probability. Compatibility fields named `max_p_failure` still denote an experimental sampled-population score. `operational_warning_ready` is false; CAP export remains HTTP 409.

Current source PGA can support the stable-block Newmark regression where applicable. An earthquake catalog proxy is not measured local shaking or a forecast of future earthquakes. Snow, burn and erosion mechanisms retain their source/domain requirements; the interface does not fabricate those measurements.

## Automatic 72-hour horizons

`predictive_forecast_72h` uses downloaded future hourly forcing and available ensemble information. Its `views` list identifies `now`, `t24`, `t48`, `t72` and `peak`, with elapsed hour, valid time and state metadata. `paths.p50.hourly` contains the sector minimum-FS trajectory. If an actual ensemble is absent, deterministic-only forcing is explicitly identified; there is no invented 1.5× storm masquerading as P90.

```text
GET /api/v1/scans/{scan_id}/forecast/t24
GET /api/v1/scans/{scan_id}/forecast/t48
GET /api/v1/scans/{scan_id}/forecast/t72
GET /api/v1/scans/{scan_id}/forecast/peak
```

Each route materializes a full, state-bound scan payload with projected per-cell columns, uncertainty, learned scores, GeoJSON and conditional routing. It reuses the acquired provider snapshot. The browser replaces the map and inspector with that returned state; it does not merely change a time label.

`peak` means the hour of minimum sector nominal FS on the P50 forcing path, explicitly labeled `minimum_sector_nominal_fs_p50`. It is not a claim that all hourly ML predictions have been searched or that failure will occur at that time. Missing future forcing leaves current columns intact and shows a forecast-specific reason. Ensemble quantiles describe forcing, not calibrated landslide uncertainty or coherent storm members.

## Inspect the displayed state

```text
GET /api/v1/inspect/{scan_id}/{h3_index}?scenario_id={scenario_id}
GET /api/v1/scans/{scan_id}/scenarios/{scenario_id}/cells/{h3_index}
GET /api/v1/scans/{scan_id}/scenarios/{scenario_id}/runout/{h3_index}
GET /api/v1/scans/{scan_id}/scenarios/{scenario_id}
```

Use the displayed `scenario_id`, including for forecast views. The inspection alias can select the latest state when no scenario is supplied; explicit identity avoids ambiguity. Retained states are bounded in memory; expired identifiers return 404. Provider refresh creates a new scan.

The runout payload contains three declared friction/xi/depth scenarios, their coordinates, peak velocities, corridor cell IDs and stop reasons. Release is hypothetical unless independently established. H3 routing has no entrainment or mass conservation; boundary stops can truncate travel.

## State-grounded, multi-turn Qwen chat

```json
{
  "message": "Explain this cell's uncertainty and source quality, then fly to it.",
  "scan_id": "returned-scan-id",
  "scenario_id": "displayed-state-id",
  "selected_h3_index": "cell-in-this-scan",
  "autonomous": true,
  "history": [
    {"role": "user", "content": "What controls the mechanical margin?"},
    {"role": "assistant", "content": "Prior verified response."}
  ]
}
```

Submit to `POST /api/v1/chat` (`/api/chat` and `/api/v1/agent/chat` remain aliases). History is bounded to twelve messages. The server validates the selected cell and supplies current column/envelope, mathematical features, forecast/state identity, conditional corridor and provider provenance. In autonomous mode, chat cannot invent forcing changes; returned map actions are limited to validated inspection/fly commands.

The response includes `clean_markdown`, `briefing_guard`, active IDs, `bound_claims`, `numeric_audit_passed` and `ui_action.actions`. Qwen returns structured prose with numeric JSON-pointer bindings; the server verifies/render these values. Valid output is `VERIFIED_LLM_PROSE`. When the local model service cannot be reached, `VERIFIED_TEMPLATE` is an explicit deterministic fallback with service-error metadata. A responding service without compatible weights is `MODEL_UNAVAILABLE`; invalid generated prose is withheld after retry. Templates and model prose are visibly distinct.

The frontend retains conversation messages with state identity, displays numeric citations, and rejects stale responses/actions after a coordinate, horizon or selected-cell change. Actual Qwen weights and CUDA/QLoRA execution require target-machine testing; mocked protocol checks do not establish language-model quality.

The default local model service is `http://localhost:11434`. Set `TERRA_QWEN_URL` before starting the backend to select another Ollama-compatible endpoint, such as `http://127.0.0.1:11435` for `scripts/serve_qwen_copilot.py --port 11435`. The [RTX 5070 Ti guide](docs/windows_5070ti.md) covers native QLoRA, adapter serving and optional GGUF/Ollama export. Both the assistant request context and trained context are capped at 8192 tokens; the direct adapter server rejects an overflowing prompt plus reserved completion without truncation.

## Research compatibility and publication

Optional sourced site records (`ground_columns`) can override cohesion, friction angle, unit weight, depth, head, conductivity and diffusivity at their exact H3 cell. Records retain units, source/method and validity dates where applicable. Missing fields remain labeled priors; invalid overrides are disclosed per cell.

Legacy scenario and replay endpoints remain research utilities. They are not the autonomous frontend workflow. Raw corpora, cached providers, generated reports and legacy/optional weights stay outside Git. The small approved runtime ML artifact is included; see its [model card](artifacts/MODEL_CARD.md). Generated local training bundles include source-reviewed materials and are not tracked source mirrors. See [training workflow](docs/training_workflow.md) and [source/API critique](docs/global_source_audit.md).
