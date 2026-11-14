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

`GET /api/v1/connections` returns redacted deployment-level assistant/Earthdata configuration with `requires_personal_key=false`. It makes no provider request and does not prove availability. There is no public key-writing route. The installation owner uses `python scripts/configure_assistant.py` and restarts with `--env-file .env`; shared visitors need no keys. [Connection setup](docs/assistant_connections.md).

A scan returns:

- `scan_id`, `scenario_id`, response schema/version, location and source timestamps.
- `source_status`: actual availability, live/cache/fallback modes, provider errors and recorded latency, including `nisar` and `nasa_lhasa`. Provider-channel counts do not imply independent measurements or successful responses.
- `live_inputs`: the actual observed model window, including 24h liquid rain, preceding nonoverlapping 144h rain, 24h mean surface moisture, soil properties and provenance, catalog-based PGA proxy and modeled river discharge.
- `h3_geojson`, `terrain_geojson`, full physical and learned cell records, a coherent selected hotspot, conditional runout and automatic forecast metadata.

Requested display/H3 resolution does not improve native source accuracy. Polar terrain limits, provisional coastal masking and provider outages remain explicit domain/coverage limitations. Coordinates are accepted globally; a cell outside available terrain coverage cannot receive invented topography.

## Physical columns and learned scores

`physical_assessment.cells` exposes corrected slope/TWI, uncapped `fs_mid`, `fs_low`, `fs_high`, `fs_interval`, pressure initialization, material/hydraulic inputs, joint covariance, Jacobian, variance allocation, critical rainfall deficit and provenance. Missing boreholes use declared reference priors; they do not disable the physical column or conditional routing. Exactly zero driving shear has a specific status and undefined FS.

`physics_informed_assessment.cells` joins the same cells' mathematical and source features. It includes the conditional `score`, `learned_trigger_score`, `mechanical_admissibility`, four ablation scores and model explanations. The learned score and its mechanical coupling must not be confused with one another or with globally calibrated event probability. Compatibility fields named `max_p_failure` still denote an experimental sampled-population score. `operational_warning_ready` is false; CAP export remains HTTP 409.

Current source PGA can support the stable-block Newmark regression where applicable. An earthquake catalog proxy is not measured local shaking or a forecast of future earthquakes. Snow, burn and erosion mechanisms retain their source/domain requirements; the interface does not fabricate those measurements.

## Selectable models and radar context

The primary UI offers Terra In House Model, NASA LHASA and Rainfall baseline. The same scan retains five model IDs for compatibility and advanced diagnostics:

| ID | Displayed estimate |
|---|---|
| `terra_physics_ml` | Experimental physics-informed initiation score; default |
| `nasa_lhasa_2` | Official NASA coarse raster where available, otherwise a separately labeled unvalidated reference index |
| `physics_only` | Conditional physical reference/FOSM estimate |
| `weather_terrain_ml` | Weather + terrain ablation |
| `rainfall_only` | Rainfall-only ablation |

`prediction_models` is the model catalog. Cell records and GeoJSON properties expose `prediction_scores` by model ID. Selecting a model changes the client map/readout and assistant context without requiring a second provider scan. `spatial_support` records DEM and H3 support separately: nominal 30 m terrain inputs do not turn resolution-9 H3 cells into 30 m pixels.

`nasa_lhasa.cells` joins official, reference and susceptibility fields to H3 IDs. `official_score`, `reference_score`, `active_score`, provenance and native support remain distinct; official zero is valid. The live services do not declare their exact served LHASA version or issue/valid time. The compatibility ID `nasa_lhasa_2` does not assert exact version-2.0 reproduction. Catalog peaks expose `official_peak` and `reference_peak` separately; official cells take precedence for NASA critical-cell ranking when present. Repeated pixel values are not downscaled NASA predictions.

`nisar` carries real granule/collection/acquisition metadata and optional measured subsets. Inspection and cell records include `nisar_coherence`, `nisar_los_velocity_mm_yr`, `nisar_polarimetric_moisture_index`, granule count, latest acquisition and measurement provenance. Null fields mean no supported measurement. The legacy-named moisture index is an uncalibrated radar ratio. `radar_feature_context` is a versioned sidecar, not new trained SAR weights; `radar_constraint` reports whether an explicitly calibrated site strength was applied or rejected. Full reference columns remain available with metadata-only or missing radar. [Source semantics and optional subset configuration](docs/nasa_radar_comparison.md).

## Automatic 72-hour horizons

`predictive_forecast_72h` uses downloaded future hourly forcing and available ensemble information. Its `views` list identifies `now`, `t24`, `t48`, `t72` and `peak`, with elapsed hour, valid time and state metadata. `paths.p50.hourly` contains the sector minimum-FS trajectory. If an actual ensemble is absent, deterministic-only forcing is explicitly identified; there is no invented 1.5× storm masquerading as P90.

```text
GET /api/v1/scans/{scan_id}/forecast/t24
GET /api/v1/scans/{scan_id}/forecast/t48
GET /api/v1/scans/{scan_id}/forecast/t72
GET /api/v1/scans/{scan_id}/forecast/peak
```

Each route materializes a full, state-bound scan payload with projected per-cell columns, uncertainty, learned scores, GeoJSON and conditional routing. It reuses the acquired provider snapshot. The browser replaces the map and inspector with that returned state; it does not merely change a time label.

NASA Tomorrow may be shown with the T+24h view as a separately labeled daily product whose exact temporal alignment is unknown. It is not a guaranteed exact T+24h prediction. NASA scores are not fabricated for +48h/+72h; unavailable NASA forecast values do not disable the in-house physical forecast.

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

## State-grounded, multi-turn assistant chat

The model receives one selected physical column, its full covariance/sensitivities, the six largest absolute learned feature contributions and compact provider summaries. Full grids, complete SHAP decompositions and routing geometry remain in the scan/inspection APIs. Recent whole conversation messages are bounded to 2,400 UTF-8 bytes; `history_context` reports any omitted older turns. The local adapter server enforces an 8,192-token context with a 1,536-token completion reserve. Hosted calls retain the compact evidence/history and completion cap, subject to the selected provider's context limits.

```json
{
  "message": "Explain this cell's uncertainty and source quality, then fly to it.",
  "scan_id": "returned-scan-id",
  "scenario_id": "displayed-state-id",
  "selected_h3_index": "cell-in-this-scan",
  "active_prediction_model": "terra_physics_ml",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "nasa_product": "today",
  "autonomous": true,
  "history": [
    {"role": "user", "content": "What controls the mechanical margin?"},
    {"role": "assistant", "content": "Prior verified response."}
  ]
}
```

Submit to `POST /api/v1/chat` (`/api/chat` and `/api/v1/agent/chat` remain aliases). History is bounded to twelve messages. The server validates the selected cell and active model and supplies current column/envelope, mathematical features, forecast/state identity, conditional corridor and provider provenance. In autonomous mode, chat cannot invent forcing changes.

Send the personal credential in **`X-LLM-API-Key`**. A body `api_key` is accepted for compatible clients; the header takes precedence. Providers are `openai`, `gemini`, `anthropic`, `openrouter`, `groq` and `custom_openai`. The custom provider also requires `base_url`, a public HTTPS base URL without query parameters, embedded credentials or redirects. Public chat uses request credentials only: no server-key or local-model fallback. With no key, verified deterministic templates and navigation remain active. Credentials are excluded from serialization, validation errors and application logs. The response's `model_provenance` identifies provider/model, request credential scope and verified-response versus unchecked availability.

For NASA, `nasa_product` selects the official relative `today` or `tomorrow` product independently of Terra's hourly forecast. Its selected score and action ranking use that product's official values; reference indices are not substituted. Product selection is part of the browser's conversation identity and stale-response guard.

An initial navigation request can omit scan/state/cell IDs. For example, `{"message":"Take me to 11.548, 76.134, switch to NASA LHASA and then back to TERRA-SHIELD, and inspect the steepest cliff","autonomous":true}` returns a validated UI plan. The browser performs the scan once, waits for its returned grid, and resolves later cell selectors against that new state. A plan does not assert that source acquisition or inspection already succeeded.

`ui_action.actions` supports ordered chains of these commands:

| Type | Main arguments |
|---|---|
| `scan_coordinates` | `latitude`, `longitude`, optional `label` |
| `search_location` | `query` |
| `select_prediction_model` | `model`: one of the five IDs above |
| `set_time_horizon` | Terra: `now`, `t_24h`, `t_48h`, `t_72h`, `peak_72h`; NASA: `nasa_today`, `nasa_tomorrow` |
| `set_map_layer` | `layer`: `probability`, `fs_mid`, `lhasa_2`, `nisar_radar`, `slope`, `pore_pressure` |
| `fly_to_cell` | `h3_index` or symbolic `target`; optional `pitch`, `bearing`, `zoom` |
| `inspect_cell` | `h3_index` or symbolic `target` |
| `compare_cells` | `primary_h3`/`secondary_h3`, or `primary_target`/`secondary_target` |

Symbolic targets are `steepest`, `riskiest`, `selected`, `lowest_fs` and `flattest`. Fixed H3 IDs must belong to the bound state. Actions carry originating nullable `scan_id`/`scenario_id` and `state_scope`; post-scan actions use `chain` scope. The executor validates the origin, runs each command in order, reports completion/failure and cancels stale work after a user state change. UI action horizon names map to the shorter forecast route keys above.

The response includes `clean_markdown`, `briefing_guard`, active IDs, `bound_claims`, `numeric_audit_passed`, `thought_steps` and `ui_action.actions`. Each public step has `step`, `phase` and `reasoning` fields containing short intent, evidence or action summaries. They are not private model deliberation or proof that an action ran. Execution results are recorded separately by the browser. The selected provider returns structured prose with numeric JSON-pointer bindings, verified and rendered by the server. Valid output is `VERIFIED_LLM_PROSE`; no-key/provider-unavailable responses retain `VERIFIED_TEMPLATE`, while invalid generated prose is withheld after retry. Templates and model prose are visibly distinct.

The frontend retains conversation messages with state/model/product identity, displays numeric citations, and rejects stale responses/actions after a coordinate, horizon, selected-cell, provider or model change. **AI Model & API Key** keeps personal settings in browser session storage. Follow [assistant connections](docs/assistant_connections.md). Configuration is not a successful live provider test, and mocked protocol checks do not establish a provider/model's answer quality. Legacy direct Python/CLI assistant transports are separate; their environment credentials do not fund public chat.

## Refresh measured radar evidence

`POST /api/v1/scans/{scan_id}/radar/refresh` accepts `{"scenario_id":"current-condition-state-id"}`. It rechecks NISAR processing and retrieves supported ready subsets, reuses the retained weather/terrain/other sources, and returns a newly identified full scan. The previous scan remains immutable. The browser replaces its base scan and invalidates stale conversation/actions. This route shares the bounded analysis admission budget.

`nisar.measurement_access` reports authentication, processing jobs, retry timing, measured availability and no-valid-pixel/coverage states. Discovery metadata and processing jobs are not measurements. Native masks and acquisition dates remain attached to returned measured pixels. Refreshing radar does not claim that the other providers were refreshed or that the approved classifier was retrained with SAR features.

For existing local installations, the default local model service is `http://localhost:11434`. The adapter server can choose loopback 8001 when 11434 is occupied and writes `artifacts/qwen_copilot_endpoint.json` for dynamic discovery. An explicit constructor URL or `TERRA_QWEN_URL` overrides that discovery. Only safe loopback endpoint metadata is accepted. The [archived RTX 5070 Ti guide](docs/legacy/windows_5070ti.md) preserves training, restart commands and optional GGUF/Ollama export; local GPU training is not required for the website. Both the historical trained context and direct-server context are capped at 8192 tokens, with up to 1536 completion tokens reserved for the structured reply; overflow is rejected without truncation. Actual Windows/CUDA/QLoRA execution remains a separate, unverified target-machine requirement.

## Research compatibility and publication

Optional sourced site records (`ground_columns`) can override cohesion, friction angle, unit weight, depth, head, conductivity and diffusivity at their exact H3 cell. Records retain units, source/method and validity dates where applicable. Missing fields remain labeled priors; invalid overrides are disclosed per cell.

Legacy scenario and replay endpoints remain research utilities. They are not the autonomous frontend workflow. Raw corpora, cached providers, generated reports and legacy/optional weights stay outside Git. The small approved runtime ML artifact is included; see its [model card](artifacts/MODEL_CARD.md). Generated local training bundles include source-reviewed materials and are not tracked source mirrors. See [training workflow](docs/training_workflow.md) and [source/API critique](docs/global_source_audit.md).
