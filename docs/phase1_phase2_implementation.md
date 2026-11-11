# Phase 1/2 record: corrected terrain, reference physics and v1 fusion

Historical implementation record, 10 November 2026. Terrain `metric_specific_area_bounded_1000m_v1` and physical columns `pedotransfer_reference_column_v1` remain the foundation. The **v1 fixed-offset model, metrics and frontend descriptions below are superseded** by [feature-based v2 and the live frontend](feature_fusion_phase3.md). The 6 November measurement-only live contract is also superseded. Read [current_status.md](current_status.md) for the active result.

## Terrain and providers

Both raster axes use local Mercator ground spacing, with the cosine-of-latitude correction. Raster rows increase southward; aspect is clockwise from north. Flat raster cells retain zero slope. Analytical plane tests cover multiple latitudes and azimuths.

TWI is `ln((A/b)/tan(slope))`, where `b=sqrt(H3 cell area)` is the declared effective contour width. Contributing paths have a fixed 1000 m length limit. The requested viewport has an additional 1600 m terrain halo, including sampling/derivative support, so overlapping cells in 1.2 km and 3.5 km scans receive the same bounded catchment. This is a bounded hydrologic index, not whole-basin drainage delineation. The numerical tangent regularization applies only to TWI, not mechanical slope.

All 867 retained terrain locations were rebuilt from decoded AWS Terrarium tiles. Original derived caches remain archived. Raw tiles are retained under `data/source_catalogs/terrarium_tiles`; corrected products are under `terrain_metric_v1`. Terrain is later topography, not an event-date survey. The elevation-based ocean filter remains provisional and can exclude below-sea-level land.

SoilGrids uses WCS directly, the property-specific map argument, a valid local Goode Homolosine PROJ definition, and thickness-weighted mean predictions through 0–30 cm over a 3 km patch. The dead REST endpoint no longer consumes the request budget. The patch is regional input, not a borehole or per-cell strength measurement. Missing weather hours invalidate accumulation summaries; liquid rainfall is separated from snow water equivalent. USGS queries explicitly separate a 300-second recent catalog window from 30-day background context. PGA remains an attenuation proxy, not an accelerometer observation. Overpass requests buildings and quarries, intersects geometry with cells, and fails over using normal TLS verification. Coastal GloFAS requests select land cells.

## Reference physical column

Every retained observation and live terrain cell can receive a conditional reference column without boreholes. Valid supplied site records override priors at their own cell. Invalid supplied records are disclosed, not silently reinterpreted as measurements at another cell.

Saxton–Rawls hydraulics and the Iverson pulse response are published equation families. Texture-to-strength coefficients, slope-depth coefficients, initial storage, basal drainage and covariance are **project reference assumptions**, not universal published transfer fits. Their values and sources are exposed per column. Bedrock joints, liquefaction and erosion failure are separate applicability domains.

Antecedent storage is advanced over six days of hourly liquid forcing, including infiltration capacity, basal drainage, ET and overflow. Surface moisture updates only its 7 cm support. Incompatible moisture/porosity pairs retain the raw conflict and use water-balance initialization. Retained totals without hourly pulses are explicitly reconstructed uniformly. Missing forcing uses a labeled nominal scenario and enlarged forcing uncertainty, not an observed dry-weather claim.

FS has no display cap. Joint delta-method covariance spans cohesion, friction, unit weight, depth, conductivity, diffusivity, initial head, slope and rain. The API exposes the Jacobian, covariance, `J Sigma J^T`, variance allocations, and `[FS_low, FS_mid, FS_high]`. Reliability integrates a bounded slope distribution by quadrature, then uses conditional log-FS FOSM for the remaining parameters. This avoids the ratio singularity near zero slope. The sensitivity envelope includes both the nominal FS and the mixture fifth/ninety-fifth quantiles; distribution quantiles are also exposed separately. The unconditional local FOSM variance and linear-normal diagnostic remain available with their limitations. These are parameter-conditioned reference diagnostics, not measured event frequencies or guaranteed interval coverage. Exactly zero driving shear has its own status rather than an artificial slope floor.

Runout uses the standard xi convention in m/s², exact constant-edge Voellmy energy integration, travel-time quadrature, and explicit release-from-rest authorization. Three declared friction/depth scenarios produce a corridor. Release is conditional; a returned trajectory does not assert failure. Routing is on a coarse H3 terrain graph, without entrainment, mass conservation or momentum traversal over rises. Boundary stops are flagged as truncated.

## Archived v1 target correction and four-way experiment

The 71 training positives labeled only by failure footprints are retained as unresolved failure extents and excluded from supervised initiation fitting. They are not converted to negatives or assigned invented crowns. The remaining 93 manual-point initiation proxies have 93 matching temporal-absence controls. Points within 300 m still do not establish surveyed crowns; controls remain weather-selected inventory absences.

Full source event/spatial components and holdout/calibration reservations are constructed before feature and target selection. Training has 186 rows in 10 components. All 16 original rainfall holdout rows remain reserved. The four models use identical five outer folds, grouped inner calibration, and the same development holdout:

| Model | Component CV AUC | Brier |
|---|---:|---:|
| Rainfall-Only | 0.8952 | 0.0864 |
| Physics-Only | 0.5931 | 0.4413 |
| Weather+Terrain ML | 0.8880 | 0.0967 |
| Physics-Informed ML | 0.6662 | 0.3451 |

The fusion model receives source observables plus uncapped FS/envelope, shear/friction ratio, cohesion stability number, pore-pressure ratio, and critical pressure/rainfall deficits. Monotonic trees learn corrections to a FOSM log-odds base margin; grouped offset calibration preserves marginal monotonicity. This physical-prior assumption is part of the model, not a post-hoc probability union. A smaller minimum Hessian is necessary for the physical-offset learner because near-zero prior probabilities otherwise prevent fold trees from splitting. All three final fusion folds use nonconstant splits.

This table records the fixed-offset implementation, whose optimization/design limitation was subsequently corrected. It is **not** a scientific refutation of physics-informed ML. The [v2 comparison](feature_fusion_phase3.md) uses the same cohort/folds/holdout and separates learned scores from conditional coupling. Parameter priors and imperfect report locations remain important. No model was selected from holdout results. Weather extends to the event-date evening and can follow actual failure; this is retrospective development evaluation, not prospective warning validation.

## Archived v1 co-pilot and frontend contract

Scans expose `scan_id`, `scenario_id` and a schema version. Stored states are copied and bounded; subsequent scenarios do not overwrite earlier inspections. Scenario evaluation reuses the original provider snapshot. Cell inspection, conditional runout and scenario retrieval have state-specific HTTP routes.

`/api/chat`, `/api/v1/chat` and `/api/v1/agent/chat` accept the active identifiers. The co-pilot receives the selected column/envelope, fusion score, feature evidence, mechanical counterfactuals, release corridor and source provenance. Server-derived `ui_action.actions` support `fly_to_cell`, `inspect_cell` and `apply_scenario`.

Qwen returns structured prose with numeric JSON-pointer bindings. The server verifies and renders those bindings as canonical metric/value pairs. Valid local output is `VERIFIED_LLM_PROSE`. Confirmed absence of compatible weights yields the multi-paragraph `VERIFIED_TEMPLATE`; service errors and invalid generated prose have separate failure statuses and do not masquerade as absent weights. The protocol is tested with mocked Ollama responses; actual Qwen weights and Windows CUDA were not exercised here.

The regenerated corpus has 594 distinct cell/date traces after full-component holdout embargo and deduplication. No padding restores the old nominal count of 600. The primary file is `sft_context_trajectories.jsonl`; old filenames remain byte-identical compatibility aliases. Calibration rows may appear in this instruction corpus, so it is not independent LLM evaluation. QLoRA has an 8192-token context and rejects silent trace truncation.

## Reproduction

```sh
python scripts/rebuild_metric_terrain.py --fetch  # retained locations only; restartable tile cache
python scripts/train_physics_informed.py         # offline; rebuild all historical columns, four-way fit
python scripts/regenerate_sft.py
python scripts/report_physics_context.py
python scripts/check_training_artifacts.py       # network disabled
python scripts/finalize_readiness.py
python scripts/sync_training_bundle.py
```

These report filenames are regenerated local artifacts; their version and model hashes identify the run, so a current file need not contain the archived v1 numbers above. CPU engineering checks, predictive validity and CUDA verification remain separate. The browser frontend was built subsequently and now follows the live-first [FRONTEND_SPEC.md](FRONTEND_SPEC.md), with no manual weather controls or replay-first defaults.
