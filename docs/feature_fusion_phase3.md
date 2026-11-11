# Feature-based fusion and live global command center

Updated 11 November 2026. Active model: `physics_informed_initiation_v2_features`. Terrain: `metric_specific_area_bounded_1000m_v1`. Physical columns: `pedotransfer_reference_column_v1`.

The target remains a global, multi-causal platform competitive with LHASA 2.0. The primary matched temporal experiment, broader proxy benchmark, physical applicability and live interface are separate evidence tracks.

## Why the fixed physical offset was removed

The former fusion initialized every row with `logit(p_fosm)` at unit weight. Its implementation clipped probabilities to `[1e-12, 1−1e-12]`, so the smallest initial logit was approximately **−27.63**. An unclipped probability of 1e-25 would produce approximately −57.56, but that was not the actual minimum passed by the code.

For logistic loss, `h=p(1−p)`. In the preserved 186-row frame, **34 rows** began below `h=1e-7`. The old configuration used L2 regularization and a finite update budget: 120 rounds × learning rate 0.04 × maximum step 1 permits at most 4.8 log-odds of tree correction along a row's path. Lowering `min_child_weight` could permit splits but did not remove the fixed, potentially miscalibrated prior or guarantee sufficient correction. A later shared calibration intercept could also raise moderately unstable dry slopes too aggressively.

That is an architecture/optimization diagnosis. The old AUC/Brier **0.6662/0.3451** must remain an archived result of that implementation, not a scientific rejection of physics-informed ML. Reference-parameter accuracy and statistical optimization still require separate evaluation.

## Replacement architecture

The learner starts from each fitting fold's label prevalence. It receives meteorological, corrected terrain, mechanical and trigger-interaction features rather than an unscaled FOSM training offset.

| Group | Included evidence |
|---|---|
| Source forcing/terrain | 24h liquid rain, prior 144h rain, raw surface moisture, slope, bounded specific-area TWI, curvature |
| Physical margin | FS low/mid/high, log-FS, FOSM reference score/log probability, shear/friction ratio and cohesion stability |
| Hydrology | Pressure head, pore-pressure ratio, critical pressure/rain deficit and attainability |
| Trigger interaction | Rain-to-critical-deficit ratio; rainfall-induced frictional resistance reduction in kPa |
| Seismic context | Newmark displacement only with positive source/scenario PGA and a stable-block initial state, plus availability status |
| Mechanical compatibility | Necessary gravity-driving-shear-over-cohesion condition under declared lognormal parameter moments |

Monotonic constraints follow each feature's declared direction. Nested grouped selection chooses regularization/depth inside fitting components; column-sampling weights reduce static-prior dominance. A disjoint component reserve fits each calibration offset. Holdout results do not select models or coefficients.

The API exposes the **learned trigger score** and **mechanical admissibility** separately. The displayed conditional fusion score multiplies them. This factor is a project assumption-conditioned necessary-condition estimate. It is neither calibrated release probability nor a replacement for uncertainty in friction, depth, hydrology or structural model error. Native tree SHAP explains learned margins; its additive terms do not include the separate mechanical factor as if it were a learned causal contribution.

## Benchmark A: matched temporal discrimination

Source observations, labels, dates, all five outer-fold memberships and the original holdout manifest remain unchanged. There are **186 rows (93 positive, 93 matched absence controls), ten components and sixteen development-holdout rows**. Seventy-one unresolved footprint positives stay outside initiation supervision, without invented crowns or stable relabeling.

| Component CV | ROC-AUC | Brier ↓ |
|---|---:|---:|
| Rainfall-Only | 0.8952 | 0.0864 |
| Physics-Only | 0.5931 | 0.4413 |
| Weather+Terrain ML | 0.9043 | 0.0911 |
| Feature fusion: learned trigger score | 0.8940 | 0.0965 |
| Physics-Informed ML: conditional mechanical coupling | 0.9026 | 0.1183 |

The learned feature model reaches the requested approximate temporal AUC/Brier range. The conditional score's Brier remains outside 0.08–0.11 and worse than rainfall-only. Its higher AUC does not negate that calibration loss.

Matched pairs share coordinates, so terrain distributions are identical by label. This design evaluates temporal discrimination at reported failure sites; it cannot establish spatial discrimination across a rainy landscape. Controls are weather-selected inventory absences, not surveyed stable slopes. Point labels may not be initiation crowns; event-date weather can follow failure; the holdout has been inspected. Those limitations are not repaired by a model architecture change.

## Benchmark B: global proxy coverage and unlabeled storm grids

All **1,611 contexts** remain available, including **1,263 monsoon-rain**, **45 coseismic**, **28 cryosphere**, **9 post-wildfire**, **16 anthropogenic cut/mining** and **250 unclassified-trigger** records. These counts include labeled and unlabeled records; they are not independent events or complete mechanism validation sets.

A secondary, grouped report/extent-versus-inventory-absence task uses **510 eligible rows in 34 components** after original holdout, legacy and calibration embargoes. Its target differs from surveyed initiation, and its metrics cannot be substituted for Benchmark A.

| Multi-regime proxy component CV | ROC-AUC | Brier ↓ |
|---|---:|---:|
| Rainfall-Only | 0.7842 | 0.1772 |
| Physics-Only | 0.4795 | 0.4960 |
| Weather+Terrain ML | 0.7713 | 0.1816 |
| Feature fusion: learned trigger score | 0.7767 | 0.1847 |
| Physics-Informed ML: conditional mechanical coupling | 0.6866 | 0.2451 |

This test does not establish global superiority for fusion. The mechanical coupling remains a substantial source of error on these proxy labels. Non-rainfall sample counts and the regolith column's applicability limit mechanism-specific conclusions; the wider global architecture remains in scope.

The twelve Monaco/Wayanad retained storm-grid cases are **unlabeled spatial consistency diagnostics**. Flat/low-slope cells under the same rain reveal implausible behavior and compare model sensitivity, but lack of a reported failure does not supply a verified non-failure label. No neighborhood negatives, AUC or FPR are invented from that assumption.

The retained Monaco diagnostic sweep at added rain `[0,1,10,50,100,300]` remains below the requested steep-cliff >0.85 target and does not hold the flat cell arbitrarily close to zero at every setting. Recent diagnostics show roughly **0.005→0.03** for the flat harbor and **0.18→0.80** for the cliff. The original 0.822 flat-cell result is substantially reduced, but those remaining gaps are visible. They are not fixed by geographic exceptions, score caps or adjusting labels/control dates.

## Live-first frontend and automatic forecasts

[FRONTEND_SPEC.md](FRONTEND_SPEC.md) is the product contract. `frontend/` uses Vite, React 18, TypeScript, Tailwind, MapLibre and Lucide. The full-viewport world map loads Esri imagery and Terrarium terrain immediately, with pitch 55 and exaggeration 1.35. The obsidian glass HUD uses Inter and JetBrains Mono. Compiled Terrarium terrain is not uniformly measured at 30 m accuracy everywhere.

Map clicks, a map-center button, coordinates and place-name search request a live scan. There are no manual weather sliders, fixed-city dropdown or replay-first boot. Existing H3 cells are selectable; map layers expose conditional fusion, FS, slope and pressure. Read-only source telemetry distinguishes fresh, cached, fallback and unavailable values.

The current state and automatic `T+24h`, `T+48h`, `T+72h` and peak-forecast views are driven by incoming source weather/ensembles. Displayed map, physical column, runout and co-pilot evidence must share the active state identity. Missing future forcing must have a specific reason while retaining current physical analysis. A projected FS crossing is not a precise failure-time prediction or a validated warning. Ensemble quantiles are used only when the provider returns the required ensemble evidence. Missing ensemble information does not create a synthetic high-rain member; a disclosed deterministic forecast may remain available. Projected forcing uncertainty remains a declared assumption rather than an empirically validated interval.

The live API contract is:

- `POST /api/v1/scan` accepts `{latitude, longitude, query?}` and legacy `lat`/`lon`; unnamed coordinates do not require a successful place-name lookup.
- `GET /api/v1/health` exposes engine/model readiness without acquiring environmental providers.
- `GET /api/v1/inspect/{scan_id}/{h3_index}?scenario_id=...` inspects the exact displayed state.
- `predictive_forecast_72h.views` exposes `now`, `t24`, `t48`, `t72` and `peak` metadata; `GET /api/v1/scans/{scan_id}/forecast/{view_key}` retrieves the corresponding full state-bound Scan.
- `POST /api/v1/chat` accepts the active IDs, `selected_h3_index`, bounded user/assistant `history` and `autonomous: true`. Autonomous mode permits inspection/fly actions and does not mutate weather.

The co-pilot retains bounded conversation context and selected-cell evidence, displays verified prose/template status and bound numeric citations, and executes validated inspection/fly map actions. Service problems and invalid prose retain explicit metadata. Deterministic fallback must not be presented as generated Qwen prose. Actual QLoRA weights require separate target-machine testing.

## Reproduction and artifact boundaries

```sh
python scripts/train_physics_informed.py
python scripts/benchmark_global_context.py
python scripts/regenerate_sft.py
python scripts/report_physics_context.py
python scripts/report_storm_grid.py
python scripts/check_training_artifacts.py
python scripts/finalize_readiness.py
python scripts/sync_training_bundle.py
```

These commands require the retained source corpus and current local models. `report_storm_grid.py` reads explicitly exported retained response fixtures; it is not live product acquisition. Regenerate those fixtures with `scripts/export_frontend_replays.py` when needed for research diagnostics. They are not the frontend boot default.

Local reports include `physics_informed_ablation_report.json`, `global_context_benchmark.json`, `storm_grid_diagnostics.json`, `context_observation_coverage.json` and `reference_verification.json`. Training frames, models and generated reports remain outside Git. Source-only CI checks deterministic contracts with two explicit artifact-dependent skips; full verification requires the local corpus.

The Windows bundle retains the **1,611-row context**, **186-row initiation experiment** and **594 distinct instruction traces**, plus corrected terrain and source archives. `run_training.bat --qlora` runs CUDA checks, the XGBoost/global experiments, instruction regeneration and 8192-token Qwen2.5 QLoRA. CPU checks do not certify actual Windows/CUDA/QLoRA execution, globally calibrated warning skill or an LHASA benchmark reproduction.
