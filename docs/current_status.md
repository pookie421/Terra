# Current status — 14 November 2026

TERRA-SHIELD targets a global, autonomous landslide hazard and forecast platform competitive with LHASA 2.0. The current development implementation combines a live-first global map, source ingestion, prior-based physics, feature-based ML, automatic forecast states and a selected-cell co-pilot. Independent global forecasting skill and sustained operational coverage remain open. [The roadmap](../ROADMAP.md) and [source audit](global_source_audit.md) define that evidence program.

## Active contracts

| Area | Current behavior | Evidence gap |
|---|---|---|
| Frontend | Full-screen satellite/terrain map, three primary models, advanced comparisons, compact Options, automatic forecast horizons and cell inspection | Integration at tested locations does not establish uniform global coverage, equal score calibration or predictive validity |
| Terrain | Metric Mercator spacing on both axes; corrected aspect; no mechanical slope floor; specific-area TWI with bounded 1 km contributing paths | Compiled DEM accuracy, bounded drainage support, provisional elevation land mask, polar coverage and independent GIS comparison |
| Physics | `pedotransfer_reference_column_v1`, antecedent storage, transient response, uncapped FS, sensitivity envelope and joint FOSM | Strength/depth/state/covariance priors and structural error need independent validation |
| ML | `physics_informed_initiation_v2_features`; ordinary prevalence initialization, source/mechanical/interaction features, grouped inner selection/calibration | Sample selection, uncertain labels and conditional coupling still limit interpretation |
| NASA comparison | Native 30-arcsecond ImageServer sampling, preserved categorical susceptibility and separately labeled coarse reference fallback | Served model version and issue/valid time are undeclared; the fallback is an unvalidated project index, not a reproduced LHASA 2.0 model |
| NISAR | Public CMR discovery, acquisition/product metadata, optional local/authenticated raster subsets and `nisar_context_features_v1` | Metadata is not a radar measurement; the active 25-feature classifier has no newly fitted SAR weights or demonstrated SAR skill gain |
| Runout | Conditional Voellmy release, three scenarios, corridors, unique asset intersections and boundary flags | Coarse routing without entrainment or mass conservation; no independent motion validation |
| Forecast | Automatic source-driven 72-hour column/ML projections and selectable horizon states | Forecast-as-issued skill, uncertainty coverage and useful warning lead time remain unverified |
| Co-pilot | Local or operator-configured hosted compatible API; verified numeric bindings, public summaries and eight ordered actions; planning before the first scan | Protocol tests do not establish a particular model's answer quality, semantic/tool reliability or global hazard skill |
| Research reproduction | Canonical offline CPU hazard-model fitting, shared-cohort benchmarks, model/data hashes and source-linked instruction diagnostics; hosted APIs are the current assistant direction | Complete reproduction needs the retained corpus; the retired Windows/QLoRA path remains unverified on its target hardware |

The frontend follows [FRONTEND_SPEC.md](FRONTEND_SPEC.md): no manual perturbation sliders, fixed-city dropdown or replay-first boot. Legacy analyst/replay APIs and fixtures remain reproducibility tools; they do not drive the live product by default.

Missing boreholes select exposed reference priors, preserving physical columns, uncertainty and conditional routing. Site records override compatible fields at their own cell. Missing model weights, incomplete future forcing, invalid site records and unsupported mechanism inputs have specific statuses. They do not erase otherwise available analysis.

## NASA, radar and app-control integration

The primary map choices are Terra In House Model (the existing Physics-Informed ML), NASA LHASA and Rainfall baseline. Physics-only and weather/terrain comparisons remain advanced diagnostics, while all views preserve the same scanned location and inspected cell. H3 resolution 9 averages about **0.105 km² per cell**; these cells are not 30 m pixels. The nominal 30 m DEM input, coarser environmental forcing, H3 aggregation and NASA's roughly 1 km geographic pixels have different support. Display detail is not evidence of superior prediction.

`nasa_lhasa.py` samples Today, Tomorrow and Global Landslide Susceptibility at native support under one 3.5-second budget. The public endpoints do not declare an exact served LHASA version or issue/valid time; the upstream repository's newer version is not assigned to those rasters. Official output and an explicitly unvalidated project fallback remain separate. In mixed coverage, the NASA critical-cell selection uses official values when any exist, rather than ranking them against unlike fallback indices. The relative Tomorrow product is not guaranteed to align with exactly T+24h; NASA T+48h/T+72h predictions are not fabricated. This display is not a matched LHASA skill benchmark.

`nisar_processor.py` discovers real granules and reads explicitly described L-SAR subsets where available. Coherence, LOS time-series velocity and the uncalibrated HV/HH index retain units, acquisition timing and provenance. Metadata-only scans leave measured fields empty and retain the full broad reference FOSM distribution. No arbitrary uncertainty multiplier or coherence-to-strength law is introduced. A documented, real-granule-anchored site cohesion calibration may update the physical column and its existing mechanistic ML inputs; explicit ground observations take precedence. The separate radar feature context does not turn the approved model into a SAR-trained classifier.

The co-pilot returns brief public `thought_steps` describing intent, available evidence and planned actions. Validated action chains can scan coordinates, search a place, select a model/horizon/layer, fly to or inspect a cell, and compare cells. New-grid targets resolve after acquisition; stale state/model replies and superseded chains are rejected. Plans do not claim execution before the browser reports results. `VERIFIED_LLM_PROSE`, `VERIFIED_TEMPLATE` and withheld/unavailable outcomes remain distinct.

The regenerated `agentic_app_v2` instruction corpus retains the same **594 source cell/date identities and component exclusions**. It teaches the app schema and honest missing NASA/radar evidence without attaching present-day satellite or nowcast values to historical event dates. Earlier corpus/report bytes remain in local history. Model weights, cohorts, benchmark labels and the results below are unchanged. Hosted assistant configuration replaces local fine-tuning as the deployment direction; no new language-quality result is claimed. See [NASA and radar methods](nasa_radar_comparison.md), [the active research workflow](training_workflow.md) and [the archived Windows guide](legacy/windows_5070ti.md).

## Benchmark A: unchanged matched temporal evaluation

All **1,611 retained observations** have reference context. The initiation experiment preserves **93 manual-point proxies and 93 matched temporal-absence controls**, **ten components**, the same five outer folds and the original **sixteen-row development holdout**. The **71 unresolved footprint positives** remain preserved outside initiation supervision.

| Component CV | ROC-AUC | Brier |
|---|---:|---:|
| Rainfall-Only | 0.8952 | 0.0864 |
| Physics-Only (FOSM) | 0.5931 | 0.4413 |
| Weather+Terrain ML | 0.9043 | 0.0911 |
| Feature fusion: learned trigger score | 0.8940 | 0.0965 |
| Physics-Informed ML: conditional coupling | 0.9026 | 0.1183 |

The v1 fixed-offset AUC/Brier **0.6662/0.3451** is an archived architecture result. Thirty-four of 186 rows started with logistic Hessian below 1e-7; the probability clip limited initial logits to approximately **−27.63**, rather than the raw −57.5 example. The new classifier learns from features with ordinary fold-prevalence initialization. Its improvement supports the optimization diagnosis; it does not demonstrate that physical priors are already accurate or that fusion beats rainfall-only in Brier.

Temporal pairs share coordinates, so static terrain cannot by itself discriminate their labels. Quiet-weather control selection, report-point uncertainty, retrospective forcing that may follow failure, ten components and an inspected holdout further limit interpretation. These are development metrics, not annual/global event probabilities or a reproduced LHASA benchmark.

## Benchmark B: multi-regime proxy evaluation and spatial diagnostics

The full context corpus retains rainfall, coseismic, cryosphere, post-fire, anthropogenic and unclassified event regimes. A secondary report/extent-versus-inventory-absence experiment fits **510 eligible rows in 34 components**, retaining original holdout/legacy/calibration embargoes.

| Multi-regime proxy component CV | ROC-AUC | Brier |
|---|---:|---:|
| Rainfall-Only | 0.7842 | 0.1772 |
| Physics-Only (FOSM) | 0.4795 | 0.4960 |
| Weather+Terrain ML | 0.7713 | 0.1816 |
| Feature fusion: learned trigger score | 0.7767 | 0.1847 |
| Physics-Informed ML: conditional coupling | 0.6866 | 0.2451 |

This broadens development evidence without converting report footprints into surveyed initiation labels. It still favors rainfall-only on aggregate. Small non-rainfall regimes and unclassified triggers prevent strong mechanism-specific conclusions. The full multi-causal architecture remains in scope; equivalent mechanism validation is not claimed.

Twelve retained Monaco/Wayanad storm-grid cases are **unlabeled consistency diagnostics**. No neighboring terrain cells become invented non-failure labels, and no AUC/FPR is calculated from presumed stable flats. The retained Monaco calculation reduced the old 0.822 flat-cell behavior substantially, but the latest diagnostic sweep still misses the requested showcase targets: the flat cell reaches about 0.03 and the cliff about 0.80 under high added rain. These gaps remain disclosed rather than fitted away with geographic exceptions.

## Verification and remaining work

Deterministic tests cover equations, extreme-prior optimizer behavior, provider contracts, grouping, state immutability, Newmark applicability and mocked co-pilot responses. Two retained-corpus tests explicitly skip without their required local feature frame or corrected terrain cache. The bundled runtime model is checked separately without provider acquisition. Full verification additionally checks feature/model/data hashes, unchanged source labels/manifests, fold isolation, metrics and **594 distinct source cell/date instruction traces**. The instruction corpus is not an independent assistant evaluation.

A live arbitrary-coordinate backend check at **30.2357, 78.3429**, without a place-name query, returned **489 H3 cells**, reference physical columns, the actual v2 feature model and automatic 72-hour projected states. The check took approximately **16.9 seconds**; eleven provider channels were live, supplied coordinates were labeled separately, and Overpass TLS / EONET wildfire timeout failures were disclosed. A projected T+24h state and exact-cell inspection also returned successfully. This is one integration/availability check, not a global coverage or forecast-skill result.

Frontend build/type checks, live endpoint availability, CPU engineering consistency, target-device execution and predictive skill are separate results. The root completion report should identify their actual outcomes. Git excludes raw corpora, historical/optional weights, generated reports and portable mirrors. The approved small runtime model is now included with its [model card](../artifacts/MODEL_CARD.md) and compatible constraints; prepared local bundles carry source-reviewed training materials separately.

[Feature-fusion details](feature_fusion_phase3.md), [training workflow](training_workflow.md) and [physical methods](physics_sources.md) describe the active implementation.

[Research history](research_history.md) preserves the Monaco defect, retired measurement-only response, failed fixed-offset fusion, earlier source experiments and model-conditioned control diagnostics. It links the complete original records at an immutable commit. Do not mix their contracts or cohort sizes with the active experiment.
