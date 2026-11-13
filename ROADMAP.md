# TERRA-SHIELD global roadmap

## Target

Build a global landslide hazard assessment and forecasting platform competitive with **NASA LHASA 2.0**, with inspectable physical columns, uncertainty, triggering scenarios and conditional downstream exposure. Regional studies are validation units within a global program. The product ambition spans multiple continents, climates, terrain and material regimes, with repeatable global refresh and accessible analysis.

The scientific starting point is [LHASA 2.0's published study](https://doi.org/10.3389/feart.2021.640043) and [NASA's implementation](https://github.com/nasa/LHASA). Pin the exact comparator version and inputs for each experiment. A more detailed interface, finer H3 grid or longer provider list does not demonstrate greater forecasting skill.

## Current foundation

Corrected metric terrain, full prior-based physical columns, joint uncertainty, conditional Voellmy corridors, feature-based four-way ML, state APIs, CLI and a live-first global browser interface form the current implementation. The browser follows [FRONTEND_SPEC.md](docs/FRONTEND_SPEC.md): full-screen satellite terrain, arbitrary-coordinate scanning, automatic source-driven forecast horizons, read-only telemetry and selected-cell Qwen chat. Manual weather sliders and replay-first city presets are absent from the product workflow.

Removing the fixed FOSM training offset improves the temporal experiment; the conditional mechanical factor still worsens Brier, and the broader proxy benchmark does not establish superiority over rainfall-only. Scheduled global infrastructure, independent predictive validation and evaluation of the selected hosted assistant remain open. The former local Windows/QLoRA plan is [archived](docs/legacy/local_llm_training.md), with target-hardware execution unverified. See [current evidence](docs/current_status.md) and [the architecture/evaluation report](docs/feature_fusion_phase3.md).

## 1. Establish input credibility

Apply the [source/API audit](docs/global_source_audit.md) to every consumed field. Publish units, native support, model/product version, issue/valid/retrieval times, transformations, provenance, assumptions and unknowns. Check accuracy separately from endpoint availability. Audit existing coefficients as critically as new providers.

Priorities include independent terrain-tool comparisons and a land mask, source/depth-aware soil uncertainty, as-issued rainfall/moisture archives, validated material/depth priors, event timing and initiation geometry. Evaluate IMERG and SMAP assimilation products as explicit candidates alongside existing weather inputs; they are not currently integrated. Acquire representative wet non-events as well as dry controls. Preserve old sources and failed cases.

Reference physics must remain available without boreholes, with uncertainty and honest provenance. Site measurements should replace compatible priors where available. Conditional estimates remain distinguishable from observations and validated probabilities.

## 2. Freeze a fair global benchmark

Predeclare an untouched geographical and chronological evaluation across multiple continents and climatic/geological regimes. Register dataset IDs, event grouping, control sampling, source vintages, spatial support, forecast horizons, metrics and operating thresholds before evaluation. Include poorly reported regions and quantify inventory completeness.

| Requirement | Completion evidence |
|---|---|
| Comparable task | Shared rainfall-triggered target, spatial grid/aggregation, issue times and event windows for Terra and a pinned LHASA 2.0 comparator |
| Inputs available at issue time | As-issued products and latency masks; retrospective final/reanalysis runs reported separately |
| Independent evaluation | Whole storms/events, spatial blocks and future time periods excluded from fitting, calibration and threshold selection |
| Honest false alarms | Representative space/time backgrounds, including wet non-events; report prevalence, exclusions, coverage and forecasted area/time |
| Useful skill | Precision-recall, Brier/reliability, event detection at attainable fixed false-alarm budgets, lead time and component-level uncertainty |
| Added value | Four-way ablation plus matched-data comparisons; separate source improvements from algorithm improvements |
| Global applicability | Results by region, regime, season and input availability, with an explicit supported-domain mask |

Report common-coverage comparisons and each system's full-domain performance separately. Missing inputs must remain in coverage accounting. Do not improve a headline metric by dropping difficult regions, changing controls, moving report pins to steep cells or tuning to the test set. Conditional runout and other triggers need separate benchmarks; they do not count as improved rainfall-nowcast skill.

## 3. Build global operating infrastructure

Move from query-time point acquisition to scheduled bulk/tile ingestion, versioned geospatial storage and incremental updates. Cache static terrain/soil independently of forcing; share acquisition and derived products across users. Size storage, compute and refresh budgets from measured land-domain and supported resolution counts.

Measure cold/warm latency distributions, source age, global missingness, refresh lag, rate limits, retries and daily cost under concurrent use. Replace reliance on best-effort public point endpoints where scale or terms require a supported delivery arrangement. Retain reproducible issue-time snapshots and product revisions. Test provider outages and degraded operation without disguising stale/model-prior values as fresh measurements.

Latitude limits, antimeridian behavior, coasts, islands, below-sea-level land, steep rock terrain and regional mapping gaps need explicit tests. An arbitrary-coordinate scan is only one part of global readiness; scheduled coverage and sustained refresh must be demonstrated.

## 4. Validate the physical contribution

Compare numerical equations against analytic limits and an external implementation such as TRIGRS in compatible domains. Evaluate material/depth/hydrologic priors on independent observations, and test parameter uncertainty coverage and structural errors. Diagnose the error added by conditional mechanical coupling, separately from learned feature-fusion behavior. Change assumptions based on evidence, preserving failed ablations and unchanged holdouts.

Validate routing, friction/depth uncertainty, boundary truncation, deposition and exposure with independent runout inventories. Add rock-slope, deep-slide, seismic, snow/thaw, post-fire and erosion capability only with appropriate models and regime-specific evidence. More mechanisms do not automatically improve the rainfall benchmark.

## 5. Verify and operate the global interface

The live product contract is [FRONTEND_SPEC.md](docs/FRONTEND_SPEC.md). The MapLibre satellite/terrain canvas fills the viewport immediately. A user can scan an unscanned map point, map center, coordinate pair or place name. H3 inspection, source telemetry, conditional runout, physical uncertainty and automatic current/+24h/+48h/+72h/peak forecast views share the active state identity. There are no manual perturbation sliders or fixed replay defaults in this workflow.

Continue end-to-end coverage across unnamed mountains, valleys, coasts and remote terrain, including provider failures and source-domain limits. Measure cold/warm scan and forecast latency, rendering, mobile/keyboard access and request cancellation. Aggregate wide-area views progressively instead of rendering every fine cell globally at once.

Co-pilot chat uses bounded multi-turn context and active selected-cell evidence; autonomous product actions use validated chains without inventing weather. Evaluate numeric grounding, semantic claims and tool behavior on held-out prompts for the actual hosted provider/model. Distinguish a verified deterministic template from verified LLM prose and preserve service/verification errors. Changing the prose provider does not establish hazard-model improvement.

Frontend build/type checks and a successful live location check establish software integration at tested inputs. They do not validate global forecasting skill, uniform provider coverage or model probabilities.

## 6. Reproduce and assess deployment

Reproduce the fixed four-way hazard experiment on CPU using the [active workflow](docs/training_workflow.md), preserving source/cohort identity, environment, hashes and save/reload parity. Evaluate hosted assistant quality independently from hazard prediction. If the historical CUDA/local-LLM path is revisited, record actual target-device execution and memory measurements; CPU reports cannot certify those runs.

Prospective evaluation and external scientific review should precede operational warning claims. Define calibrated thresholds, responsibilities and partner workflows against measured evidence. CAP export is currently disabled (HTTP 409). No global performance parity or operational-readiness date is claimed.
