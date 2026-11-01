# TERRA-SHIELD

A research prototype combining local terrain mechanics, five landslide trigger regimes, constrained XGBoost and a tool-grounded local Qwen assistant. The FastAPI backend returns H3 polygons, scenario scores, provisional runout paths and source provenance. There is no production frontend in this repository.

**Research use only.** Scores, causal shares, velocities and routes have not been validated for warnings, evacuation or engineering design. Passing `pretraining_ready` is a project regression gate, not evidence of operational safety. The current result and measured metrics are in [the full audit](artifacts/full_system_bug_audit.md) and [its JSON](artifacts/full_system_bug_audit.json). They are regenerated, not manually set to pass.

The original code is **MIT-licensed open source**, including commercial use. See [LICENSE](LICENSE). See [third-party terms](THIRD_PARTY_NOTICES.md) for datasets, imagery, dependencies and model weights.

## Latest measured baseline (2026-11-01)

The v5 model was rebuilt after cryospheric/fuel boundary fixes and finite-block lateral resistance. Spatial out-of-fold ROC-AUC is **0.8441**; the unchanged 48-row event-holdout ROC-AUC is **0.7700**, with Recall@40 **0.9167**. The requested PINLS filter removes no current negatives, so its secondary holdout has identical scores. Thirteen separate spatial blocks (66 rows) calibrate conformal label intervals; q is approximately 1 and mean holdout interval width is 1.0. These intervals do not provide precise failure probabilities.

**The latest requested pretraining gates are not all met.** The authoritative flag is [artifacts/readiness.json](artifacts/readiness.json). Full, randomized and boundary audits must match the current source/model/data fingerprint. Engineering invariants are reported separately from the latest mandatory score, latency and scientific-provenance requirements. See [boundary results](artifacts/literature_boundary_audit.md), [full scenarios](artifacts/full_system_bug_audit.md), [randomized sweeps](artifacts/randomized_global_blind_audit.md), and [exact numbered source](artifacts/core_code_audit.md). No target is made green by changing observed scores or hiding failed cases.

## Run locally

Use Python 3.11 or 3.12 for a new environment. The CPU baseline can run on a Mac; XGBoost does not use Apple MPS. Optional geospatial decoders are separate.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Optional SoilGrids raster decoding:
python -m pip install -r requirements-geo.txt
python -m unittest discover -s tests -v
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

API documentation is at `http://127.0.0.1:8000/docs`. For example:

```bash
curl 'http://127.0.0.1:8000/api/v1/scan?query=Wayanad&lat=11.548&lon=76.134&rain_delta_mm=180'
```

The model is loaded from `artifacts/global_multi_causal_xgb.pkl` if available; missing model/provider states are disclosed. Only load trusted pickle artifacts. The optional agent connects to local Ollama; deterministic tool execution and briefing generation remain available without it. CAP export produces a private **Test** message.

## Source architecture

`backend/realtime` fetches providers concurrently, bounds request times and caches responses. `backend/physics` computes local cell states and movement. `backend/ml` constructs source-linked examples and trains/evaluates the model. `backend/api` serializes strict JSON; `backend/llm` orchestrates tools and verifies the briefing.

| Input | Implemented source | Interpretation |
|---|---|---|
| Terrain | AWS Terrarium tiles at zoom 13 | Latitude-dependent grid spacing; not verified uniform 30 m NASADEM/SRTM. H3 resolution 9 cells aggregate samples. |
| Weather | Open-Meteo forecast/archive APIs | Default best-match model selection; ECMWF is not guaranteed. Scenario rain is added to the fetched baseline. |
| Hydrology | NASA POWER `GWETROOT`, `PRECTOTCORR` | Coarse source estimates with dates/status; not cell soil sensors. |
| Historical events | NASA COOLR point/polygon catalog and cited report records | Mixed positional precision, reporting coverage and event footprints. |
| Natural events | NASA EONET | Event metadata; no event does not prove absence of a hazard. |
| Earthquakes | USGS FDSN | Catalog measurements plus a marked attenuation proxy for PGA, not recorded site shaking. |
| Soil | ISRIC SoilGrids if decoding succeeds | Otherwise explicit pedotransfer/topographic estimates. |
| Vegetation/burn context | Esri imagery RGB and EONET context | Optical proxies, not calibrated multispectral NDVI or field burn severity. |
| Exposure | OSM Overpass | Incomplete mapped roads/buildings; missing exposure is marked as a proxy. DEM TWI is a channel proxy, not household evidence. |

Every scan, including an empty terrain response, exposes `location.nasa_telemetry` and `location.reliability_report`. NASA telemetry contains POWER values, EONET event records, COOLR count, local cached 100 km count and nearest cached event distance. Terrain provenance is explicitly `srtm_terrarium_elevation_range_m`, not NASADEM. No NASA logo or endorsement is used.

The live COOLR count uses `where=1=1`, an envelope of ±0.45 degrees, WGS84 and `esriSpatialRelIntersects`. This envelope is approximately regional: it is **not an exact 50 km circle**, especially at high latitude. If the live query fails, the count is the deduplicated partial local catalog within 50 km, with an explicit fallback marker. A diagnostic on 2026-10-31 found 40,154 event-point records globally but zero in the Wayanad envelope; the separate report-point layer returned 32 there. See [the retained NASA query responses](artifacts/nasa_count_probe.json). These are different layers and their counts are not substituted. The independent 100 km count uses Haversine distance to cached report/cell coordinates; buffer locations remain approximate. Legitimate zero counts remain zero and missing nearest distance remains null. Query parameters, response count, source URL and fetch time are retained when available.

Reliability identifies source availability and proxy use. It currently remains `TIER_2_PARTIAL_PROXY`, even if all feeds respond, because material strength, road-cut geometry, alpine snowpack and runout parameters include assumptions. Provider availability cannot certify operational confidence.

## Physics and its limits

The v5 engine implements M1-shaped post-fire likelihood, Gartner-shaped conditional volume, unsaturated suction stress, bounded basal perching, separate FOSM/seismic/fire probability components and an adaptive physical blend. The original mechanical FS remains available separately from the requested probability-derived post-fire yield index. Shared training/inference features now use equivalent water input, rain plus 1.35 times melt.

[Equation provenance and domains](docs/physics_sources.md) distinguishes published equations from the requested input proxies and scenario factors. In particular, the ice reduction factors are not Mamot's published strength criterion, cell area/relief are not measured burned watersheds, and the perching approximation is not a TRIGRS solver. An exact domain-checked Mamot helper is included for verified joint inputs; live soil cells lack those inputs.

Terrain uses local slopes and local FS without neighborhood overwriting. Saxton–Rawls organic matter coefficients now receive percent, converted from the input fraction. Curvature, soil strength, root weakening, alpine snowpack and toe-cut geometry still include scenario assumptions. Rapid movement requires dynamic yield ≤1; FS >1.25 resolves to zero motion. Runout remains an approximate downslope energy graph.

## ML, aggregation and validation

Fourteen shared features include local FS, the FOSM component, pressure head, Newmark displacement, local slope, TWI, saturation, equivalent water, melt, PGA, thaw, burn severity, toe weakening and curvature. `slope_max_deg` is a legacy feature name: it now stores the local cell slope. Monotonic constraints are negative for FS, positive for trigger features and unconstrained for curvature. Isotonic calibration uses spatially grouped held-out folds.

The live score is

$$G=\sigma((\theta-8.5)/0.65)\,\sigma((2.15-FS)/0.18),\qquad w=.35+.50\sigma((1.05-FS)/.06),\qquad P=G[(1-w)P_{XGB}+wP_{phys}].$$

This gate is a requested research design choice; there is no established universal rule that all landslides or debris-flow impacts are impossible below 8°. The final gated blend is not itself calibrated. The gate alone does not mathematically guarantee Delhi's <0.1% target: FS and the model output also matter.

Critical P05 FS and P95 score summarize cells with effective slope ≥24°, falling back to all cells if none qualify. Whole-viewport percentiles are separate. Hotspot ranking weights probability by exposure and bounded Newmark/velocity intensity. Thus critical percentiles, top-hotspot values and viewport extrema are different statistics. Live causal percentages normalize stress proxies; they are heuristic attribution, not causal inference; TreeSHAP explains raw booster margins, not the calibrated/gated output.

The observation dataset restores 125 dated, cited trigger-unclassified COOLR positives. All five known causal regimes are present, with severe imbalance: few fire and toe-cut cases. Positive polygons/point buffers/report locations vary in precision; 60-day pre-event controls are **inventory-absence pseudo-negatives**, not surveyed intact slopes. Forty simulated controls (25 high-rain flat, 15 dry steep) receive weight 0.25 in booster fits and never enter isotonic fitting or CV scoring.

The frozen `data/event_holdout_manifest.json` selects eight source events per rain, seismic and cryospheric regime, one positive and one matched control each: **48 rows from 24 groups**. Whole event groups and their coarse spatial blocks are excluded from training; the former Brazil holdout remains reserved. The balanced sample changes the benchmark population, so its scores cannot be compared directly with the old Brazil-only result. Selection is fixed without inspecting model scores. Repeated development against any holdout limits its independence.

Stage 1 reports pooled raw constrained-XGBoost out-of-fold predictions in spatial CV (earlier reports averaged fold AUCs, so the summaries differ). Stage 2 evaluates **ungated isotonic event-buffer scores**, a different quantity from the live gated score. Both Recall at the top 10% and Recall@40 are reported with their actual K; Recall@40 covers most of a 48-row holdout and should not be overinterpreted. 95% percentile bootstrap intervals resample whole spatial blocks for Stage 1 and whole matched event pairs for Stage 2, conditional on fixed predictions; they do not include model refitting or label uncertainty. Stage 3 checks 80 monotonic probes and synthetic counterfactuals. Stage 4 tests four movement branches. These tests do not prove generalization, calibrated rare-event probabilities or five-regime forecasting performance.

## Data rebuild and Mac baseline

Large Parquet, JSONL, raw API snapshots and model binaries are intentionally ignored by Git. Lightweight split manifests, metadata and validation reports are tracked. Raw report-text catalogs and third-party mapped geometries remain local pending source-specific redistribution review. A code-only clone can run offline regression tests and fetch new observations, but cannot exactly reproduce the cached baseline without the original source snapshot. Live services and their data can change.

```bash
# First acquisition in a code-only checkout:
python backend/ml/build_real_labeled_dataset.py --output data/real_global_landslide_observations.parquet --max-events 450
# Refresh the retained observation/source snapshot with the audited local physics:
python backend/ml/build_real_labeled_dataset.py --audit-refresh
python scripts/run_mac_baseline_and_verify.py --skip-dataset-build --provider-timeout 12
python scripts/check_training_artifacts.py
python scripts/prepare_public_release.py
python scripts/sync_windows_trainer.py
python scripts/full_system_bug_audit.py
python scripts/randomized_global_blind_audit.py
python scripts/literature_boundary_audit.py
python scripts/finalize_readiness.py
python scripts/sync_windows_trainer.py
```

The full and randomized audits return nonzero for failed engineering invariants. The boundary audit also returns nonzero for unmet mandatory targets. `finalize_readiness.py` reports engineering readiness separately and grants pretraining readiness only when all three fingerprints and the latest mandatory targets pass. `--offline` cannot grant readiness. Live services can time out; missing terrain is a failed/incomplete check, never a fabricated flat control.

The randomized suite records its two seeds and selections before acquisition. Each sweep selects two locations per archetype and adds all five named plain controls, for 15 rows per sweep. Baseline and perturbation use the same frozen live telemetry. Raw snapshots are retained locally, and source availability is recorded per row. The [audit protocol](docs/randomized_audit_protocol.md) records the causal-activation test correction and replay procedure. A random draw from a named pool is a scenario stress test, not independent labeled global validation. The agent audit exercises the deterministic fallback and numeric guard; it does not certify an untrained Qwen model.

The original project was preserved before cleanup in `../Terra_Full_Archive_Backup`; `artifacts/archive_verification.json` records SHA-256 hashes for all 108 files. That sibling backup is local and excluded from publication. The full audit verifies it when run in this stabilization workspace; a fresh clone needs its own archival provenance and cannot pretend to have this backup.

## Windows RTX 5070 Ti

After reviewing the current audit, copy the **entire local** `windows_5070ti_trainer` folder, including ignored local datasets, to the PC. A Git clone alone does not include those files. `scripts/sync_windows_trainer.py` mirrors shared physics/ML/ingestion source, datasets, metadata and reports; the audit checks source parity.

`RUN_TRAINING_5070TI.bat` creates an environment, installs dependencies, checks CUDA and runs Optuna/XGBoost plus the requested QLoRA stage. Its PyTorch wheel index is configured for CUDA 13.0. Native Windows driver, CUDA, Unsloth and GGUF conversion compatibility must be verified on that PC; they have not been executed on the Mac. To run XGBoost alone, use `python train_all_on_5070ti.py --skip-dataset-build` in the bundle environment.

Qwen training requires 600 distinct source-linked cell/date trajectories, not 600 independent landslides. The export preserves adapter/tokenizer first, attempts merged weights and then Q4_K_M GGUF. Conversion failure retains the adapter and records the error. `IMPORT_ARTIFACTS_MAC.sh` imports the trusted returned artifacts. GPU retraining clears readiness and needs fresh evaluation; a copied baseline audit does not certify the new weights.

## Briefing guard and release checks

The displayed narrative is generated from a deterministic telemetry-bound template. Every formatted numeric token must match a JSON numeric value at its displayed precision; unsupported numeric prose and arbitrary road names are replaced by the approved template. If even the template fails verification, numeric prose is withheld. `briefing_guard` records the decision. This prevents model-added claims in this output channel, but cannot make erroneous upstream data or equations true.

Run `python -m unittest discover -s tests -v` for network-independent tests. The GitHub Actions workflow runs those tests on Python 3.11/3.12; remote CI has not yet run. Provider and artifact integration checks are separate. `scripts/prepare_public_release.py` removes generated caches and workstation paths while `.gitignore` keeps large/private runtime artifacts out of a public source release. No remote repository is created or pushed by these scripts.

## v5 added outputs and limitations

Cells expose `conformal_interval_90`, `uncertainty_width`, `bivariate_class`, `cryospheric_eligible`, `effective_burn_severity` and `lateral_resistance_kpa`. Conformal intervals target binary labels under exchangeable-block assumptions, **not** confidence intervals for true failure probability. The four bivariate classes use explicit project display thresholds.

Top-hotspot summaries expose `cascading_dbi` and `impoundment_status`. Channel intersections show a Larsen full-cell failure-volume scenario and viewport accumulation, but DBI is null unless real dam height, deposited volume and full catchment geometry are supplied. Terrain does not establish those quantities. The Milledge citation supplied in the directive was incorrect; the implemented finite-block term is an adaptation, while live thaw and post-fire yield factors remain scenario proxies. Read the source/domain distinctions before treating this as literature-locked physics.

Publication copies of the audit JSON summarize embedded raw EONET event arrays with record counts and SHA-256 hashes. Metrics, checks and readiness flags are unchanged. Original full audit JSON remains in the ignored local `archive_snapshots/pre_github_publication/` folder.
