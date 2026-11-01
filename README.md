# TERRA-SHIELD

**Explore a slope. Understand its triggers. Inspect the evidence.**

TERRA-SHIELD is a research platform for on-demand, multi-causal landslide analysis. It combines live environmental sources, H3 terrain cells, slope mechanics, constrained XGBoost, provisional movement estimates and a local Qwen co-pilot. The ambition is a global terrain explorer that helps researchers and, eventually, meteorological and emergency-management teams ask: **where might a slope fail, why, how could it move, and what changes under a different weather or earthquake scenario?**

Developed toward the NASA Space Apps Challenge vision; independently built, with no NASA endorsement. NASA datasets are attributed through source telemetry, without NASA logos.

**Current stage: pre-Phase-4 research backend and interactive CLI.** A production map frontend, verified RTX 5070 Ti training run and operational warning system are not delivered yet. Scores, causal shares and runout paths are research outputs—not validated forecasts, evacuation advice or engineering design inputs.

[Vision and roadmap](ROADMAP.md) · [Current evidence](docs/current_status.md) · [API guide](API_USAGE.md) · [Physics sources and assumptions](docs/physics_sources.md) · [Windows trainer](windows_5070ti_trainer/README.md)

## What works today

- **On-demand terrain:** place-name or coordinate scans, Terrarium elevation tiles, H3 resolution-9 cells, slope/curvature and a downslope drainage graph. Source coverage and geocoding availability limit where a scan can succeed.
- **Five trigger regimes:** hydro-meteorological, seismic/tectonic, cryospheric, post-wildfire/biotic, and anthropogenic/fluvial. Their representation in the data is uneven; implementing five mechanisms is not evidence of equal predictive skill across them.
- **Local mechanics and ML:** mechanical Factor of Safety (FS), pore-pressure and suction terms, bounded Newmark displacement, grouped isotonic XGBoost, raw-booster TreeSHAP and a mechanically gated research score.
- **Movement scenarios:** four kinematic branches, Cruden–Varnes velocity classes, provisional downslope routes and mapped-exposure intersections. Speeds, travel times and impact footprints are not independently validated.
- **Live source provenance:** individual `LIVE`, `CACHED`, `FALLBACK` or `PROVIDED` statuses, NASA telemetry, mapped geology and modeled river discharge. Missing terrain returns `UNAVAILABLE`, not invented geometry.
- **Interactive CLI and API:** scan, forecast/stress scenarios, six synthetic sanity checks, six localized metric cards, and a tool-using local Qwen assistant with deterministic fallback.
- **Portable training workflow:** shared source/data synchronization, CUDA XGBoost tuning, QLoRA entry points and adapter-first GGUF export handling. GPU execution remains to be verified on the target PC.

## Latest measured baseline — v6, 2026-11-01

| Evidence | Recorded result | Interpretation |
|---|---:|---|
| Engineering checks | 16/16 pass | Implemented regression contracts |
| Offline tests | 26 pass | Runtime and deterministic regressions |
| Latent-bug / kinematics checks | 8/8 and 4/4 pass | Defined synthetic/API controls |
| Monotonicity | 0 violations / 80 comparisons | Specified feature sweeps |
| Spatial CV ROC-AUC | 0.8546 | Pooled raw-booster out-of-fold scores |
| Original event-holdout ROC-AUC | 0.7396 | Ungated isotonic event-buffer scores |
| Holdout Recall@40 | 0.9167 | Selects 40 of 48 rows; not a rare-event operational metric |
| Randomized snapshot replay | 28 hazard-evaluated rows; 2 fallback-only rows | Recorded provider inputs, not 30 independent labeled validations |
| Successful named CLI samples | Wayanad 4.00 s; Zermatt 4.14 s | Both exceeded the sub-three-second ambition |
| Readiness | Engineering/pretraining `true` | Existing code/data/model fingerprint only |

[Readiness JSON](artifacts/readiness.json), [validation report](artifacts/scientific_validation_report.json), [full audit](artifacts/full_system_bug_audit.md), [randomized audit](artifacts/randomized_global_blind_audit.md) and [boundary audit](artifacts/literature_boundary_audit.md) preserve the evidence. **`all_requested_targets_passed` is false.** Empirical score cutoffs do not control engineering readiness; failed scientific/performance targets remain visible. A fresh clone or newly trained model does not inherit the local baseline's certification.

## Try it locally

Use Python 3.11 or 3.12. The CPU baseline runs on a Mac; XGBoost does not use Apple MPS.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/terra_cli.py
# Or a one-shot live scan:
python scripts/terra_cli.py "Zermatt, Switzerland" --non-interactive
```

The interactive menu provides another scan, forecast/custom stress testing, Qwen questions and the sanity suite. Public APIs require network access but no user API keys in the current integrations. Optional SoilGrids raster decoding needs `requirements-geo.txt`.

**A Git clone contains source, manifests and reports, not the local datasets or trained weights.** Without the trusted XGBoost artifact, the API discloses physical-only scoring. Dataset acquisition and training instructions are below. Ollama/Qwen is optional; the deterministic briefing fallback remains available.

To start the API:

```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
curl 'http://127.0.0.1:8000/api/v1/scan?query=Wayanad&lat=11.548&lon=76.134&rain_delta_mm=180'
```

Browse local `/docs` for the schema. CAP exports are **Test/Private**, not transmitted public warnings.

## Data and architecture

```text
Public terrain/weather/catalog/map providers
                  ↓
     Async ingestion + cache + source status
                  ↓
       Local H3 terrain and five-regime physics
                  ↓
     Shared 14 features → grouped calibrated XGBoost
                  ↓
       Mechanical gate + physical/ML blend
                  ↓
   API / CLI / tool-grounded Qwen / future map UI
```

| Input | Source | What it actually represents |
|---|---|---|
| Terrain | AWS Terrarium tiles, zoom 13 | Latitude-dependent sampling; not verified uniform 30 m NASADEM |
| Weather and soil moisture | Open-Meteo | Model-derived weather/forecasts; availability varies |
| Root wetness and precipitation | NASA POWER | Coarse, latest available daily estimates—not local sensors |
| Historical landslides | NASA COOLR and cited event reports | Mixed footprints, point buffers and report locations |
| Natural-event context | NASA EONET | Dated event records, not a complete local hazard census |
| Earthquakes | USGS catalog | Catalog data with a project attenuation proxy for site PGA |
| Soil | SoilGrids when available | Otherwise disclosed soil/pedotransfer estimates |
| Vegetation and burn context | Esri RGB imagery and EONET | Optical/context proxies, not measured pre-fire NDVI |
| River discharge | Open-Meteo GloFAS | Modeled flow, not a local gauge measurement |
| Bedrock | Macrostrat | Mapped units; strength adjustments are uncalibrated scenario proxies |
| Roads/buildings/waterways | OSM Overpass | Incomplete mapped exposure, not surveyed household occupancy |

The live COOLR count uses an approximately regional ±0.45° envelope. Its fallback is a deduplicated partial local catalog within 50 km. These populations differ and are identified in provenance. Legitimate zero counts stay zero; the separate cached 100 km count and nearest-event distance are not substituted for the live query. See [retained query diagnostics](artifacts/nasa_count_probe.json).

Provider deadlines are 2.2 seconds each. Geocoding, terrain-dependent imagery and local processing can make the complete scan longer. Reliability remains `TIER_2_PARTIAL_PROXY` because successful data retrieval does not remove material, snowpack or runout assumptions.

## Physics and score interpretation

In v6, **`fs_dynamic` is the mechanical Mohr–Coulomb FS**. The unverified lateral-resistance term and probability-derived post-fire FS override were removed. Staley-shaped post-fire likelihood remains separate in the FOSM/seismic/fire probability union. Mamot-based strength retention uses a disclosed soil-to-joint temperature proxy within the implemented temperature domain; median-reference lapse, baseline cryospheric eligibility and assumed alpine snow floors are not observed permafrost. Voellmy steps now use inclined distance and an explicit flow-depth denominator.

For ML probability `m`, physical probability `p`, slope `theta`, and mechanical `FS`, the deployed blend is:

```text
G = sigmoid((theta - 8.5)/0.65) * sigmoid((2.15 - FS)/0.18)
base = 0.65*m + 0.35*p
w = 0.35 + 0.50*sigmoid((1.05 - FS)/0.06)
yield_blend = (1-w)*m + w*p
score = G * (max(base, yield_blend) if FS <= 1.05 else base)
```

The gate and mixture are research design choices. **The final score is not separately calibrated.** Low local initiation susceptibility does not establish immunity from debris arriving from upslope. Causal percentages normalize stress proxies; they are not identified causal effects. TreeSHAP explains booster margins, not the final gated score.

Critical P05 FS/P95 score summarize cells with effective slope ≥24°, or all cells if no cells qualify. Whole-viewport percentiles and the exposure/kinematics-ranked hotspot are separate statistics. The first two metric cards show FS and score; the next four represent distinct mechanisms.

Ensemble-scaled conformal intervals target **binary event labels**, not confidence intervals on true failure probability. Calibration uses 66 cells in 13 reserved spatial blocks, but cell exchangeability is not established. Holdout label coverage was 85.42%, mean width 0.8835, and live examples returned `[0,1]`. Narrower intervals are an objective, not an implemented guarantee.

Unknown dam height, deposited volume and full catchment geometry remain null; channel intersections alone do not establish a real blockage or operational DBI. See [equation provenance and limitations](docs/physics_sources.md) and the [verbatim numbered source dump](artifacts/core_code_audit.md).

## Dataset, training and reproducibility

The CPU training partition contains 534 rows: 304 positives and 230 negatives, including 40 explicitly simulated regularization controls. Those controls have weight 0.25 in booster fitting and are excluded from isotonic fitting and CV scoring. Source-derived negatives are inventory-absence proxies, not surveyed stable slopes. All five causal regimes are represented, but post-fire and toe-cut evidence is sparse.

The frozen original holdout contains 48 rows from 24 matched event groups across rain, seismic and cryospheric regimes. Whole event groups/coarse blocks are embargoed from training; calibration occupies separate blocks. Repeated development against this holdout limits its independence. Stage 1, Stage 2 and live scores measure different quantities and must not be conflated.

```bash
# First acquisition in a source-only checkout:
python backend/ml/build_real_labeled_dataset.py --output data/real_global_landslide_observations.parquet --max-events 450
# Refresh a retained local observation snapshot:
python backend/ml/build_real_labeled_dataset.py --audit-refresh
python scripts/run_mac_baseline_and_verify.py --skip-dataset-build --provider-timeout 12
python scripts/check_training_artifacts.py
python scripts/sync_windows_trainer.py
```

Parquet/JSONL, model binaries, raw provider snapshots and source report text remain local. Their absence from Git is intentional. Live acquisition cannot reproduce an old cached snapshot exactly. Third-party data and base models retain their source terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

On the RTX 5070 Ti, copy the **entire local trainer folder**, including ignored datasets, and follow its [README](windows_5070ti_trainer/README.md). The workflow includes CUDA XGBoost and optional QLoRA on 600 source-linked cell/date trajectories—not 600 independent landslides. Export preserves adapters before attempting merged weights and GGUF. Native Windows/CUDA/Unsloth/converter compatibility remains unverified. Retraining/importing weights clears readiness and requires fresh evaluation.

## Audits and co-pilot boundaries

Run offline regressions with `python -m unittest discover -s tests -v`. GitHub Actions is configured for Python 3.11/3.12; inspect the actual workflow result rather than inferring CI success from the local tests.

The local integration sequence is `full_system_bug_audit.py`, `randomized_global_blind_audit.py`, `literature_boundary_audit.py`, then `finalize_readiness.py` under `scripts/`. Combined engineering readiness requires all three matching implementation/data/model fingerprints. Empirical thresholds are reported separately. The boundary script can exit nonzero for descriptive targets while its engineering checks pass. Missing terrain is explicitly marked as a fallback-runtime check and is not counted as hazard validation.

The full audit also verifies the original sibling archive recorded in `artifacts/archive_verification.json`, excluding mutable Finder `.DS_Store` metadata. A fresh clone does not possess that local backup. Do not regenerate its hashes to manufacture a pass. See [randomized audit protocol](docs/randomized_audit_protocol.md).

Qwen's valid numeric-grounded prose is preserved as `VERIFIED_LLM_PROSE`. Offline fallback is `VERIFIED_TEMPLATE`; rejected numeric prose is replaced or withheld. Percentage forms, rounding, numbered lists and standard labels are handled explicitly. **Numeric parity is not semantic verification** of road names, causal interpretations or recommendations. Real trained-Qwen evaluation is still pending.

## Next milestone

Phase 4 is an interactive map workbench: search, terrain/score/uncertainty layers, cell inspection, scenario comparison, provisional runout overlays and role-aware briefings with visible source quality. Before any real warning use, the project needs independent event/region validation, stronger labels and exposure data, reliable abstention, and expert review. The [roadmap](ROADMAP.md) defines those workstreams and acceptance evidence.

Original code uses the existing [MIT License](LICENSE), including commercial use. No source/model license is replaced by it.
