# TERRA-SHIELD

Global physics-informed landslide hazard intelligence.

TERRA-SHIELD combines live environmental inputs, terrain, conditional slope mechanics, uncertainty, machine learning and downstream movement estimates. **The competitive target is NASA LHASA 2.0**, across multiple continents, climates and triggering regimes. Regional studies are validation units within that global program.

The browser application is a live-first 3D global analysis interface. It opens satellite terrain, accepts a map click, map-center scan, place name or coordinates, and requests automatic current and 72-hour forecast assessments. Manual weather sliders and replay-first city presets are absent from the product workflow. [The frontend specification](docs/FRONTEND_SPEC.md) defines that contract.

Global predictive skill, sustained global refresh and operational warning reliability require independent evidence. Current development results establish neither LHASA parity nor calibrated global failure probabilities.

[Current evidence](docs/current_status.md) · [Feature fusion and frontend](docs/feature_fusion_phase3.md) · [Source/API critique](docs/global_source_audit.md) · [Methods](docs/physics_sources.md) · [API](API_USAGE.md) · [Global roadmap](ROADMAP.md)

## Run the platform

Python 3.11 or 3.12 and Node.js 22 are recommended.

```sh
git clone https://github.com/pookie421/Terra.git
cd Terra
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`. Open its printed local URL. The full-screen satellite/terrain canvas uses an obsidian HUD with source telemetry, automatic forecast horizons, model comparisons, cell mechanics, conditional runout and a state-grounded Qwen co-pilot. Click an unscanned coordinate or use **Scan Current Map Center**; a place name is optional. Clicking a returned hexagon inspects that exact cell.

```sh
curl -X POST http://127.0.0.1:8000/api/v1/scan \
  -H 'Content-Type: application/json' \
  -d '{"latitude":11.548,"longitude":76.134}'
```

The interactive API schema is at `http://127.0.0.1:8000/docs`. The CLI also supports live coordinate entry:

```sh
python scripts/terra_cli.py --lat 11.548 --lon 76.134 --non-interactive
python scripts/terra_cli.py "42.26 S, 173.62 E" --details --plain
```

**Learned weights and the retained training corpus are local assets.** Learned scores require `artifacts/physics_informed_models.pkl`; a source checkout does not contain a globally validated model. The physical reference column and conditional routing can operate without learned weights or boreholes, with provenance and uncertainty. Missing weights and provider failures are disclosed. Prepare the local artifacts using the [training workflow](docs/training_workflow.md).

## Analysis contract

- Metric terrain derivatives, H3 connectivity and a bounded specific-catchment-area wetness index.
- Full `pedotransfer_reference_column_v1` columns with optional site overrides, uncapped FS, `[FS_low, FS_mid, FS_high]`, joint FOSM variance, Jacobian, covariance and field provenance.
- Antecedent water balance and transient infiltration driven by source forcing. The product's forecast views use incoming numerical weather/ensemble forcing rather than invented user rainfall.
- Feature-based `physics_informed_initiation_v2_features` fusion, with ordinary fold-prevalence initialization. Learned trigger scores and conditional mechanical coupling remain separately inspectable.
- Three conditional Voellmy rheology/depth trajectories, corridors, boundary truncation and mapped asset intersections.
- Scan/scenario identities bind map layers, selected-cell evidence, forecast states and numeric co-pilot claims. Local Qwen/Ollama output distinguishes verified prose, verified templates and explicit service/verification errors.

The current physical column describes a shallow-regolith reference. Seismic, snow/thaw, post-fire and anthropogenic/fluvial research modules remain part of the global architecture; their applicability and data requirements are separate. A regolith column does not establish rock-joint, deep-slide, liquefaction or bank-erosion behavior. Provider and terrain-domain gaps stay visible, including Web Mercator latitude limits and the provisional low-elevation land mask.

## Where the numbers come from

| Channel | Current evidence | Interpretation |
|---|---|---|
| Elevation | AWS Terrarium compiled DEM | Terrain derivatives are project calculations; native accuracy varies |
| Weather and forecast | Open-Meteo numerical products/ensembles | Model estimates with issue-time, aggregation and latency constraints |
| Wetness context | NASA POWER | Coarse modeled relative wetness, not observed slip-plane head |
| Soil | SoilGrids WCS or disclosed fallback | Predicted soil properties or a project topographic prior |
| Strength, depth, initial head | Reference priors or compatible site records | Assumptions/model states unless independently measured |
| Events | USGS, COOLR, EONET | Catalogs/inventories with differing completeness, geometry and timing |
| Geology and rivers | Macrostrat and GloFAS | Mapped bedrock and modeled discharge, not measured toe strength |
| Imagery and assets | Esri RGB and OSM | Visible-band proxies and incomplete mapped exposure |

Authentic API values are not automatically local measurements. Fine H3 rendering does not improve coarse input resolution. Published equations do not validate project transfer coefficients. The [source audit](docs/global_source_audit.md) records these distinctions and the required evidence for changing inputs. Attribution implies no institutional endorsement.

## Measured development results

Benchmark A preserves **186 matched observations, ten event/spatial components, five outer folds and the original sixteen-row development holdout**. All 1,611 retained source observations have reference columns. Seventy-one unresolved footprint positives remain outside initiation fitting without invented crowns or stable labels.

| Same-cohort component CV | ROC-AUC | Brier ↓ |
|---|---:|---:|
| Rainfall-Only | 0.8952 | 0.0864 |
| Physics-Only (FOSM) | 0.5931 | 0.4413 |
| Weather+Terrain ML | 0.9043 | 0.0911 |
| Feature fusion: learned trigger score | 0.8940 | 0.0965 |
| Physics-Informed ML: conditional mechanical coupling | 0.9026 | 0.1183 |

The old fixed-FOSM-offset result, AUC 0.6662/Brier 0.3451, is superseded. Its optimization/design problem is not evidence against physics-informed ML as a class. Removing the offset improves temporal discrimination substantially. Mechanical coupling still worsens Brier relative to the learned trigger score and rainfall-only.

Benchmark B evaluates a different, multi-regime report/extent-versus-absence proxy task on 510 eligible rows in 34 components. Learned fusion reaches AUC/Brier **0.7767/0.1847**, conditional fusion **0.6866/0.2451**, and rainfall-only **0.7842/0.1772**. These results do not establish global superiority. Unlabeled storm-grid diagnostics remain separate from observed non-failure classification. See [the complete interpretation](docs/feature_fusion_phase3.md).

Quiet-weather controls, uncertain initiation locations, post-failure event-date forcing, assumed parameters and an inspected holdout limit these metrics. A fair LHASA comparison still requires shared tasks, spatial/temporal support, inputs available at issue time, representative backgrounds and explicit coverage.

## Verification and portable training

```sh
python -m unittest discover -s tests -v
python -m compileall -q backend scripts training
# Requires the retained local corpus, weights and reports:
python scripts/check_training_artifacts.py
python scripts/finalize_readiness.py
# Frontend:
cd frontend
npm run build
npx tsc --noEmit
```

Source-only CI reports two explicit skips for learned-artifact regressions. Full local verification checks feature extraction, source/model hashes, component isolation, reported metrics and 594 instruction traces; it is separate from live provider availability and predictive validation.

`windows_5070ti_trainer/` remains a full local training bundle. **`run_training.bat --qlora` runs CUDA checks, XGBoost, the global proxy benchmark, instruction regeneration and Qwen2.5 QLoRA with an 8192-token context.** Native Windows/RTX 5070 Ti and actual QLoRA execution still require target-machine verification. [Training instructions](docs/training_workflow.md) cover commands and artifact expectations.

Root source is canonical. Synchronize portable copies with `python scripts/sync_training_bundle.py`. Raw corpora, provider archives and learned weights remain outside Git; prepared local bundles carry source-reviewed assets separately. See [contributing](CONTRIBUTING.md), [third-party terms](THIRD_PARTY_NOTICES.md) and the [MIT license](LICENSE).
