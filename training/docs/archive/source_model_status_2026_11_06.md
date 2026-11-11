> Historical record. Superseded by the [current status](../current_status.md); statements below are not the active live contract.

# Active source and fusion models — 10 November 2026

The current model is the four-way corrected-terrain experiment documented in [phase1_phase2_implementation.md](../phase1_phase2_implementation.md). The standalone six-feature inventory classifier below is archived. It does not control live initiation ranking.

# Rainfall source model and application status

Evaluation: 6 November 2026. Feature contract: `rainfall_source_windows_v2`.

**Monaco correction:** live scoring now requires all six source features and abstains if raw moisture is missing. The fold-local imputation described below remains part of the retained research training procedure. The 14-feature assumption-based physics/ML path is excluded from live use. Current FS requires explicit sourced ground inputs, and classifier/physics comparisons join the same H3 cell. See [monaco_input_audit.md](../monaco_input_audit.md) and the [current API contract](../../API_USAGE.md). Existing research metrics and model weights have not been retuned to change Monaco's score.

The working CLI and API have a rainfall statistical path independent of the physical column's ground-data requirements. Live location analysis remains the default. Saved provider snapshots are an explicit reproducibility option. Multi-mechanism reference equations and the 14-feature XGBoost baseline remain available for archived research comparisons.

## Scope of the four changes

| Item | Change | Remaining limitation |
|---|---|---|
| 1: incompatible inputs | Show raw volumetric moisture, estimated porosity and physical saturation separately; retain terrain and weather without physics; align source weather definitions. | Statistical availability does not resolve conflicting physical measurements and priors. |
| 2: ML availability | Fit a six-feature rainfall model without requiring FS, strength, saturation or runout. Optional surface moisture is imputed inside each training fold. | Rain and terrain remain required. Other hazards are outside this statistical model's scope. |
| 7: baseline comparison | Compare prevalence, rainfall-only logistic, source logistic and shallow XGBoost on identical event/spatial folds. Preserve holdout and calibration reservations. | Labels and controls remain imperfect; this is retrospective development evaluation. |
| 10: accurate outputs | Show evidence, unavailable reasons, coherent cell comparisons, source status and experimental scores. Add live refresh, original-state reset and JSON export. | No operational alert threshold, validated motion estimate or failure-time prediction is established. |

## Input contract

The model uses latest-24h liquid rain, the preceding nonoverlapping 144h liquid rain, latest-24h mean raw 0–7 cm volumetric moisture, slope, TWI and profile curvature. Moisture is never divided by estimated porosity for this model. Missing moisture stays missing, with median imputation and a missingness indicator learned inside each training partition. Required rain and terrain are never imputed.

Both modes use one extractor and exactly 168 contiguous hourly timestamps. Open-Meteo precipitation timestamps are preceding-hour sums: selected stamps are `(cutoff−168h, cutoff]`. Live liquid rain is `rain + showers`; historical archive `rain` already includes showers. Missing live showers cannot silently become zero. Definitions follow the [forecast documentation](https://open-meteo.com/en/docs) and [archive documentation](https://open-meteo.com/en/docs/historical-weather-api).

Live cutoff is the current complete UTC hour. Retained archives end at 23:00 UTC on the label date, so that is the training cutoff; the missing next-midnight hour is not fabricated. These features can include weather after the landslide. This evaluates retrospective event-date classification, not advance forecasting. Common extraction does not eliminate reanalysis-to-live-model distribution shift.

The labeled H3 cell, or nearest retained cell when absent, supplies terrain. No steepest-cell or outcome-based selection is used. Raw payload hashes accompany every derived row.

## Coverage and comparison

Across all 1,611 retained source rows, 1,072 moisture values exceed estimated porosity, 74 lack moisture and 465 are compatible. No density tuning or saturation clipping repairs them.

The rainfall subset has 885 labeled rows. Full event/spatial connected components are formed before filtering. Holdout, legacy-holdout and calibration components exclude 612 rows from this training experiment. The model trains on **257 rows: 164 positive, 93 negative, 14 grouping components**. Of these, 148 have usable physics and **109 additional rows** are usable through raw source features. All 16 original rainfall holdout rows remain reserved. There are no source-extraction omissions in these selected partitions.

Five outer component folds evaluate each candidate. Three inner component folds fit sigmoid calibration, with preprocessing inside each fitted pipeline. Event-group and region-block overlap is zero in every outer fold. Component identifiers are dependence proxies, not proof of complete meteorological independence.

| Same 257-row component CV | ROC-AUC | Average precision | Brier ↓ |
|---|---:|---:|---:|
| Outer-training prevalence only | 0.3455 | 0.5685 | 0.2440 |
| Rainfall-only logistic | **0.8293** | **0.9302** | **0.1633** |
| Source logistic, predeclared application model | 0.7768 | 0.8770 | 0.1918 |
| Shallow source XGBoost | 0.7958 | 0.8987 | 0.1829 |

The application model improves on the constant-prevalence baseline, but the simpler rainfall-only model performs better. Extra terrain/moisture features have not established an improvement. The application model was declared before scoring; no holdout winner was selected. A prevalence-only model can have pooled AUC below 0.5 because its constant differs between folds with different class prevalence.

On the common 148-row physics-available subset, source logistic has AUC 0.7061 and Brier 0.2255; the FOSM diagnostic has AUC 0.2624 and Brier 0.6555. All common-subset comparators are in the report. This is a different cohort from the original 258-row multi-regime v14 evaluation; its AUC is not a like-for-like improvement over v14's 0.5848.

The predeclared model's conditional component-bootstrap AUC interval is **0.6052–0.9124**. This resamples fixed predictions and excludes model-retraining uncertainty. The repeatedly inspected 16-row rainfall holdout gives AUC 0.8438 and Brier 0.1495, with only eight negatives. It remains development evidence.

## What still needs better evidence

Reported points and mapped footprints are not necessarily initiation crowns. Negative dates indicate inventory absence, not surveyed stable slopes. Their external weather selection can make rainfall discrimination easier. Sample prevalence is 63.8%, so scores are not annual, regional or personal landslide probabilities. Prospective forecasting requires independently collected event timing, representative controls, forecast-as-issued inputs and an untouched regional/time partition. Physical modeling additionally needs supported strength, depth and water-state measurements.

## Reproduction and interface

```sh
python scripts/train_rainfall_baseline.py
python scripts/build_replay_snapshot.py
python scripts/check_training_artifacts.py
python scripts/finalize_readiness.py
python scripts/sync_training_bundle.py
python scripts/terra_cli.py "Wayanad, India" --interactive
python scripts/terra_cli.py --lat 11.548 --lon 76.134 --json > scan.json
```

Training and verification use retained local data without network access. Generated data and weights remain local and are included in portable training bundles. `rainfall_baseline_report.json` records row/fold membership, out-of-fold predictions, metrics, environment versions and source/model hashes. The original model and SFT corpus remain separate. `--snapshot data/replays/chooralmala_2024.json` explicitly replays historical weather and later retained terrain, with provenance and no network access. `--output` exports an API response, not a reusable raw-provider snapshot.

CPU engineering checks, live provider availability and predictive validity are separate results. See [current_status.md](../current_status.md) and [training_workflow.md](../training_workflow.md).
