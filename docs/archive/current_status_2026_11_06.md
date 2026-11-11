> Historical record. Superseded by the [current status](../current_status.md); statements below are not the active live contract.

# Active implementation — 10 November 2026

Phases 1 and 2 now use corrected metric terrain, reference columns with joint uncertainty, conditional Voellmy corridors, state-bound co-pilot and a four-way physics-informed ablation. All 1,611 historical observations receive reference columns. The initiation experiment uses 186 matched observations and the original 16-row rainfall holdout; 71 unresolved footprint positives are separated. Fusion has not outperformed rainfall-only. See [the implementation and measured results](../phase1_phase2_implementation.md).

The earlier status below is retained as archived research and describes superseded live behavior.

# Evaluation and research status

**Latest correction — 6 November, Monaco:** the screenshot's values were reproduced and traced to different selections, assumed strength/depth/initial head, rounded tiny probabilities and legacy clipping. The live physical path now requires sourced ground inputs; absent values stay null. The classifier is labeled as inventory classification and includes coordinates, selection rules and model explanations. See [the detailed investigation](../monaco_input_audit.md). All v14 reference results below describe archived research; they are no longer the live physical inference path.

**6 November application update:** the live CLI/API now include an independently available rainfall source model, retained terrain when physics is unavailable, explicit water-input diagnostics, baseline comparisons and JSON export. See [source_model_status.md](../source_model_status.md) for the six-feature contract, 257-row rainfall evaluation and its limitations. The existing multi-mechanism physics and 14-feature model remain separate. The following reference results are preserved, not replaced by the new rainfall cohort.

Reference evaluation: **5 November 2026**, CPU model `reference_columns_v14`.

The subsequent live CLI integration check fetched **509 Wayanad terrain cells** and returned the independent rainfall assessment using fresh weather. Physical saturation remained unavailable because moisture exceeded the soil prior's porosity. SoilGrids, Overpass and wildfire-feed failures were disclosed; other responding sources included NASA POWER and COOLR. The timestamped `artifacts/live_source_provider_check.json` records this single availability check. It is separate from deterministic verification and is not evidence of predictive skill.

The refactor establishes a testable research baseline. It does not establish a reliable warning system. Event/spatial component CV gives ML ROC-AUC **0.5848** and Brier **0.2619**; the same-fold training-prevalence-only baseline has better Brier **0.2482**. The physical FOSM diagnostic performs poorly against these labels. These results do not justify global probability claims, and no coefficients or control dates were changed to meet desired scenario scores.

## Implemented and verified

SoilGrids SOC units are corrected; partial raw fields are completed before fractions/porosity are derived; fallback soil is static. Incompatible moisture is unavailable instead of clipped to saturation. Mechanics share exactly the ingestion porosity. The explicit column interface accepts material parameters without live-wrapper clipping. Diffusivity is `Ks/C` in m²/s and the Iverson response includes the declared slope/time convention.

The .30 physical-score jump, FS 1.05 blend switch, curvature uplift, scarp weakening, batch TWI normalization, invented snow floors, and inferred ice/root/toe strength are retired. Newmark no longer has yield floors or displacement caps and reports applicability. Separate physics and ML scores replace the gated union; the default prevalence prior is unset. Hotspot fields identify one coherent H3 cell while legacy viewport percentile keys remain available.

The deterministic verification run passes **142 tests**: 6 reference-equation tests, 2 explicitly archived API replays, and 134 other regressions covering the current source/ground-input contract. All run with network connections disabled. Artifact checks independently recompute training features, check source IDs/dates/labels, recheck fold isolation, reproduce reported metrics, and verify model/SFT hashes. The CPU XGBoost train/save/reload smoke passes. These checks establish implementation consistency, not predictive validity.

## Data and isolation

The retained v12 source provides **1,611 nonsynthetic observations**, including 475 positive labels and 745 inventory-absence negative labels. Synthetic/model-conditioned v13 anchors and date replacements are not used. Dates and labels of the retained observations are unchanged. Negative labels are externally sampled temporal inventory absences, not random global backgrounds or field-surveyed stable slopes.

Only **465/1,611 (28.9%)** rows have compatible complete physics inputs; **1,146** retain explicit unavailable-input reasons. Exclusion creates availability selection and materially limits representativeness. A correct equation cannot repair incompatible measurements or coarse report locations.

After the frozen holdout/legacy spatial embargo and conformal reservation, the model fits **258 rows (155 positives, 103 negatives), spanning 27 event/spatial connected components**. A separate **51-row** component reserve supports descriptive conformal residual intervals. The five outer evaluation folds and inner sigmoid-calibration folds keep components intact. Verified overlap in event IDs and spatial blocks is zero. The identifiers are source grouping proxies, not proof that all meteorological event dependence is captured.

There are no synthetic training anchors, no modeled-FS/probability control acceptance, and no assumed 15% field prior. Fixed baseline parameters are declared before scoring. Optional Optuna uses seed 42 inside outer training partitions and was not run for this baseline.

## Measured scores

| Evaluation / score | Scored rows | ROC-AUC | Average Precision | Brier |
|---|---:|---:|---:|---:|
| Component CV: calibrated ML | 258 | 0.5848 | 0.6754 | 0.2619 |
| Component CV: FOSM diagnostic | 258 | 0.3519 | 0.5290 | 0.5931 |
| Component CV: training-prevalence baseline | 258 | 0.3878 | 0.5515 | 0.2482 |
| Scorable original holdout: calibrated ML | 28 | 0.7974 | 0.7564 | 0.2203 |
| Scorable original holdout: FOSM diagnostic | 28 | 0.5692 | 0.5241 | 0.4580 |

CV positive prevalence is **60.08%**; scorable holdout prevalence is **46.43%**. AP and Brier describe those sampled populations. The prevalence baseline uses only each outer training fold's label mean. Bootstrap intervals are conditional on fixed predictions, not retraining uncertainty: ML CV AUC has a broad 95% component-bootstrap interval **0.2732–0.8071**.

The original holdout still contains all **48 IDs/dates/labels**. Only **28** can be scored (13 positive, 15 negative); **20 remain unavailable**. Its ML AUC 0.7974 describes that available subset of repeatedly inspected development data. It is not an untouched external validation or a directly comparable improvement over the old 48-row result. All omitted IDs/reasons are retained in the report.

For calibrated ML, CV TPR at attainable FPR <=1%, <=5%, <=10% is **1.94%, 1.94%, 25.16%** (achieved FPR 0%, 0%, 9.71%). Holdout TPR is **15.38%, 15.38%, 30.77%** (achieved FPR 0%, 0%, 6.67%). With only 15 holdout negatives, one false positive changes FPR by 6.67 percentage points. Even zero false positives has a one-sided binomial 95% upper bound of 18.10%, assuming independent negatives; spatial dependence further limits that interpretation. No ROC interpolation or deployment threshold tuning is used.

## Artifacts and portable training

`artifacts/reference_baseline_report.json` contains metrics, fold/component membership, out-of-fold predictions, environment versions, source/model hashes, and coverage. `artifacts/reference_verification.json` separates equation tests, frozen input replay, and artifact checks. `readiness.json` records CPU engineering verification separately from predictive, operational, Windows CUDA, and QLoRA readiness. Original v13 source/data/models/reports remain archived under `artifacts/refactor_baseline_v13/`.

Both `training/` and `windows_5070ti_trainer/` are generated from one canonical source, with matching source/data/model manifests. Windows dependencies and PyTorch are installed before actual CUDA checks. XGBoost is the default stage; QLoRA is optional and separate. CUDA fallback to CPU fails the requested CUDA smoke. The RTX 5070 Ti, Windows batch execution, and QLoRA have **not** been tested on this Mac CPU host.

The regenerated instruction corpus has **600 distinct source cell/date traces**: **313 modeled and 287 unavailable**. Frozen and legacy XGBoost holdout groups are excluded; other calibration source rows may appear. These are not 600 independent events. Unavailable examples do not invent probabilities, runout velocity, or future failure time.

## Work still needed for predictive claims

No external TRIGRS executable comparison, untouched open-data warning benchmark, live provider availability check, or prospective forecast evaluation was run for this refactor. Published equations are checked in their explicit reference domains. Basin inputs, soil strength/depth, initial groundwater, catchment geometry, and event labels remain major limitations. The live wrapper retains documented pedotransfer and degree-day assumptions; hourly forecasts remain scenario projections.

A useful next scientific experiment is a predeclared regional evaluation with complete event inventories, independent background sampling, temporally aligned moisture/forcing, and an untouched geographical/temporal test partition. Improving data support and comparing against simple baselines should precede more elaborate models or a long QLoRA run. Passing engineering checks must not be represented as operational reliability.

Reproduction commands and target-environment caveats are in [training_workflow.md](../training_workflow.md); equations, units, and remaining approximations are in [physics_sources.md](../physics_sources.md). Archived v13 scenario reports and the old modeled-quiescence pair table are superseded for current readiness.
