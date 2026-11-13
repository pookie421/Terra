# Research history and retired experiments

This page preserves the important failures and design changes without presenting old contracts as current setup instructions. Use [current status](current_status.md), [physical methods](physics_sources.md), [feature-fusion results](feature_fusion_phase3.md) and the [source audit](global_source_audit.md) for the active implementation.

The complete original records remain available at the immutable commit `0a5912bffb22c9f157a563db6aef5a4a04cad165`. Their model versions, cohorts and assumptions differ; their metrics must not be combined or presented as a continuous improvement on one untouched test set.

## Earlier reference baseline and source classifier

The November 2026 `reference_columns_v14` baseline retained 1,611 source observations, but only 465 had compatible complete inputs for its physical wrapper. Its multi-regime fitting cohort had 258 rows in 27 event/spatial components; a separate 51-row reserve supported descriptive conformal residual intervals. The original holdout retained 48 IDs, of which only 28 were scorable. Excluding the other records created a coverage limitation, not evidence of stability.

| Historical v14 component CV | ROC-AUC | Brier |
|---|---:|---:|
| Calibrated ML | 0.5848 | 0.2619 |
| Physical FOSM diagnostic | 0.3519 | 0.5931 |
| Outer-training prevalence baseline | 0.3878 | 0.2482 |

The prevalence baseline had better Brier than ML. The ML AUC's conditional component-bootstrap interval was broad, 0.2732–0.8071. The inspected, scorable holdout subset was development evidence, not independent validation.

A later six-feature rainfall-source experiment used raw rainfall/moisture/terrain rather than requiring a complete mechanical column. Its distinct cohort had 257 rows in 14 components and retained the original sixteen-row rainfall holdout.

| Historical 257-row source-model CV | ROC-AUC | Brier |
|---|---:|---:|
| Rainfall-only logistic | 0.8293 | 0.1633 |
| Predeclared source logistic | 0.7768 | 0.1918 |
| Shallow source XGBoost | 0.7958 | 0.1829 |

Extra source features did not establish an improvement over rainfall-only. Event-day forcing could follow failure, and inventory-absence controls were not surveyed stable slopes. These scores were never annual or globally calibrated failure probabilities.

Full records: [v14 status and coverage](https://github.com/pookie421/Terra/blob/0a5912bffb22c9f157a563db6aef5a4a04cad165/docs/archive/current_status_2026_11_06.md), [earlier physical methods](https://github.com/pookie421/Terra/blob/0a5912bffb22c9f157a563db6aef5a4a04cad165/docs/archive/physics_sources_2026_11_06.md), and [six-feature source experiment](https://github.com/pookie421/Terra/blob/0a5912bffb22c9f157a563db6aef5a4a04cad165/docs/archive/source_model_status_2026_11_06.md).

## Monaco defect and the retired measurement-only response

The original classifier assigned **0.8215765** to H3 cell `893969a6a5bffff`, with a **3.6982° slope** and **TWI 18.1711**, under the retained low-rainfall inputs. The old steep cliff cell `893969a401bffff` had a **41.1147° slope** and **FS 1.3816947**. A separately displayed “physics-selected” cell, `893969a4607ffff`, had a 10.9195° slope and FS 3.7814181; selection actually came from a legacy ML/exposure ranking.

These were different quantities at different cells. The important defects were not the appearance of valid H3 IDs or an FS above three. They included:

- Assumed material properties and regolith depth presented too much like supported local values.
- An unobserved initial pressure head defaulting to zero.
- Misleading hotspot selection and comparisons between different cells.
- FS clipping at five that zeroed uncertainty derivatives, plus tiny values rounded to apparent zeros.
- Missing-data placeholders, equal causal shares and warning metadata without corresponding evidence.

The first response required in-situ ground inputs and left physics unavailable without them. That measurement-only live contract is **retired**. Current scans retain full conditional columns, uncertainty and routing through explicitly labeled reference priors; measured site inputs can replace compatible fields. Assumptions are not relabeled observations.

Subsequent corrections rebuilt metric terrain from raw tiles, corrected geographic spacing and drainage support, removed mechanical slope/display caps, and exposed storage, covariance, applicability and source provenance. These were general input/mathematical changes rather than Monaco-specific score gates. The current retained-input storm diagnostics still reach approximately 0.03 on the flat cell and 0.80 on the cliff under high added rainfall; desired showcase targets were not fitted into the model.

Full record: [original Monaco investigation and superseded response](https://github.com/pookie421/Terra/blob/0a5912bffb22c9f157a563db6aef5a4a04cad165/docs/monaco_input_audit.md).

## Fixed-offset v1 fusion

The corrected-terrain initiation experiment separated 71 unresolved failure-footprint positives from supervised initiation fitting. It used 93 manual-point proxies and 93 matched temporal-absence controls, ten components, five shared outer folds and the same sixteen-row rainfall development holdout. Neither the points nor the control dates established surveyed initiation crowns or stable terrain.

| Historical fixed-offset v1 CV | ROC-AUC | Brier |
|---|---:|---:|
| Rainfall-Only | 0.8952 | 0.0864 |
| Physics-Only (FOSM) | 0.5931 | 0.4413 |
| Weather+Terrain ML | 0.8880 | 0.0967 |
| Physics-Informed ML | 0.6662 | 0.3451 |

The fusion model learned corrections to a FOSM log-odds base margin. Near-zero physical priors impaired tree optimization; this failed architecture is not a scientific refutation of physics-informed ML. The active v2 model uses ordinary fold-prevalence initialization and physical/source interaction features. Its [same-cohort comparison](feature_fusion_phase3.md) reports learned and conditional mechanical scores separately. Conditional coupling still worsens Brier, and the broader proxy benchmark still favors rainfall-only.

The terrain and physical reference contracts established during this stage remain active: metric Mercator spacing on both axes, bounded drainage support, antecedent water balance, uncapped FS, a joint covariance envelope and conditional Voellmy routing. Published equation families do not validate project strength/depth/covariance coefficients. The [current methods](physics_sources.md) describe those assumptions and applicability limits.

Full record: [Phase 1/2 implementation and v1 experiment](https://github.com/pookie421/Terra/blob/0a5912bffb22c9f157a563db6aef5a4a04cad165/docs/phase1_phase2_implementation.md).

## Retired model-conditioned controls

The v13 pair diagnostic selected replacement negative dates through a modeled quiescence screen. This changed the evaluation population using the model under assessment. It is not the active control-selection method and cannot support a claimed predictive improvement. Original source dates, labels, failed cases and holdout reservations remain preserved; current fitting does not search control dates to satisfy physical scores.

Full record: [retired holdout-pair diagnostic](https://github.com/pookie421/Terra/blob/0a5912bffb22c9f157a563db6aef5a4a04cad165/docs/holdout_pair_diagnostics.md).

## Retired local-LLM training and engineering protocols

The Windows RTX 5070 Ti / Qwen2.5 QLoRA plan was retired as the deployment direction on 14 November 2026 in favor of operator-configured hosted assistant APIs. Canonical training, export and local-serving code remains available for research installations. Its presence and CPU protocol tests do not establish actual Windows/CUDA execution, measured target VRAM, successful adapter export or language quality.

The [original combined training workflow](legacy/local_llm_training.md) and [hardware guide](legacy/windows_5070ti.md) preserve those implementation details. The [retired randomized audit protocol](legacy/randomized_audit_protocol.md) retains its old named-location scenario criteria, including arbitrary score thresholds; those are historical engineering checks, not current scientific gates or training targets. The [active CPU hazard-model workflow](training_workflow.md) and [assistant connection guide](assistant_connections.md) describe the current paths. No model weights, source labels, cohort memberships or scientific results were changed by this documentation reorganization.

## What these records do not establish

Successful equation tests, provider availability, a browser demonstration, local CPU fitting or instruction-trace generation are separate engineering results. None establishes untouched global predictive skill, calibrated probabilities, forecast lead time, runout accuracy, actual Windows/CUDA/QLoRA execution or operational warning readiness. The target remains a global LHASA 2.0 comparison using shared issue-time inputs, independent events/regions/time, representative backgrounds and explicit missingness.
