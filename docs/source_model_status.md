# Active source and feature-fusion model status

The active model contract is **`physics_informed_initiation_v2_features`**, distributed as the approved small runtime artifact `artifacts/physics_informed_models.pkl`. The physical column remains `pedotransfer_reference_column_v1`. Different version names reflect different components; the model must not be relabeled v1 to match an older document.

The four ablations remain Rainfall-Only, Physics-Only, Weather+Terrain ML and Physics-Informed ML. Learned trigger scores and conditional mechanical coupling are reported separately. [Current metrics](current_status.md) and [the architecture/evaluation report](feature_fusion_phase3.md) describe both.

## Inputs and fitting

Source inputs are latest-24h liquid rain, preceding nonoverlapping 144h rain, raw surface moisture, slope, bounded specific-area TWI and curvature. Mechanical features include the FS envelope, log-FS/FOSM features, shear/friction and cohesion terms, pressure, critical pressure/rain deficits and applicable Newmark displacement. Interaction features include rain-to-critical-deficit ratio and rainfall-induced frictional resistance reduction. Units, missingness and unattainable thresholds remain explicit.

Fitting uses ordinary **fold-prevalence initialization**. There is no per-row `base_margin=logit(p_fosm)`. Monotonic trees learn source/mechanical relationships; grouped inner selection controls regularization/depth and static-prior column-sampling weights. Disjoint component folds calibrate offsets. Tree explanations describe learned log-odds; the exposed mechanical factor is applied separately to the learned trigger score.

The mechanical factor estimates a necessary gravity-over-cohesion condition under declared lognormal moments. It is an assumption-conditioned compatibility calculation, not calibrated event frequency. Its impact on Brier and spatial behavior is measured separately; it is not hidden inside a claimed universal probability law.

## Timing, scope and evidence

The shared source extractor requires 168 contiguous hourly timestamps. Live liquid rain is rain plus showers; archive rain includes showers. Raw moisture retains its support and is not divided by estimated porosity as an ML feature. Missingness belongs inside the declared fitting procedure. Terrain must match the corrected metric version.

Historical event-date forcing can follow failure. Common extraction does not eliminate reanalysis-to-live-model shift. Automatic live 72-hour views use source forecasts; their numerical availability does not establish forecast skill. The [source audit](global_source_audit.md) covers product identity, native support, latency, priors and global delivery gaps.

The v1 fixed physical offset impaired optimization; its old 0.3451 Brier is not a refutation of physics-informed ML. Removing that offset improves the matched temporal experiment. Current conditional coupling still worsens Brier relative to the learned score; the multi-regime proxy test does not establish superiority over rainfall-only.

The [research history](research_history.md) preserves the old six-feature model and failed fixed-offset v1 experiment, with commit-pinned links to their full records. They do not control current live ranking. Root trainers preserve source dates/labels and holdout membership; regenerated frames, reports, historical/optional weights and instruction data remain local artifacts. The exact active runtime model is the published exception, with [provenance and compatibility](../artifacts/MODEL_CARD.md).
