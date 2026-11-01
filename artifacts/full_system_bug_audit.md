# Full system audit

Generated UTC: 2026-11-01T08:54:12.635275+00:00

**engineering_ready = true**

This is the full-audit engineering result; combined pretraining readiness also requires the randomized audit. Empirical targets below are descriptive and do not control readiness.

| Scenario | Critical P05 FS | Critical P95 score | Top velocity m/s | Result |
|---|---:|---:|---:|---|
| Wayanad | 0.935 | 85.920% | 14.35 | PASS |
| Kaikoura | 0.179 | 99.677% | 13.74 | PASS |
| Zermatt | 0.275 | 99.994% | 1.33 | PASS |
| Malibu | 1.323 | 94.655% | 0.00 | FAIL |
| New Delhi | 4.088 | 0.000% | 0.00 | PASS |

Critical percentiles use steep candidate cells (effective slope >=24 degrees), falling back to the viewport when none exist. These are not whole-viewport percentiles.

| Validation | Value | Result |
|---|---:|---|
| Spatial ROC-AUC (>=0.84) | 0.8546 | PASS |
| Event holdout ROC-AUC (>=0.80) | 0.7396 | FAIL |
| Recall@40 (descriptive) | 0.9167 | Reported |
| Monotonicity violations | 0.0 / 80.0 | PASS |
| Kinematics | 4 / 4 | PASS |
| Latent checks | 8 / 8 | PASS |

The 48-row balanced holdout has 24 event groups across rain, seismic and cryospheric regimes with matched temporal absence proxies. It replaces the Brazil-only benchmark and is not comparable with its old scores. Recall@40 covers most of this small set. Stage 2 scores ungated isotonic probabilities; the live gated blend is not independently calibrated.

| Engineering check | Result |
|---|---|
| physics_and_kinematics | PASS |
| local_features_and_briefing_guard | PASS |
| untouched_archive | PASS |
| vector_3_feature_order | PASS |
| vector_4_runout | PASS |
| vector_5_attribution | PASS |
| vector_6_sanitizer | PASS |
| dataset_isolation | PASS |
| vector_8_windows_export | PASS |
| source_compilation | PASS |
| vector_1_geojson | PASS |
| vector_2_live_shap | PASS |
| live_cards_and_kinematic_invariant | PASS |
| all_api_endpoints | PASS |
| vector_7_portability | PASS |
| mit_license | PASS |

| Empirical metric | Estimate | 95% cluster interval |
|---|---:|---|
| stage1 roc_auc | 0.8546 | 0.7251–0.9386 |
| stage1 pr_auc | 0.9162 | 0.7995–0.9760 |
| stage1 brier_score | 0.1526 | 0.0857–0.2464 |
| stage2 roc_auc | 0.7396 | 0.6258–0.8702 |
| stage2 pr_auc | 0.7732 | 0.6751–0.9097 |
| stage2 brier_score | 0.2179 | 0.1605–0.2696 |
| stage2 recall_at_k | 0.1667 | 0.1250–0.2083 |
| stage2 recall_at_40 | 0.9167 | 0.7917–1.0000 |

Intervals resample whole spatial blocks or matched event pairs, conditional on fixed predictions; they exclude refitting and label uncertainty.
- Source footprint/point proxies and unverified temporal absences are not surveyed global ground truth.
- Requested scarp/fire coefficients and alpine snowpack floors are scenario assumptions, not calibrated laws.
- Native TreeSHAP explains raw booster margins, not isotonic or mechanically gated scores.
- Windows CUDA/Unsloth execution is not verified on this Mac host.
