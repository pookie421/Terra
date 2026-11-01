# Full system audit

Generated UTC: 2026-10-31T12:51:14.077200+00:00

**engineering_ready = true**

This is the full-audit engineering result; combined pretraining readiness also requires the randomized audit. Empirical targets below are descriptive and do not control readiness.

| Scenario | Critical P05 FS | Critical P95 score | Top velocity m/s | Result |
|---|---:|---:|---:|---|
| Wayanad | 0.935 | 75.833% | 14.35 | FAIL |
| Kaikoura | 0.537 | 99.709% | 13.88 | PASS |
| Zermatt | 0.225 | 99.890% | 2.37 | PASS |
| Malibu | 0.311 | 99.163% | 8.59 | PASS |
| New Delhi | 4.067 | 0.000% | 0.00 | PASS |

Critical percentiles use steep candidate cells (effective slope >=24 degrees), falling back to the viewport when none exist. These are not whole-viewport percentiles.

| Validation | Value | Result |
|---|---:|---|
| Spatial ROC-AUC (>=0.84) | 0.8584 | PASS |
| Event holdout ROC-AUC (>=0.80) | 0.7682 | FAIL |
| Recall@40 (descriptive) | 0.8750 | Reported |
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
| stage1 roc_auc | 0.8584 | 0.7448–0.9228 |
| stage1 pr_auc | 0.9123 | 0.8167–0.9590 |
| stage1 brier_score | 0.1536 | 0.1042–0.2327 |
| stage2 roc_auc | 0.7682 | 0.6780–0.8759 |
| stage2 pr_auc | 0.8076 | 0.7322–0.9078 |
| stage2 brier_score | 0.2058 | 0.1523–0.2532 |
| stage2 recall_at_k | 0.2083 | 0.1667–0.2083 |
| stage2 recall_at_40 | 0.8750 | 0.8333–1.0000 |

Intervals resample whole spatial blocks or matched event pairs, conditional on fixed predictions; they exclude refitting and label uncertainty.
- Source footprint/point proxies and unverified temporal absences are not surveyed global ground truth.
- Requested scarp/fire coefficients and alpine snowpack floors are scenario assumptions, not calibrated laws.
- Native TreeSHAP explains raw booster margins, not isotonic or mechanically gated scores.
- Windows CUDA/Unsloth execution is not verified on this Mac host.
