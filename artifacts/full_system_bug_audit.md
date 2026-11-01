# Full system audit

Generated UTC: 2026-11-01T03:13:08.002944+00:00

**engineering_ready = true**

This is the full-audit engineering result; combined pretraining readiness also requires the randomized audit. Empirical targets below are descriptive and do not control readiness.

| Scenario | Critical P05 FS | Critical P95 score | Top velocity m/s | Result |
|---|---:|---:|---:|---|
| Wayanad | 0.998 | 65.088% | 14.35 | FAIL |
| Kaikoura | 0.565 | 99.548% | 13.74 | PASS |
| Zermatt | 0.268 | 99.688% | 1.58 | PASS |
| Malibu | 0.311 | 98.984% | 8.59 | PASS |
| New Delhi | 4.417 | 0.000% | 0.00 | PASS |

Critical percentiles use steep candidate cells (effective slope >=24 degrees), falling back to the viewport when none exist. These are not whole-viewport percentiles.

| Validation | Value | Result |
|---|---:|---|
| Spatial ROC-AUC (>=0.84) | 0.8441 | PASS |
| Event holdout ROC-AUC (>=0.80) | 0.7700 | FAIL |
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
| stage1 roc_auc | 0.8441 | 0.7159–0.9293 |
| stage1 pr_auc | 0.9024 | 0.7884–0.9666 |
| stage1 brier_score | 0.1640 | 0.0957–0.2617 |
| stage2 roc_auc | 0.7700 | 0.6744–0.8711 |
| stage2 pr_auc | 0.8005 | 0.7238–0.8983 |
| stage2 brier_score | 0.2062 | 0.1524–0.2584 |
| stage2 recall_at_k | 0.2083 | 0.1667–0.2083 |
| stage2 recall_at_40 | 0.9167 | 0.8750–1.0000 |

Intervals resample whole spatial blocks or matched event pairs, conditional on fixed predictions; they exclude refitting and label uncertainty.
- Source footprint/point proxies and unverified temporal absences are not surveyed global ground truth.
- Requested scarp/fire coefficients and alpine snowpack floors are scenario assumptions, not calibrated laws.
- Native TreeSHAP explains raw booster margins, not isotonic or mechanically gated scores.
- Windows CUDA/Unsloth execution is not verified on this Mac host.
