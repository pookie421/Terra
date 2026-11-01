# Pre-Phase-4 final results

engineering_ready = true; pretraining_ready = true. Scope: checked research engineering only.

| Engineering check | Result |
|---|---|
| Full-audit checks | 16/16 |
| Offline regression tests | 17/17 |
| Latent checks | 8/8 |
| Kinematic branches | 4/4 |
| Model monotonicity probes | 0/80 violations |
| Randomized location/scenario pairs | 30/30 |
| Paired-cell monotonic violations | 0 |
| Kinematic contradictions | 0 |
| Agent numeric/scan parity | 30/30 |
| Named plain controls at +300 mm | 10/10 across two sweeps |
| Source, MIT and Windows bundle parity | PASS |

The full-audit scenario score targets are retained separately:

| Region | Critical FS | P95 score | Top speed m/s | Scenario target |
|---|---:|---:|---:|---|
| Wayanad | 0.935 | 75.83337% | 14.35 | MISS: score below 80% |
| Kaikoura | 0.537 | 99.70878% | 13.88 | PASS |
| Zermatt | 0.225 | 99.88986% | 2.37 | PASS |
| Malibu | 0.311 | 99.16282% | 8.59 | PASS |
| New Delhi | 4.067 | 0.00012% | 0.00 | PASS |

Kaikoura selected-hotspot Newmark displacement: 18.773 cm. Critical statistics use effective slopes ≥24°, falling back to all cells. Scores are exploratory, not calibrated operational probabilities.

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

Stage 2: 48 rows, 24 matched event groups. Recall@K uses K=5; Recall@40 selects 40/48 rows. Intervals resample clusters with fixed predictions and exclude refitting/label uncertainty.

Final randomized sweeps: seeds [20261031, 320950895]. These results replay the saved live acquisitions after correcting the strict-increase assertion; selections and source inputs are unchanged.

| Sweep | Location | Perturbation | Critical FS | P95 score | Invariants |
|---|---|---|---:|---:|---|
| 1 | Darjeeling | rain +64.3 mm; burn 76.6% | 0.321 | 99.94158% | PASS |
| 1 | Petropolis | rain +89.1 mm; burn 85.2% | 0.300 | 99.98810% | PASS |
| 1 | Langtang | rain +42.4 mm; burn 90.6% | 0.318 | 99.99597% | PASS |
| 1 | Taroko | PGA 0.295 g | 0.636 | 99.81605% | PASS |
| 1 | Zermatt | rain +202.8 mm | 0.739 | 99.07913% | PASS |
| 1 | Chamonix | rain +347.7 mm | 0.810 | 93.81451% | PASS |
| 1 | Glenwood Canyon | rain +79.1 mm; burn 60.6% | 0.310 | 99.83999% | PASS |
| 1 | Blue Mountains | PGA 0.339 g | 0.921 | 77.61214% | PASS |
| 1 | Ramban | rain +75.2 mm; burn 86.7% | 0.303 | 99.89640% | PASS |
| 1 | Chuquicamata | rain +76.1 mm; burn 93.5% | 0.301 | 99.99267% | PASS |
| 1 | New Delhi | rain +300.0 mm | 4.072 | 0.00012% | PASS |
| 1 | Kansas | rain +300.0 mm | 5.000 | 0.00000% | PASS |
| 1 | Amsterdam | rain +300.0 mm | 5.000 | 0.00000% | PASS |
| 1 | Riyadh | rain +300.0 mm | 5.000 | 0.00000% | PASS |
| 1 | Manaus | rain +300.0 mm | 4.262 | 0.00011% | PASS |
| 2 | Darjeeling | rain +90.6 mm; burn 77.2% | 0.301 | 99.99497% | PASS |
| 2 | Petropolis | rain +98.9 mm; burn 83.2% | 0.300 | 99.99341% | PASS |
| 2 | Kaikoura | rain +322.3 mm | 0.702 | 99.82095% | PASS |
| 2 | Banos | rain +112.1 mm | 0.849 | 92.11953% | PASS |
| 2 | Zermatt | rain +49.1 mm; burn 66.2% | 0.531 | 99.70976% | PASS |
| 2 | Aoraki | rain +75.2 mm; burn 63.5% | 0.320 | 99.74616% | PASS |
| 2 | Montecito | rain +196.7 mm | 1.475 | 62.85423% | PASS |
| 2 | Blue Mountains | warming +8.0 C | 0.206 | 99.99664% | PASS |
| 2 | Chuquicamata | rain +41.6 mm; burn 76.2% | 0.602 | 98.24695% | PASS |
| 2 | Geirangerfjord | rain +61.9 mm; burn 72.5% | 0.341 | 97.26021% | PASS |
| 2 | New Delhi | rain +300.0 mm | 4.072 | 0.00012% | PASS |
| 2 | Kansas | rain +300.0 mm | 5.000 | 0.00000% | PASS |
| 2 | Amsterdam | rain +300.0 mm | 5.000 | 0.00000% | PASS |
| 2 | Riyadh | rain +300.0 mm | 5.000 | 0.00000% | PASS |
| 2 | Manaus | rain +300.0 mm | 4.262 | 0.00011% | PASS |

Cold provider acquisition: 7.88–14.15 seconds. SoilGrids and OSM were unavailable and marked as proxies; other listed feeds responded. The 2-second cold-start objective remains unmet.

Published-equation domains and scenario approximations: docs/physics_sources.md. Replay and causal activation protocol: docs/randomized_audit_protocol.md. Windows CUDA/QLoRA execution remains to be verified on the target GPU.
