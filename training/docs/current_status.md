# Evaluation and research status

Reference evaluation: **2 November 2026**, CPU baseline with `live_first_storm_wetting_v8` features. These are recorded development results, not certification of future weights or an operational deployment.

## Model evaluation

The training partition contained 546 rows: 304 positive and 242 negative, including 40 training-only synthetic anchors. Those anchors were excluded from calibration and scoring. The original holdout contained 48 rows from 24 event groups.

| Metric | Result |
|---|---:|
| Spatial CV ROC-AUC | 0.8428706 |
| Spatial CV PR-AUC | 0.8925781 |
| Spatial CV Brier score | 0.1601865 |
| Event-holdout ROC-AUC | 0.6979167 |
| Event-holdout PR-AUC | 0.7363478 |
| Event-holdout Brier score | 0.2001038 |
| Holdout Recall@40 | 0.8750 |
| Monotonicity violations | 0 / 80 |
| Kinematics branch controls | 4 / 4 passed |
| Deterministic unit tests | 64 passed |

Spatial CV measures raw-booster out-of-fold scores. Holdout evaluation measures calibrated event-buffer scores. Neither validates the final live gated blend. Recall@40 selects most of this small holdout. Repeated development against the holdout limits independence; a new untouched external evaluation is needed.

## Live scenario checks

The evaluation used nine fresh acquisitions with TTL cache reads disabled. Paired scenarios shared each freshly acquired baseline. All five perturbed regions produced complete 72-hour projections. This is a preselected scenario experiment, not blind labeled prediction validation.

| Scenario | Critical FS | Critical score | Hotspot speed, m/s | Hotspot Newmark, cm | Benchmark result |
|---|---:|---:|---:|---:|---|
| Wayanad, +180 mm | 1.0121 | 73.12% | 14.348 | 0 | Missed FS and score thresholds |
| Kaikōura, +0.35 g | 1.0021 | 75.42% | 7.180 | 18.773 | Missed FS, score and speed thresholds |
| Zermatt, +8°C | 0.3615 | 99.31% | 7.217 | 0 | Passed defined scenario thresholds |
| Malibu, 85% burn and +65 mm | 1.3212 | 96.08% | 8.592 | 0 | Passed defined scenario thresholds |
| New Delhi, +300 mm | 3.8256 | 2.0237% | 0 | 0 | Missed stable-ground score threshold |

The Wayanad targets were FS <.95, score ≥85% and speed >10 m/s; Kaikōura used FS <.65, score ≥95%, displacement ≥10 cm and speed >10 m/s. Zermatt used FS <.85 and score ≥85%; Malibu used score ≥90%, speed ≥5 m/s and available forecasting. Delhi required score <.01% and zero motion. These scenario cutoffs are regression criteria, not observational ground truth.

| Additional location | Critical FS | Score | Evaluation |
|---|---:|---:|---|
| Grand Canyon, Arizona | 1.4364 | 23.39% | Exceeded 15% stable-context threshold |
| Ooty, Tamil Nadu | 1.6176 | 8.97% | Descriptive result |
| Palakkad, Kerala | 5.0000 | 0.095% | Below 5% threshold |
| Chooralmala, Wayanad | 1.2819 | 16.79% | Descriptive result |

Chooralmala had no live geocoder match and used explicitly disclosed fallback coordinates. SoilGrids and Overpass failures were retained in provenance.

The zero-trigger Wayanad control scored 11.04% but had FS 1.3109, below its >1.35 target. The Zermatt control passed with FS 1.9557 and score 10.16%. An injected 220 mm storm beginning at T+18 over 12 hours started from a synthetic 10.44% score and reached 65% at T+29, outside the evaluated T+20–28 interval.

## Interpretation

Stable-ground false alarms remain, including Delhi and Grand Canyon. The mechanical gate can remain partly open despite high mechanical FS. Probabilities, causal shares, intervals and movement estimates require further calibration and independent validation. Some uncertainty intervals remain broad.

Combined engineering/pretraining readiness is **false**. Complete fingerprint-matched integration audits have not been re-established after subsequent source changes. A passing unit suite does not establish scientific validity. End-to-end GPU training, model conversion, forecast skill and operational warning use remain unverified.

## Reproduction

Run `python -m unittest discover -s tests -v` for deterministic checks. With locally acquired data and fitted weights, use the scripts below to generate reports under `artifacts/`:

- `scripts/train_baseline.py`: model fitting and empirical evaluation.
- `scripts/check_training_artifacts.py`: feature, formula and split-isolation checks.
- `scripts/phase4_forecast_audit.py`: fresh region, control and forecast scenarios.
- `scripts/full_system_bug_audit.py`: integration invariants and live API checks.
- `scripts/randomized_global_blind_audit.py`: property-based scenario sampling; replay must be explicitly identified.
- `scripts/literature_boundary_audit.py`: mechanism boundary checks.
- `scripts/finalize_readiness.py`: combines fingerprint-matched audit evidence.

Raw provider snapshots, generated reports and weights are local outputs. Fresh acquisition can differ from this reference run. See [scientific assumptions](physics_sources.md) and [audit protocol](randomized_audit_protocol.md).
