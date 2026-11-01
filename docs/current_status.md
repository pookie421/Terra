# Current project status — v6, 2026-11-01

This is the measured local research baseline prepared for GitHub publication. It describes existing artifacts, not a new acquisition or GPU training run. The [README](../README.md) explains usage; the [roadmap](../ROADMAP.md) separates future ambition from current capability.

## Engineering evidence

`engineering_ready = true`, `pretraining_ready = true`, and `all_requested_targets_passed = false` in [readiness.json](../artifacts/readiness.json). Full-system, randomized and boundary reports match fingerprint `770283f5d49ff96235a8725a62024d2806689a477e190027de7e855a6b990025`.

- 16/16 engineering checks; 26 offline regression tests.
- 8/8 latent-bug checks; 4/4 kinematics controls; 6/6 CLI sanity checks.
- Zero monotonicity violations in 80 comparisons.
- Randomized replay: 28 hazard-evaluated rows and 2 explicitly unavailable-terrain runtime checks, using retained provider snapshots. This is not independent historical validation.
- Source, local datasets, model and readiness reports mirrored into the Windows bundle.

Readiness is limited to the audited local artifacts and implemented invariants. Ignored datasets/models and the sibling archive are absent from a Git clone. New training requires fresh evaluation. The external scientific review work in the roadmap remains open.

## Measured empirical validation

| Metric | Value |
|---|---:|
| Spatial CV ROC-AUC | 0.8546485 |
| Spatial CV PR-AUC | 0.9162266 |
| Spatial CV Brier score | 0.1526382 |
| Original holdout ROC-AUC | 0.7395833 |
| Original holdout PR-AUC | 0.7731659 |
| Original holdout Brier score | 0.2179379 |
| Holdout Recall@40 | 0.9166667 |
| Holdout Recall@5 | 0.1666667 |
| Holdout size | 48 rows / 24 matched event groups |
| Holdout label-interval coverage | 0.8541667 |
| Mean holdout interval width | 0.8835158 |

Stage 1 measures raw-booster out-of-fold scores. Stage 2 measures ungated isotonic event-buffer scores. Neither directly validates the final live gated blend. The holdout has been observed during development; a new untouched evaluation is needed. Ninety-percent nominal label intervals do not guarantee spatial coverage or confidence in true failure probability.

## Five scenario snapshots

| Perturbed scenario | Critical FS | Critical score | Hotspot speed (m/s) |
|---|---:|---:|---:|
| Wayanad | 0.9350 | 85.92% | 14.35 |
| Kaikōura | 0.1793 | 99.68% | 13.74 |
| Zermatt | 0.2754 | 99.99% | 1.33 |
| Malibu | 1.3228 | 94.65% | 0.00 |
| New Delhi | 4.0879 | 0.000112% | 0.00 |

The [full audit](../artifacts/full_system_bug_audit.json) retains perturbations and assertions. Malibu fails the earlier mechanical-FS/velocity targets: separate post-fire likelihood does not force mechanical yield. City scores do not determine engineering readiness.

## Live CLI evidence and missing data

Successful named runs: [Wayanad](../artifacts/cli_wayanad.txt), 509 cells in 3.997 seconds; [Zermatt](../artifacts/cli_zermatt.txt), 530 cells in 4.135 seconds. Both showed `[0,1]` hotspot label intervals. The earlier [Zermatt terrain timeout](../artifacts/cli_zermatt_dem_timeout.txt) is retained. Its output preceded the final fix that preserves successful weather/river/geology telemetry in empty-terrain responses; it is failure-history evidence, not the current response contract.

The final cold coordinate runs took 3.272 s (Blue Mountains), 3.076 s (Wayanad) and 3.062 s (Delhi). A 2.2 s deadline per provider is not a three-second end-to-end guarantee. The final Blue Mountains warming check created no cryospheric eligibility and left FS unchanged.

## What changed in v6

Mechanical FS no longer contains lateral resistance or a post-fire probability-derived override. Median lapse reference, Mamot-based joint retention through a disclosed proxy, dimensionally corrected Voellmy steps, the non-reducing near-yield probability blend and ensemble-scaled residual intervals are implemented. Core metric cards are pinned before distinct mechanism cards. Valid Qwen prose survives numeric verification. Provider deadlines/cancellation and source badges are explicit; modeled river discharge and mapped geology are exposed. The interactive CLI and unavailable-terrain telemetry path are included.

GPU training, real trained-Qwen evaluation and GGUF conversion have not executed on this host. Export failure handling was tested with an injected converter failure. No production frontend or operational alert system is claimed.
