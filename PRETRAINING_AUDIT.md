# Current pretraining evidence (v6)

The authoritative combined result is [readiness.json](artifacts/readiness.json). Full, randomized and boundary audits must match the same source/model/data fingerprint. Engineering readiness reflects deterministic invariants; empirical city scores and benchmark decimals do not gate it. This is not operational certification.

Read the [full audit](artifacts/full_system_bug_audit.md), [randomized sweeps](artifacts/randomized_global_blind_audit.md), [boundary and literature audit](artifacts/literature_boundary_audit.md), [exact numbered source](artifacts/core_code_audit.md), and [equation/proxy distinctions](docs/physics_sources.md).

The dataset, CPU baseline and 600 source-linked Qwen trajectories were regenerated. Spatial CV ROC-AUC is .8546; original event holdout AUC is .7396 and Recall@40 is .9167. These are source-location proxy evaluations, not global field validation. Conformal calibration uses 66 cells in 13 separate spatial blocks. Ensemble-scaled intervals target binary labels, can be [0,1], and do not guarantee spatial coverage or quantify true-probability confidence. Mean holdout width is .8835 and observed label coverage is .8542.

Provider deadlines are 2.2 seconds each; sequential geocoding/terrain-dependent imagery and local computation mean total named scans can exceed three seconds. Fallback-only scenarios are separated from hazard-evaluated scenarios. The Mac has not executed CUDA/Unsloth training or real GGUF conversion; conversion failure handling was tested. Starting a trainer or importing returned weights clears readiness until the new artifacts are audited.
