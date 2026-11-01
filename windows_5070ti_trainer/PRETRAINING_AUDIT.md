# Current pretraining evidence

The authoritative combined result is [readiness.json](artifacts/readiness.json). Full, randomized and v5 boundary audits must match the same source/model/data fingerprint. Engineering checks and requested scientific/performance targets are reported separately. The latest pretraining requirements are not all met; no GPU run has been performed.

Read the [full audit](artifacts/full_system_bug_audit.md), [randomized sweeps](artifacts/randomized_global_blind_audit.md), [boundary and literature audit](artifacts/literature_boundary_audit.md), [exact numbered source](artifacts/core_code_audit.md), and [equation/proxy distinctions](docs/physics_sources.md).

The v5 dataset/model and 600 source-linked Qwen trajectories were regenerated. Stage 1 AUC is .8441; unchanged Stage 2 AUC .7700, Recall@40 .9167. PINLS removes no current negatives. Conformal calibration uses separate blocks and produces essentially vacuous label intervals. Source-linked trajectories are not independent landslide events. CUDA training has not run on the Mac. Starting either trainer or importing returned weights clears readiness; evaluate new artifacts again.
