# Hazard-model research and reproducibility

The website includes the approved CPU runtime model and physical solver. **Using the map does not require retraining, a GPU or a language-model download.** Follow [application setup](https://github.com/pookie421/Terra/blob/main/docs/SETUP.md) for a fresh clone and [assistant connections](https://github.com/pookie421/Terra/blob/main/docs/assistant_connections.md) for optional hosted generated prose. This page is for reproducing the retained hazard-model experiments.

Root source is canonical. Preserve original observations, labels, dates, source manifests, event/spatial components and holdout membership. Hardware or architecture changes do not authorize different controls, inferred initiation crowns, holdout tuning or the removal of failed cases. Consult [current results](https://github.com/pookie421/Terra/blob/main/docs/current_status.md), the [source audit](https://github.com/pookie421/Terra/blob/main/docs/global_source_audit.md) and [physical methods](https://github.com/pookie421/Terra/blob/main/docs/physics_sources.md) before scientific changes.

## Required research inputs

A Git clone contains the small approved model, source and test fixtures; it does not contain the raw research corpus, provider archives or complete training frames. Provision the source-reviewed retained inputs and their manifests separately. Installing dependencies or possessing the runtime model cannot reconstruct them.

Use the Python environment from the setup guide, then install the research requirements:

```sh
python -m pip install -r requirements-training.txt -c constraints-runtime.txt
```

The active experiments use corrected metric terrain, retained historical weather and conditional reference columns. All **1,611 source observations** retain context. Initiation fitting uses **186 matched observations, ten components, five shared outer folds and sixteen development-holdout rows**. **Seventy-one unresolved footprint positives** remain outside initiation supervision. The broader proxy benchmark uses **510 rows in 34 components** under its declared exclusions. These are different tasks, not interchangeable evaluations.

## Reproduce the CPU hazard models

From the canonical repository root, with the original corpus and raw terrain tiles already provisioned:

```sh
python scripts/rebuild_metric_terrain.py
python scripts/train_physics_informed.py --device cpu
python scripts/benchmark_global_context.py --device cpu
python scripts/report_physics_context.py
```

The terrain rebuild uses the retained tile cache by default and fails explicitly when a required tile is absent. Its separate `--fetch` option acquires terrain for retained locations; use it only when deliberately preparing missing research inputs. It is not part of offline reproduction. Classifier fitting disables provider acquisition and rebuilds source-linked physical contexts from retained inputs.

The four-way trainer fits rainfall-only, physics-only, weather/terrain ML and feature-fusion models with shared grouped evaluation. The learner uses ordinary fold-prevalence initialization, not a per-row FOSM `base_margin`; learned trigger and conditional mechanical scores remain separately inspectable. `scripts/train_rainfall_baseline.py` is a compatibility entry point into this trainer, not an independent NASA model reproduction.

Fitting replaces the local `artifacts/physics_informed_models.pkl` only after save/reload prediction parity succeeds on the configured device. It writes the feature frame and ablation report with dataset/model hashes, fold membership, environment and exclusions. Preserve the published model and its checksum before fitting if you need that exact release. A newly fitted artifact is a new experiment; do not reuse the published checksum or imply automatic approval for publication.

Relevant outputs include `physics_informed_ablation_report.json`, `global_context_benchmark.json`, `context_observation_coverage.json` and the model-bound physical-context report. Generated research outputs remain local. The approved public runtime artifact has a separate [model card and checksum manifest](https://github.com/pookie421/Terra/blob/main/artifacts/MODEL_CARD.md).

## Verify the complete retained experiment

The complete checker also verifies source-linked instruction diagnostics. Regenerating these traces requires no language-model fit or provider request and does not demonstrate assistant quality:

```sh
python scripts/regenerate_sft.py
python scripts/check_training_artifacts.py
python scripts/finalize_readiness.py
```

The artifact checker runs deterministic tests with network access disabled and checks feature/model/data hashes, grouping, metric parity and **594 distinct source cell/date instruction traces**. It requires the complete retained inputs. The readiness report must follow a successful checker on the same source/data/model fingerprint; old reports are not substitutes for a current pass.

For source-only deterministic checks, including environments without the retained research corpus:

```sh
python -m unittest discover -s tests -v
```

Artifact-dependent tests disclose missing-input skips. A passing source-only suite, matching hashes, CPU model parity and frontend build are engineering evidence. They do not establish calibrated global probabilities, forecast lead time, independent runout accuracy, actual CUDA execution or LHASA parity. Current fusion still underperforms rainfall-only on the broader proxy benchmark; retain that result alongside improvements in the matched temporal experiment.

## Hosted assistant and historical local training

The website uses request-scoped personal API keys with state binding, numeric verification and deterministic fallback. Configure your provider/model through **AI Model & API Key** as described in [assistant connections](https://github.com/pookie421/Terra/blob/main/docs/assistant_connections.md). Hazard prediction stays on the backend; changing the prose model does not retrain or improve the hazard classifier. Evaluate numeric grounding, semantic accuracy and action completion on held-out prompts for the actual provider/model used.

The former combined Windows/QLoRA plan is preserved in [the archived workflow](https://github.com/pookie421/Terra/blob/main/docs/legacy/local_llm_training.md) and [RTX 5070 Ti guide](https://github.com/pookie421/Terra/blob/main/docs/legacy/windows_5070ti.md). Canonical local-serving/training scripts remain available for existing research installations. Their retention does not make GPU training a website dependency or certify target-hardware execution, VRAM use, exports or language quality. The [retired randomized audit protocol](https://github.com/pookie421/Terra/blob/main/docs/legacy/randomized_audit_protocol.md) likewise records historical engineering criteria, not current score targets.

## Refresh portable research copies

`training/` and `windows_5070ti_trainer/` are ignored generated bundles, not independently maintained implementations. Never patch their code separately. Review the inventory before refreshing them:

```sh
python scripts/sync_training_bundle.py --plan
python scripts/sync_training_bundle.py
```

Synchronization uses canonical root bytes and hashed manifests. It includes only selected source-reviewed research inputs; a fresh clone still needs those inputs provisioned. Unreadable files or stale copies must not be relabeled as a successful sync. Keep raw corpora, provider archives, unapproved weights, keys and local notes outside Git. See [research history](https://github.com/pookie421/Terra/blob/main/docs/research_history.md) for retired experiments and preserved failures.
