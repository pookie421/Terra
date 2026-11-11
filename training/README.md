# Feature-based physics fusion, global context and Windows/Qwen training

The active experiment uses corrected metric terrain, conditional reference columns and a four-way grouped ablation. See [implementation and results](https://github.com/pookie421/Terra/blob/main/docs/feature_fusion_phase3.md). The original source observations, labels, dates and holdout manifest are preserved. Unresolved footprints are separated from initiation supervision rather than converted to stable negatives.

The target is a global system competitive with LHASA 2.0. This small retrospective experiment is a development check within that program, not evidence of global parity. Follow the [source and API audit](https://github.com/pookie421/Terra/blob/main/docs/global_source_audit.md) when changing inputs, assumptions or datasets.

A Git clone contains portable source and small test fixtures, not raw observations, cached provider archives, generated training frames or learned weights. Reproducing the retained results requires those source-reviewed local materials. `sync_training_bundle.py` copies them into prepared local bundles; pushing the source mirror does not upload them. Source-only CI can verify deterministic contracts, with two explicit learned-artifact skips; full artifact verification requires the retained corpus and model outputs.

From the canonical repository root:

```sh
python scripts/rebuild_metric_terrain.py --fetch
python scripts/train_physics_informed.py
python scripts/benchmark_global_context.py
python scripts/regenerate_sft.py
python scripts/report_physics_context.py
python scripts/check_training_artifacts.py
python scripts/finalize_readiness.py
python scripts/sync_training_bundle.py
```

The optional terrain stage is canonical-root preparation: its job list uses preserved original derived-cache locations. The prepared Windows bundle already includes the corrected metric terrain and raw tiles; the nightly `run_training.bat --qlora` path does not invoke that legacy job-list stage or require obsolete derived caches in its current data directory. The terrain stage downloads only tiles required by retained locations and is restartable. Subsequent stages use local raw weather/terrain with network connections disabled. All 1,611 observations receive rebuilt physical context. Initiation fitting uses 186 matched observations, ten grouping components, identical five outer folds and the original sixteen-row rainfall holdout. Full component embargo precedes target/feature selection. This is retrospective development evaluation, not an untouched prospective benchmark.

`train_rainfall_baseline.py` now delegates to the four-way trainer. Both `training/` and `windows_5070ti_trainer/` are generated from root source/data/models, with matching hash manifests. Edit root source and rerun synchronization. Original v14 model/results remain archived and do not control live inference.

## Windows RTX execution

Use Python 3.11 or 3.12 and run `run_training.bat` from the copied bundle. It installs the pinned numerical stack and PyTorch CUDA build before actual XGBoost/PyTorch kernel and train/save/reload smoke tests. Requested CUDA execution fails if it silently falls back to CPU.

```sh
python train.py --smoke-only --check-torch
python train.py --device cuda
python train.py --device cpu --smoke-only
```

The feature-based model uses fold-prevalence initialization, never a per-row unscaled FOSM `base_margin`. Learned trigger and conditional mechanical scores remain separate in the report.

Default training rebuilds the current four-way experiment, the multi-regime proxy benchmark and the instruction corpus. No dataset acquisition or control-date search occurs during fitting. Candidate regularization/depth settings and feature-group weights are predeclared; grouped inner selection stays inside each fitting partition and disjoint components calibrate offsets. Optional legacy Optuna flags are rejected. CPU verification does not establish that native Windows, the RTX 5070 Ti or QLoRA works on the target machine. Cross-device prediction comparisons require declared numerical tolerances and a fresh verification report.

## CUDA XGBoost plus Qwen2.5 QLoRA

The primary instruction file is `data/sft_context_trajectories.jsonl`. It has 594 distinct source cell/date traces after full-component holdout exclusion and deduplication. The former nominal-600 filename is a compatibility alias, not a claim of 600 distinct examples. Traces include physical envelopes, mathematical context, conditional runout, fourteen-source provenance and JSON-pointer numeric bindings. Calibration rows may appear; this is not independent LLM evaluation.

Run **`run_training.bat --qlora`** from the prepared `windows_5070ti_trainer/` directory to execute the complete requested path: install dependencies, verify actual CUDA kernels, fit XGBoost, run the global proxy benchmark, regenerate the instruction corpus, then train Qwen2.5 QLoRA. It does not drop or defer the QLoRA stage. For an intentional LLM-only resume, `python train.py --llm-only` skips the XGBoost stage but still performs its requested CUDA/PyTorch checks. The model and SFT context are 8192 tokens; tokenization preflight rejects oversized examples rather than silently truncating tool evidence. Adapter weights are retained before optional merged/GGUF conversion, and conversion outcomes are recorded separately. Actual Qwen weights, QLoRA and Windows CUDA were not exercised on this CPU host.

Artifact checks re-extract corrected features, verify hashes and unchanged source labels/manifests, inspect component isolation, reproduce metrics, reconstruct model explanations, verify numeric bindings and run deterministic equation/provider/state tests with network disabled. Engineering readiness and predictive validity remain separate fields.

## Live application integration

The browser product follows [FRONTEND_SPEC.md](https://github.com/pookie421/Terra/blob/main/docs/FRONTEND_SPEC.md): arbitrary-coordinate/map-click scans and automatic source-driven forecast horizons. Research replay/export scripts and analyst perturbation APIs are not the frontend boot workflow. Keep original source weather distinct from user-generated research perturbations.

The installed fusion contract is `physics_informed_initiation_v2_features`; the physical reference remains `pedotransfer_reference_column_v1`. Keep version names and model/SFT hashes consistent when copying artifacts to the live API or Ollama workflow. Successful CPU fitting, a browser build and a CUDA smoke each verify different things; none alone certifies predictive skill or an operational warning system.

## Current bundle inventory and preservation

`python scripts/sync_training_bundle.py --plan` reviews the selected inventory without copying or reading payload contents. Synchronization excludes only the obsolete top-level `terrain_v13_*.json` derived caches. Canonical originals remain untouched; earlier bundle copies move to ignored `archive_snapshots/legacy_terrain_v13/` without being reread or claimed freshly verified. Observations, original benchmark/holdout data, current metric terrain, raw Terrarium tiles, historical weather, seismic archives and retained multi-causal source materials remain selected.

Each copied file is atomically staged and SHA-checked against the actual source bytes read. Blocked canonical reads fail with a timeout; stale destination bytes or an archived hash cannot substitute for an unreadable source. The current manifest covers exactly the expected inventory, and both portable copies must match. Previous manifests stay in ignored archives. A failed synchronization does not issue a fresh manifest.

The bundle includes the current global-context benchmark and storm-grid diagnostic reports alongside the four-way experiment and artifact verification. These reports remain development evidence, not a globally validated operational model.

## QLoRA environment and export evidence

Unsloth is imported before TRL/Hugging Face training libraries so its patches run in the intended order. The [upstream Windows installation guide](https://unsloth.ai/docs/get-started/install/windows-installation) describes Core installation with working PyTorch and an actual CUDA test before Unsloth. The prepared launcher performs kernel checks again after QLoRA dependencies are installed. `requirements-qwen.txt` is not a frozen, target-verified environment; actual resolved package, GPU and CUDA versions are recorded in `training_run.json`.

That record also captures the dataset/fusion hashes, actual row count, maximum tokenized example length, base revision when exposed, completion status and export outcomes. The 8192-token preflight runs with the actual target tokenizer and rejects oversize traces. CPU corpus-format checks do not certify target tokenization or training.

Existing adapter, merged and GGUF exports are preserved under `previous_exports/` before replacement. New adapter weights are saved before optional conversions. A conversion failure is recorded explicitly and leaves the trained adapter available; file existence alone does not prove a fresh successful GGUF export. Loading the resulting model through Ollama and evaluating its real multi-turn/tool behavior are separate checks.
