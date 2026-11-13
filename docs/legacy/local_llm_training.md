# Archived: feature fusion, Windows CUDA and Qwen2.5 training

> **Historical local-LLM plan, retired 14 November 2026.** This is the previous combined hazard-model/QLoRA workflow, retained as a research and implementation record. Hosted assistant APIs are now the deployment direction; these local GPU instructions are not website prerequisites. Target Windows/CUDA execution, memory use and language quality were not certified by CPU tests. Use the [current hazard-model workflow](../training_workflow.md), [application setup](../SETUP.md) and [assistant connections](../assistant_connections.md) for active guidance.

The active experiment uses corrected metric terrain, conditional reference columns and a four-way grouped ablation. [Implementation and results](https://github.com/pookie421/Terra/blob/main/docs/feature_fusion_phase3.md) describe its evidence. Original source observations, labels, dates and holdout membership remain preserved. Unresolved footprints are separated from initiation supervision rather than converted to stable negatives.

The target is a global system competitive with LHASA 2.0. This retrospective development experiment does not establish global parity. Follow the [source/API audit](https://github.com/pookie421/Terra/blob/main/docs/global_source_audit.md) when changing inputs, assumptions or datasets. Hardware optimization does not justify changing cohorts, choosing outcomes or concealing the broader benchmark's inferior fusion results.

A clone includes canonical source, small test fixtures and the approved pretrained runtime artifact. **Starting the application does not require training**: use the [setup guide](https://github.com/pookie421/Terra/blob/main/docs/SETUP.md) and [model card](https://github.com/pookie421/Terra/blob/main/artifacts/MODEL_CARD.md). Raw observations, provider archives and generated training frames remain local. Reproducing the retained results requires the source-reviewed corpus, not just the runtime model.

## Prepare source-linked inputs

The commands below are research reproduction, not a fresh-clone setup path. First provision the retained corpus and original manifests at the canonical root.

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

The optional terrain stage downloads only tiles required by the retained locations and is restartable. Prepared Windows bundles already include current metric terrain and raw tiles; the default `run_training.bat` does not rerun the legacy derived-cache job list. Subsequent fitting uses retained local weather/terrain with provider acquisition disabled.

All 1,611 observations retain rebuilt physical context. Initiation fitting uses 186 matched observations, ten components, identical five outer folds and the original sixteen-row development holdout. Seventy-one unresolved footprints remain separate from initiation supervision. Full-component embargo precedes target/feature selection. Event-date forcing can follow failure, report points are uncertain, and absence controls are not surveyed stable slopes; this is not an untouched prospective benchmark.

`train_rainfall_baseline.py` delegates to the four-way trainer. The model uses ordinary fold-prevalence initialization, never a per-row FOSM `base_margin`. Learned trigger and conditional mechanical scores remain separate. Grouped inner selection stays inside fitting partitions and disjoint components calibrate offsets; legacy Optuna flags are rejected. GPU execution does not authorize new controls or holdout-driven tuning.

## Native Windows 11 / RTX 5070 Ti

The target is an NVIDIA RTX 5070 Ti, Blackwell `sm_120`, with 16 GB VRAM. The [hardware guide](https://github.com/pookie421/Terra/blob/main/docs/legacy/windows_5070ti.md) gives the complete launcher, diagnostic and serving commands. Use Python 3.11 or 3.12 and a compatible NVIDIA driver.

From a prepared `windows_5070ti_trainer/` bundle:

```bat
run_training.bat
run_training.bat --smoke-only
run_training.bat --smoke-test
```

- The default run prepares the environment, fits CUDA XGBoost, rebuilds the linked instruction corpus, downloads the committed base/tokenizer files, runs the QLoRA gate and full fit, then starts the direct local assistant server. `--qlora` remains an explicit alias for this full path.
- `--smoke-only` checks XGBoost's requested CUDA execution and same-model save/reload prediction parity without fitting the retained corpus.
- `--smoke-test` installs/checks the QLoRA stack, tests the longest complete trajectory, then exits without a corpus fit.
- `--qlora` runs the complete CUDA XGBoost experiment, broader proxy benchmark, instruction regeneration, mandatory QLoRA memory gate and full adapter fit.

`run_training.bat --llm-only` intentionally skips classifier fitting for an existing verified instruction corpus. Use `--no-serve` to save the full training outputs and exit, or `--xgboost-only` to omit QLoRA. `run_training.bat --cpu` is the explicit XGBoost CPU path; it cannot be combined with QLoRA. Equivalent canonical entry points live in `scripts/`. The [hardware guide](https://github.com/pookie421/Terra/blob/main/docs/legacy/windows_5070ti.md) explains recorded 7B-to-3B memory fallback and explicit base-model selection.

The launcher preserves a working CUDA installation or repairs it using official PyTorch **2.7.1 / CUDA 12.8** wheels from `https://download.pytorch.org/whl/cu128`. It checks actual kernels and reports GPU name, memory, capability, compiled architectures and native BF16 support. Version strings or CUDA visibility alone do not certify `sm_120` execution. QLoRA additionally exercises bitsandbytes NF4 and optimizer kernels; silent CPU or FP16 fallback is rejected.

`requirements-qwen.txt` pins Transformers 4.57.6, PEFT 0.17.1, bitsandbytes 0.48.2, Accelerate 1.10.1, Safetensors 0.6.2, Hugging Face Hub 0.36.0, Datasets 4.3.0 and TRL 0.24.0. The actual bounded-loss trainer remains native Transformers/PEFT. Torch is bootstrapped separately and checked again after installation. Native fused PyTorch **SDPA** is the default. Unsloth, Triton and source-compiled `flash-attn` are not mandatory dependencies. The standalone Qwen trainer's optional `--attention-backend flash_attention_2` requires an importable build, a successful active-GPU forward/backward kernel check and the same longest-trajectory gate; no source compilation or hidden fallback is attempted.

CUDA XGBoost uses `tree_method="hist"`, `device="cuda"`, and rejects an effective CPU device. The same saved model must reproduce predictions exactly after reload on the same configured device. Optional CPU/CUDA inference comparisons have explicit tolerances; independently trained CPU and GPU trees are not claimed bit-identical. The ensemble, ablation report and SHA-256 artifact manifest retain the existing source/cohort identity.

## QLoRA memory policy and instruction evidence

The primary instruction file is `data/sft_context_trajectories.jsonl`: 594 distinct source cell/date traces after full-component holdout exclusion and deduplication. The former nominal-600 filename is a compatibility alias, not 600 distinct examples. `agentic_app_v2` adds public action/evidence summaries, all eight validated application action types and explicit NASA/NISAR interpretation while preserving source IDs, cohort order and exclusions. Traces retain physical envelopes, conditional routing and numeric JSON-pointer bindings. Missing historical comparator/radar evidence remains missing; today's products are not attached to earlier event dates. Earlier corpus bytes and reports are preserved in local `data/instruction_history/`. Calibration rows may appear; these traces are not an independent assistant evaluation.

Every run verifies the retained corpus/report/fusion binding, resolves an immutable Qwen2.5 base revision and tokenizes complete trajectories. Tokenizer/config snapshots are prepared before full weight download; local tokenizer paths avoid unintended online metadata probes in offline mode. The context limit remains **8192 tokens**, and oversize trajectories fail before model loading rather than being truncated. Short examples use **dynamic per-batch padding**, not static 8192-token padding; real EOS tokens remain supervised, and only padding/masked positions use label `-100`.

The enforced training policy is BF16, TF32-enabled FP32 matmul, NF4 four-bit weights with double quantization and BF16 compute, batch size one, eight-step gradient accumulation, length grouping, non-reentrant gradient checkpointing, `paged_adamw_8bit`, pinned host memory and zero dataloader workers. The loss averages causal-token losses within each microbatch; the trainer averages those microbatches over the accumulation window. No worker-spawn or tokenizer-pickling path is needed on Windows.

A full `[1, 8192, 152064]` FP32 logits tensor would occupy about 4.98 decimal GB before its gradient-related storage. Training therefore bypasses the normal CausalLM logits/loss path. It calls the PEFT-instrumented transformer backbone, keeps the frozen dense vocabulary head in BF16, and computes shifted cross-entropy in bounded token chunks. Backward recomputes one chunk at a time, instead of retaining all chunk logits/graphs. Embedding/head dtypes are restored after PEFT preparation; normalization parameters retain their stability treatment. Real CPU tests compare this loss and its gradients against the dense reference, including padding and chunk boundaries. That mathematical check does not measure CUDA VRAM or certify target kernels.

**Every full fit first runs the mandatory memory gate.** It performs two real paged-optimizer steps on the longest complete trajectory, with eight accumulated forward/backward passes per step. It checks Torch allocated/reserved peaks below **14.5 GB = 14,500,000,000 bytes** and samples free device memory. Trainable adapter parameters, training mode and Python/NumPy/Torch RNG are restored even on failure; smoke updates are not extra training exposure for the longest trace. A failed kernel, non-finite loss/update or memory check stops the fit. Paged optimizer state does not page activations or guarantee that every workload will fit.

`training_attempts.json`, `smoke_run.json`, `vram_smoke_test.json` and `training_run.json` under `artifacts/qwen25_terra_lora/` record the actual requested/selected base, dataset/fusion hashes, token lengths, policies, package/device versions, measured peaks and outcomes. A genuine memory failure may retry the complete gate with the declared 3B fallback; provenance or kernel failures do not silently change the base. Prior records/checkpoints are preserved. Corpus-format checks, CPU loss parity and mocked protocol tests are not native Windows/5070 Ti training, measured target VRAM or language-quality evidence. Inspect the target run's records before claiming those checks passed.

## Serve or export the adapter

Training first saves the adapter and tokenizer with hashes. It does not try to materialize a merged 7B model on the 16 GB GPU, and it does not assume Unsloth-only export methods exist. Existing exports are preserved and never relabeled as newly successful outputs.

The default launcher starts direct serving after a successful full fit. To restart the saved adapter later:

```bat
run_copilot_server.bat
```

The generated bundle includes the helper; canonical source supports `python scripts/serve_qwen_copilot.py --auto-port`. It serves Ollama-compatible `/api/tags`, `/api/generate` and `/api/chat` on loopback 11434, using 8001 if auto-port selection finds an existing listener. It records a safe local endpoint for dynamic backend discovery. The saved adapter is `artifacts/qwen25_terra_lora/adapter/`. Serving uses its exact base revision, native BF16/NF4 SDPA, an 8192-token context and a 1536-token completion budget; overflow is rejected without dropping evidence. `TERRA_QWEN_URL` remains an explicit override.

For an optional Ollama import, use the separate export helper after installing compatible external converters and providing sufficient host RAM:

```bat
python export_qwen_ollama.py --merge-cpu --llama-cpp D:\llama.cpp --create
```

Merging/conversion is explicit CPU work, separate from the training VRAM gate. Training's `Modelfile` is a pending recipe until a verified GGUF exists; direct adapter serving needs no conversion. The export helper preserves prior files and records merge, conversion, quantization, model/template validation and import results. Externally supplied GGUF files do not prove binding to this adapter. No Qwen base weights, adapter, merged weights or GGUF outputs are committed to Git. Upstream base-model and source-data terms remain applicable.

The backend's selected scan/state/cell and numeric bindings remain authoritative: a running service does **not** automatically produce `VERIFIED_LLM_PROSE`. Only accepted generated output receives that status; fallback, unavailable-model and rejected-prose outcomes remain explicit. Real multi-turn semantic/tool evaluation is a separate requirement.

## Synchronization and offline verification

Root source is canonical. `training/` and `windows_5070ti_trainer/` are ignored generated bundles, not independent tracked implementations. Never patch their scripts separately. Run `python scripts/sync_training_bundle.py --plan` to review the inventory, then run the script without `--plan` to refresh both copies.

Synchronization excludes obsolete top-level `terrain_v13_*.json` derived caches while retaining canonical originals and preserving earlier bundle copies under ignored archives. Observations, original benchmark/holdout data, current metric terrain, raw Terrarium tiles, historical weather, seismic archives and retained multi-causal materials remain selected. Copies are atomic and hashed against actual canonical bytes; unreadable files fail explicitly. Both current manifests must match; stale destinations or prior hashes cannot substitute for successful source reads.

Run the deterministic suite and complete retained-artifact verification where their inputs are present:

```sh
python -m unittest discover -s tests -v
python scripts/check_training_artifacts.py
python scripts/finalize_readiness.py
```

The checks re-extract current features, verify data/model hashes and grouping, reproduce metrics, reconstruct explanations and verify numeric bindings with provider acquisition disabled. Separate CPU Torch tests validate causal loss, gradients, accumulation and PEFT connectivity. Legacy full-system audit entry points can perform live scans and write historical report formats; they are not a substitute for the current artifact checker. Complete standalone system/randomized audits also require prepared mirrors and their declared inputs.

The bundle carries current benchmark/diagnostic reports as development evidence. A matching manifest, CPU test suite, browser build, CUDA smoke and prospective predictive evaluation establish different things; none alone certifies a global warning system. The active fusion remains `physics_informed_initiation_v2_features`, and the physical reference remains `pedotransfer_reference_column_v1`. The live product follows [FRONTEND_SPEC.md](https://github.com/pookie421/Terra/blob/main/docs/FRONTEND_SPEC.md): automatic arbitrary-coordinate scans, source-driven forecast states and exact state-bound inspection/chat.
