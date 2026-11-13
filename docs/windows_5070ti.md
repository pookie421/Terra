# Windows 11 / RTX 5070 Ti training

The canonical implementation is in `scripts/`. Generate the portable `training/` and `windows_5070ti_trainer/` directories with `python scripts/sync_training_bundle.py`; edit root source and regenerate them together. A fresh Git clone does not include the retained research corpus or Qwen weights. Application setup needs neither retraining nor a GPU: see [SETUP.md](SETUP.md).

## Run on the target machine

Use 64-bit Python 3.11 or 3.12, a current NVIDIA driver supporting CUDA 12.8, and the prepared bundle. From its directory:

```bat
run_training.bat
```

Double-clicking the launcher or running it without flags performs the full path: prepare the environment, verify CUDA kernels, fit the fixed XGBoost experiment and broader proxy benchmark, regenerate the linked instruction corpus, download the immutable Qwen snapshot, run the mandatory memory gate, fit QLoRA, save the adapter and start the local assistant server. It uses the prepared retained corpus; it does not acquire new environmental observations or search for control dates. Dependencies and missing Hugging Face files require internet access.

| Optional command | Effect |
|---|---|
| `run_training.bat --smoke-test` | QLoRA longest-trajectory memory gate, then exit |
| `run_training.bat --smoke-only` | CUDA XGBoost hardware/save-reload test, then exit |
| `run_training.bat --cpu --smoke-only` | Explicit CPU XGBoost diagnostic |
| `run_training.bat --xgboost-only` | Fit XGBoost and regenerate linked instructions without QLoRA/serving |
| `run_training.bat --llm-only` | Use the existing verified instruction corpus for QLoRA and serving |
| `run_training.bat --no-serve` | Complete the full fit and save outputs, then exit |
| `run_training.bat --base-model Qwen/Qwen2.5-3B-Instruct` | Explicitly choose the smaller base |
| `run_training.bat --no-model-fallback` | Require the requested base to pass; do not retry with another model |

The default requested base is `Qwen/Qwen2.5-7B-Instruct`. A genuine memory failure can trigger an explicitly logged retry with `Qwen/Qwen2.5-3B-Instruct`. That retry retokenizes every complete trace and repeats the same memory gate; it does not truncate data or turn a failed 7B check into a 7B success. Data/provenance, provider and kernel errors do not trigger a model substitution. Inspect `training_attempts.json` and the selected base/revision in the run metadata.

The launcher retains a working CUDA build or installs `torch==2.7.1` from the official `cu128` index. It prints actual GPU name, total/free VRAM, compute capability and native BF16 support. Checks execute CUDA matrix forward/backward, fused SDPA and bitsandbytes NF4/paged-optimizer kernels. Blackwell requires CUDA 12.8+ and compatible architecture code; a reported CUDA version alone is insufficient. CPU fallback is rejected for requested CUDA execution. Failures replace current evidence reports while preserving earlier reports under `previous_checks/`.

The pinned native stack includes Transformers 4.57.6, PEFT 0.17.1, bitsandbytes 0.48.2, TRL 0.24.0, Datasets 4.3.0, Hugging Face Hub 0.36.0 and XGBoost 2.1.4. The bounded-loss implementation uses native Transformers/PEFT training; installing TRL does not switch it back to a full-logits SFT loss. Native HDF5 support is included for optional NISAR subsets. Triton, Unsloth and source-compiled FlashAttention are not required. Default attention is PyTorch fused SDPA with the quadratic math fallback disabled. Optional `--attention-backend flash_attention_2` on the Python trainer requires an already installed package and a successful active-GPU backward test.

## Memory contract

| Setting | Value |
|---|---|
| Base / quantization | Qwen2.5-7B-Instruct, NF4, double quantization, BF16 compute |
| Maximum sequence | 8192; reject oversize traces, never truncate |
| Batch / accumulation | 1 / 8; group by actual token length |
| Padding | Longest sequence in the current batch only |
| Precision | BF16, no FP16; TF32 enabled, float32 matmul precision `high` |
| Checkpointing | Enabled, `use_reentrant=False`, KV cache disabled |
| Optimizer | `paged_adamw_8bit` |
| Data loader | Zero workers, pinned memory, no persistent workers |

At 8192 tokens and vocabulary 152,064, one dense FP32 logits tensor alone is 4,982,833,152 bytes (4.98 GB / 4.64 GiB). The trainer calls the decoder directly and applies the frozen BF16 language-model head in 128-token chunks. FP32 cross-entropy is limited to each chunk; backward recomputes each chunk instead of retaining every chunk's loss graph. PEFT's frozen embeddings and output head are restored to BF16 after k-bit preparation. Tests compare shifted, masked loss and LoRA gradients against dense reference calculations and verify eight-step Trainer accumulation. This avoids the full vocabulary logits allocation without changing the causal loss.

The mandatory smoke performs **two optimizer updates, each with eight forward/backward passes**, on the single longest complete trace. It checks allocated and reserved peaks against **14.5 GB = 14,500,000,000 bytes**, also requiring at least 1 GB sampled free device memory. Adapter weights, training mode and Python/NumPy/Torch random state are restored afterward; the smoke does not add training updates to that trajectory. Failure stops the full fit and writes evidence. Paged optimizer state does not page activations or guarantee that every future workload fits.

Every run resolves and records an immutable base revision and retokenizes all **594 complete source-linked traces** in the current corpus. Tokenizer/config preparation precedes full-weight download. `snapshot_download` resumes committed files; the tokenizer is loaded from its resolved local snapshot path so `--local-files-only` does not rely on a library's unsolicited metadata lookup. All entry paths, including LLM-only and direct Python runs, verify the retained corpus against its SFT report and actual fusion artifact. Custom corpora without matching provenance explicitly omit a fusion binding.

Review `training_attempts.json`, `vram_smoke_test.json`, `smoke_run.json` and `training_run.json` under `artifacts/qwen25_terra_lora/`. They record the actual base/revision, corpus/model hashes, maximum tokenized length, policy, versions and execution status. Tokenizer or CPU checks do not certify an 8192-token GPU workload. Actual Windows/5070 Ti QLoRA and measured target VRAM remain target-machine checks.

## XGBoost and artifact evidence

CUDA fits use `tree_method="hist"`, `device="cuda"`. The hardware smoke checks the effective device, saves/reloads the same model and requires exact prediction equality. An optional CPU/CUDA inference comparison on that same saved model uses declared tolerances; separately trained CPU and GPU trees are not claimed bit-identical. The real ensemble is also reloaded and checked exactly on the retained training and holdout rows before publication. `training_artifact_manifest.json` binds saved artifacts and the ablation report by SHA-256. No holdout labels, cohorts or experiment settings are changed for hardware optimization.

## Serve the adapter directly

The default full run starts serving after saving `artifacts/qwen25_terra_lora/adapter/`, including the tokenizer and binding metadata. To restart it later from the prepared bundle:

```bat
run_copilot_server.bat
```

The helper uses the training environment (`.venv` by default; an already active virtual environment is respected). Canonical source also supports `python scripts/serve_qwen_copilot.py --auto-port`. The server exposes Ollama-compatible `/api/generate`, `/api/chat`, `/api/tags` and `/healthz` on loopback port **11434**, trying **8001** when auto-port selection encounters an existing listener. It leaves that listener running. If both ports are occupied, select an available `--port`.

The running server writes `artifacts/qwen_copilot_endpoint.json`. The backend discovers this safe loopback endpoint dynamically, including prepared local bundle locations; an explicit constructor URL or `TERRA_QWEN_URL` takes precedence. This lets an already running app discover the newly trained adapter without editing backend source. The exact recorded base revision, NF4/BF16 adapter and native fused SDPA are used. Context is **8192 tokens**, with up to **1536 reserved completion tokens** for the structured multi-action response. Overflow is rejected without truncation.

The backend's state and numeric checks remain authoritative. A running model does not automatically earn `VERIFIED_LLM_PROSE`: only prose passing those checks does. An unreachable service uses `VERIFIED_TEMPLATE`; invalid generated prose is explicitly withheld as `WITHHELD_UNVERIFIED_LLM_PROSE`.

## Optional Ollama import

Training writes a `Modelfile` recipe marked **pending verified GGUF export**. It is not an importable Qwen adapter claim or a fresh successful GGUF artifact. Direct serving uses the saved adapter immediately after a successful fit. The optional conversion helper merges on CPU, not the 16 GB GPU. With a separately installed llama.cpp converter/quantizer and at least 24 GiB available host RAM:

```bat
python export_qwen_ollama.py --merge-cpu --llama-cpp D:\llama.cpp --create
```

The helper merges on CPU, converts to GGUF, quantizes to Q4_K_M, validates Qwen architecture/template metadata and writes a ChatML `Modelfile` before `ollama create`. Its default output directory is `artifacts/ollama_qwen/`, including `ollama_export_manifest.json` and `Modelfile`. Earlier exports are preserved. Alternatively pass `--merged-dir` or an existing `--gguf`; an externally supplied GGUF is explicitly marked as having unverified adapter binding. Conversion, import and real assistant-quality evaluation are separate results. No Qwen weights or generated adapters are committed to Git.

## Upstream compatibility references

- [PyTorch release wheels, including 2.7.1 / CUDA 12.8](https://pytorch.org/get-started/previous-versions/).
- [bitsandbytes 0.48.2 native Windows CUDA/architecture matrix](https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation).
- [XGBoost 2.1.4 Blackwell and CUDA 12.8 release notes](https://xgboost.readthedocs.io/en/release_3.0.0/changes/v2.1.0.html).
- [Ollama import](https://docs.ollama.com/import) and [Modelfile reference](https://docs.ollama.com/modelfile).

These upstream support statements motivate the selected stack. Only actual target execution can establish that its installed wheels, driver and available memory work together. None of these engineering checks establishes global predictive skill or LHASA parity; see [current research results](current_status.md).
