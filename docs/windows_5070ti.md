# Windows 11 / RTX 5070 Ti training

The canonical implementation is in `scripts/`. Generate the portable `training/` and `windows_5070ti_trainer/` directories with `python scripts/sync_training_bundle.py`; edit root source and regenerate them together. A fresh Git clone does not include the retained research corpus or Qwen weights. Application setup needs neither retraining nor a GPU: see [SETUP.md](SETUP.md).

## Run on the target machine

Use 64-bit Python 3.11 or 3.12, a current NVIDIA driver supporting CUDA 12.8, and the prepared bundle. From its directory:

```bat
run_training.bat --smoke-test
run_training.bat --qlora
```

The first command loads Qwen and runs only the longest-trajectory memory test. The second fits the shared-cohort XGBoost experiment, regenerates its linked instruction corpus, repeats the mandatory memory test, and trains QLoRA. It does not download environmental observations or search for new control dates. Hugging Face base weights and dependencies are downloaded when absent. For an intentional Qwen-only run, use `run_training.bat --llm-only`. `run_training.bat --smoke-only` tests XGBoost hardware only; it does not certify Qwen memory. `run_training.bat --cpu --smoke-only` is an explicit CPU diagnostic.

The launcher retains a working CUDA build or installs `torch==2.7.1` from the official `cu128` index. It prints actual GPU name, total/free VRAM, compute capability and native BF16 support. Checks execute CUDA matrix forward/backward, fused SDPA and bitsandbytes NF4/paged-optimizer kernels. Blackwell requires CUDA 12.8+ and compatible architecture code; a reported CUDA version alone is insufficient. CPU fallback is rejected for requested CUDA execution. Failures replace current evidence reports while preserving earlier reports under `previous_checks/`.

The pinned native stack uses Transformers 4.57.6, PEFT 0.17.1, bitsandbytes 0.48.2 and XGBoost 2.1.4. No Triton, Unsloth, TRL or source-compiled FlashAttention dependency is required. Default attention is PyTorch fused SDPA with the quadratic math fallback disabled. Optional `--attention-backend flash_attention_2` on the Python trainer requires an already installed package and a successful active-GPU backward test.

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

The tokenizer-only audit of the current 594-row corpus found a maximum of **3600 tokens**, with zero truncations, using base revision `a09a35458c702b33eeacc393d103063234e8bc28`. That is evidence about this corpus, not an 8192-token GPU certification. Every actual run resolves and records an immutable base revision and retokenizes its complete corpus. Review `vram_smoke_test.json`, `smoke_run.json` or `training_run.json` under `artifacts/qwen2_5_7b_lora/`. They include corpus/model hashes, policy, versions and execution status. No native Windows/5070 Ti training or measured target VRAM result was available on the CPU development host.

## XGBoost and artifact evidence

CUDA fits use `tree_method="hist"`, `device="cuda"`. The hardware smoke checks the effective device, saves/reloads the same model and requires exact prediction equality. An optional CPU/CUDA inference comparison on that same saved model uses declared tolerances; separately trained CPU and GPU trees are not claimed bit-identical. The real ensemble is also reloaded and checked exactly on the retained training and holdout rows before publication. `training_artifact_manifest.json` binds saved artifacts and the ablation report by SHA-256. No holdout labels, cohorts or experiment settings are changed for hardware optimization.

## Serve the adapter directly

After a successful fit, use the same environment as training (`.venv` by default; an already active virtual environment is respected):

```bat
.venv\Scripts\activate
python serve_qwen_copilot.py
```

From the canonical root, use `python scripts/serve_qwen_copilot.py`. The server exposes Ollama-compatible `/api/generate`, `/api/chat`, `/api/tags` and `/healthz` on `127.0.0.1:11434`. Stop an existing Ollama listener first, or use `--port 11435` and set `TERRA_QWEN_URL=http://127.0.0.1:11435` in the backend's environment before starting FastAPI. No backend source edit is needed. The exact recorded base revision, NF4/BF16 adapter and native fused SDPA are used; prompt plus reserved completion must fit 8192 tokens without truncation.

The backend's state and numeric checks remain authoritative. A running model does not automatically earn `VERIFIED_LLM_PROSE`: only prose passing those checks does. An unreachable service uses `VERIFIED_TEMPLATE`; invalid generated prose is explicitly withheld as `WITHHELD_UNVERIFIED_LLM_PROSE`.

## Optional Ollama import

The training stage saves the adapter first and does not merge a full 7B model on the 16 GB GPU. With a separately installed llama.cpp converter/quantizer and at least 24 GiB available host RAM:

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
