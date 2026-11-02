# Training workflows

This directory provides a portable research training environment for CUDA XGBoost and optional Qwen QLoRA fine-tuning. No particular GPU model is required by the project; compatibility and available memory depend on the selected dependencies and model configuration. The reference evaluation used the CPU workflow. End-to-end GPU training and model conversion have not been independently verified.

## CPU baseline

From the repository root, acquire observations and fit the baseline:

```bash
python backend/ml/build_real_labeled_dataset.py --output data/real_global_landslide_observations.parquet --max-events 450
python scripts/train_baseline.py --skip-dataset-build --provider-timeout 15
python scripts/check_training_artifacts.py
```

To recompute physics features from retained local observations, use the dataset builder's `--audit-refresh` option. Reports and learned weights are written to `artifacts/` and remain outside version control.

## Portable CUDA environment

From the repository root, synchronize shared source and locally acquired data:

```bash
python scripts/sync_training_bundle.py
```

Copy the complete `training/` directory to the training environment if necessary. A Git clone alone does not contain the required datasets or model weights.

Create and activate a Python environment, install a CUDA-enabled PyTorch build compatible with the environment, then install the training dependencies. The optional QLoRA stack has additional platform requirements; validate its installation before starting a long run.

```bash
cd training
python -m pip install -r requirements.txt
python train.py --skip-dataset-build
```

For optional QLoRA training:

```bash
python -m pip install -r requirements-qwen.txt
python train.py --skip-dataset-build --train-llm
```

`run_training.bat` is a Windows launcher. It uses the active environment when one is provided, checks CUDA visibility, installs the declared project dependencies and runs both training stages. It does not choose a hardware-specific PyTorch wheel.

## Evaluation contracts

Training requires sufficient complete labeled observations and positive-bearing spatial blocks. Synthetic regularization controls are weighted only in base-booster fitting and excluded from calibration and evaluation. Event groups and reserved spatial blocks must remain isolated. Inventory-absence negatives and report-location positives remain imperfect labels.

Stage 1 reports raw-booster spatial cross-validation. Stage 2 evaluates calibrated event-buffer scores. The deployed mechanical gate and mixture are evaluated separately. Repeated tuning does not constitute nested cross-validation or independent evidence of generalization.

QLoRA uses source-linked cell/date trajectories, not an equivalent number of independent landslide events. Export saves adapters and tokenizers before attempting merged weights and GGUF conversion; `export_manifest.json` records outcomes. Failed conversion retains adapter weights.

## Importing artifacts

On a shell-compatible inference environment, `./import_artifacts.sh` at the repository root imports local artifacts from `training/artifacts/` and optionally registers a GGUF with Ollama. Load only trusted serialized model files. Importing or retraining weights invalidates prior readiness until the new artifacts are evaluated.

See the repository's scientific methods and evaluation documentation. Original code uses MIT; source data and base models retain their own terms.
