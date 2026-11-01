# TERRA-SHIELD portable GPU trainer

Current v6 baseline: spatial CV ROC-AUC **0.8546**, original event-holdout ROC-AUC **0.7396**, Recall@40 **0.9167** (40 of 48 rows selected). The local baseline has engineering/pretraining readiness **true** under invariant-based gates, while `all_requested_targets_passed` remains **false**. These reports do not certify new GPU weights or a source-only clone.

Conformal calibration reserves 66 cells in 13 separate blocks. Ensemble-scaled intervals target binary labels, can span [0,1], and do not guarantee spatial coverage. The latest host has not executed CUDA, Unsloth training or actual GGUF conversion.

This folder is a local portable snapshot for research training on an RTX 5070 Ti. Review `artifacts/full_system_bug_audit.md` and its JSON before training. A passing baseline audit is not an operational validation or a Windows CUDA test.

Copy the entire **local folder**, including `data/*.parquet` and `data/chatml_trajectories.jsonl`, to Windows. These files and model binaries are ignored by Git; a source-only clone needs its own dataset acquisition. Run `RUN_TRAINING_5070TI.bat` to create an environment, check CUDA and run XGBoost/QLoRA. The launcher is configured for the PyTorch CUDA 13.0 wheel index. Native Windows driver, Unsloth and converter execution must be tested on the target host. For XGBoost only, run `python train_all_on_5070ti.py --skip-dataset-build` in the prepared environment.

Training requires at least 450 complete labeled rows, five coarse blocks and three positive-bearing blocks. The stabilized split uses local terrain features, a fixed 48-row rain/seismic/cryosphere holdout, complete spatial-block embargoes and the reserved former Brazil holdout. The observational set restores 125 cited trigger-unclassified positives. Forty simulated controls receive weight 0.25 only in booster fits; they never enter calibration or scoring. Negative source-date rows are unverified 60-day inventory-absence proxies. Five causal regimes are represented but fire/toe-cut support remains sparse.

Stage 1 reports raw-booster spatial CV; Stage 2 reports ungated calibrated event-buffer scores. Training and inference share equivalent-water features and revised M1/suction/perching physics. The live backend applies a separate mechanical gate and adaptive blend; see `docs/physics_sources.md` for equation/proxy distinctions. Neither stage establishes global calibration or validates live operational decisions. GPU tuning uses its CV objective repeatedly; the selected tuning score is optimistic and is not nested CV.

QLoRA needs 600 distinct source-linked cell/date trajectories (not 600 independent events). Export saves the adapter and tokenizer first, then attempts merged 16-bit weights and Q4_K_M GGUF. `export_manifest.json` records each outcome; conversion failure retains adapter weights. Only import trusted model artifacts back to the Mac.

The root workspace's `scripts/sync_windows_trainer.py` mirrors shared source and current reports. `python scripts/check_training_artifacts.py` verifies local dataset/formula contracts. New training resets `pretraining_ready` to false until the new artifacts receive a fresh full-system audit. Source code uses MIT License; source data and base models keep their own terms (see LICENSE and THIRD_PARTY_NOTICES.md).
