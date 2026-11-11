# Contributing

TERRA-SHIELD welcomes improvements to scientific methods, provider reliability, usability and reproducibility.

## Local setup

Create a Python environment and install `requirements.txt`. Run `python -m unittest discover -s tests -v` before submitting changes. Deterministic tests should use fixtures or mocked providers; live checks should record failures and source provenance explicitly.

For the web client, install dependencies with `npm ci` in `frontend/`, then run `npm run build` and `npx tsc --noEmit`. Check the live API contract and UI state/selected-cell identities; fixtures and a TypeScript build do not establish live provider availability.

Shared ML, physics and ingestion code is maintained under `backend/`. After changing those modules, run `python scripts/sync_training_bundle.py` to refresh the portable training copy. Do not edit the mirrored implementation independently.

## Product contract

Read [FRONTEND_SPEC.md](docs/FRONTEND_SPEC.md) before changing the browser workflow. It is live-first and automatic: arbitrary coordinates/map clicks, source-derived current and 72-hour forecast states, read-only forcing and an obsidian satellite/terrain HUD. Do not restore manual weather sliders, fixed-city presets or a replay-first boot. Legacy research scenarios and fixtures remain separate reproducibility tools.

## Scientific changes

Describe the mechanism, cite primary sources and distinguish published equations from project assumptions. Include appropriate boundary or regression checks. Feature construction must remain label-blind; evaluation events and calibration blocks must remain isolated from fitting. Retain failed cases and disclose changes to evaluation populations.

The target is a global system competitive with LHASA 2.0. Follow the [source/API audit contract](docs/global_source_audit.md#required-record-for-future-source-or-coefficient-changes) for every provider or coefficient change. Critique native support, timing, uncertainty and applicability as well as API availability. Compare benchmark systems on shared tasks, inputs available at issue time and explicit coverage; a higher score on a different cohort is not evidence of superiority.

The source-only test suite explicitly skips two regressions requiring local trained artifacts. Full local verification additionally requires `python scripts/check_training_artifacts.py`; do not represent a source-only CI pass as artifact or predictive validation. Keep prior-based physical calculations available with honest provenance when site measurements are absent.

A pull request should explain the resulting behavior, relevant validation and remaining limitations. Improvements to numerical targets alone are not sufficient evidence of scientific correctness.

## Repository contents

Keep credentials, workstation paths, local notes, raw provider responses, source dumps, generated reports, datasets and model weights outside version control. Publish concise evaluation summaries with their protocol and limitations. Third-party data and model licenses continue to apply.

Do not make operational warning, calibrated probability or globally validated performance claims without supporting independent evidence.
