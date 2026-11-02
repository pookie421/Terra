# Contributing

TERRA-SHIELD welcomes improvements to scientific methods, provider reliability, usability and reproducibility.

## Local setup

Create a Python environment and install `requirements.txt`. Run `python -m unittest discover -s tests -v` before submitting changes. Deterministic tests should use fixtures or mocked providers; live checks should record failures and source provenance explicitly.

Shared ML, physics and ingestion code is maintained under `backend/`. After changing those modules, run `python scripts/sync_training_bundle.py` to refresh the portable training copy. Do not edit the mirrored implementation independently.

## Scientific changes

Describe the mechanism, cite primary sources and distinguish published equations from project assumptions. Include appropriate boundary or regression checks. Feature construction must remain label-blind; evaluation events and calibration blocks must remain isolated from fitting. Retain failed cases and disclose changes to evaluation populations.

A pull request should explain the resulting behavior, relevant validation and remaining limitations. Improvements to numerical targets alone are not sufficient evidence of scientific correctness.

## Repository contents

Keep credentials, workstation paths, local notes, raw provider responses, source dumps, generated reports, datasets and model weights outside version control. Publish concise evaluation summaries with their protocol and limitations. Third-party data and model licenses continue to apply.

Do not make operational warning, calibrated probability or globally validated performance claims without supporting independent evidence.
