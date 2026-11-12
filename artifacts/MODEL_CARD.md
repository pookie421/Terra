# Bundled runtime model

The small pretrained artifact in this directory makes the current ML engine available after a normal Git clone. No retraining, GPU, retained corpus or legacy model downloads are required for runtime inference.

| Item | Value |
|---|---|
| File | `physics_informed_models.pkl` |
| Size | 901,814 bytes |
| SHA-256 | `ff70502e82cdd3cb57dbfc141478336c144234f6ed6eb1d059393aac377fcf0c` |
| Contract | `physics_informed_initiation_v2_features` |
| Terrain contract | `metric_specific_area_bounded_1000m_v1` |
| Physical contract | `pedotransfer_reference_column_v1` |
| Components | Rainfall-Only, Physics-Only, Weather+Terrain ML, Physics-Informed ML |
| Execution | CPU inference; CUDA and Qwen are optional, separate paths |
| Machine-readable record | [runtime_manifest.json](runtime_manifest.json) |

## Install and check

Keep the file at the repository-root `artifacts/physics_informed_models.pkl` path. Install the compatible runtime environment:

```sh
python -m pip install -r requirements.txt -c constraints-runtime.txt
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/api/v1/health`. Both `model_ready` and `physics_reference_available` should be true. The model loads when the backend starts; restart the server after replacing it. The full physical engine is source under [`backend/physics/`](../backend/physics/), not a missing model download.

The artifact uses Python/joblib serialization and XGBoost memory snapshots. Use the pinned **XGBoost 2.1.4**, **scikit-learn 1.6.1**, **NumPy 2.0.2**, **SciPy 1.13.1** and **Joblib 1.5.3** stack. The retained fit used Python 3.9.6 on CPU; published installation checks target Python 3.11 and 3.12. This is not a promise of compatibility with arbitrary later versions. XGBoost distinguishes portable model IO from version-sensitive pickle snapshots in its [serialization documentation](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).

Verify the file before loading it:

```sh
python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('artifacts/physics_informed_models.pkl').read_bytes()).hexdigest())"
```

Only load trusted artifacts from this repository/version. Pickle is executable serialization; a matching checksum identifies these bytes, not the safety of an unrelated replacement.

## Training provenance

The retained source pool contains 1,611 observations. The active initiation fit uses **186 rows: 93 manual-point initiation proxies and 93 matched temporal-absence controls**, across **ten components**. Five outer component folds provide the development comparison; sixteen rows remain in the inspected development holdout. Seventy-one unresolved failure footprints are excluded from initiation supervision, rather than treated as verified crowns or stable neighboring terrain.

The model has 25 source, mechanical and interaction features. Grouped inner selection and disjoint calibration are fitted within training partitions. The learned trigger score uses ordinary fold-prevalence initialization. A separate declared gravity-release admissibility calculation couples the learned score to reference mechanics. [Methods and evaluation](../docs/feature_fusion_phase3.md) explain the cohort, assumptions and limitations.

The manifest records hashes for the model, feature frame, retained observation corpus and holdout membership. It contains aggregate evidence rather than source rows. The fitted object contains trees, offsets, fitting diagnostics and hashed group identifiers; it does not package the raw provider archive, report text, source coordinates or training examples. Structural inspection is not a formal privacy guarantee.

Source families include NASA COOLR reports/inventories, Open-Meteo historical weather and compiled Terrarium terrain. NASA describes COOLR as a collection of NASA, volunteer and external contributions, and permits research use of the Landslide Reporter Catalog with citation: Juang, Stanley & Kirschbaum (2019), *Using citizen science to expand the global map of landslides: Introducing COOLR*, [doi:10.1371/journal.pone.0218657](https://doi.org/10.1371/journal.pone.0218657). See [NASA's source description](https://gpm.nasa.gov/applications/landslides/coolr), [research-use FAQ](https://gpm.nasa.gov/applications/landslides/reporter-faq) and the [project source audit](../docs/global_source_audit.md).

## What the numbers establish

| Evaluation | Rainfall-only AUC / Brier | Coupled fusion AUC / Brier |
|---|---|---|
| 186-row temporal development CV | 0.8952 / 0.0864 | 0.9026 / 0.1183 |
| Separate 510-row multi-regime proxy experiment | 0.7842 / 0.1772 | 0.6866 / 0.2451 |

The second experiment fits separate benchmark models; it is not the shipped artifact's training cohort. Rainfall-only has better Brier in both comparisons and outperforms fusion in the broader experiment. These retrospective, sampled-population results do not establish global probability calibration, untouched predictive skill or LHASA parity. Report pins, absence controls, timing uncertainty, only ten initiation components, hydrologic/material priors and source-domain differences limit interpretation. Full metrics and omissions remain in [current status](../docs/current_status.md); failed architectures remain in [research history](../docs/research_history.md).

Use this artifact for research and inspection of conditional estimates. It is not an operational warning model, a borehole measurement, a validated runout model or a certification that an individual slope is safe. Physics and uncertainty remain assumption-conditioned; FOSM alone does not calibrate event probability.

## Distribution and scope

This exact runtime artifact is the explicit published exception to the project's default exclusion of local weights. Legacy models, raw corpora, provider archives, adapters, merged Qwen models and generated training bundles remain local. No custom fine-tuned Qwen weights are included or implied.

The root MIT license covers original project code and documentation. This model card records provenance and availability; it does not relicense upstream inventories, imagery or third-party model/data rights. Those terms and required acknowledgements remain applicable. No NASA or other source-provider endorsement is implied. See [third-party notices](../THIRD_PARTY_NOTICES.md).
