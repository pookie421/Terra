# Monaco — correction status, 10 November 2026

The measurement-only abstention described below is superseded. Corrected raw-tile terrain, uncapped reference columns, FS envelopes, fusion scores and conditional release corridors now run without boreholes. The original 0.8215765 inventory score is preserved only as a forensic comparison. Corrected slopes and the new nominal cliff FS differ from the old assumptions; they were not forced to reproduce the old 1.3817. See [current implementation](phase1_phase2_implementation.md) and `artifacts/phase1_phase2_report.json` for the retained-input storm regression.

The investigation below records the historical defect and previous response, not the current engineering contract.

# Monaco: conflicting displays and unsupported physical inputs

Audit date: 6 November 2026. A fresh public-provider scan reproduced the screenshot's selected IDs and numerical values. Its complete input snapshot and before/after responses are retained locally in `artifacts/monaco_input_audit/`. The two screenshot cells are also retained in a small regression fixture.

## What the displayed values actually represented

| Display | Cell centre | DEM slope | Result and reason |
|---|---|---:|---|
| Inventory classifier selection, `893969a6a5bffff` | 43.7479233, 7.4402005 | 3.6982° | Score 0.8215765. TWI 18.1711 strongly increases this classifier's output. |
| Former “physics-selected” cell, `893969a4607ffff` | 43.7279142, 7.3834621 | 10.9195° | FS 3.7814181 under estimated strength/depth and an assumed zero initial pressure head. Selection actually came from the legacy ML/exposure ranking. |
| Lowest FS in the old 277-cell calculation, `893969a401bffff` | See retained response | 41.1147° | FS 1.3816947, showing that the old “physics-selected” label did not identify the least stable modeled cell. |

Both hexadecimal IDs are valid resolution-9 H3 grid cells. They are not placeholders. H3 packs the cell address into a hexadecimal identifier; unused resolution digits account for the repeated `f` suffix. See the [H3 index definition](https://h3geo.org/docs/library/index/cell/). The interface now shows coordinates, distance from the query, slope and an inspectable map link alongside the ID.

The classifier output and FS were different model quantities at different locations. FS is a resistance-to-driving-stress calculation, not a failure probability; a value above 3 is mathematically possible. The problem was treating an assumption-based column as a supported local stability result. The required material and pressure inputs are explicit in the [USGS TRIGRS formulation](https://pubs.usgs.gov/of/2002/ofr-02-424/ofr-02-424-508.pdf).

## Specific defects found

1. **Unsupported material inputs.** SoilGrids was unavailable. The wrapper supplied static texture/density priors, then inferred cohesion 8.5839 kPa, friction 26.3135° and depth 2.3765 m for the displayed FS cell. These were not site strength or depth measurements.
2. **Zero initial pressure head.** Groundwater pressure was not observed. The wrapper defaulted initial head to zero; its short rain response supplied only approximately 1.04e-10 m at that cell. Surface moisture does not establish that hydraulic initial condition.
3. **Misleading cell selection.** “Physics-selected” used the legacy calibrated ML/exposure ranking, not minimum FS. The separate source classifier selected another cell. The interface made the pair too easy to read as one assessment.
4. **False-looking zero display.** The FOSM value at FS 3.781 was approximately 8.98e-14 and was rounded to `0.000`. This number also depended on assumed uncertainty. It was never evidence of zero real-world failure probability.
5. **A second, actual numerical zero problem.** At the classifier-selected cell the legacy wrapper clipped FS to 5 and zeroed uncertainty derivatives at the clip, producing zero FOSM spread and a numerical zero tail probability. The current public path does not use that wrapper or its FS cap.
6. **Additional placeholders.** Missing-data responses had zero-valued metrics and equal 20% causal shares. Nonfinite JSON was converted to zero. CAP export could turn absent information into generic severity/certainty labels. These behaviors have been removed from the live interface.

## Corrections

Live ingestion no longer completes missing soil data with terrain priors. The live physical path requires explicit, sourced, time-valid cohesion, friction, unit weight, depth and pressure head, evaluated with the reference equation. Missing inputs return null with a reason. A real supplied zero remains valid; missing head never becomes zero. FS is not capped to force an expected range. No default uncertainty, velocity, failure time, exposure weight or causal percentage is supplied.

`cell_comparisons` joins classifier and physical values using the same H3 cell. Statistical selection is highest classifier output; physical selection is lowest FS among complete ground columns. Their coordinates and selection rules are explicit. Tiny nonzero values use scientific notation. The optional assistant uses a deterministic evidence briefing, and CAP warning export is unavailable without an operationally validated model.

The classifier's weights and scores were not tuned to suppress Monaco. The exact fold-level logistic/calibration decomposition shows that TWI is the main positive influence, while the two rain features reduce the log-odds relative to their fitted centring. Its labels include reported points and mapped failure footprints rather than consistently verified initiation crowns. A high output on a low-slope accumulation cell therefore exposes a target/label limitation. It is not a validated initiation-risk estimate. The live response explicitly leaves hazard probability and verdict null.

On the same retained Monaco inputs, the classifier remains 0.8215765 and FS is **unavailable**, because site ground inputs are missing. A separate fresh live scan confirms the same abstention behavior while retaining terrain and weather. This is a correction to the system's evidence requirements, not evidence that Monaco is safe or unsafe.

## Verification and remaining work

Regression checks cover each missing ground parameter, invalid provenance, expired/future dates, genuine zero observations, unclipped FS, rainfall scenarios, unchanged initial pressure, exact per-fold classifier explanations, null serialization, same-cell comparisons, blocked CAP export and assistant claim containment. Existing equation and historical-data/model parity checks remain separate; the v14 wrapper is accessible only as an explicitly marked archived reference regression.

The current engineering checks do not certify an emergency-warning system. Site measurements, reliable initiation labels, representative controls, prospective evaluation and an independently validated warning policy remain necessary. The parameter and API contracts are in [API_USAGE.md](../API_USAGE.md).
