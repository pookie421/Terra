# TERRA-SHIELD vision and roadmap

## Ambition

Build a global, on-demand terrain intelligence workbench that connects slope stability, triggering mechanisms, movement scenarios and source evidence in one inspectable workflow. A user should be able to search a place, inspect a slope, understand the locally relevant mechanisms, and compare a baseline against rain, earthquake, warming or burn scenarios without manually assembling environmental inputs.

The intended audiences are researchers, meteorological analysts and, after independent validation, local emergency-management teams. The project began with a NASA Space Apps Challenge ambition. It is independently developed and does not imply NASA or WMO endorsement.

The five mechanisms are hydro-meteorological, seismic/tectonic, cryospheric, post-wildfire/biotic and anthropogenic/fluvial. The long-term goal is useful decision support across these regimes—not a promise of exact failure probabilities, velocities or arrival times at every point on Earth.

## Current foundation — implemented

The v6 backend provides H3 terrain, asynchronous public data feeds, local mechanical FS, constrained/group-calibrated XGBoost, raw-booster SHAP, provisional kinematics/runout, provenance, a CLI, FastAPI endpoints, a numeric-audited Qwen tool workflow and a portable GPU-training bundle. The [current evidence](docs/current_status.md) records what passed and what did not. No production frontend or target-PC GPU execution is claimed.

## Next: verify the RTX 5070 Ti workflow

- Execute CUDA XGBoost/Optuna on the copied local dataset; record hardware, dependency versions, split IDs and artifact hashes.
- Train the requested Qwen2.5 QLoRA adapter, retain export manifests, test GGUF loading and real tool-call behavior in Ollama.
- Re-evaluate new weights on frozen evaluation sets; never reuse CPU baseline readiness to certify GPU artifacts.
- Treat tuning CV as a development metric. Plan a fresh, untouched evaluation before drawing generalization conclusions.

Completion evidence: reproducible target-host logs, usable model artifacts, explicit conversion outcomes and fresh artifact-matched audit reports.

## Phase 4: build the terrain explorer

The proposed frontend stack is Next.js, MapLibre GL and deck.gl, subject to measured rendering and accessibility needs. This is planned work, not a shipped interface.

1. Search a place or select coordinates and display scan coverage, timestamps and provider status.
2. Toggle local FS, research score, uncertainty, slope and trigger layers; distinguish source susceptibility from downstream impact.
3. Inspect a hexagon with the six localized metric cards, source/proxy information and separate critical-region/hotspot statistics.
4. Compare the same baseline snapshot against rain, PGA, warming and burn changes. Mark forecast scenarios separately from current conditions.
5. Overlay provisional runout and mapped roads/buildings, with unknown geometry/exposure shown as unknown.
6. Add the tool-grounded co-pilot and meteorological/local-EOC briefing views. Camera actions may use validated UI commands; prose must not silently change analysis inputs.

Completion evidence: end-to-end tests against real API payloads, visible missing-data states, keyboard-accessible inspection, traceable scenario inputs and measured rendering/latency distributions. Frame-rate and scan-time ambitions become measured targets, not marketing guarantees.

## Scientific work before stronger claims

| Workstream | What must improve | Evidence required |
|---|---|---|
| Terrain and hydrology | Independently review metric pixel spacing, derivative conventions, H3 sampling, specific-area/TWI units and viewport dependence | Analytical terrain fixtures and comparison with established geospatial tools |
| Labels and splitting | Separate initiation footprints from report pins; strengthen negative labels and scarce fire/toe-cut support | Source-linked datasets and untouched regional/event/temporal tests |
| Mechanisms | Validate material, snow/ice, post-fire runoff and toe-erosion assumptions | Domain-specific studies and sensitivity/ablation results |
| Probability and uncertainty | Evaluate the final deployed blend, base rates, abstention and spatial dependence | Reliability/precision-recall analysis, calibration and uncertainty coverage under shift |
| Kinematics and exposure | Validate routes, speeds and travel times; replace missing geometry/occupancy with surveyed evidence where possible | Independent runout/impact inventories and uncertainty bounds |
| New live feeds | Assess historical/live consistency of river/geology inputs and their adjustments | Reproducible feature pipelines and out-of-sample ablations |
| Co-pilot | Evaluate claims beyond numeric matching, tool fidelity and fallback behavior | Held-out grounded dialogue tests with expert review |

A green engineering audit does not close these research questions. We will not tune geographic exceptions, select features using labels, or redefine empirical success to improve a showcase table.

## Reliability and eventual decision support

Reduce sequential acquisition costs, instrument cold/warm latency percentiles, and preserve live/cache/fallback distinctions. Keep provider outages visible. Version scientific inputs and outputs so a briefing can be traced to a specific scan and model. Define human-review and abstention procedures before any partner pilot.

Any future operational deployment requires external geotechnical review, local validation, documented responsibilities and appropriate institutional authorization. CAP exports remain Test/Private in the present project. There is no committed deployment date or operational-readiness claim.
