# Randomized audit protocol

The 30-location pool and archetypes are fixed before selection. Seed 20261031 draws two locations per archetype; a fresh cryptographic random seed supplies the second draw. Every sweep adds all five named plain controls, yielding 15 rows per sweep. Selections and seeds are saved before data acquisition. Failed rows are retained.

Each location uses a frozen real-provider snapshot for baseline, randomized perturbation, causal control and agent parity checks. Raw inputs remain local in `artifacts/randomized_provider_snapshots.jsonl`; their hashes and provider timestamps are in the report. `--replay-report` together with `--replay-snapshots` reuses the original selections and telemetry; perturbations are regenerated under the current declared eligibility rules without claiming a new live acquisition. API/schema/briefing checks still execute during replay.

The plain controls require P95 score <.01 and zero maximum velocity after +300 mm. Every paired cell is checked for score decreases beyond 1e-6. Kinematic checks enforce Class 1 below 16 mm/year when FS >1.25 and M1 <.5, and mechanical yield ≤1 for speeds above .05 m/s. GeoJSON rings are matched to H3 coordinates, JSON values are finite, reliability provenance is present and the agent's returned scan matches the direct scenario scan.

Causal controls require the injected trigger to appear in telemetry and an eligible steep cell to have valid positive corresponding attribution plus a matching metric card. Cells with no steep eligible terrain are explicitly not applicable. A diagnostic also counts increased causal shares. This count is not a strict gate: physical heads can already be capped and normalized shares can plateau or change as other mechanisms activate.

The agent is forced to its deterministic fallback for repeatable tool/numeric-parity checks. This does not validate a newly trained Qwen model. Road-cut archetypes receive random trigger tests and existing toe-distance physics checks, but no claim of observed excavation geometry is made when OSM is unavailable.

This is a property-based scenario audit over a named pool, not a blind labeled prediction experiment. Full-system, randomized, boundary, forecast and surgical-regression engineering audits must pass on identical implementation/model/data fingerprints for combined readiness. Empirical benchmark scores are reported separately, including low scores and wide uncertainty intervals.

The protocol restricts randomized burning to rain/vegetated-fire archetypes with median RGB vegetation proxy >=.20 and warming to cryospheric archetypes. It separately tests the warm Blue Mountains +8 C counterexample, so restricting random draws cannot hide that boundary regression. Random samples are scenario tests, not evidence of historical failures.
