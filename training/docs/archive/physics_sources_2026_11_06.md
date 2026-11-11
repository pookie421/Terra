> Historical record. Superseded by the [current status](../current_status.md); statements below are not the active live contract.

# Active physical implementation — 10 November 2026

The current live path is `physics/reference_context.py` through `api/reference_assessment.py`. It has no measurement-only blockade or FS clip. It uses depth-limited surface storage assimilation, antecedent water balance, Iverson pulses, joint FS/log-FS FOSM and conditional Voellmy corridors. Strength/depth and covariance coefficients are declared project priors; published hydraulic/mechanical equation families do not make those coefficients measured or universally validated. See [the active equations and assumptions](../phase1_phase2_implementation.md).

The v14 description below is archived; its zero-head, clipped-FS and three-parameter uncertainty descriptions are not the current live contract.

# Reference equations, live approximations, and limits

Version `reference_columns_v14`, 5 November 2026. Equation verification, input integration, and predictive validation are separate. The pure column functions are in `backend/physics/reference_column.py`; the H3 pedotransfer wrapper is in `multi_causal_engine.py`. Passing the equation tests does not validate the wrapper's estimated material properties or observed warning skill.

## Soil and initial water state

SoilGrids `soc` in dg/kg is divided by 10 to obtain g/kg; organic matter fraction is `soc_g_kg * 1.724 / 1000`. Bulk density in cg/cm³ is multiplied by 10 for kg/m³. Missing raw fields are filled before deriving all fractions and porosity. See the [ISRIC unit conventions](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html).

The fallback is a static topographic prior, not a soil measurement. Live fallback uses the cell nearest the scan center, without viewport averages or weather. That one soil payload is currently shared across the scan. Historical rows use the labeled H3 cell, or nearest retained source coordinate, rather than the steepest nearby cell. Historical and live support are therefore still approximations; report coordinates need not identify initiation crowns.

Both hydrology and mechanics use `n = theta_s = 1 - bulk_density / 2.65`, with density in g/cm³. Absolute saturation is `S = theta / n` only when `0 <= theta <= n`. Missing or incompatible values remain unavailable. NASA relative wetness is a separate index. Neither index nor saturation is interpreted as observed groundwater head. The default initial head is an explicit zero-head scenario.

## Explicit column interface

```python
from backend.physics.reference_column import ColumnParameters, solve_column
column = ColumnParameters(
    cohesion_kpa=5, friction_angle_deg=35, unit_weight_kn_m3=18,
    conductivity_m_s=1e-5, diffusivity_m2_s=1e-4,
    depth_m=1, slope_deg=30, initial_pressure_head_m=0.1,
)
result = solve_column(column, [0, 21600, 86400], [(0, 21600, 5e-6)])
```

Depth `Z` is vertical, head `psi` is in metres, cohesion in kPa, unit weight in kN/m³, conductivity in m/s, and diffusivity in m²/s. For each infiltration pulse, the semi-infinite response is:

```text
s = 4 D0 cos²(theta) t / Z²
R(s) = sqrt(s/pi) exp(-1/s) - erfc(1/sqrt(s)); R(0) = 0
psi = psi0 + sum[Z (flux/Ks) (R(s_since_start) - R(s_since_end))]
FS = [c' + max(gamma Z cos²(theta) - gamma_w psi, 0) tan(phi')]
     / [gamma Z sin(theta) cos(theta)]
```

The factor 4 and slope transformation belong to this response convention. `D0*t/Z²` is dimensionless; substituting a conductivity alone is dimensionally wrong. The contact constraint removes tensile friction. The explicit solver has no pedotransfer, head cap, or FS display clip. Pulses must be ordered, nonoverlapping, and no larger than `Ks`. See [Iverson (2000)](https://doi.org/10.1029/2000WR900090) and [TRIGRS version 2](https://pubs.usgs.gov/publication/ofr20081159). This implements the semi-infinite analytical limit, not TRIGRS' full finite-depth/unsaturated solver.

The live wrapper retains estimated texture-dependent conductivity/strength and slope-dependent depth. It uses explicit capacity `C` in 1/m when supplied; otherwise `hc=1/(alpha*gamma_w)`, `C=theta_s/hc`, and `D0=Ks/C`. The latter is a characteristic-capacity approximation, not a measured retention derivative. Live infiltration is capped at `Ks`, head at `Z*cos²(theta)`, and reported FS at `[0.1,5]`. Moist slab weight is `(bulk_density + S*theta_s)*9.81`, fixed through rainfall perturbations. There is no median-TWI normalization, curvature slope uplift, scarp weakening, synthetic root/ice strength, or inferred toe-strength multiplier. Degree-day snowmelt and pedotransfer bounds remain disclosed approximations.

Hourly pressure uses finite pulse superposition, including recession from the preceding storm. No arbitrary exponential drainage or synthetic snow floor is applied. Temperature lapse adjustment requires an explicit reference elevation. Forecast crossings are scenario outputs, not validated failure times. Historical enrichment still uses a retrospective composite of storm and moisture extrema, rather than a fully simultaneous observed column state.

## Seismic reference

For a stable initial block, `ac/g=(FS_static-1)*sin(theta)`. For `0 < r=ac/PGA < 1`:

```text
log10(D_cm) = 0.215 + 2.341 log10(1-r) - 1.438 log10(r)
```

The log10 residual standard deviation is 0.510. `PGA <= ac` yields zero displacement; `FS_static <= 1` is outside this stable-block regression and yields an unavailable value. No .05 g floor, .60 g ceiling, or 120 cm displacement cap remains. See the [USGS presentation of Jibson (2007)](https://pubs.usgs.gov/sim/3077/downloads/3077_pamphlet.pdf).

The separate `0.335*(1-exp(-0.048*D_cm**1.565))` relation comes from a Northridge inventory calibration, not a universal probability law or the 2007 displacement regression. See [USGS OFR 01-324](https://pubs.usgs.gov/of/2001/0324/report.pdf). Pseudo-static loading is an explicit independent `kh` input; the wrapper no longer silently applies `0.6*PGA` as well.

Legacy numeric Newmark fields and the 14-feature matrix retain zero when the regression is unavailable. Use `newmark_applicable` and nullable `newmark_displacement_reference_cm` to distinguish that placeholder from modeled zero displacement.

## Post-fire reference

For the 15-minute Staley M1 relation:

```text
logit(P) = -3.63 + R15 * (0.41*T + 0.67*(dNBR/1000) + 0.70*Kf)
```

`T` is the catchment fraction both burned at moderate/high severity and sloping at least 23°, dNBR is the scaled catchment mean, and Kf is catchment soil erodibility. `R15` is millimetres in 15 minutes: intensities 16, 24 and 40 mm/h imply accumulations 4, 6 and 10 mm. The [USGS M1 documentation](https://ghsc.code-pages.usgs.gov/lhp/pfdf/guide/models/s17.html) defines these variables and coefficients. The wrapper requires explicit basin variables and 15-minute accumulation; cell imagery and three-hour rain do not establish them.

Gartner's conditional volume is:

```text
ln(V_m3) = 4.22 + 0.39*sqrt(I15_mm_per_hour)
           + 0.36*ln(burned_area_km2) + 0.13*sqrt(basin_relief_m)
```

This requires actual basin area/relief inputs; there is no H3-area substitution. See the [USGS Gartner model documentation](https://ghsc.code-pages.usgs.gov/lhp/pfdf/guide/models/g14.html). Missing basin inputs leave compatibility numeric outputs at zero with `postfire_basin_inputs_available=false`; zero is then unavailable, not an assessed absence of hazard.

## Scores, uncertainty, and movement

`p_failure_physical` denotes the FOSM diagnostic only. Its material spreads, normal approximation, and 0.05 minimum FS spread are assumed, not calibrated. `p_failure_seismic` and `p_failure_m1` stay separate. `p_xgb_calibrated` is an unshifted sigmoid score on the sampled inventory/absence population. `p_combined` is null. There is no mechanical gate, physical union, or default 15% field prior.

Legacy `p_failure` uses the ML score when a matching model is available, otherwise the disclosed FOSM diagnostic. Marginal viewport percentiles are not one physical cell. Added `hotspot_*` fields share one H3 index. Display thresholds and exposure-weighted ranking remain interface conventions, not validated warning thresholds or a WMO endorsement.

Conformal intervals describe binary-label residuals on separately reserved components; spatial exchangeability and global coverage are unproved. The API's normal spread interval uses calibrated-fold disagreement alone and has no calibrated coverage claim. SHAP explains raw booster margins; stress allocations are not identified causal effects.

No surveyed runout geometry is inferred. The default velocity remains an unavailable compatibility zero; an explicit travel-length scenario can evaluate a simple Coulomb energy relation. Existing terrain routing and display thresholds remain experimental, not a mass-conserving runout model. Ice, root strength, erosion, mapped dam geometry, and observed motion require separate evidence before supporting warnings.

## Evaluation protocol

All fitting/calibration folds use connected components of event identifiers and spatial blocks. Controls are retained externally sampled dates with inventory absence, never accepted by modeled FS or physical probability. Synthetic anchors are removed from this baseline. Original holdout IDs/dates/labels are preserved, including unscorable records. This holdout has been inspected during development.

ROC-AUC, Average Precision, Brier score, attainable empirical FPR points, coverage and omissions are reported separately for physics and ML. AP/Brier depend on the sampled prevalence. LHASA-inspired metric names do not constitute a NASA benchmark reproduction. Independent geographical/temporal evaluation and an external solver comparison have not been completed; current measured results are in [current_status.md](../current_status.md).
