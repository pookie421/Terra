# Scientific methods and assumptions

TERRA-SHIELD combines published equation forms with project-specific parameterizations. It is not a validated reproduction of an operational USGS model. The following describes the current implementation; source availability, equation applicability and predictive validation are separate questions.

## Terrain and features

Terrain is sampled from Terrarium tiles into H3 cells. Derivatives use metric pixel spacing. Local slope and FS are not overwritten by neighborhood maxima. TWI normalization depends on the sampled batch, leaving a context difference between historical event-cell features and viewport inference. Feature construction does not select terrain using labels.

Training and live inference use the same 14-feature schema. The legacy `rain_24h_mm` model column represents equivalent water, `rain + 1.35*melt`; API rainfall telemetry retains rainfall itself. `p_f_fosm` contains the FOSM component, while the physical blend uses the mechanism union.

## Slope mechanics and hydrology

`fs_dynamic` is the mechanical Mohr–Coulomb Factor of Safety. The lateral-resistance compatibility field is zero; a probability-derived post-fire yield index does not govern mechanical FS. Negative effective normal stress is clipped before friction is evaluated.

The unsaturated-strength term uses a van Genuchten suction-stress expression with assumed parameters, bounded apparent cohesion and saturation/storm tapers. These bounds and tapers are project choices. See [Lu and Godt 2008](https://doi.org/10.1029/2008WR006976) and [Lu et al. 2010](https://doi.org/10.1029/2009WR008646).

Let `W = rain + 1.35*snowmelt`. The initial wetting calculation applies:

```text
convergence = 1 + ((TWI / median(TWI))^0.28 - 1) * clip(W / 35, 0.08, 1)
storm_wetting = (1 - initial_saturation) * clip(W / 210, 0, 0.85)
local_saturation = clip((initial_saturation + storm_wetting) * convergence, 0.10, 0.99)
mobile_water = clip(W / 15, 0.04, 1) * clip((local_saturation - field_capacity) / max(1 - field_capacity, 0.15), 0, 1)
```

Positive head combines mobile-water storage and an approximate transient infiltration response, bounded by layer depth. These rainfall-conditioned heuristics do not solve a mass-conserving groundwater model or prove gravity drainage. Real antecedent groundwater can remain positive during dry weather. Restricted basal drainage, homogeneous properties and suction treatment require independent hydrologic validation.

[USGS TRIGRS version 2.0](https://pubs.usgs.gov/publication/ofr20081159) provides relevant physical context. The implemented coefficients are not verified transcriptions of that solver.

## Seismic displacement

Yield acceleration uses `clip((FS_static - 1) * sin(theta) * 0.75, 0.05, 0.60)` in units of g. `FS_static` retains current material, wetness and mapped-geology adjustments. The `0.75` factor, acceleration bounds and 120 cm displacement cap are project assumptions.

The displacement regression and empirical seismic probability are based on Newmark/Jibson forms. The multiplier is not part of the standard [critical-acceleration relation](https://pubs.usgs.gov/of/1998/ofr-98-113/ofr98-113.html), and the result is not a site-specific validation of [Jibson 2007](https://pubs.usgs.gov/publication/70029910).

## Post-fire and cryospheric mechanisms

The post-fire likelihood uses the Staley M1 logistic structure, with sub-hourly intensity and basin inputs estimated from coarser cell data. Root weakening, the vegetation mask and the slope activation threshold are project assumptions. RGB imagery is not measured dNBR or pre-fire NDVI. See the [USGS model documentation](https://ghsc.code-pages.usgs.gov/lhp/pfdf/api/models/staley2017.html) and [Staley et al.](https://doi.org/10.1016/j.geomorph.2016.10.019).

Conditional debris volume follows the [Gartner model form](https://ghsc.code-pages.usgs.gov/lhp/pfdf/guide/models/g14.html), substituting cell-based area and relief proxies for watershed measurements. It is not observed deposited volume or routed mass. Surface runoff may activate movement over a mechanically stable slab; this is distinct from Mohr–Coulomb failure.

Cryospheric eligibility uses baseline snow, temperature, latitude and elevation before scenario warming. Snow/cycle floors require an explicit eligible warming scenario; elevation alone does not demonstrate ice-filled joints. Temperature lapse and soil-to-joint mappings remain assumptions. [Mamot et al.](https://tc.copernicus.org/articles/12/3333/2018/) provides an ice-joint strength criterion within a specified temperature/stress domain; using its retention behavior for unsurveyed terrain remains a proxy, not evidence of local joint ice.

## Scores, attribution and uncertainty

The physical score is `1 - (1 - P_FOSM)*(1 - P_seismic)*(1 - P_M1)`. Dependence between mechanisms can invalidate this union approximation.

For model score `m`, physical score `p`, slope `theta` and mechanical `FS`:

```text
effective_FS = min(FS, 1.40 - 0.65*p)
G = sigmoid((theta - 8.5)/0.65) * sigmoid((1.35 - effective_FS)/0.12)
base = 0.65*m + 0.35*p
w = 0.35 + 0.50*sigmoid((1.05 - FS)/0.06)
yield_blend = (1-w)*m + w*p
score = G * (max(base, yield_blend) if FS <= 1.05 else base)
```

The final score is not separately calibrated. The effective-FS cap leaves the screen partly open even at high mechanical FS; stable-ground false alarms remain in the reference evaluation. The mixture does not algebraically guarantee monotonicity across arbitrary changes to physical inputs.

Critical P05 FS/P95 score summarize cells with effective slope ≥24°, or all cells when none qualify. Viewport percentiles and the exposure/kinematics-ranked hotspot are separate statistics. Low local initiation susceptibility does not establish immunity from upslope runout.

Causal shares combine normalized physical stress proxies with grouped absolute TreeSHAP contributions. Shared features are allocated across groups by a display rule. These are not identified causal effects; TreeSHAP explains raw booster margins rather than the final gated score.

Binary-label conformal intervals and the nominal 90% probability-spread proxy are separate outputs. The latter combines calibrated-fold disagreement with `phi(beta)*clip(sigma_FS,.03,.18)*(1-P_seismic)*(1-P_M1)`. That bounded term is a heuristic, not a delta-method standard error. Independence, normality, spatial exchangeability and nominal coverage are unverified. Neither interval is guaranteed narrow.

## Hourly projections and movement

The 72-hour solver uses fixed initial hydraulic properties, linear pulse superposition, a dry-hour drainage approximation and bounded snow storage. It is TRIGRS-inspired, not a full transient groundwater solver. Five forecast days are requested before selecting the first 72 future hours. Missing future soil temperature falls back to air with explicit provenance; missing air keeps forecasting unavailable.

[Open-Meteo](https://open-meteo.com/en/docs) liquid rain combines rain and showers, or subtracts snowfall water equivalent from total precipitation. Initial snow storage assumes a density ratio of 0.1. Hourly rain intensity is a sub-hourly proxy. Ensemble P50/P90 values are pointwise forcing quantiles, not coherent members or probabilities of failure. Missing spread uses a disclosed deterministic scaling.

Explicit storm injection resets forecast initial rain, intensity and pressure head to zero and labels the result `SYNTHETIC_CALM_NOT_LIVE`. Antecedent moisture and other triggers remain; the original scan and observed modeled score are retained separately. Threshold crossings are scenario results, not validated warning lead times.

Runout follows a simplified downhill energy graph with inclined distance and an explicit flow-depth term. Unknown dam geometry and catchment properties remain unavailable. Channel intersections alone do not establish impoundment. Speeds, routes, exposure and travel times require independent evaluation.

## Data quality and evaluation

NASA POWER wetness is retained as a dimensionless fraction and converted only at percent consumers. GloFAS provides modeled discharge; Macrostrat provides mapped units rather than measured strength. Neutral historical defaults for unavailable feeds create a training/inference context difference. OSM exposure and optical proxies are incomplete.

Forty simulated regularization anchors enter booster fitting at weight 0.25 and are excluded from calibration and scoring. Source-derived negatives represent inventory absence, not surveyed stability. A separately reported negative-filtered cohort changes the evaluation population and must not replace the original holdout as evidence of improvement.

See [evaluation results](current_status.md) and the [roadmap](../ROADMAP.md) for remaining validation work.
