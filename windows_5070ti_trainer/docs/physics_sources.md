> Current implementation: v6. The final v6 section supersedes earlier v5 descriptions of lateral resistance, probability-derived FS, lapse reference and uncertainty calibration. See [current status](current_status.md) and [roadmap](../ROADMAP.md) for validation limits and independent review priorities.

# Equation provenance and scope (physics v5)

The engine mixes published equation forms with requested scenario approximations. It is not a validated reproduction of a USGS operational model. Source availability, model applicability and predictive validation are distinct questions.

| Component | Implemented form | Evidence and applicability |
|---|---|---|
| Post-fire likelihood | `sigmoid(-3.63 + (.41 PropHM23 + .67 dNBR/1000 + .70 Kf) R15)`; active only for burn > .05 and slope ≥12° | Staley M1 logistic structure. Rain accumulation is in mm per 15 minutes. Inputs here are cell-scale proxies, not surveyed basin variables. |
| Conditional debris volume | `exp(4.22 + .39 sqrt(i15) + .36 ln(Bmh) + .13 sqrt(relief))` | Gartner emergency-model form: intensity mm/h, burned catchment area km², relief m. Output is conditional volume, not expected deposited volume or routed mass. |
| Ice-filled joint shear strength | `c=53.3−73.5T`, `mu=.42−.21T`, `tau=c+sigma*mu` (kPa) | Mamot Eqs. 4–6, implemented as a separate domain-checked helper. Requires known ice-filled joints, −8 to −0.5°C and 100–400 kPa normal stress. |
| Live cryospheric weakening | Cohesion factor `clip(1−.12*dT−.10*cycles,.14,1)` and friction factor `clip(1−.08*dT−.05*cycles,.40,1)` | Requested scenario factors, **not Mamot's published criterion**. Warming is an explicit slider delta; freeze-thaw and alpine snow floors are proxies. |
| Basal perching | Storage from infiltration minus assumed basal drainage, bounded by modeled layer head | Inspired by restricted basal drainage; **not the TRIGRS transient PDE solution**. Basal/saturated conductivity ratio defaults to .01. |
| Unsaturated strength | van Genuchten suction stress converted to apparent cohesion with tan(phi) | Alpha=.1 kPa⁻¹ and n=2 are assumed. Cap=8 kPa; smooth taper from effective saturation .75 to .85. Storm/melt suppresses suction smoothly over 0–25 mm; positive perched storage removes it. |

Staley source: [USGS model API](https://ghsc.code-pages.usgs.gov/lhp/pfdf/api/models/staley2017.html), [paper](https://doi.org/10.1016/j.geomorph.2016.10.019). The supplied 15-minute coefficients are implemented directly. `R15=max(.25*peak_3h_intensity_mmhr,.12*rain_24h_mm)` estimates sub-hourly rain from coarser inputs. Burned steep-area fraction, dNBR and erodibility are requested approximations. A burn proxy is not a measured dNBR raster. The 12° mask is a project design choice.

Gartner source: [USGS model guide](https://ghsc.code-pages.usgs.gov/lhp/pfdf/guide/models/g14.html). Here `Bmh=max(.105*burn,.01)` and `relief=max(Z*50*sin(slope),10)` substitute cell-area/depth proxies for watershed measurements. The paper's published uncertainty is not a confidence interval for these substitutions. No volume accuracy is claimed.

Mamot source: [open paper](https://tc.copernicus.org/articles/12/3333/2018/). The standalone published criterion is not automatically applied to shallow soil cells: joint ice, temperature and normal stress are not observed there. Treating hundreds of kPa of joint strength as shallow soil cohesion would be an unsupported extrapolation. Live weakening remains explicitly labeled a scenario approximation.

Basal drainage source: [Baum et al. TRIGRS, USGS 2008](https://pubs.usgs.gov/of/2008/1159/). The bounded storage term is added to the existing approximate Iverson response. This is not a coupled water balance solver, does not resolve perched-water geometry and can double-count water redistribution; it needs hydrologic validation.

Suction sources: [Lu & Godt 2008](https://doi.org/10.1029/2008WR006976), [Lu et al. 2010 closed-form SSCC](https://doi.org/10.1029/2009WR008646). The closed van Genuchten expression is associated with the latter paper; the 2008 paper motivates unsaturated slope stability. The saturation and storm tapers are project assumptions.

## Mechanical FS versus runoff yield index

`fs_mohr_coulomb` is the slope force-balance ratio, including the v5 finite-block lateral adaptation. `fs_postfire_yield_index` is a requested probability transformation: when M1 > .5, `clip(1.05−1.5*(M1−.5),.3,2.5)`; otherwise the inactive value is 5. `fs_dynamic` is the minimum of these values. Thus a low displayed dynamic value in a burn scenario may represent runoff entrainability rather than slab instability. It must not be interpreted as an independent Mohr–Coulomb calculation. The API exposes both values and a provenance note.

FOSM is computed from the **Mohr–Coulomb** component, avoiding reusing the probability-derived yield index as independent FOSM evidence. The physical score is `1−(1−FOSM)(1−Jibson)(1−M1)`. This independence approximation can overcount dependent mechanisms. It is not a calibrated global failure probability.

The old burn-dependent slab-depth, hydraulic-conductivity, soil-cohesion and driving-force multipliers have been removed. Root weakening `1−.90*burn` remains a stated proxy. Organic matter fractions are converted to percent before the Saxton–Rawls coefficient equations.

## Shared features and live scores

Both training and live feature construction use `rain + 1.35*melt` in the legacy `rain_24h_mm` model column. API weather rainfall remains actual rainfall plus the scenario increment. The `p_f_fosm` feature remains the separate FOSM component; the three-mechanism union is used in the physical blend. Only the rainfall feature name is a compatibility name for equivalent water.

The live blend uses `w=.35+.50*sigmoid((1.05−FS)/.06)` and `G*((1−w)*P_XGB+w*P_phys)`, with the retained mechanical gate `G=sigmoid((slope−8.5)/.65)*sigmoid((2.15−FS)/.18)`. Neither the gate nor the blend is empirically calibrated. Increasing physical weight does **not** algebraically guarantee monotonicity if the physical score is below the learned score. Randomized paired tests measure this explicitly; failed monotonic checks must remain visible.

Terrain remains local. Neighborhood maxima do not overwrite a flat cell's slope or FS. Local TWI normalization depends on the sampled batch; event-cell features and viewport inference therefore have a remaining context difference. No feature operator selects terrain based on the observed label.

## Interpretation limits

Live API causal shares normalize the physical stress proxies. The diagnostic helper can also combine these with absolute SHAP contributions; native TreeSHAP is exposed separately in API diagnostics. These are explanatory scores, not identified causal percentages. Runout is a simplified downhill energy graph, not a continuum debris-flow solver. Rapid branches require the combined yield index ≤1; stable values >1.25 resolve to zero motion. Snowpack, strength, basal properties and exposure are often assumed. The 3D UI, universal <2-second cold-start scan, uniform 30-m terrain and operational early-warning certification remain roadmap items.

## v5 boundary, finite-block, sampling and uncertainty audit

Cryospheric eligibility is evaluated on observed snow and **baseline** lapse-adjusted temperature before scenario warming: snow >.005 m, baseline temperature <=2.5 C, or elevation >=max(1800,4600-55|latitude|) m. The latitude line, .40 m elevation-scaled snow floor and minimum three cycles are requested scenario assumptions, not observed permafrost. Warming cannot create cryospheric eligibility in a warm, snow-free hill. Effective burn is raw burn times clip(vegetation/.22,0,1); this boundary uses the same vegetation proxy in training and live scans. Esri RGB is not pre-fire NDVI, so the fuel mask remains uncertain.

The supplied Milledge DOI [10.1029/2011JF002186](https://doi.org/10.1029/2011JF002186) belongs to Pudasaini's 2012 two-phase flow paper. The relevant finite-block source is [Milledge et al. 2014, MD-STAB](https://pmc.ncbi.nlm.nih.gov/articles/PMC4508911/), DOI 10.1002/2014JF003135. The requested lateral term `(2Z/W)*(c + .5*(1-sin(phi))*max(0,gamma*Z*cos(theta)-gamma_w*psi)*tan(phi))/kappa` is implemented in wet static, dynamic and dry-baseline strength, with partial derivatives included in FOSM. This is an **adaptation**, not a verified verbatim paper equation or the complete MD-STAB active/passive wedge model. Width 35 m and kappa=1 are defaults. Negative effective normal stress is clipped to avoid tensile friction. No curvature-to-kappa calibration is invented.

PINLS applies the requested prior FS >1.20 to negatives after identical feature construction for all labels. Both low (<15 degrees) and moderate/high slopes are retained and counted. The original 48 holdout row IDs remain fixed; a separately named filtered cohort is also reported. Filtering evaluation negatives selects an easier/different target population, so it cannot establish an unbiased improvement. The [Do et al. DOI](https://doi.org/10.1007/s40808-026-02884-8) was located, but its primary full text and exact threshold were not verified. We do not attribute 1.20 or improved scores to that paper.

Split conformal follows the finite-sample order-statistic construction in [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511). Thirteen fixed-hash spatial blocks (66 rows) are reserved before PINLS filtering, booster fitting, isotonic calibration and hyperparameter tuning. Nonconformity is the within-block maximum of |binary label - deployed gated score|. For n calibration blocks, rank=ceil((n+1)*.9); an unavailable order statistic yields the conservative bound q=1. The API interval is [max(0,p-q),min(1,p+q)]. Coverage requires exchangeable blocks, an unverified assumption for global spatial and scenario shifts. These are intervals containing future **binary labels**, not confidence intervals for true failure probability. Intersect with {0,1} for a classification prediction set; empty sets are possible. The fitted v5 q is almost 1, so intervals are effectively vacuous. This correctly exposes limited information rather than manufacturing narrow confidence bands.

[Schlogl et al. 2025](https://nhess.copernicus.org/articles/25/1425/2025/) motivates joint susceptibility/uncertainty displays. The four requested names are used; susceptibility >=.5 and interval width >=.5 are project display thresholds, not universal cutoffs from that paper. This implementation is not a reproduction of its ensemble uncertainty method.

For source-cell failure **scenarios**, [Larsen et al. 2010 Fig.2a](https://gis.ess.washington.edu/grg/publications/pdfs/Larsen_et_al_2010.pdf) gives V=10^(-.836)*A^1.332 for all materials (A m2, V m3). An H3 cell area is not an observed scar. The [Ermini-Casagli DBI as reproduced in a primary 2020 study](https://nhess.copernicus.org/articles/20/3179/2020/) is log10(Ac*H/V), using consistent SI units here. Equivalently Ac km2 and V millions of m3 can be used together; mixing km2 with m3 produces a six-order error. Empirical domains below 2.75 and above 3.08 are labeled, not site-certified. A TWI>=14.5 intersection exposes the source-area volume scenario and viewport-only contributing area. **Live DBI remains null** because dam height, deposited volume and full upstream catchment are unobserved. Supply these inputs to `cascading_screen(...,dam_geometry=...)` to calculate DBI; no dam height or deposition fraction is invented.

Optional SoilGrids and Overpass calls have separate two-second deadlines; COOLR participates in the concurrent acquisition barrier. Other providers, terrain decoding, imagery and response construction still consume time. Cold end-to-end timing is measured separately and cannot be inferred from the optional deadline tests.

## v6 directive changes

The lateral adaptation and probability-derived FS override are removed. Mechanical FS is Mohr-Coulomb; M1 enters the physical probability union separately. Mamot retention uses the requested mapped joint-temperature proxy and reference stress, not measured joint temperature or verified joint ice. Median-elevation lapse correction is a viewport assumption. Voellmy routing now uses inclined step length and explicit positive flow depth.

GloFAS supplies modeled regional discharge, not local gauges (https://open-meteo.com/en/docs/flood-api). Macrostrat supplies mapped lithology, not measured material strength. The requested +/-1.5 strength adjustments and discharge-driven toe scaling are scenario assumptions and are disclosed in the API. Missing feeds apply neutral adjustments. Historical training lacks these feeds and uses neutral defaults; this train/live context difference needs validation.

Ensemble-scaled calibration now uses per-cell residuals on the separately reserved calibration set. Cell dependence prevents a formal spatial 90% coverage claim. Wide intervals remain possible; widths are never capped to make the display look informative. Numeric prose validation permits the requested percentages, standard labels and rounding tolerance. It is not a semantic proof of factual road names or evacuation advice.
