# Research, methods & data sources

TERRA-SHIELD brings terrain, weather, physical slope calculations and model comparisons into one interactive landslide exploration workspace. This guide connects the [public preview](https://terra-production-289057fe.up.railway.app/) and [project overview](../README.md) to the research and source products behind each result.

**Start here:** [Read a result](#reading-a-result) · [Choose a model](#prediction-models) · [Follow a scan](#how-a-scan-becomes-an-analysis) · [Understand the physics](#physical-methods) · [Inspect the ML](#the-active-machine-learning-engine) · [NASA and NISAR](#nasa-comparison-and-measured-radar) · [Find a paper](#research-library) · [Check a source](#data-catalogue) · [Review the evidence](#evaluation-and-reproducibility) · [Reproduce or extend](#reproducibility-and-research-priorities) · [Questions](#common-research-questions)

Implementation references were checked against canonical source code on **15 November 2026**. Selected primary publications and official provider documentation were checked on the same date. The retained benchmark values below are existing development results, not a new experiment. Provider availability, terms and product metadata can change.

| Your question | Useful starting point |
|---|---|
| “What does this number mean?” | [Reading a result](#reading-a-result) and [uncertainty](#what-the-uncertainty-envelope-contains) |
| “Where did the number come from?” | [Data catalogue](#data-catalogue), [source provenance](#a-source-record-that-can-be-audited), and the application's Sources panel |
| “What is the model actually calculating?” | [Physical methods](#physical-methods), [the active ML engine](#the-active-machine-learning-engine), and linked implementation files |
| “Is NASA or NISAR really connected?” | [NASA comparison and measured radar](#nasa-comparison-and-measured-radar) |
| “How well has it been tested?” | [Evaluation](#evaluation-and-reproducibility), [model card](../artifacts/MODEL_CARD.md) and [current status](current_status.md) |
| “How can I investigate or improve it?” | [Reproducibility and research priorities](#reproducibility-and-research-priorities), [training workflow](training_workflow.md) and [roadmap](../ROADMAP.md) |

## Reading a result

A scan is a structured estimate assembled from sources with different support. The map helps you find a location and compare the returned estimates; its visual detail does not certify the accuracy of the underlying soil, weather or hazard model.

| What you see | What it means | What to inspect next |
|---|---|---|
| Terra initiation score | A learned trigger score multiplied by the declared mechanical admissibility factor. A larger score ranks a cell higher under this model and state. | Selected model, source completeness, physical envelope and the benchmark population. Do not read a score of 0.8 as an established 80% chance of failure. |
| Factor of safety, FS | Available shear resistance divided by modeled driving shear for a reference regolith column. Below 1 means the assumed resistance is insufficient in that calculation. | Low/central/high envelope, material priors, depth, slope and pore-pressure head. An FS above 1 does not survey rock joints or rule out another failure mechanism. |
| FS low / central / high | A sensitivity envelope that contains the nominal result and the modeled fifth/ninety-fifth distribution quantiles. | The assumed standard deviations and covariance. These endpoints are not independently demonstrated confidence limits for a particular mountain. |
| NASA Today / Tomorrow | The selected official daily raster product sampled at its own native support. | Product availability, native pixels, retrieval time and the service's declared or undeclared metadata. |
| Radar coherence | A dimensionless measured radar observable when a valid raster sample is present. | Acquisition pair, polarization, source posting and masks. Low coherence alone does not identify landslide motion. |
| Runout corridor | A possible route under a specified release and rheology scenario. | Velocity, flow depth, friction, stop reason and whether the route reaches the scan boundary. |
| Gray or unavailable | No supported value for that field, source, model or time. | The reason and provider status. A missing value must remain distinct from a genuine zero. |

**A practical reading order:** check the active model and time; inspect source support; select a cell; read its physical envelope; then explore forecast or conditional movement. Use NASA's daily controls in its own view and Terra's hourly horizon controls in Terra analysis. Export retains the analysis/state identifiers and source records needed to discuss the same result later.

A few terms recur throughout the guide:

| Term | Meaning in this project |
|---|---|
| Observation | A quantity measured by an instrument, subject to processing, support and quality limitations. A radar coherence raster is derived from instrument data; it is not a borehole strength test. |
| Model product | An estimate produced by an upstream physical/statistical model, such as numerical rainfall, SoilGrids texture or GloFAS discharge. Authentic delivery does not turn it into a local observation. |
| Proxy | A substitute indicator, such as catalogue-derived shaking or an RGB image index. Its relationship to the desired variable requires separate evidence. |
| Reference prior | An exposed project assumption used when direct site inputs are absent. It keeps conditional calculations usable and must not be labelled measured. |
| Native support | The source's spatial footprint, temporal accumulation, depth and sampling geometry. Displaying it in a smaller hexagon does not create finer information. |
| Calibration | A fitted relationship evaluated for a defined population. Parameter uncertainty propagation and a verified API response do not by themselves provide probability calibration. |

## Prediction models

| Main model | What the application uses | How to interpret it |
|---|---|---|
| **Terra In House Model** | An XGBoost source/physical feature model, coupled to an explicitly reported mechanical admissibility calculation. The active feature contract is `physics_informed_initiation_v2_features`. | A research initiation score supported by a conditional physical column, uncertainty envelope and source provenance. It is not an established globally calibrated probability. |
| **NASA LHASA** | Independently sampled NASA Earthdata GIS hazard and susceptibility services. The LHASA 2.0 research paper is the scientific comparison reference. | An external model view, retained separately from Terra. The served endpoint does not establish its exact model version or issue time. A Terra reference fallback is labelled separately and is not an official NASA prediction. |

The rainfall-only model remains a useful baseline. Physics-only and weather-plus-terrain diagnostics help isolate the contribution of individual components. These diagnostics do not become additional independent observations.

NASA's [LHASA repository](https://github.com/nasa/LHASA) and [Stanley et al. (2021)](https://doi.org/10.3389/feart.2021.640043) describe a global data-driven nowcasting system. Connecting to its map services is different from reproducing its training, establishing a matched comparison or demonstrating parity. The application records native raster samples instead of presenting repeated H3 samples as independent NASA predictions. See the [NASA and radar integration guide](nasa_radar_comparison.md).

**NISAR is a source integration, not a third prediction model.** The code supports discovery, configured local products and automatic authenticated Harmony variable/spatial subsets. Pending provider jobs, missing authorization, native NoData and measured pixels are reported separately. Discovery metadata, acquisition times and product names are not measured deformation. The published Terra classifier has not been refitted with NISAR-labelled training features. Automatic extraction and a successful single-location measured check do not establish global measured SAR coverage. [ASF's product guide](https://nisar-docs.asf.alaska.edu/products-overview/) documents the products; the [integration guide](nasa_radar_comparison.md) records the application's actual handling.

## How a scan becomes an analysis

~~~mermaid
flowchart TD
    A[Place name or exact coordinate] --> B[Terrain tiles and H3 terrain cells]
    A --> C[Weather and environmental context]
    B --> D[Reference material and hydrologic column]
    C --> D
    D --> E[FS envelope and joint sensitivities]
    C --> F[Shared source features]
    E --> G[Learned trigger score and exposed mechanical factor]
    F --> G
    C --> H[Automatic provider-forced forecast states]
    D --> H
    E --> I[Conditional Voellmy routing]
    A --> J[Independent NASA daily raster sampling]
    A --> K[NISAR discovery and authenticated raster subsets]
    K --> L[Measured radar context and provenance]
    L --> M[Map inspection and verified assistant evidence]
    G --> M
    H --> M
    I --> M
    J --> M
~~~

The diagram separates the active classifier's inputs from radar evidence. NISAR context is available to inspection and the assistant; the shipped classifier has no fitted SAR weights. A separately documented, valid site calibration can constrain a compatible physical input, which then changes existing mechanical features. The application does not manufacture a generic coherence-to-strength conversion.

The main implementation path is [global ingestion](../backend/realtime/global_ingestor.py) → [reference assessment](../backend/api/reference_assessment.py) → [feature-based assessment](../backend/api/physics_informed_assessment.py), with [automatic forecasts](../backend/api/reference_forecast.py) and [conditional routing](../backend/physics/kinematics.py). NASA and radar have their own [comparison integration](nasa_radar_comparison.md). The browser and co-pilot bind outputs to the exact scan/state/cell rather than combining numbers from different requests.

## Physical methods

The live pipeline builds H3 terrain cells, retrieves source forcing and context, constructs a reference soil column, calculates stability and sensitivities, scores initiation, and routes conditional movement. Automatic 72-hour states use provider forecasts. A cell remains tied to its exact scan and scenario identifiers throughout the map and assistant.

| Component | Implemented relation or method | Evidence boundary |
|---|---|---|
| Terrain and drainage | Metric terrain derivatives and bounded specific contributing area; `metric_specific_area_bounded_1000m_v1`. TWI follows the contributing-area/slope index family. | Terrarium is a compiled elevation product. A drainage index is not a measured groundwater table or direct evidence of initiation. |
| Hydraulic properties | Saxton–Rawls texture/organic-matter pedotransfer relations. | These support hydraulic estimates, not the project's cohesion, friction or regolith-depth coefficients. |
| Reference column | `pedotransfer_reference_column_v1`: source soil properties plus declared strength, depth and hydrologic priors. | Cohesion, friction angle, depth and piezometric head are reference assumptions rather than borehole observations. |
| Infiltration | Iverson analytical pulse response with pulse superposition/recession, conductivity limitation and declared head bounds. Antecedent storage includes capacity, drainage, ET and overflow. | A simplified conditional column model; the code is not a complete TRIGRS implementation. Surface moisture does not directly measure full-depth saturation. |
| Stability | Effective-stress infinite-slope factor of safety, with vertical depth and explicit pressure-head convention. | Appropriate as a regolith reference calculation; not a complete model of rock cliffs, deep rotational failures, liquefaction or river undercutting. |
| Uncertainty | Joint local delta method `J Sigma J^T`; bounded-slope integration and conditional log-FS treatment near flat terrain. | Parameter/covariance sensitivity is not empirical event-frequency calibration or proven interval coverage. Structural error and spatial correlation remain additional uncertainty. |
| Initiation ML | Grouped fitting/calibration and monotonic trees using source, physical and interaction features. Mechanical admissibility is exposed separately from the learned score. | The coupling is an assumption-conditioned compatibility calculation; its effect must be evaluated rather than presumed beneficial. |
| Runout | Voellmy-family energy integration on the H3 graph with three friction/flow-depth reference scenarios. | Conditional on release. Centerlines/corridors are not a mass-conserving flow solution; entrainment, deposition, lateral spreading and complete terrain outside the scan are absent. |
| Other mechanisms | Applicable Newmark/Jibson displacement, Staley post-fire likelihood and Gartner conditional-volume relations remain in the codebase. | These require their own stable-block, shaking, catchment, severity and rainfall inputs. Earthquake magnitude, RGB imagery or a discharge ratio cannot replace them. |

The central stability expression is:

```text
FS = [c + max(gamma Z cos²(theta) - gamma_w psi, 0) tan(phi)]
     / [gamma Z sin(theta) cos(theta)]
```

Here `Z` and pressure head `psi` are metres, `c` is kPa, and unit weights are kN/m³. Conductivity is m/s and diffusivity m²/s. The physical output includes `[FS_low, FS_mid, FS_high]`, the nominal state, parameter spreads, Jacobian and variance contributions. Exactly zero driving shear has undefined FS; FS is not clipped to a cosmetic range. The [physical methods reference](physics_sources.md) gives the transient response, uncertainty conventions and routing limitations; the [API contract](../API_USAGE.md) documents returned fields.

Strength/depth transfer coefficients, initial storage, drainage/ET, covariance and runout scenarios are **project reference assumptions**. Their provenance and sensitivity remain visible even when no site measurements exist. Published methods make conditional global calculations possible; they do not make those coefficients locally measured or universally validated.

### Terrain, scale and drainage

The [terrain processor](../backend/realtime/dem_processor.py) decodes Terrarium elevations, mosaics the required tiles, calculates derivatives in metres and aggregates terrain into H3 cells. For Web Mercator tiles the ground spacing includes latitude; the implementation applies the local cosine factor to both metric axes. Slope follows the elevation-gradient magnitude, while aspect uses a circular mean when aggregated.

The bounded wetness index uses the following implemented form:

~~~text
specific_area = accumulated_upstream_area / local_contour_width
TWI = ln(specific_area / max(tan(slope), 0.005))
~~~

Contribution travels downhill along paths up to a fixed **1,000 m** support, with a halo intended to include possible contributing origins. The tangent lower bound regularizes the drainage index only; it is not a mechanical slope floor. Specific area has units of metres, and the logarithmic index depends on its stated unit/reference convention. This index is a bounded terrain descriptor, not a complete catchment simulation, groundwater observation or universal measure of drainage efficiency.

H3 resolution 9 averages roughly **0.105 km² per hexagon**, with area varying by location. Those cells cover terrain hundreds of metres across. The map's terrain pixels, the analysis cells, the soil patch, the weather grid and NASA's native pixels are different supports. For example, many displayed hexagons may share one weather history or one NASA pixel. Counting those hexagons does not create that many independent observations. The [H3 statistics](https://h3geo.org/docs/core-library/restable/) and [terrain registry](https://registry.opendata.aws/terrain-tiles/) provide the indexing/product background.

Known terrain limitations include the compiled source mosaic's varying vintage and vertical quality, interpolation/void handling, a provisional elevation-based land mask and Web Mercator's polar limits. Coastlines, very low land, cliffs, dateline sectors and drainage near domain boundaries require particular care. The [global source audit](global_source_audit.md) records these issues.

### Hydrology and the pressure response

The current source window requires **168 complete hourly timestamps** ending at a declared UTC cutoff. The latest 24 hours and preceding nonoverlapping 144 hours remain separate features. For live forecast-style weather the extractor adds rain and showers; for archive weather it uses the archive rain definition. Missing showers, gaps or duplicate required timestamps cannot silently become dry hours. See [the shared extractor](../backend/ml/rainfall_features.py).

Hydraulic properties use texture, organic matter and density through the Saxton–Rawls equation family. Sand/clay enter as fractions, while the cited hydraulic coefficients use **percent organic matter** internally. Soil organic carbon is converted to an organic-matter estimate with the project's fixed factor 1.724. That factor and a particle density of 2.65 g/cm³ are assumptions, not measured mineralogy or carbon composition for every site. The implementation bounds hydraulic/material parameters; these bounds are part of the model definition and should be included when reproducing a column.

The pressure response uses:

~~~text
t* = 4 D cos²(theta) t / Z²
R(t*) = sqrt(t*/pi) exp(-1/t*) - erfc(1/sqrt(t*))
head contribution for a pulse = Z (infiltration_flux / Ks) [R(t*_start) - R(t*_end)]
~~~

Here Z is **vertical depth**, D is diffusivity in m²/s, Ks is conductivity in m/s, time is seconds and head is metres. The response follows the analytical infiltration method discussed by [Iverson (2000)](https://pubs.usgs.gov/publication/70022499). The reference wrapper limits infiltration to conductivity and head to its declared geometric bound. The network-free [column solver](../backend/physics/reference_column.py) exposes the underlying pulse calculation independently.

NOW initialization is simplified and worth distinguishing from a full hydrologic model:

1. The preceding 144 hours advance reference storage hourly, with infiltration capacity, gravity drainage above field capacity, evapotranspiration and overflow.
2. A compatible 24-hour mean surface-moisture estimate adjusts at most the top **7 cm** of storage. It does not replace the water content of a metre-scale column or observe its water table.
3. The latest 24-hour rain total is represented as a pulse over the declared duration for the NOW pressure response. It does not reproduce every within-day rainfall burst as a separate pressure pulse.
4. Automatic forecast calculations retain that current pulse, propagate its recession and superpose future hourly forcing.

Uniform antecedent-total reconstruction is labelled when full historical pulses are absent. Numerical availability under a reference reconstruction is a different result from complete source-driven forcing. This implementation does not reproduce TRIGRS' complete finite-depth, unsaturated or boundary-condition treatment; [the USGS TRIGRS reference](https://www.usgs.gov/software/trigrs) is a comparison and method source, not a claim that Terra runs the original solver.

### Strength, depth and coefficient register

These values explain exactly how a usable reference column exists without a borehole. **The formula being explicit does not make its coefficients locally validated.** The current register below comes from [reference context](../backend/physics/reference_context.py), [the material engine](../backend/physics/multi_causal_engine.py) and [soil-property derivation](../backend/realtime/soilgrids.py). Sand S and clay C in the strength rows are fractions; theta is the actual slope angle.

| Quantity | Current default relation or value | Units / interpretation | Status |
|---|---|---|---|
| Porosity | 1 − bulk_density / 2.65 | Fraction; density in g/cm³ | Fixed mineral particle-density approximation |
| Friction angle | clip(24 + 14 S − 8 C, 18, 39) | Degrees | Project texture-to-strength prior |
| Soil cohesion | clip(4 + 16 C − 3.5 S, 2, 20) | kPa | Project texture-to-strength prior |
| Regolith depth | clip(2.8 exp(−0.85 tan(theta)), 0.60, 3.00) | Vertical metres | Project slope-dependent depth prior |
| Saturated conductivity | Hydraulic pedotransfer estimate, bounded to 10⁻⁷…10⁻³ | m/s | Published equation family with project bounds |
| Characteristic hydraulic capacity | From the declared retention approximation; diffusivity D = Ks / capacity | Capacity in m⁻¹; D in m²/s | Approximation, not measured diffusivity |
| Retention alpha | 0.10 by default | kPa⁻¹ | Project reference input |
| Basal drainage conductivity ratio | 0.01 of Ks in antecedent storage | Dimensionless | Project drainage assumption |
| Evapotranspiration | 2 | mm/day | Fixed reference storage loss |
| Surface-moisture update depth | min(Z, 0.07) | m | Deliberately shallow storage support |
| Cohesion standard deviation | 0.4 c + 0.5 | kPa | Declared uncertainty moments |
| Friction standard deviation | 4 | Degrees | Declared uncertainty moments |
| Unit-weight / depth spread | 8% / 35% of the nominal values | Their corresponding units | Declared uncertainty moments |
| Conductivity / diffusivity spread | 50% / 60% | Their corresponding units | Declared uncertainty moments |
| Initial-head spread | 0.15 Z cos²(theta) + 0.02 | m; positive spread even at zero nominal head | Declared uncertainty moments |
| Slope / 24-hour-rain spread | 1 degree / (0.2 rain + 0.5 mm) | Degrees / mm | Reference uncertainty; incomplete forcing has a broader rain spread |
| Correlation | cohesion–friction −0.25; conductivity–diffusivity +0.6; other default off-diagonal terms zero | Dimensionless | Project covariance structure |

The default material path does not apply unsurveyed root reinforcement, rock-joint weakening, bedrock-to-regolith strength adjustments or river-toe forces merely because context APIs returned a record. Explicit compatible ground inputs take precedence at their stated cell and time. Missing boreholes retain the broad reference calculation, complete sensitivity and conditional routing.

These assumptions need independent comparison against strength tests, depth surveys, hydraulic measurements and groundwater trajectories across representative regimes. A SoilGrids mean at 0–30 cm cannot validate a regolith-depth prior extending to 3 m. Likewise, a narrower numerical interval achieved by changing covariance is not evidence of improved prediction.

### What the uncertainty envelope contains

The local delta method evaluates a Jacobian J of FS with respect to the declared parameter vector and reports:

~~~text
variance(FS) ≈ J Sigma J^T
allocation_i = J_i (Sigma J)_i
~~~

Sigma is the declared joint covariance. Individual allocations can be negative because of covariance; they are signed contributions to this approximation, not causal percentages. The general first-order propagation method is described by [NIST TN 1297](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty). Terra's parameter moments, distributions and interpretation are its own reference model.

A direct linear-normal approximation of FS becomes unreliable near zero driving shear. The current implementation separately integrates a **truncated slope distribution from 0 to 89.999 degrees** with 32-point quadrature and conditional log-FS FOSM. It reports the unconditional local variance as a diagnostic rather than pretending that a divergent flat-slope Taylor expansion proves a large physical failure chance. Exactly zero driving shear has an explicitly undefined FS column, while otherwise computable columns remain available.

The displayed central FS is the **nominal calculation**, not necessarily the median of the integrated distribution. The returned distribution quantiles are separate. Low/high bounds include the nominal FS and modeled fifth/ninety-fifth quantiles. Contact loss, saturation bounds, parameter limits and omitted mechanisms restrict a local approximation; none of this establishes observed 90% interval coverage or calibrated event frequency.

### Forecast and runout interpretation

The [forecast code](../backend/api/reference_forecast.py) uses 72 complete future forcing records and maintains fixed current material priors. It calculates pressure/FS trajectories, rolling source windows and selected horizon states. Rainfall quantiles come from [ensemble ingestion](../backend/realtime/forecast_weather.py), with the deterministic liquid fraction or an explicit temperature phase proxy where needed.

P50 and P90 are **hour-by-hour pooled forcing quantiles**, not necessarily realizable trajectories of a single ensemble member. A P90 rain path is not a 90% landslide interval or a demonstrated worst case. If ensemble spread is missing, deterministic forcing is labelled; missing required liquid-rain records cannot become zero. The peak horizon selects the minimum sector nominal FS along the P50 trajectory. A possible threshold crossing is a conditional trajectory result, not an exact failure-time prediction. Forecast uncertainty also uses a declared 20% forcing-scale spread.

The current 72-hour projection does not forecast an earthquake, a rock-joint failure, future erosion or post-fire catchment runoff from weather alone. Those mechanisms remain within the global development objective and need appropriate observations/models rather than invented future inputs.

The conditional route evolves an energy head E = v²/(2g):

~~~text
dE/ds = sin(beta) − mu cos(beta) − 2gE/(xi h)
~~~

Distances and flow depth h are metres, velocity is m/s, mu is dimensionless and xi is m/s². Constant-edge integration avoids an arbitrary large-step drag update. The [reference runout wrapper](../backend/api/reference_assessment.py) exposes three scenarios: (mu, xi, h) = **(0.1, 1000, 1.0), (0.2, 500, 0.5), (0.3, 200, 0.25)**. These are project reference scenarios, not locally fitted rheology. [RAMMS friction documentation](https://ramms.ch/ramms-debrisflow/friction-parameters/) provides the model-family background and emphasizes calibration; Terra is not the RAMMS solver.

The route follows lower H3 neighbors and can start from rest when a release scenario is explicitly enabled. It does not conserve a release volume, resolve lateral flow, entrain material, model deposition or traverse arbitrary uphill rises with momentum. Boundary termination limits travel distance; mapped road/building intersections are exposure context, not occupancy, damage or loss estimates.

## The active machine-learning engine

The exact shipped contract is **physics_informed_initiation_v2_features**. [The approved model card](../artifacts/MODEL_CARD.md), [runtime manifest](../artifacts/runtime_manifest.json) and [implementation](../backend/ml/physics_informed.py) identify the artifact, feature order and fitting environment. Physics columns use their own pedotransfer_reference_column_v1 version; the two version names describe different components.

### Exact feature inventory

The model has **25 features**. Internal names are preserved here so a reviewer can trace a displayed result back to the implementation.

| Group | Features | Meaning / important unit |
|---|---|---|
| Source rainfall | `rain_24h_mm`; `rain_prior_6d_mm` | Latest 24-hour and preceding 144-hour liquid totals, mm |
| Source moisture | `soil_moisture_0_7cm_mean_m3m3` | Raw volumetric 0–7 cm mean, m³/m³; missingness is allowed through the declared model procedure |
| Terrain | `slope_deg`; `twi`; `profile_curvature` | Degrees, declared wetness index and metric terrain derivative |
| Physical envelope | `fs_mid`; `fs_low`; `fs_high` | Dimensionless nominal/reference-envelope quantities |
| Mechanical ratios | `shear_friction_ratio`; `cohesion_stability_number`; `pore_pressure_ratio`; `critical_pressure_ratio_deficit` | Declared stress/angle combinations, dimensionless |
| Rain deficit feature | `critical_rainfall_deficit_ratio` | Despite its internal name, the current implementation stores required rain / 1000: **metres of water**, not a dimensionless ratio |
| Physical reliability transforms | `fosm_reference_score`; `fosm_log10_p`; `log_fs_mid` | Reference FOSM probability-like score and logarithmic transforms; no empirical calibration is implied |
| Column/trigger context | `critical_rain_deficit_mm`; `pressure_head_m`; `newmark_displacement_cm`; `newmark_available` | mm, m, cm and an availability flag; unsupported displacement stays missing |
| Interactions and admissibility | `rain_to_critical_deficit_ratio`; `effective_shear_stress_surge`; `gravity_release_admissibility`; `critical_rain_attainable` | Rain/deficit ratio, rain-induced frictional-resistance loss in kPa, necessary-condition factor and attainability flag |

The “effective shear stress surge” field is a pressure-induced **loss of frictional resistance**, not a directly observed new shear load. Unattainable rain thresholds carry an explicit flag/missing value. Stable-block Newmark displacement requires positive supported PGA and an admissible initial state; an unsupported case is not encoded as measured zero shaking.

**No NISAR coherence, velocity, polarimetric ratio, NASA hazard label or assistant-generated value is among these 25 features.** Radar has a separate versioned context contract. Historical radar features cannot be created by attaching today's acquisition to an old event date.

### Fitting, explanation and mechanical coupling

The current trees use a binary-logistic objective and histogram training, with ordinary training-fold prevalence as the initial score. The feature-based architecture does not inject a per-row logit(FOSM) training offset. The retained small implementation uses 120 estimators, learning rate 0.04, monotonic feature constraints and bounded tree complexity; physics-informed inner selection considers declared depth/regularization candidates. These settings are source-code behavior, not a guarantee that they are optimal globally.

Component-disjoint inner partitions fit the learner and calibrate intercepts; outer components provide the reported development comparison. Static-prior groups receive a different **feature-sampling weight**, rather than a cosmetic rescaling of feature values. Native tree contributions describe the learned log-odds margin. They are not percentages of real-world causation, and they do not contain the separate mechanical multiplier.

Conceptually the returned in-house score is:

~~~text
learned_score = mean_across_fitted_folds(sigmoid(tree_margin + calibration_offset))
Terra_score = learned_score × mechanical_admissibility
~~~

The admissibility calculation estimates whether modeled gravity-driven shear can exceed cohesion after friction loss, using declared uncertainty moments. This necessary condition provides a physically motivated compatibility factor; it is not an empirical non-failure label, an independently calibrated release forecast or proof that adding mechanics improves a learned classifier. Its effect is tested separately in the results below.

The four diagnostic components are useful because they answer different questions:

| Component | Inputs / calculation | Purpose |
|---|---|---|
| Rainfall-Only | The two rain totals | Establish what rainfall alone explains on the same sampled cohort |
| Physics-Only | The reference physical reliability score | Inspect the prior-based mechanics without learned feature fusion |
| Weather+Terrain ML | Six source/terrain features | Compare a statistical model without derived physical features |
| Physics-Informed ML | All 25 features plus exposed admissibility coupling | Test whether conditional mechanics improve the source model |

The uncoupled **learned trigger score** is also retained so a poor mechanical coupling cannot be hidden by reporting only the final score. All models must use shared cohorts and held-out components for a meaningful comparison. The [active source/model contract](source_model_status.md) and [architecture report](feature_fusion_phase3.md) document the fitting details.

## NASA comparison and measured radar

### What “NASA LHASA” means in this application

The primary NASA view reads the official Earthdata GIS **Today**, **Tomorrow** and categorical susceptibility services at their native raster support. It does not relabel Terra's output as NASA's, infer extra lead times or treat repeated H3 samples as independent NASA pixels. The [NASA implementation](../backend/realtime/nasa_lhasa.py) keeps official values, project reference fallback and missing coverage separate.

The scientific comparator is **LHASA 2.0 as described by Stanley et al. (2021)**. The public raster service does not declare enough model-version or issue/valid-time information to identify every served pixel as that exact research release. A current upstream software version is not assigned to an undeclared raster. The service's spatial extent, masks and product semantics govern coverage; a global research model and an accessible public endpoint are different engineering objects. See [the methods guide](nasa_radar_comparison.md).

“Tomorrow” is a relative daily product, not a user-selectable exact T+24-hour issued forecast. NASA's view therefore has Today/Tomorrow controls rather than Terra's 48/72-hour controls, physical-column headline or runout. Terra physics remains inspectable by explicitly switching models at the location.

A fair LHASA comparison still requires a shared evaluation cohort, compatible event/background definitions, source data available at issue time, matching windows/support and coverage accounting. Connecting a NASA map layer establishes integration, not parity, superiority or reproduction of its validation.

### What NISAR contributes now

NISAR is a real radar source, with several different levels of evidence:

| Stage | Evidence available | What the application may claim |
|---|---|---|
| CMR discovery | Collection/granule identity, product type/maturity, acquisition interval and footprint | A product was discovered and may cover a point |
| Authenticated Harmony job | Advertised variable IDs, bounded area, submitted/reused processing job and provider status | Extraction is requested, pending or ready |
| Raster download and decoding | Actual subset bytes, georeferencing, native posting and masks | A raster was obtained and sampled |
| Valid sample at an H3 location | Finite in-range observable and its measurement provenance | A measured radar observable exists for that location/support |
| Separate documented site calibration | A real granule anchor, versioned relationship and sourced compatible physical input | That particular accepted calibration may constrain the column |
| Fitted SAR predictor | Requires a dated, isolated training/evaluation programme | **Not provided by the current approved classifier** |

The automatic path discovers GUNW/GCOV products, queries collection-advertised variable identifiers and requests bounded spatial/variable GeoTIFF subsets through [Harmony](https://nisar-docs.asf.alaska.edu/harmony/). It reuses matching jobs, including recovery through provider labels after a process restart. It does not download an unrestricted multi-gigabyte HDF5 for every map click. Source files are [nisar_processor.py](../backend/realtime/nisar_processor.py), [nisar_harmony.py](../backend/realtime/nisar_harmony.py) and [radar_context.py](../backend/physics/radar_context.py).

The two automatic observable paths retain distinct meanings:

- **GUNW coherence:** dimensionless co-polarized coherence from a stated acquisition pair. Native posting comes from the sampled raster; the official GUNW guide describes different layer postings, so the application must retain actual layer support rather than assign every product one universal resolution. Vegetation, viewing geometry, noise and temporal decorrelation can all affect coherence. [ASF GUNW guide](https://nisar-docs.asf.alaska.edu/gunw/).
- **GCOV polarimetric ratio:** HVHV / HHHH linear power from the same granule and matching native grid, with mask union and a positive denominator. The ratio is dimensionless, can exceed 1, and is not a volumetric soil-moisture measurement or generic vegetation-water index. [ASF GCOV guide](https://nisar-docs.asf.alaska.edu/gcov/).
- **LOS displacement or velocity:** supported only through suitable explicitly described measured products/time series, with units, reference, sign and timing. A single interferogram is not automatically an annual velocity estimate.

A sector can have valid samples while the selected point is masked. The interface reports **sampled H3 cells**, not independent radar pixels or percent of physically surveyed ground. Its acquisition dates describe the measured observable, not the newest unrelated granule in the discovery list. The newest product's mask may differ from an older product's; “newest” alone does not establish quality or local applicability.

Earthdata access is deployment-owned. A visitor's AI key does not grant NASA/ASF authorization. Pending jobs, rejected authorization, download/decoding failure, no compatible product and native NoData remain explicit. A completed provider job is not yet a measured result if its raster cannot be retrieved within the bounded request. A radar-only refresh reuses the current environmental forcing and creates a new immutable analysis state if source evidence changes.

Without a documented site relation, radar does not manufacture cohesion, friction, strain, groundwater head or narrower FOSM variance. The full broad reference column remains available. Future SAR-trained features need acquisition-time matching, spatial/event/time isolation, meaningful missingness and an ablation against the existing model; the present integration does not establish a SAR skill gain.

## Research library

These references identify the scientific foundations and comparison products. “Implemented” means a stated relation is used or adapted; it does not imply full reproduction of the cited system or its validation.

| Citation label in this guide | How to read it |
|---|---|
| Implemented or adapted | A particular equation, algorithm or product is used in the linked source. Only that stated use is claimed. |
| Comparison reference | An external system or benchmark informs the research objective. Its published skill does not transfer to Terra. |
| Applicability reference | A paper explains the required inputs/domain or a mechanism that needs further implementation/validation. Its presence does not mean the live interface predicts that mechanism. |

### Hydrology, soils and mechanics

| Reference | Role in TERRA-SHIELD |
|---|---|
| **Beven, K. J. & Kirkby, M. J. (1979).** *A physically based, variable contributing area model of basin hydrology.* Hydrological Sciences Bulletin, 24(1), 43–69. [DOI: 10.1080/02626667909491834](https://doi.org/10.1080/02626667909491834). | Contributing-area/wetness-index background. Terra's bounded terrain index is not a reproduction of the complete TOPMODEL hydrologic system. |
| **Saxton, K. E. & Rawls, W. J. (2006).** *Soil Water Characteristic Estimates by Texture and Organic Matter for Hydrologic Solutions.* Soil Science Society of America Journal, 70(5), 1569–1578. [DOI: 10.2136/sssaj2005.0117](https://doi.org/10.2136/sssaj2005.0117). | Implemented hydraulic pedotransfer foundation. |
| **Iverson, R. M. (2000).** *Landslide triggering by rain infiltration.* Water Resources Research, 36(7), 1897–1910. [DOI: 10.1029/2000WR900090](https://doi.org/10.1029/2000WR900090); [USGS record](https://pubs.usgs.gov/publication/70022499). | Analytical transient infiltration foundation. |
| **Baum, R. L., Savage, W. Z. & Godt, J. W. (2008; revised 2009).** *TRIGRS—A Fortran program for transient rainfall infiltration and grid-based regional slope-stability analysis, version 2.0.* USGS Open-File Report 2008–1159. [Report and software](https://pubs.usgs.gov/of/2008/1159/). | Infinite-slope/transient-pressure method reference and applicability context. Terra implements its own simplified solver. |
| **Poggio, L. et al. (2021).** *SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty.* SOIL, 7, 217–240. [DOI: 10.5194/soil-7-217-2021](https://doi.org/10.5194/soil-7-217-2021). | Research behind the predicted soil-property source, including its native uncertainty and global scale. |
| **Baum, R. L., Godt, J. W. & Savage, W. Z. (2010).** *Estimating the timing and location of shallow rainfall-induced landslides using a model for transient, unsaturated infiltration.* JGR Earth Surface, 115, F03013. [DOI: 10.1029/2009JF001321](https://doi.org/10.1029/2009JF001321). | Related hydrologic benchmark and future comparison: the current simplified Terra column does not implement the paper's full unsaturated treatment. |
| **NIST Technical Note 1297, Appendix A.** [Law of propagation of uncertainty](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty). | General background for first-order covariance propagation; not a validation of Terra's distributions or event probabilities. |
| **RAMMS / WSL–SLF.** [Voellmy debris-flow friction documentation](https://ramms.ch/ramms-debrisflow/friction-parameters/) and [Debrisflow manual](https://ramms.ch/wp-content/uploads/RAMMS_DBF_Manual.pdf). | Authoritative description of the dry-friction and velocity-squared-drag model family. Terra's graph routing and coefficients are its own conditional approximation, not RAMMS output. |

### Additional mechanism references

| Reference | Applicability |
|---|---|
| **Jibson, R. W. (2007).** *Regression models for estimating coseismic landslide displacement.* Engineering Geology, 91(2–4), 209–218. [DOI: 10.1016/j.enggeo.2007.01.013](https://doi.org/10.1016/j.enggeo.2007.01.013); [USGS record](https://pubs.usgs.gov/publication/70029910). | Regional Newmark displacement regressions with stated domains and substantial residual scatter; not site-specific design. The related [USGS SIM 3077 report](https://pubs.usgs.gov/sim/3077/) is a separate regional application. |
| **Jibson, R. W., Harp, E. L. & Michael, J. A. (2000).** *A method for producing digital probabilistic seismic landslide hazard maps.* Engineering Geology, 58(3–4), 271–289. [DOI: 10.1016/S0013-7952(00)00039-9](https://doi.org/10.1016/S0013-7952(00)00039-9); [USGS record](https://pubs.usgs.gov/publication/70022381). | Northridge-derived displacement-to-failure relation retained in legacy mechanics. Regional calibration is not evidence of a universal failure probability. |
| **Staley, D. M. et al. (2017).** *Prediction of spatially explicit rainfall intensity–duration thresholds for post-fire debris-flow generation in the western United States.* Geomorphology, 278, 149–162. [DOI: 10.1016/j.geomorph.2016.10.019](https://doi.org/10.1016/j.geomorph.2016.10.019); [USGS model documentation](https://ghsc.code-pages.usgs.gov/lhp/pfdf/guide/models/s17.html). | Post-fire model with catchment, burn-severity, soil and duration-specific rainfall requirements. |
| **Gartner, J. E., Cannon, S. H. & Santi, P. M. (2014).** *Empirical models for predicting volumes of sediment deposited by debris flows and sediment-laden floods in the transverse ranges of southern California.* Engineering Geology, 176, 45–56. [DOI: 10.1016/j.enggeo.2014.04.008](https://doi.org/10.1016/j.enggeo.2014.04.008); [USGS model documentation](https://ghsc.code-pages.usgs.gov/lhp/pfdf/guide/models/g14.html). | Conditional regional volume relations; not a globally calibrated release-volume law. |

### Global models, inventories and machine learning

| Reference | Role |
|---|---|
| **Stanley, T. A. et al. (2021).** *Data-Driven Landslide Nowcasting at the Global Scale.* Frontiers in Earth Science, 9, 640043. [DOI: 10.3389/feart.2021.640043](https://doi.org/10.3389/feart.2021.640043). | LHASA 2.0 scientific reference, including its data-driven global nowcasting approach. Its study results do not transfer to Terra's model. |
| **Juang, C. S., Stanley, T. A. & Kirschbaum, D. B. (2019).** *Using citizen science to expand the global map of landslides: Introducing the Cooperative Open Online Landslide Repository (COOLR).* PLOS ONE, 14(7), e0218657. [DOI: 10.1371/journal.pone.0218657](https://doi.org/10.1371/journal.pone.0218657). | Event-inventory foundation; reported points and missing reports require uncertainty and reporting-bias treatment. |
| **Chen, T. & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* KDD, 785–794. [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785); [author manuscript](https://arxiv.org/abs/1603.02754). | Gradient-boosted tree engine for the active model and ablations. |
| **Lundberg, S. M. & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions.* NeurIPS, 30. [Author manuscript](https://arxiv.org/abs/1705.07874). | SHAP explanation framework. Native tree contributions explain the learned margin; the separate mechanical multiplier is not a learned causal contribution. |

## Data catalogue

“Live” means a source was requested for the scan, not that every value was directly observed at that coordinate and instant. A usable record needs product/version, units, native spatial/time/depth support, acquisition or issue/valid time, retrieval time, transformations, uncertainty and fallback provenance. The [global source audit](global_source_audit.md) contains the detailed critique and delivery gaps.

| Source and official documentation | Use and native support | Interpretation / terms to retain |
|---|---|---|
| **AWS / Mapzen Terrarium** — [dataset registry](https://registry.opendata.aws/terrain-tiles/), [source attribution](https://github.com/tilezen/joerd/blob/master/docs/attribution.md) | Elevation tiles, slope, curvature and drainage; compiled elevation sources with varying native resolution/vintage. | Tile pixel spacing and H3 spacing are not local survey accuracy. Retain underlying source attribution and licences. |
| **Esri World Imagery** — [mosaic metadata guidance](https://support.esri.com/en-us/knowledge-base/how-to-view-the-world-imagery-basemap-metadata-in-arcgi-000018129) | Online basemap and visible-band image context. Acquisition dates/resolutions vary across the mosaic. | RGB-derived greenness/burn indices are not multispectral NDVI or dNBR. Preserve Esri/source credits and service terms. |
| **Open-Meteo Forecast** — [API](https://open-meteo.com/en/docs) | Hourly weather, shallow soil-moisture context and automatic forecast forcing. Forecast/grid support varies by selected upstream model. | Numerical weather/land-model output, not a rain gauge or local pore-pressure measurement. Scan-center forcing may be shared by many cells. |
| **Open-Meteo historical products** — [historical weather](https://open-meteo.com/en/docs/historical-weather-api), [historical forecast](https://open-meteo.com/en/docs/historical-forecast-api) | Retrospective extraction and issue-time research inputs. | Reanalysis, historical forecasts and live forecasts have different availability/semantics. A retrospective event-date series may contain information after failure. |
| **Open-Meteo ensemble** — [ensemble API](https://open-meteo.com/en/docs/ensemble-api) | Automatic 72-hour forcing spread where returned. | Ensemble spread is conditional on the supplied weather models; it is not total landslide forecast uncertainty. |
| **NASA POWER wetness** — [meteorological sources](https://power.larc.nasa.gov/docs/methodology/data/sources_meteorology/), [hydrology parameter definitions](https://power.larc.nasa.gov/docs/methodology/data/sources_hydrology/) | Terra requests GWETROOT/GWETPROF wetness indices. POWER documents MERRA-2 and a lower-latency GEOS-IT continuation on a 0.5° × 0.625° meteorological grid. | Keep the actual parameter/source/date. These wetness indices are not volumetric surface moisture, piezometric head or an automatically connected SMAP product. |
| **ISRIC SoilGrids** — [layer units](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html), [access/policy](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_02.html), [scale guidance](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_04.html) | Global 250 m predicted soil properties; the application aggregates a patch and thickness-weights available shallow layers for its reference column. | Predicted texture, carbon and bulk density are not borehole strength. Bulk density cg/cm³ × 10 → kg/m³; SOC dg/kg ÷ 10 → g/kg. CC BY 4.0; cite SoilGrids and transformations. |
| **USGS FDSN** — [earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/) | Earthquake origin/magnitude/depth/time catalogue. | Catalogue attenuation-derived local PGA is a project proxy, not station shaking or a ShakeMap sample. Event solutions can be revised. |
| **NASA EONET** — [API](https://eonet.gsfc.nasa.gov/docs/v3), [scope](https://eonet.gsfc.nasa.gov/what-is-eonet) | Natural-event context, including wildfire/landslide event records. | A fire event does not establish a burn-severity map, perimeter or complete inventory; no returned event is not proof of absence. |
| **NASA COOLR** — [repository overview](https://gpm.nasa.gov/applications/landslides/coolr), [reporter FAQ](https://gpm.nasa.gov/applications/landslides/reporter-faq) | Contributed reports and inventories for historical context/training research. | Report locations may be impact/report points rather than initiation crowns. Preserve timing/position confidence and original inventory terms before redistribution. |
| **GloFAS via Open-Meteo** — [flood API](https://open-meteo.com/en/docs/flood-api) | Modeled river-discharge context at approximately 5 km support. | Point sampling does not resolve local bank stress. Terra's discharge-ratio index is not a return period or direct landslide probability. |
| **Macrostrat** — [data services](https://macrostrat.org/docs/data-services), [maps](https://www.macrostrat.org/docs/maps) | Mapped geology/lithology context at source-dependent map scale. | Bedrock map labels do not measure weathered regolith strength. Retain source-map citation and source-specific rights. |
| **OpenStreetMap / Overpass** — [API](https://wiki.openstreetmap.org/wiki/Overpass_API), [copyright and ODbL](https://www.openstreetmap.org/copyright) | Mapped roads, buildings and other exposure geometry. | Mapping completeness varies. Geometry does not establish occupancy, vulnerability or engineered road-cut geometry. Attribute contributors and respect database terms. |
| **NASA LHASA Earthdata GIS** — [Today](https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/LHASA_Hazard_Today/ImageServer), [Tomorrow](https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/LHASA_Hazard_Tomorrow/ImageServer), [susceptibility](https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/Global_Landslide_Susceptibility/ImageServer) | External native raster comparison, approximately 30 arc-second support and limited service extent. | Record service metadata and native pixels. Exact served version/issue time is undeclared; Tomorrow is a relative product, not an arbitrary +24/+48/+72-hour forecast. Susceptibility categories are not probabilities. |
| **NISAR / ASF / Earthdata** — [products](https://nisar-docs.asf.alaska.edu/products-overview/), [Earthdata access](https://nisar-docs.asf.alaska.edu/earthaccess/), [Harmony](https://nisar-docs.asf.alaska.edu/harmony/), [GUNW](https://nisar-docs.asf.alaska.edu/gunw/) | CMR discovery and optional measured raster ingestion/subsetting. Product, acquisition pair, polarization, quality masks and physical units determine support. | Coherence is dimensionless; a single interferogram is not a velocity time series. LOS interpretation needs reference/sign/time conventions. A power ratio is not vegetation water content. Credentials alone do not provide continuous global measured coverage. |
| **Earthdata CMR** — [search API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html), [token help](https://urs.earthdata.nasa.gov/documentation/for_users/user_token) | Collection/granule discovery and access metadata. | Keep discovery/retrieval timestamps separate from measured-product freshness. Tokens stay server-side and out of source control. |
| **Open-Meteo geocoding** — [API](https://open-meteo.com/en/docs/geocoding-api) | Place-name lookup for navigation. | A missing name does not invalidate raw coordinates. Place centroids do not define a landslide initiation footprint. |

Open-Meteo's [terms](https://open-meteo.com/en/terms) and [plans](https://open-meteo.com/en/pricing) distinguish data attribution from access to its hosted API: the free service has non-commercial usage/rate restrictions and no uptime guarantee. A public commercial deployment needs an appropriate access arrangement. Do not infer unrestricted imagery, inventory or source-map redistribution from a public endpoint.

NISAR [urgent-response products](https://nisar-docs.asf.alaska.edu/urgent-response/) offer a distinct expedited pathway when available; they are not a user-triggered guarantee of a fresh acquisition anywhere. ASF's [current product overview](https://nisar-docs.asf.alaska.edu/products-overview/) distinguishes Beta, Provisional and Validated collections; record maturity and processing version, and avoid treating changes between processing generations as landscape change. IMERG and SMAP appear in the LHASA research context and future source planning. They must not be listed as active Terra measurements merely because NASA POWER or LHASA services are connected.

### Depth, time and unit support in practice

The source table is a catalogue, not a claim that all channels independently drive the approved classifier. The exact 25-feature inventory above defines its learned inputs. Other channels provide contextual or conditional-mechanism evidence, with their own applicability rules.

| Displayed or stored quantity | Native/returned unit | Transformation or support used here | Main caution |
|---|---|---|---|
| Current rain totals | mm per hourly interval → mm accumulation | 24 hours plus the preceding 144 hours; no overlap | The UTC cutoff and product definition matter as much as the number |
| Future liquid rain | mm per hour | Rain + showers when present; otherwise a declared precipitation/snow-water partition where supported | A snow-phase proxy adds uncertainty; missing required values are not dry weather |
| Open-Meteo surface moisture | m³/m³ | Mean of available required 0–7 cm values over 24 hours | Numerical land-model state, not a local probe or deep water table |
| POWER GWETROOT/GWETPROF | Wetness index/fraction, potentially shown as percent | Separate relative-wetness context | Do not average it with volumetric water content and call the result saturation |
| SoilGrids sand/clay | g/kg | Divide by 1000 for fractions, or by 10 for displayed percent | Globally predicted texture, not surveyed slip-plane material |
| SoilGrids bulk density | cg/cm³ | Multiply by 10 for kg/m³; divide kg/m³ by 1000 for g/cm³ | Distinguish mapped integer units from conventional units |
| SoilGrids organic carbon | dg/kg | Divide by 10 for g/kg; infer organic-matter fraction with the declared 1.724 factor | Organic carbon is not itself a measured cohesion input |
| GloFAS river discharge | m³/s | Daily modeled discharge context from the provider's selected river/grid | Not a surveyed bank shear stress or local flood return period |
| Catalogue-derived PGA | g | Project event/distance attenuation proxy | Not measured station shaking or an official ShakeMap sample |
| Radar coherence / HVHV:HHHH | Dimensionless | Native masked raster values / same-grid power ratio | Neither is soil strength or direct annual deformation |
| LOS displacement / velocity | mm / mm/year when supported | Requires the product/time-series reference and sign convention | LOS is a viewing-direction quantity, not automatically downslope motion |
| FS / pore-pressure head | Ratio / m | Derived from the declared material, depth and water-state reference | Derived calculation, not a separate measurement source |

For example, a SoilGrids bulk-density raster value of **130 cg/cm³** becomes **1,300 kg/m³ = 1.30 g/cm³**. A SOC value of **150 dg/kg** becomes **15 g/kg**, then a reference organic-matter fraction of 15 × 1.724 / 1000 ≈ **0.02586**. These are unit/derivation examples, not observations from a scanned site. [ISRIC's layer definitions](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html) document the mapped units.

SoilGrids retrieval currently requests four properties across three shallow layers, averages valid raster pixels in a **3 km patch** and thickness-weights the **0–5, 5–15 and 15–30 cm** layers by 5:10:15. A property needs all three layers for its completed shallow mean. Partial delivery is preserved; missing fields can use explicitly labelled topographic priors. The current path does not propagate SoilGrids' provided prediction quantiles into a measured site distribution. [ISRIC's scale guidance](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_04.html) recommends broad regional/global applications and cautions against local/farm interpretation.

Weather and soil may be shared across many cells. A hillslope map can therefore vary because of terrain and reference geometry while its forcing/material source support remains coarser. The mathematical precision of a displayed FS does not imply that each decimal digit reflects independent local information.

### Timing, latency and upstream changes

Keep four clocks separate: **measurement/acquisition time**, **forecast issue time**, **valid/accumulation time** and **retrieval time**. Downloading an older image today does not make it today's observation. A catalogue's latest granule may not be the one supplying a particular measured cell value. An API cache needs its original source age, not a refreshed observation label.

Open-Meteo live forecast output, archived forecast output and historical reanalysis are distinct products. Shared extraction code aligns feature definitions; it does not remove model-resolution changes, retrospective assimilation, revisions or forecasting error. Historical event-date extraction can include weather after the failure time when only a reported date is known. That limitation belongs in evaluation, not just the source footer.

POWER's official documentation now describes MERRA-2 together with GEOS-IT low-latency continuation. Terra must retain the returned product/date rather than assume every current value belongs to one immutable upstream generation. POWER also documents other hydrology products, including SMAP and IMERG; their availability upstream does not mean Terra requests them. The exact parameter/product path determines what is connected. [POWER source documentation](https://power.larc.nasa.gov/docs/methodology/data/sources_meteorology/) and [hydrology definitions](https://power.larc.nasa.gov/docs/methodology/data/sources_hydrology/).

NASA/ASF product maturity and processing generations matter too: Beta, Provisional and Validated collections can differ. Sensor repeat cadence does not guarantee a usable local acquisition at every revisit, and processing/access latency is additional to sensing time. Product version changes, masks or geocoding changes must not be mistaken for landscape change.

### A source record that can be audited

A defensible exported value should retain this information, where the provider actually declares it:

| Record field | Why it matters |
|---|---|
| Provider and endpoint/product identity | Distinguishes an authentic upstream product from a project fallback or unrelated NASA product |
| Collection, granule, model/run and processing version | Identifies the source generation and permits later revision checks |
| Variable, units and depth/polarization/reference | Prevents mixing soil moisture with wetness indices or LOS velocity with displacement |
| Native footprint/grid/posting and aggregation | Shows the scale of evidence before it is displayed on H3 cells |
| Acquisition/issue/valid interval and retrieval time | Separates freshness, lead time and caching |
| Quality mask, unavailable reason and coverage | Prevents NoData or an empty catalogue query from becoming a confirmed physical absence |
| Transformation and coefficients | Makes unit decoding, depth averaging, phase partition and derived ratios reviewable |
| Uncertainty and applicability | Distinguishes provider uncertainty from project spread and omitted mechanisms |
| Live/cache/fallback/local status and origin | Prevents a fallback from silently becoming a live measurement |
| Content checksum where bytes are retained | Identifies the exact raster/model payload; a hash does not establish scientific validity |
| Scan/state/cell identity | Binds map, inspection, forecast and assistant claims to the same computed result |

Not every current provider supplies every field, and not every missing field has been resolved in this implementation. The source audit records those gaps. Undeclared source versions and valid times should remain undeclared, rather than inferred from a repository release or the time of the HTTP request.

## Software and assistant

| Layer | Primary references | What it contributes |
|---|---|---|
| Browser workspace | [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs/), [React](https://react.dev/), [Vite](https://vite.dev/guide/) | Interactive imagery/terrain, H3 overlays, cell inspection and application UI. |
| Spatial indexing | [H3 documentation](https://h3geo.org/docs/), [resolution statistics](https://h3geo.org/docs/core-library/restable/) | Hierarchical hexagonal indexing. Resolution 9 averages about 0.105 km² per hexagon; H3 cells are not 30 m measurement pixels. |
| Geospatial processing | [Rasterio](https://rasterio.readthedocs.io/), [GDAL](https://gdal.org/), [pyproj](https://pyproj4.github.io/pyproj/stable/), [PROJ](https://proj.org/), [h5py](https://docs.h5py.org/) | Raster/vector access, coordinate transforms and scientific product decoding. Native dependency versions affect reproducibility. |
| ML and numerics | [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/stable/), [NumPy](https://numpy.org/doc/), [SciPy](https://docs.scipy.org/doc/scipy/) | Training, calibration, diagnostic metrics, tree contributions and physical numerical calculations. |
| API and transport | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [HTTPX](https://www.python-httpx.org/) | Stateful scan/chat API and concurrent provider access. Runtime/deployment requirements are in the [setup](SETUP.md) and [deployment](deployment.md) guides. |
| Co-pilot | [Connection guide](assistant_connections.md), [retained Qwen research](legacy/local_llm_training.md) | Personal-key OpenAI, Gemini, Anthropic, Groq, OpenRouter or compatible hosted prose, guarded by the server's state/numeric checks. Deterministic verified templates/actions remain available without a key. |

The LLM explains computed state; it does not supply replacement measurements, factor-of-safety values or hazard labels. `VERIFIED_LLM_PROSE` means the returned claims passed the application's numeric/state checks; `VERIFIED_TEMPLATE` identifies deterministic fallback. Neither status validates the underlying geophysical model. The public transport uses request-scoped personal keys; the browser keeps them in session storage, and the server does not persist or fund them. Adapter/security contracts are tested offline; a given provider/model's actual availability and answer quality require its own authenticated evaluation.

Dependency manifests specify installable package constraints. Upstream packages, model revisions and data products retain their own licences; the root software licence does not relicense them. The retired GPU-specific fine-tuning workflow is retained under [legacy documentation](legacy/local_llm_training.md), not required for live scans or a hosted co-pilot.

## Evaluation and reproducibility

The scientific objective is a defensible global validation programme. Current experiments are useful development evidence, with proxy labels and restricted cohorts. They do not establish operational warning skill, global probability calibration or superiority to NASA LHASA.

| Retained experiment | Rainfall-only AUC / Brier | Coupled Terra fusion AUC / Brier |
|---|---|---|
| Initiation cohort: 186 fitting rows, 93 manual point proxies plus 93 weather-selected temporal absence controls; 10 components, 5 outer folds | 0.8952 / 0.0864 | 0.9026 / 0.1183 |
| Separate multi-regime proxy experiment: 510 eligible rows, 34 components | 0.7842 / 0.1772 | 0.6866 / 0.2451 |

### How to read the metrics

ROC-AUC measures ranking discrimination on the stated positive/background sample; Brier is the mean squared error between the returned score and those sample labels. Brier depends on calibration and the sampled class balance. A high AUC on a balanced report/absence cohort is not a global annual landslide probability, and it does not establish a useful public-warning false-alarm rate.

The complete retained component comparison makes the effect of coupling visible:

| Component | 186-row temporal CV: AUC / Brier | Separate 510-row proxy CV: AUC / Brier |
|---|---:|---:|
| Rainfall-Only | 0.8952 / 0.0864 | 0.7842 / 0.1772 |
| Physics-Only | 0.5931 / 0.4413 | 0.4795 / 0.4960 |
| Weather+Terrain ML | 0.9043 / 0.0911 | 0.7713 / 0.1816 |
| Learned fusion before mechanical coupling | 0.8940 / 0.0965 | 0.7767 / 0.1847 |
| Coupled Physics-Informed ML | 0.9026 / 0.1183 | 0.6866 / 0.2451 |

These are retained results from [current status](current_status.md) and [feature-fusion evaluation](feature_fusion_phase3.md), not scores recomputed for this documentation update. The 510-row experiment fits separate benchmark models; it is not the training cohort of the shipped 186-row artifact.

Higher AUC and lower Brier are preferable. The coupled model has worse Brier than rainfall-only in both experiments and worse discrimination in the broader proxy experiment. The learned, uncoupled fusion and the other ablations are reported separately in the [full architecture/evaluation report](feature_fusion_phase3.md). The earlier failed fixed-offset experiment is preserved in [research history](research_history.md).

The initiation reconstruction retained 1,611 contexts but used only 186 rows for fitting; 71 unresolved footprints were excluded from supervised fitting and 16 development-holdout contexts had been inspected. Report positions, uncertain failure time, weather-selected controls and possible post-failure forcing restrict interpretation. Such controls are not confirmed non-failures, and inspected development data are not an untouched final test.

Future comparisons need shared cohorts, whole-event/spatial/time separation, representative backgrounds, issue-time inputs, coverage/missingness accounting and calibration checks. Forecast availability is not forecast validation. Source-provider outages, fallback paths and failed experiments remain part of the evidence.

| Reproducibility entry point | Contents |
|---|---|
| [Active model card](../artifacts/MODEL_CARD.md) and [runtime manifest](../artifacts/runtime_manifest.json) | Exact approved runtime artifact, checksum, feature compatibility, provenance and limitations. |
| [Current status](current_status.md) and [active source/model contract](source_model_status.md) | Present engineering/scientific state and separation of learned scores from mechanical coupling. |
| [Feature-fusion evaluation](feature_fusion_phase3.md) | Architecture, cohorts, ablations, metrics and unresolved scientific work. |
| [Training/rebuild workflow](training_workflow.md) | Canonical scripts and local reproducibility; the published runtime model avoids requiring a retrain for installation. |
| [Research history](research_history.md) | Commit-pinned historical methods, old models and unsuccessful experiments. |
| [Legacy randomized audit protocol](legacy/randomized_audit_protocol.md) | Historical sampling/audit procedure, retained for provenance. |
| [Roadmap](../ROADMAP.md) | Global validation and operational development goals. |

Tests establish mathematical/engineering consistency for covered cases; they do not establish geographic skill. Frontend readiness, source delivery, predictive validity and operational readiness are separate claims. Raw corpora, private notes, provider archives, unapproved weights and credentials remain outside Git; the approved small runtime artifact is the documented exception.

## Reproducibility and research priorities

### Follow a result back to its implementation

| Question | Canonical entry point |
|---|---|
| Which source window entered the model? | [rainfall_features.py](../backend/ml/rainfall_features.py), [global_ingestor.py](../backend/realtime/global_ingestor.py) |
| Which elevations and metric derivatives were used? | [dem_processor.py](../backend/realtime/dem_processor.py) and the returned terrain version |
| Which soil values were predicted, converted or replaced? | [soilgrids.py](../backend/realtime/soilgrids.py) and property-level source provenance |
| What values, spreads and covariance defined FS? | [reference_context.py](../backend/physics/reference_context.py), [reference_column.py](../backend/physics/reference_column.py) |
| Which exact feature order and fitting method are active? | [physics_informed.py](../backend/ml/physics_informed.py), [runtime manifest](../artifacts/runtime_manifest.json) |
| Which pressure/forcing path created a future state? | [reference_forecast.py](../backend/api/reference_forecast.py), [forecast_weather.py](../backend/realtime/forecast_weather.py) |
| Which release assumptions created a route? | [reference_assessment.py](../backend/api/reference_assessment.py), [kinematics.py](../backend/physics/kinematics.py) |
| Which NASA pixels or NISAR rasters supplied a value? | [nasa_lhasa.py](../backend/realtime/nasa_lhasa.py), [nisar_processor.py](../backend/realtime/nisar_processor.py), [nisar_harmony.py](../backend/realtime/nisar_harmony.py) |
| Which claims and actions did the co-pilot verify? | [state_copilot.py](../backend/llm/state_copilot.py), [assistant guide](assistant_connections.md), [API reference](../API_USAGE.md) |

The approved runtime artifact is **901,814 bytes**, SHA-256 **ff70502e82cdd3cb57dbfc141478336c144234f6ed6eb1d059393aac377fcf0c**. It is included for CPU inference after a normal clone. Its model card records fitting provenance, aggregate corpus/holdout hashes and compatible dependencies. A matching artifact hash identifies the bytes; it does not validate a new site or make untrusted pickle files safe. The [setup guide](SETUP.md) explains installation without retraining, and [training_workflow.md](training_workflow.md) explains research reproduction.

To reproduce a scientific result, retain more than a screenshot: the code commit, model hash, source/product metadata, exact scan/state/cell identifiers, acquisition/issue/cutoff time, transformations, ground overrides, masks, forecast forcing path, component grouping, exclusions and evaluation script/report. Record unavailable providers and rejected inputs alongside successes. Tests using fixtures must be labelled separately from live delivery checks.

### What a stronger global evaluation needs

| Priority | Required evidence | Why it comes before a stronger performance claim |
|---|---|---|
| Better event labels | Dated initiation areas/crowns and timing/position uncertainty | Report or impact points do not consistently identify the failed source slope |
| Representative backgrounds | Spatially and temporally sampled non-event exposure, with known observation opportunity | A catalogue omission or weather-selected quiet day is not confirmed non-failure |
| Whole-event, spatial and time isolation | All related records/neighboring footprints kept in one partition; final evaluation selected before outcome inspection | Prevents event duplication, geographic leakage and holdout-driven choices |
| Forecasts as issued | Archived model runs and source latency available before the prediction time | Retrospective reanalysis or later rain can leak information after failure |
| Prior and hydrology tests | Independent strength/depth/groundwater data plus pressure/uncertainty coverage checks | Mechanistic equations cannot compensate for incorrect local parameters |
| Matched LHASA comparison | Shared cohorts, support/time conventions and reported missingness | Independent published metrics are not directly comparable |
| NISAR ablations | Correct dated acquisitions, quality masks, availability controls and held-out regimes | Demonstrates whether radar adds useful information beyond metadata/terrain correlation |
| Runout validation | Independent release areas, mapped runout/velocities and boundary-aware comparison | Conditional routing availability does not establish movement accuracy |
| Operating coverage | Prospective latency, outage, provider-cost and regional coverage records | A successful demonstration is not a sustained 24/7 service guarantee |

Regional case studies support this global programme. They should not become location-specific score gates, favorable terrain/date searches or invented stable labels. Failed experiments, poor calibration, unavailable sources and unsupported regimes remain part of the research record. The current broader fusion result underperforms rainfall-only; improving it requires evidence and controlled comparisons rather than a cosmetic threshold.

## Common research questions

<details>
<summary><strong>Are the numbers real?</strong></summary>

Some are measured observables, some are upstream model predictions, some are project-derived proxies and some are declared priors. The source catalogue and each scan's provenance distinguish them. An authentic API response establishes delivery of that product, not perfect local accuracy. A complete physical column is useful conditional reasoning; it is not a borehole log.

</details>

<details>
<summary><strong>Why calculate physics without boreholes?</strong></summary>

Published hydraulic/mechanical methods and explicit broad reference priors allow a conditional column and sensitivity envelope at remote terrain. Missing in-situ measurements should not permanently disable the platform. The obligation is to label assumptions, expose uncertainty/applicability and validate them, while allowing compatible site observations to replace them.

</details>

<details>
<summary><strong>Does more map detail mean better predictions than NASA?</strong></summary>

No. Smaller display cells, nominal DEM posting and 3D imagery describe presentation or input support. Predictive advantage needs a matched independent evaluation. NASA pixels, weather grids and soil predictions retain their native support even when displayed on a fine map.

</details>

<details>
<summary><strong>Is NISAR just a reference link?</strong></summary>

The implementation supports real authenticated subset extraction, downloaded raster decoding, valid per-cell samples and source hashes/provenance. Whether a particular scan has measured pixels depends on acquisition coverage, permissions, processing, transfer and native masks. That measured context is distinct from fitting SAR features into the current classifier, which has not occurred.

</details>

<details>
<summary><strong>Does the AI assistant create the hazard prediction?</strong></summary>

The scientific engine computes it. A personal-key provider may explain the active evidence and propose the authorized action plan, while the server checks numeric bindings and app commands. Deterministic verified templates/actions work without a key. Verification of prose is not validation of the hazard model.

</details>

<details>
<summary><strong>Can the 72-hour view tell me the exact time a slope will fail?</strong></summary>

It projects the declared current material/water state under downloaded future forcing. Its threshold crossings and peak horizon are conditional model outputs. Forecast-as-issued skill, interval coverage and useful warning lead time have not been independently established, and future non-weather triggers are not inferred from rainfall alone.

</details>

<details>
<summary><strong>Can I reproduce everything from GitHub?</strong></summary>

The physical source, tests, approved small runtime model, model card, checksum manifest, methods and aggregate results are published. Raw corpora, provider archives, unapproved/legacy weights and credentials remain local. Reproducing the fitted research cohorts needs the retained source data and its rights/provenance; using the included runtime model does not require retraining.

</details>

## Credits and terms

Original TERRA-SHIELD code and documentation are [MIT-licensed](../LICENSE). Read [third-party notices](../THIRD_PARTY_NOTICES.md) for the distinction between project code, dependencies, observations, imagery, inventories and model assets. Credit the original product, provider, version/acquisition and processing changes when exporting results; check each original inventory's redistribution terms before publishing raw records.

The project builds on the work of NASA, ISRO, ASF, USGS, ISRIC, ECMWF/GloFAS, Open-Meteo, terrain contributors, Esri, Macrostrat, OpenStreetMap contributors and the open-source scientific/geospatial communities. Their inclusion is source attribution, not institutional endorsement.
