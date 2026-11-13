# Research, methods & data sources

TERRA-SHIELD brings terrain, weather, physical slope calculations and model comparisons into one interactive landslide exploration workspace. This guide connects the [public preview](https://terra-production-289057fe.up.railway.app/) and [project overview](../README.md) to the research and source products behind each result.

**Navigation:** [Prediction models](#prediction-models) · [Physical methods](#physical-methods) · [Research library](#research-library) · [Data catalogue](#data-catalogue) · [Software](#software-and-assistant) · [Evaluation](#evaluation-and-reproducibility) · [Credits and terms](#credits-and-terms)

Implementation references describe the current repository. External documentation and bibliographic metadata were reviewed on **14 November 2026**; provider availability, licences and product versions can change.

## Prediction models

| Main model | What the application uses | How to interpret it |
|---|---|---|
| **Terra In House Model** | An XGBoost source/physical feature model, coupled to an explicitly reported mechanical admissibility calculation. The active feature contract is `physics_informed_initiation_v2_features`. | A research initiation score supported by a conditional physical column, uncertainty envelope and source provenance. It is not an established globally calibrated probability. |
| **NASA LHASA** | Independently sampled NASA Earthdata GIS hazard and susceptibility services. The LHASA 2.0 research paper is the scientific comparison reference. | An external model view, retained separately from Terra. The served endpoint does not establish its exact model version or issue time. A Terra reference fallback is labelled separately and is not an official NASA prediction. |

The rainfall-only model remains a useful baseline. Physics-only and weather-plus-terrain diagnostics help isolate the contribution of individual components. These diagnostics do not become additional independent observations.

NASA's [LHASA repository](https://github.com/nasa/LHASA) and [Stanley et al. (2021)](https://doi.org/10.3389/feart.2021.640043) describe a global data-driven nowcasting system. Connecting to its map services is different from reproducing its training, establishing a matched comparison or demonstrating parity. The application records native raster samples instead of presenting repeated H3 samples as independent NASA predictions. See the [NASA and radar integration guide](nasa_radar_comparison.md).

**NISAR is a source integration, not a third prediction model.** The code supports discovery and measured-product ingestion, including configured local products and authenticated bounded subsets. Discovery metadata, acquisition times and product names are not measured deformation. The published Terra classifier has not been refitted with NISAR-labelled training features. In the public deployment, metadata discovery must not be described as global measured SAR coverage. [ASF's product guide](https://nisar-docs.asf.alaska.edu/products-overview/) documents the products; the [integration guide](nasa_radar_comparison.md) records the application's actual handling.

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

## Research library

These references identify the scientific foundations and comparison products. “Implemented” means a stated relation is used or adapted; it does not imply full reproduction of the cited system or its validation.

### Hydrology, soils and mechanics

| Reference | Role in TERRA-SHIELD |
|---|---|
| **Beven, K. J. & Kirkby, M. J. (1979).** *A physically based, variable contributing area model of basin hydrology.* Hydrological Sciences Bulletin, 24(1), 43–69. [DOI: 10.1080/02626667909491834](https://doi.org/10.1080/02626667909491834). | Contributing-area/wetness-index background. Terra's bounded terrain index is not a reproduction of the complete TOPMODEL hydrologic system. |
| **Saxton, K. E. & Rawls, W. J. (2006).** *Soil Water Characteristic Estimates by Texture and Organic Matter for Hydrologic Solutions.* Soil Science Society of America Journal, 70(5), 1569–1578. [DOI: 10.2136/sssaj2005.0117](https://doi.org/10.2136/sssaj2005.0117). | Implemented hydraulic pedotransfer foundation. |
| **Iverson, R. M. (2000).** *Landslide triggering by rain infiltration.* Water Resources Research, 36(7), 1897–1910. [DOI: 10.1029/2000WR900090](https://doi.org/10.1029/2000WR900090); [USGS record](https://pubs.usgs.gov/publication/70022499). | Analytical transient infiltration foundation. |
| **Baum, R. L., Savage, W. Z. & Godt, J. W. (2008; revised 2009).** *TRIGRS—A Fortran program for transient rainfall infiltration and grid-based regional slope-stability analysis, version 2.0.* USGS Open-File Report 2008–1159. [Report and software](https://pubs.usgs.gov/of//2008/1159/). | Infinite-slope/transient-pressure method reference and applicability context. Terra implements its own simplified solver. |
| **Poggio, L. et al. (2021).** *SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty.* SOIL, 7, 217–240. [DOI: 10.5194/soil-7-217-2021](https://doi.org/10.5194/soil-7-217-2021). | Research behind the predicted soil-property source, including its native uncertainty and global scale. |
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
| **NASA POWER / MERRA-2** — [data FAQ](https://power.larc.nasa.gov/docs/faqs/data/), [hydrology sources](https://power.larc.nasa.gov/docs/methodology/data/sources_hydrology/) | Coarse modeled root/profile wetness context, approximately 0.5° × 0.625° native MERRA-2 support. | Relative wetness is not SMAP, volumetric surface water content or measured piezometric head. Retain product-specific source identity. |
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

## Software and assistant

| Layer | Primary references | What it contributes |
|---|---|---|
| Browser workspace | [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs/), [React](https://react.dev/), [Vite](https://vite.dev/guide/) | Interactive imagery/terrain, H3 overlays, cell inspection and application UI. |
| Spatial indexing | [H3 documentation](https://h3geo.org/docs/), [resolution statistics](https://h3geo.org/docs/core-library/restable/) | Hierarchical hexagonal indexing. Resolution 9 averages about 0.105 km² per hexagon; H3 cells are not 30 m measurement pixels. |
| Geospatial processing | [Rasterio](https://rasterio.readthedocs.io/), [GDAL](https://gdal.org/), [pyproj](https://pyproj4.github.io/pyproj/stable/), [PROJ](https://proj.org/), [h5py](https://docs.h5py.org/) | Raster/vector access, coordinate transforms and scientific product decoding. Native dependency versions affect reproducibility. |
| ML and numerics | [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/stable/), [NumPy](https://numpy.org/doc/), [SciPy](https://docs.scipy.org/doc/scipy/) | Training, calibration, diagnostic metrics, tree contributions and physical numerical calculations. |
| API and transport | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [HTTPX](https://www.python-httpx.org/) | Stateful scan/chat API and concurrent provider access. Runtime/deployment requirements are in the [setup](SETUP.md) and [deployment](deployment.md) guides. |
| Co-pilot | [Ollama API](https://docs.ollama.com/api), [Qwen2.5 project](https://github.com/QwenLM/Qwen2.5), [connection guide](assistant_connections.md) | Optional local or operator-configured compatible hosted prose generation, guarded by the server's state and numeric checks. Deterministic verified templates/actions remain available when the model is unavailable. |

The LLM explains computed state; it does not supply replacement measurements, factor-of-safety values or hazard labels. `VERIFIED_LLM_PROSE` means the returned claims passed the application's numeric/state checks; `VERIFIED_TEMPLATE` identifies deterministic fallback. Neither status validates the underlying geophysical model. The current transport is operator-configured; a universal per-request, personal-key multi-provider interface must not be claimed until implemented and verified.

Dependency manifests specify installable package constraints. Upstream packages, model revisions and data products retain their own licences; the root software licence does not relicense them. The retired GPU-specific fine-tuning workflow is retained under [legacy documentation](legacy/local_llm_training.md), not required for live scans or a hosted co-pilot.

## Evaluation and reproducibility

The scientific objective is a defensible global validation programme. Current experiments are useful development evidence, with proxy labels and restricted cohorts. They do not establish operational warning skill, global probability calibration or superiority to NASA LHASA.

| Retained experiment | Rainfall-only AUC / Brier | Coupled Terra fusion AUC / Brier |
|---|---|---|
| Initiation cohort: 186 fitting rows, 93 manual point proxies plus 93 weather-selected temporal absence controls; 10 components, 5 outer folds | 0.8952 / 0.0864 | 0.9026 / 0.1183 |
| Separate multi-regime proxy experiment: 510 eligible rows, 34 components | 0.7842 / 0.1772 | 0.6866 / 0.2451 |

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

## Credits and terms

Original TERRA-SHIELD code and documentation are [MIT-licensed](../LICENSE). Read [third-party notices](../THIRD_PARTY_NOTICES.md) for the distinction between project code, dependencies, observations, imagery, inventories and model assets. Credit the original product, provider, version/acquisition and processing changes when exporting results; check each original inventory's redistribution terms before publishing raw records.

The project builds on the work of NASA, ISRO, ASF, USGS, ISRIC, ECMWF/GloFAS, Open-Meteo, terrain contributors, Esri, Macrostrat, OpenStreetMap contributors and the open-source scientific/geospatial communities. Their inclusion is source attribution, not institutional endorsement.
