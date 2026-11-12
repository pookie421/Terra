# Physical methods, provenance and applicability

The active path is `backend/physics/reference_context.py` through `backend/api/reference_assessment.py`, using the `pedotransfer_reference_column_v1` contract. This page describes the current equations and assumptions; [feature-fusion results](feature_fusion_phase3.md) cover the coupled ML experiment. The [source/API audit](global_source_audit.md) evaluates the evidence behind inputs and coefficients.

## Published methods and project assumptions

| Component | Reference | What the current implementation establishes |
|---|---|---|
| Hydraulic pedotransfer | [Saxton and Rawls (2006)](https://doi.org/10.2136/sssaj2005.0117) | Texture/organic-matter hydraulic estimates; this citation does not validate project strength/depth coefficients |
| Transient infiltration | [Iverson (2000)](https://doi.org/10.1029/2000WR900090) | Semi-infinite analytical pulse response with declared slope/time convention |
| Slope stability | [USGS TRIGRS v2](https://pubs.usgs.gov/of/2008/1159/) | Infinite-slope equation family and pressure coupling; not a complete TRIGRS solver reproduction |
| Soil units | [ISRIC layer definitions](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html) | Bulk density cg/cm³ × 10 gives kg/m³; SOC dg/kg ÷ 10 gives g/kg |
| Uncertainty | Joint local delta method, `J Sigma J^T`, with bounded-slope integration for reliability | Sensitivity under a declared parameter/covariance model; no empirical probability calibration or interval coverage guarantee |
| Runout | Conditional Voellmy energy integration | Declared friction/xi/flow-depth scenarios on the H3 graph; no surveyed release or mass-conserving flow solution |

Cohesion/friction/depth transfer coefficients, basal drainage, ET, initial storage, covariance and runout scenario parameters are **project reference assumptions**. They require independent tests, versioning and exposed sensitivity. They must not be described as borehole observations or universally published transfer fits.

## Column and uncertainty

Depth is vertical in metres; pressure head is metres; cohesion is kPa; unit weight is kN/m³; conductivity is m/s; diffusivity is m²/s. The transient solver uses:

```text
s = 4 D cos²(theta) t / Z²
R(s) = sqrt(s/pi) exp(-1/s) - erfc(1/sqrt(s))
FS = [c + max(gamma Z cos²(theta) - gamma_w psi, 0) tan(phi)]
     / [gamma Z sin(theta) cos(theta)]
```

Pulse superposition includes recession. The live column caps infiltration at conductivity and head at its declared geometric limit; FS itself is not clipped. Antecedent hourly forcing advances storage with capacity, drainage, ET and overflow. Surface moisture has a 7 cm update support, rather than being interpreted as a full-depth observed water table. Missing forcing or reconstructed uniform pulses retain explicit scenario labels.

The joint parameter set spans cohesion, friction, unit weight, depth, conductivity, diffusivity, initial head, slope and rain. The API exposes values, spreads, covariance, Jacobian, variance contributions and `J Sigma J^T`. Bounded-slope quadrature plus conditional log-FS FOSM avoids the local ratio singularity near flat terrain. The sensitivity envelope includes nominal FS and mixture fifth/ninety-fifth quantiles; distribution quantiles remain separate fields. Material boundaries and saturation/contact kinks limit local linearization. Unmodeled spatial correlation and structural error remain outside a nominal column interval. Exactly zero driving shear is explicitly undefined FS.

The infinite-slope regolith model is unsuitable as a complete description of bedrock cliffs, deep rotational failures, liquefaction or fluvial undercutting. A returned physical column is a conditional reference calculation; those regimes need additional models.

## Movement and other mechanisms

Voellmy xi uses m/s²; energy evolves along each constant edge with drag proportional to `2g/(xi*h)`. Release from rest is explicit. Three reference rheology/depth scenarios provide conditional centerlines and traversed-cell corridors. Routes can stop at the scan boundary and do not include entrainment, deposition, lateral spreading or momentum traversal over rises. Travel times and mapped intersections are scenario outputs, not validated arrival or loss predictions.

Legacy mechanism references include [Jibson's Newmark regression](https://pubs.usgs.gov/sim/3077/downloads/3077_pamphlet.pdf), [the USGS Staley post-fire model](https://ghsc.code-pages.usgs.gov/lhp/pfdf/guide/models/s17.html), and [Gartner conditional volume](https://ghsc.code-pages.usgs.gov/lhp/pfdf/guide/models/g14.html). Their stable-block, catchment, severity and forcing requirements remain applicable. Catalog magnitude, RGB color and a river-discharge ratio do not supply those missing measurements. Legacy compatibility zeros require their applicability flags; they are not evidence of hazard absence.

The [research history](research_history.md) preserves earlier methods, failed experiments and the superseded measurement-only response, with links to the complete original records. Current uncertainty, FS and runout semantics are those above and in the [API contract](../API_USAGE.md).
