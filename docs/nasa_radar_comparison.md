# NASA comparison and NISAR evidence

The model selector changes the displayed score and map. It offers Physics + ML, NASA LHASA, Physics reference, Weather + terrain ML and Rainfall only. Physical columns and conditional runout remain inspectable alongside the selected model. A side-by-side readout is a source comparison, not a matched predictive benchmark.

## NASA LHASA

`backend/realtime/nasa_lhasa.py` samples NASA's [Today](https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/LHASA_Hazard_Today/ImageServer), [Tomorrow](https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/LHASA_Hazard_Tomorrow/ImageServer) and [susceptibility](https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/Global_Landslide_Susceptibility/ImageServer) products under one 3.5-second budget. Native-pixel deduplication and [multipoint sampling](https://developers.arcgis.com/rest/services-reference/enterprise/get-samples/) avoid a request for every H3 cell. Sampling uses nearest-neighbor values; valid zero, NoData and provider failure stay distinct. Susceptibility classes are categorical, not probabilities.

The checked services use 30-arcsecond geographic pixels, approximately 1 km north–south; east–west width varies with latitude. Hazard coverage is approximately 60°S–60°N; the susceptibility product has a different extent. The live endpoint metadata does not declare the exact serving model version or issue/valid time. The requested comparator ID remains `nasa_lhasa_2`, but the response reports `NOT_DECLARED_BY_SERVICE` rather than assigning the [upstream repository's](https://github.com/nasa/LHASA) current release to an unversioned raster service.

The published [LHASA 2.0 method](https://doi.org/10.3389/feart.2021.640043) is a trained model with specific inputs. A rainfall/ARI formula cannot reproduce it. When official coverage is absent and the required weather/terrain inputs are present, `LHASA_REFERENCE_EMULATION` exposes a separate **unvalidated Terra reference index** with its formula, project scales and missing inputs. `official_score` stays null; `reference_score` identifies the proxy. In mixed coverage, official NASA cells determine the NASA critical-cell selection when any are available. Reference indices use a distinct map palette and are never ranked against official values as if their meanings matched.

NASA Tomorrow is a relative daily product, not a guaranteed exact T+24h forecast. The API labels that alignment. It supplies no invented NASA +48h/+72h product; those selected NASA scores remain unavailable while the in-house physical forecast continues normally.

## Spatial support

Nominal 30 m DEM inputs do not make every analysis quantity a 30 m observation. H3 resolution 9 has an average area of about **0.105 km²** and cells span hundreds of metres; their area varies. Repeating one NASA pixel across several H3 polygons adds no spatial information. Soil, rainfall, moisture and radar products retain their own support. Finer terrain geometry, inspectable mechanics and conditional routing are capabilities, not proof of improved hazard skill. [H3 statistics](https://h3geo.org/docs/core-library/restable/).

## NISAR discovery and optional measurements

`backend/realtime/nisar_processor.py` queries public NASA CMR for intersecting GUNW/GCOV validated, provisional and beta collections. Records retain collection/granule identity, acquisition times, orbit information, product maturity and available browse/data links. The latest returned set is bounded; its count is not the total mission acquisition count. Discovery is not an assurance of recent or measured deformation. See the [ASF collection and access guide](https://nisar-docs.asf.alaska.edu/earthaccess/).

Without a readable raster, `METADATA_LIVE` exposes real discovery records and null measurement fields. The complete broad reference FS/FOSM column stays available. Missing radar does not justify a made-up strain variance, a tighter envelope or an arbitrary widening coefficient.

For local measurements, place self-describing NISAR HDF5 subsets under `data/nisar/`, or set `TERRA_NISAR_DIR`. Set `TERRA_NISAR_MANIFEST` to describe GeoTIFF/HDF5 subsets explicitly. A manifest template is shown below; replace both capitalized metadata values with the actual source record:

```json
{
  "version": "nisar_subset_manifest_v1",
  "assets": [{
    "path": "coherence.tif",
    "observable": "coherence",
    "radar_band": "L",
    "units": "1",
    "granule_id": "COPY_ACTUAL_GRANULE_ID",
    "acquisition_utc": "COPY_ACTUAL_ACQUISITION_TIME",
    "band": 1
  }]
}
```

Paths resolve relative to the manifest. Readers honor CRS, pixel support, scale, masks and units. `EARTHDATA_TOKEN` enables supported authenticated subset access; tokens are not returned in scan payloads. Remote reads are restricted and bounded, and do not automatically download entire multi-GB products. A token alone does not guarantee a suitable raster is available.

Coherence is dimensionless and has several causes. A single pair's phase is not an annual velocity. The optional LOS velocity field requires a documented multi-acquisition time series, its time interval, reference, sign convention and units. The legacy-named `nisar_polarimetric_moisture_index` is an **uncalibrated HV/HH radar ratio**, not volumetric soil moisture. The retrieved implementation handles L-SAR evidence; the mission's S-band capability does not imply an S-band measurement exists for a scan.

## Physics and ML coupling

`nisar_context_features_v1` records radar observations, acquisition age and missingness separately from the active 25-feature classifier. The approved model has no newly trained SAR weights. Metadata alone leaves the physical envelope and ML score unchanged.

An optional `strength_calibration_by_h3` manifest entry can supply a granule-anchored, versioned site calibration of `cohesion_kpa`, with source, method `site_calibration` and valid dates. It must have a real measured radar anchor. Compatible accepted strength enters the existing column and recomputes FS, uncertainty and the model's physical features; explicit ground inputs take precedence. Rejected calibration leaves the reference column intact and reports the reason. No universal coherence-to-strength or velocity-to-strength coefficient is inferred. This conditional calculation requires separate validation and does not establish SAR-trained predictive skill.

Historical training rows receive only evidence appropriate to their retained dates. Today's NASA/NISAR discovery must not be attached retrospectively to earlier events. The regenerated assistant corpus teaches unavailable or not-queried context where historical products are absent, rather than inventing radar observations or NASA labels.
