# NASA comparison and NISAR evidence

The primary choices are **Terra In House Model**, **NASA LHASA** and **Rainfall baseline**. NASA has a distinct daily-product workspace and native-pixel map; Terra provides the conditional physical column, 72-hour forecast and runout. Supporting in-house comparisons remain advanced diagnostics. A side-by-side readout is a source comparison, not a matched predictive benchmark.

## NASA LHASA

`backend/realtime/nasa_lhasa.py` samples NASA's [Today](https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/LHASA_Hazard_Today/ImageServer), [Tomorrow](https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/LHASA_Hazard_Tomorrow/ImageServer) and [susceptibility](https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/Global_Landslide_Susceptibility/ImageServer) products under one 3.5-second budget. Native-pixel deduplication and [multipoint sampling](https://developers.arcgis.com/rest/services-reference/enterprise/get-samples/) avoid a request for every H3 cell. Sampling uses nearest-neighbor values; valid zero, NoData and provider failure stay distinct. Susceptibility classes are categorical, not probabilities.

The checked services use 30-arcsecond geographic pixels, approximately 1 km north–south; east–west width varies with latitude. Hazard coverage is approximately 60°S–60°N; the susceptibility product has a different extent. The live endpoint metadata does not declare the exact serving model version or issue/valid time. The requested comparator ID remains `nasa_lhasa_2`, but the response reports `NOT_DECLARED_BY_SERVICE` rather than assigning the [upstream repository's](https://github.com/nasa/LHASA) current release to an unversioned raster service.

The published [LHASA 2.0 method](https://doi.org/10.3389/feart.2021.640043) is a trained model with specific inputs. A rainfall/ARI formula cannot reproduce it. The source API retains `LHASA_REFERENCE_EMULATION` as a separate **unvalidated Terra reference index**, with formula, project scales and missing inputs; `official_score` stays null and `reference_score` identifies the proxy. The primary NASA map, headline, selected-product evidence and critical-cell ranking use official scores only. Missing NASA output is not replaced by a Terra index.

NASA Tomorrow is a relative daily product, not a guaranteed exact T+24h forecast. The interface offers **Today / Tomorrow**, displays native pixel support and hides Terra's hourly forecast and physical/runout headlines. Native footprints use service extent/pixel spacing; repeated H3 samples do not become independent NASA predictions. A clearly named switch opens Terra's separate physical analysis. NASA daily selection is included in assistant evidence and stale-response guards.

## Spatial support

Nominal 30 m DEM inputs do not make every analysis quantity a 30 m observation. H3 resolution 9 has an average area of about **0.105 km²** and cells span hundreds of metres; their area varies. Repeating one NASA pixel across several H3 polygons adds no spatial information. Soil, rainfall, moisture and radar products retain their own support. Finer terrain geometry, inspectable mechanics and conditional routing are capabilities, not proof of improved hazard skill. [H3 statistics](https://h3geo.org/docs/core-library/restable/).

## NISAR discovery and measured subsets

`backend/realtime/nisar_processor.py` queries public NASA CMR for intersecting GUNW/GCOV validated, provisional and beta collections. Records retain collection/granule identity, acquisition times, orbit information, product maturity and available browse/data links. The latest returned set is bounded; its count is not the total mission acquisition count. Discovery is not an assurance of recent or measured deformation. See the [ASF collection and access guide](https://nisar-docs.asf.alaska.edu/earthaccess/).

Without a readable raster, `METADATA_LIVE` exposes real discovery records and null measurement fields. The complete broad reference FS/FOSM column stays available. Missing radar does not justify a made-up strain variance, a tighter envelope or an arbitrary widening coefficient.

With server-side Earthdata authentication, the adapter automatically requests supported GUNW coherence and GCOV HH/HV variables through [NASA Harmony](https://nisar-docs.asf.alaska.edu/harmony/). It reads collection-advertised variable IDs, a discovered real granule and the scan's spatial window; it does not fetch the complete multi-GB HDF5 product. Selection uses the latest discovered supported acquisition, with maturity breaking equal-time ties, rather than seeking a preferred hazard outcome.

At most four jobs cover the two product families and two non-wrapping antimeridian rectangles. Request identities coalesce work; deterministic provider labels recover existing jobs after restart. Pending extraction is labelled `SUBSET_PENDING`, not measured. **Check for measurements** polls the retained job and samples ready pixels without repeating weather/terrain requests. The refresh creates a new scan identity, preserving the previously inspected state. Authentication errors, absent coverage, native NoData and unavailable extraction have separate statuses under `measurement_access`.

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

## Authenticated access and freshness

Create an [Earthdata account](https://urs.earthdata.nasa.gov/users/new) and generate a token using [NASA's instructions](https://urs.earthdata.nasa.gov/documentation/for_users/user_token). Keep `EARTHDATA_TOKEN` in the ignored local `.env`, then start the backend with `python -m uvicorn backend.api.main:app --env-file .env --host 127.0.0.1 --port 8000`. Never commit credentials or downloaded provider archives.

Retain product maturity and processing versions. Authenticated scans can now submit and reuse bounded [Harmony variable/spatial subsets](https://nisar-docs.asf.alaska.edu/harmony/) automatically. Existing manifests remain supported for approved local subsets. A successful job does not guarantee valid pixels at every sampled point, and a token does not guarantee acquisition coverage everywhere.

Standard products generally arrive 36–72 hours after acquisition. Urgent-response Level-2 processing targets about six hours for designated events, with lower-quality orbit/correction inputs and restricted tasking; it is not a universal six-hour feed. GUNW uses nominal 12-day acquisition pairs. See [ASF latency guidance](https://nisar-docs.asf.alaska.edu/urgent-response/) and [GUNW support](https://nisar-docs.asf.alaska.edu/gunw/).

A bounded authenticated check on 13 November 2026 at the unchanged coordinate `11.548, 76.134` retrieved actual 80 m provisional GUNW coherence for the 8–20 October pair. Its center H3 sample was approximately **0.6135**. Newer validated 10 November GCOV subsets were NoData at that cell and remained missing. This establishes data access and reader behavior at one location, not global coverage or predictive skill.

On 14 November, the automatic path was checked at that same requested coordinate with the manual manifest disabled. It recovered successful GUNW/GCOV jobs and sampled **0.4743816853** coherence at the requested point from the 80 m 8–20 October GUNW pair. This point sample differs from the earlier H3-center sample. The 10 November GCOV point pixels remained NoData, and the ratio remained null. Downloaded subset bytes, native support and granule/variable identities were recorded locally; no velocity, moisture or strength value was inferred.

Discovery dates and measurement dates remain separate. `observable_age_days` uses each measured observable's provenance; the shared feature age conservatively uses the oldest available measurement and stays unknown if an available measurement lacks a valid date. Metadata-only age is explicitly `DISCOVERY_ONLY`. A newer discovery must not refresh older coherence. The active classifier still has no newly trained SAR weights, and no radar-to-strength relation is inferred by this check.
