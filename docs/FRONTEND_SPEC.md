# Frontend specification

## Product and visual direction

The interface is a professional global landslide analysis workspace. The map and the user's analysis take priority over branding or system decoration.

- Use a restrained neutral dark interface, readable Inter sans-serif text, sentence-case labels, clear spacing, and subtle borders. Use tabular numbers where comparisons benefit from alignment; do not use monospace microtext throughout the interface.
- Remove logos, product wordmarks, promotional taglines, engine-active slogans, glowing indicators, target-style decoration, and other science-fiction styling. Use color sparingly for selection, meaningful hazard values, warnings, and errors.
- Keep search and the main map actions easy to find. Show connection errors when they affect use; a permanent healthy-engine banner is unnecessary.
- Show essential results first. Put technical diagnostics, source timing, identifiers, and model details in clearly named expandable sections. Avoid repeated coordinates, raw internal status strings, excessive decimal precision, and placeholder-filled panels before a scan.
- Keep the interface usable on smaller screens, with readable controls and accessible focus states. Retain provider attribution required by the map sources.

## Live global map and scanning

- Load a full-viewport MapLibre globe and terrain map immediately. Use online Esri World Imagery and AWS Terrarium elevation tiles, with Terrarium encoding, terrain exaggeration 1.35, and an initial pitch of 55 degrees. Native elevation resolution varies by upstream source.
- Start with an explorable world map and a brief instruction to search or select a location. Do not boot into a replay, a fixed location, or a location preset selector.
- Accept any place name or raw `latitude, longitude`. Clicking terrain outside an existing result grid, or choosing **Scan current view**, starts `POST /api/v1/scan` for the exact coordinate. Unnamed mountains and valleys must work without a place name.
- Render the returned H3 resolution-9 grid (`h3_geojson`) over the terrain, together with conditional runout paths (`runout_geojson`). Provide layers for the active model score, factor of safety, NASA comparison, radar coherence, slope and pore-pressure head. Use a useful legend once results are present; missing evidence must remain visually distinct from a zero score. Resolution-9 cells average about 0.105 km² and span hundreds of metres; they are not 30 m pixels.
- Select the critical initiation cell after a scan. Clicking another result cell inspects that exact cell and updates the physical column, runout, model comparison, and assistant context. Preserve the ability to focus the map on the selected or critical cell.

## Results and automatic forecast

- Present a concise area overview: the peak conditional initiation score, the minimum factor-of-safety column with `[FS_low, FS_mid, FS_high]`, and a way to inspect the critical cell. Scores are experimental initiation scores, not established calibrated probabilities.
- Offer exactly three primary prediction choices: **Terra In House Model** (the existing Physics-Informed ML, default), **NASA LHASA**, and **Rainfall baseline**. Keep physics-only and weather/terrain comparisons in advanced diagnostics, with an explicit diagnostic state if requested by a legacy assistant action. Update the map, peak readout and critical-cell target together. Keep selected-cell comparisons available without implying equal score meaning, calibration or support.
- Distinguish live NASA raster values, unvalidated coarse reference indices and missing coverage. Mixed official/reference coverage needs distinct styling and labels; choose the NASA critical cell among official values when present, using fallback-only ranking otherwise. Show the served-version/time uncertainty instead of assigning an upstream code release to an undeclared ImageServer product. Nominal 30 m terrain inputs and roughly 1 km NASA pixels do not establish a predictive-skill advantage.
- Show the current location and analysis time where they help interpretation. Grid counts, full H3 IDs, scan/state IDs, terrain ranges, and duplicate coordinate readouts belong in analysis details or export rather than permanent map chrome.
- Keep observed forcing read-only. Make 24-hour rain, preceding six-day rain, surface moisture, SoilGrids composition, USGS seismic context, and modeled GloFAS river discharge available with units and source context. Secondary soil and provider diagnostics may be collapsed.
- Fetch the automatic 72-hour forecast (`predictive_forecast_72h`) for the scanned location. Allow switching between Now, 24, 48, and 72 hours and the peak mechanical forecast, with the returned forecast trajectory and valid time available. State changes must update both the displayed map and inspected results.
- Keep NASA's daily Today/Tomorrow products distinct from Terra's automatic forecast. Tomorrow is not guaranteed to align with exact T+24h; unsupported NASA 48/72-hour states must remain visibly unavailable while the physical forecast continues.
- Do not add manual rainfall, storm-duration, PGA, warming, burn, or other fabricated forcing controls. Preserve source missingness and ensemble-availability explanations rather than inventing future inputs.

## Cell inspection and supporting detail

- Keep the selected cell's slope, elevation, factor-of-safety envelope, basal pressure head, and critical rain deficit understandable at a glance. Retain a short, honest cue that the physical column uses reference assumptions when it does.
- Make the complete physical column available through progressive disclosure: exact coordinates and H3 ID, specific TWI, provenance, mechanical admissibility, learned trigger score, Newmark applicability, joint variance `JΣJᵀ`, sensitivity allocations, Jacobian, and covariance matrix. Do not remove the underlying uncertainty or provenance to simplify the default view.
- Keep the four in-house comparisons plus the separate NASA comparison available for the selected cell. It may be a collapsed diagnostic section. A paired-cell comparison must pin two cells from the same active scan/state and preserve each model's provenance.
- Provide NISAR acquisition/product context and measured radar fields on demand. Metadata-only discovery must not look like a deformation observation. Coherence and the uncalibrated polarimetric index retain their meanings; missing radar keeps complete reference physics available. Expose any documented site-calibrated strength constraint and its source, rather than implying a generic radar-to-strength law or a SAR-trained classifier.
- Keep conditional Voellmy runout available on the map and in inspection. Present useful corridor and velocity results first; place the three rheology/depth scenarios, coefficients, stop reasons, and truncation information in details. Preserve the explanation that routing assumes release.
- Put secondary controls in a compact **Options** menu: map layers, Sources, export, advanced diagnostics and Connections. Keep primary search, scanning, the three model choices and automatic forecast easy to reach. The Sources view retains per-provider availability, live/cache/fallback status, errors, timing and provenance.

## Assistant and API integration

- Use a simple **Assistant** entry point and a multi-turn interface wired to `POST /api/v1/chat`. Model branding and implementation details do not need to dominate the conversation.
- Support operator-configured local Ollama or a hosted compatible assistant API. Shared-deployment visitors do not need personal keys. Connections shows redacted configuration, distinct from actual availability, with owner setup help. No browser key storage, global key-editing endpoint, or frontend build-time secrets. The chat provider remains independent of the physical and landslide prediction engines.
- Preserve verified prose/template status, source-bound numeric citations, model-service errors, and `ui_action` map commands. Verification details and exact identifiers can be disclosed on demand; the interface must not imply that template output came from a running language model.
- Render `thought_steps` as brief public action/evidence summaries in a collapsible **Agent steps** view. Show planned work separately from browser execution results; these summaries are not private model deliberation.
- Execute validated ordered chains for coordinate scans, place search, model/horizon/layer selection, fly-to, inspection and paired-cell comparison. Allow a location plan before the first scan. Resolve targets such as steepest, riskiest or lowest FS against each newly returned grid, and wait for scans/forecast changes before subsequent actions.
- Bind the displayed map, inspected cell, forecast state, and assistant conversation to exact scan/state/cell IDs internally. Reject stale responses after a selection or forecast change even when those IDs are not permanently visible.
- Include the active prediction model in the conversation identity. A user change must cancel an obsolete action chain; generated commands require schema/range/membership validation and must not expand the user's authorized action plan.
- Keep `/api` proxied to the local FastAPI backend at `http://127.0.0.1:8000`. Preserve arbitrary-coordinate scan, inspection, forecast, and chat integration. Validate the frontend with a production build and TypeScript checking; distinguish browser integration checks from scientific validation.
