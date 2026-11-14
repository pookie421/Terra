<div align="center">

# TERRA-SHIELD

### A clearer view of landslide hazard

An interactive portal for landslide visualization and prediction: explore terrain, compare models, and understand the conditions behind an estimate.

**[Open the live website →](https://terra-production-289057fe.up.railway.app/)**

[Explore the features](#explore-a-place) · [Research & sources](docs/RESEARCH_AND_SOURCES.md) · [Run locally](docs/SETUP.md)

[![Checks](https://github.com/pookie421/Terra/actions/workflows/ci.yml/badge.svg)](https://github.com/pookie421/Terra/actions/workflows/ci.yml)
[![Website container](https://github.com/pookie421/Terra/actions/workflows/container.yml/badge.svg)](https://github.com/pookie421/Terra/actions/workflows/container.yml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-64748b)](LICENSE)

</div>

[![The live terrain workspace with hazard hexagons, forecast controls and slope inspection](docs/images/live-workspace.jpg)](https://terra-production-289057fe.up.railway.app/)

<sub>Website capture; displayed estimates belong to that scan. Imagery © Esri, Maxar, Earthstar Geographics and contributors. Elevation © Mapzen / upstream DEM contributors.</sub>

## Explore a place

1. **Find your location.** Search a place, enter `latitude, longitude`, or select a point on the map. Remote and unnamed terrain can be explored by coordinate.
2. **Scan the landscape.** View the returned hazard hexagons over satellite imagery and 3D terrain.
3. **Compare the models.** Switch between the in-house approach and NASA's landslide products while keeping the same area in view.
4. **Look closer.** Select a hexagon to inspect slope, factor-of-safety ranges, water-pressure estimates and potential runout. Explore the automatic **72-hour forecast** and open **Sources** for the evidence behind the result.

The assistant can help navigate, inspect cells and control the map using the selected analysis. Add your own provider and model in **AI Model & API Key** for generated explanations; numeric statements are checked against the analysis. Verified templates keep map commands available without a key.

## Two models, one workspace

| Model | What you can explore |
|---|---|
| **Terra In House Model** | Environmental feeds, reference slope physics and trained machine learning combine into inspectable hazard estimates, uncertainty ranges and forecast projections. |
| **NASA LHASA** | NASA landslide products provide an independent comparison. Official raster values, reference estimates and missing coverage are identified in the interface. |

A rainfall baseline and additional physics/weather–terrain diagnostics support comparison. Each model retains its own source, spatial support and interpretation. [How the comparison works →](docs/nasa_radar_comparison.md)

## See the landscape and the evidence

| Capability | Why it helps |
|---|---|
| **Satellite imagery & 3D terrain** | Connect the analysis to mountains, valleys and surrounding terrain. |
| **Automatic current & forecast states** | Explore current conditions and provider-driven projections at 24, 48 and 72 hours. |
| **Physical columns & uncertainty** | Inspect the low, central and high factor-of-safety estimates and their reference assumptions. |
| **Potential runout corridors** | Explore conditional downstream paths and mapped exposure if release occurs. |
| **Source transparency** | Inspect weather, soil, seismic, river, NASA and optional NISAR context with availability and provenance. |

The physical engine and the small pretrained ML model are included in this repository and run on CPU. [Engine details and model card →](artifacts/MODEL_CARD.md)

Outputs are conditional research estimates for visualization and analysis. Source coverage, material assumptions and predictive performance have documented limits. The [research reference](docs/RESEARCH_AND_SOURCES.md) brings together the papers, methods, benchmark results and source qualifications.

## Build on the project

| Guide | Start here |
|---|---|
| **Research, papers & data sources** | [Research and sources](docs/RESEARCH_AND_SOURCES.md) |
| **Installation & troubleshooting** | [Local setup](docs/SETUP.md) |
| **Host your own website** | [Railway and Docker deployment](docs/deployment.md) |
| **Use the API** | [API reference](API_USAGE.md) |
| **Contribute** | [Development guide](CONTRIBUTING.md) · [Roadmap](ROADMAP.md) |

<details>
<summary><strong>Run locally or deploy your own copy</strong></summary>

Use **Python 3.11 or 3.12** and **Node.js 22.12+**. The [setup guide](docs/SETUP.md) covers installation, the bundled model, launching the API and frontend, and troubleshooting. A GPU or model-training run is unnecessary for the web application.

The root [Dockerfile](Dockerfile) packages the frontend and CPU API together. The [deployment guide](docs/deployment.md) explains persistent terrain storage, health checks, hosting limits and custom domains. The public preview currently runs on Railway trial resources; availability depends on its hosting allowance.

</details>

Original project code is [MIT licensed](LICENSE). Data products and third-party software retain their own terms and acknowledgements: [third-party notices](THIRD_PARTY_NOTICES.md).

---

<sub>A project by <strong>Team Orangeo</strong> for the <strong>2026 NASA Space Apps competition</strong>.</sub>
