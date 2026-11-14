# Set up and use TERRA-SHIELD

[Back to the project](../README.md) · [API reference](../API_USAGE.md) · [Model card](../artifacts/MODEL_CARD.md)

Use [the public map](https://terra-production-289057fe.up.railway.app/) to explore without installing anything. The steps below run your own local copy. Add generated assistant prose using your personal key in **AI Model & API Key**; GPU training is not a setup step.

The browser and analysis service run as two local processes. The physics implementation and the small pretrained four-model ML artifact are included in this repository. You do **not** need the retained training corpus, a GPU, or Ollama to start the map and analysis engines.

## Before you start

Install **Python 3.11 or 3.12**, **Node.js 22.12+**, npm, and Git. Allow Internet access for satellite imagery, terrain, weather, and other providers. Provider availability and coverage vary; a working local engine does not guarantee every upstream source responds.

XGBoost also needs its platform runtime libraries: Windows uses the [Microsoft Visual C++ Redistributable](https://xgboost.readthedocs.io/en/stable/install.html); if macOS reports a missing `libomp.dylib`, run `brew install libomp` using Homebrew before restarting Python. These are native-library prerequisites, separate from the model file.

Run commands from the repository root unless a step says otherwise.

## 1. Clone the complete repository

```sh
git clone https://github.com/pookie421/Terra.git
cd Terra
```

The active ML artifact must be present at `artifacts/physics_informed_models.pkl`. Copying only `frontend/` or installing Python dependencies alone does not provide the backend and its model.

## 2. Create a Python environment

**macOS / Linux** — use your installed 3.11 or 3.12 executable:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, invoke `.\.venv\Scripts\python.exe` in place of `python` below; changing the execution policy is unnecessary.

Install the runtime dependencies with the model-compatible constraints:

```sh
python -m pip install -r requirements.txt -c constraints-runtime.txt
```

The constraints keep the included serialized ML artifact on its declared numerical-library versions. Do not substitute an arbitrary XGBoost version and assume pickle compatibility. See the [model card](../artifacts/MODEL_CARD.md).

## 3. Start and verify the analysis service

Keep this terminal open:

```sh
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Open [the engine health endpoint](http://127.0.0.1:8000/api/v1/health). It should report `status: "ok"`, `physics_reference_available: true`, and `model_ready: true`.

![Engine installation check](./images/engine-ready.jpg)

*The health check verifies local service/model availability without acquiring environmental providers. It does not certify predictive skill or provider coverage.*

The [interactive API documentation](http://127.0.0.1:8000/docs) is available on the same service. Restart the backend after replacing or restoring weights: model loading occurs when the service starts.

## 4. Start the browser application

Open a **second terminal**. From the cloned `Terra` directory:

```sh
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite, normally [http://127.0.0.1:5173](http://127.0.0.1:5173). Keep both terminals running. Vite forwards `/api` requests to `http://127.0.0.1:8000`; the browser does not run the Python engine itself.

## 5. Scan and inspect an area

![Map workspace and inspection panels](./images/live-workspace.jpg)

*Interface capture; values belong to that scan. Imagery © Esri, Maxar, Earthstar Geographics and contributors. Elevation © Mapzen / upstream DEM contributors.*

1. **Choose a location.** Search a place name, enter `latitude, longitude`, click an unscanned point on the map, or choose **Scan map view**. A place name is optional; unnamed terrain works through coordinates.
2. **Review the area overview.** The peak initiation score and lowest central factor of safety summarize the returned area. **View critical area** focuses the selected hotspot.
3. **Inspect a cell.** Click a returned hexagon. The selected location shows slope, elevation, the low/central/high factor-of-safety estimates, pressure head, and rain deficit. Expand **Physical details** for provenance, variance sensitivities, and model internals.
4. **Change the displayed time.** Choose **Now**, **+24h**, **+48h**, **+72h**, or **Peak**. Available future states use the downloaded forecast; **Peak** is the minimum sector nominal factor of safety, not a predicted failure time.
5. **Change the map layer.** Switch between initiation score, factor of safety, slope, and pressure head. Gray means unavailable evidence, rather than zero hazard.
6. **Check supporting evidence.** Open **Options → Sources** for availability, fallbacks, timing, and errors. Review the selected-cell comparison for model diagnostics and **Potential runout** for conditional routing assumptions.
7. **Ask the Assistant.** Questions use the active scan, forecast state, and selected cell. Open response details to inspect verification and numeric citations. The download button exports the current analysis.

On smaller screens, use **Map**, **Overview**, and **Inspect** at the bottom to switch panels. Weather inputs remain automatic and read-only; there are no manual weather sliders or preset replay locations.

Reference soil assumptions keep conditional physics available without boreholes. Experimental scores, factor-of-safety envelopes, and hypothetical runout are not calibrated global failure probabilities or operational warnings. [Current evidence and limitations](./current_status.md) explain their interpretation.

## Optional: connect an assistant

Open **Assistant → AI Model & API Key** or **Options → AI Model & API Key**. Select your provider, enter a model ID and paste your personal API key into the masked field. Settings are saved only for the browser session; **Clear key** removes the active credential. Public chat uses no server-key or local-model fallback, so no backend restart or shared subscription is needed. See [Assistant models and personal API keys](assistant_connections.md).

Physical analysis, ML predictions and deterministic assistant navigation work without an AI key. Generated prose must pass state/numeric checks to receive `VERIFIED_LLM_PROSE`; deterministic responses remain `VERIFIED_TEMPLATE`. Provider configuration is distinct from a successful provider response.

NASA LHASA has its own **Today / Tomorrow** controls and native raster support. Use Terra analysis for the automatic 72-hour forecast, physical column and conditional runout. For NISAR, the radar panel reports measured pixels, processing jobs or the specific access/coverage gap. **Check for measurements** refreshes NISAR without reacquiring weather/terrain and starts a new analysis identity.

## Troubleshooting

| Symptom | Check |
|---|---|
| **Service unavailable** or `/api/v1/health` will not open | Start Uvicorn from the repository root, keep its terminal open, and check that port 8000 is available. Starting only Vite is insufficient. |
| `model_ready: false` | Check the exact artifact path, use the constrained dependency installation above, inspect backend startup errors, and restart the backend. Restore the published artifact rather than trying to train without the retained corpus. |
| Model load fails after upgrading packages | Recreate the virtual environment using `constraints-runtime.txt`; the included pickle is not a cross-version model interchange format. |
| Map is blank or source tiles fail | Check browser WebGL support and access to the imagery/elevation providers. Local backend health alone does not test online map tiles. |
| Scan has missing inputs or no learned score | Open **Sources** and the returned reason. ML requires a complete weather window and compatible metric terrain. A provider outage is distinct from a missing model. |
| Future buttons are unavailable | Inspect forecast coverage and the returned reason. Current conditions remain separate from missing future forcing. |
| Assistant uses fallback or reports provider unavailable | Open AI Model & API Key and check your provider, model, personal credential and JSON support. No backend restart is required. Physical and ML scans remain independent of generated prose. |

For CPU hazard-model reproduction, follow the [research workflow](training_workflow.md). It requires separately provisioned source-reviewed corpora; the runtime model alone cannot reconstruct that dataset. Generated `training/` and `windows_5070ti_trainer/` research directories stay local. They are not needed to install or host the application.
