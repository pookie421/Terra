# Host the website on Railway

The simplest setup is **one always-on Railway service** running the production Docker image. It serves the map application and `/api` on the same HTTPS origin, uses the approved CPU model, and stores downloaded terrain tiles on a persistent volume. A GPU, the Windows trainer, training corpora and an AI key are unnecessary for the website and deterministic assistant commands.

## Launch with a Railway address

1. Create a Railway project and connect **`pookie421/Terra`**, branch **`main`**, with the repository root as the source directory. Railway detects the root `Dockerfile`. Leave build and start commands empty so the image's entrypoint runs.
2. Add a persistent volume mounted at **`/data`**. **2 GB** is sufficient for the initial 1 GiB terrain cache budget; increase it when measured needs justify it.
3. Set **`RAILWAY_RUN_UID=0`**. Railway mounts volumes as root. The startup helper creates the cache directory, drops to UID/GID 10001, checks that it can write, and replaces itself with a single Uvicorn worker. The application runs without root privileges.
4. Set the deployment healthcheck to **`/api/v1/ready`**, allow up to **120 seconds** for startup, use **one replica**, disable **Serverless/sleep**, and use restart-on-failure. Start with resource ceilings around **2 vCPU / 4 GB RAM** if your plan supports them. These are initial limits, not measured concurrent-user capacity.
5. Use **Generate Domain**, target port **8000**, for the Railway HTTPS address. A custom domain can be attached later without rebuilding the application.

For an initial trial preview, use the account's **500 MB volume** and override **`TERRA_TERRAIN_CACHE_MAX_MB=384`**. The current trial limits RAM to **1 GB** and includes **$5 of credit**, expiring after 30 days or when used. Limited trials can restrict outbound networking. The container CI checks startup under 1 GB; peak live-scan capacity still needs measurement. Move to a paid plan with adequate headroom for ongoing hosting. The Railway hostname itself does not make compute free. [Trial limits](https://docs.railway.com/pricing/free-trial), [pricing](https://docs.railway.com/pricing), [volume permissions](https://docs.railway.com/volumes), [HTTPS domains](https://docs.railway.com/networking/public-networking).

Railway's old `railway.toml`/`railway.json` deployment configuration is deprecated for new services. This repository uses a portable Dockerfile and explicit service settings, avoiding an additional IaC SDK or expired config. [Current Railway configuration guidance](https://docs.railway.com/infrastructure-as-code).

## Included production defaults

| Variable | Docker image default | Purpose |
|---|---|---|
| `PORT` | `8000` | Railway may override this; the startup helper validates it |
| `TERRA_SERVE_FRONTEND` | `true` | Serves the built React application with the API |
| `TERRA_FRONTEND_DIST` | `/app/frontend/dist` | Built frontend directory |
| `TERRA_ALLOWED_ORIGINS` | Empty | Same-origin deployment; no cross-origin browser access is needed |
| `TERRA_TERRAIN_CACHE_DIR` | `/data/terrarium` | Writable persistent terrain cache |
| `TERRA_TERRAIN_CACHE_MAX_MB` | `1024` | Cache budget in MiB; oldest managed PNGs are evicted |
| `TERRA_MAX_CONCURRENT_ANALYSES` | `1` | Excess scans/forecast computations receive HTTP 429 and `Retry-After` |
| `TERRA_MAX_CONCURRENT_CHATS` | `2` | Separate chat admission budget |

Do not set `VITE_API_URL` for this single-origin image. Do not copy a local `.env` into the image. The Docker context is an allowlist and excludes local provider archives, training mirrors, old weights, credentials and build caches. The official `xgboost-cpu==2.1.4` distribution supplies the pinned XGBoost runtime without CUDA/NCCL dependencies. The image targets **Linux amd64**.

## Optional sources and assistant

Generated assistant prose uses each visitor's personal API key, configured in **AI Model & API Key** and supplied per request. Do not add a shared paid LLM key to the website's server settings: public chat ignores server assistant credentials. Verified templates and deterministic map commands work without a key. See [assistant setup](assistant_connections.md). No model download or GPU service is required.

Authenticated Earthdata access uses the protected server variable `EARTHDATA_TOKEN`. Supported NISAR products can be extracted automatically through NASA Harmony as bounded spatial/variable subsets. Processing may outlive the scan; the radar panel can recheck the same job without reacquiring other environmental feeds. Actual acquisition coverage, product variables, authorization and valid pixels still determine measured availability. Approved external local subsets and explicit manifests remain supported. Never put credentials or provider rasters in frontend build variables or source control.

## Verify the actual deployed service

```sh
python scripts/check_web_deployment.py https://YOUR-SERVICE.up.railway.app
```

This checks the loaded model, frontend HTML/assets, API routing, hidden-file protection and invalid-coordinate rejection. It makes **no environmental provider requests**. `/api/v1/ready` is a deployment readiness check, distinct from provider availability and scientific validity. The GitHub **Website container** workflow builds and runs the Linux image, checks native imports, non-root execution and a fresh persistent volume.

After readiness passes, a bounded live coordinate check can establish that the cloud host reaches the providers. Preserve its errors and coverage; a successful location does not establish global coverage or predictive skill.

## Current public-beta limits

Use one worker and one replica. Scan/forecast states remain in process memory, with a shared 24-state capacity. Restarts, redeployments and eviction can invalidate browser scan IDs; rescan the location to recover. A volume preserves terrain files, not active scan states. The cache quota coordinates one process only.

Admission limits bound simultaneous work; they are not per-user quotas or authentication. Visitors fund their own generated AI calls. Watch environmental-provider quotas and resource use; NISAR extraction uses the deployment's Earthdata access. Durable state is required before horizontal scaling or a promise that user sessions survive deployments. Harmony job labels allow lookup of provider work after a process restart, while active map scan IDs still expire.

The website exposes experimental conditional hazard estimates and source provenance. Hosting does not establish operational warning readiness or LHASA-equivalent predictive skill.

## Attach your own domain later

Add the hostname in Railway's domain settings and create the exact CNAME/TXT records it returns at your registrar/DNS provider. Railway provisions TLS. Start with DNS-only routing when using Cloudflare DNS, so an additional proxy timeout does not shorten the scan request budget. No registrar-specific code is required.

## Run the same image locally

With Docker installed:

```sh
docker build --platform linux/amd64 -t terra-shield .
docker run --rm --user 0 -p 8000:8000 -v terra-data:/data terra-shield
```

Open `http://localhost:8000`. Apple Silicon uses amd64 emulation for this image. Production hosting does not run the Vite development server or install the QLoRA stack.
