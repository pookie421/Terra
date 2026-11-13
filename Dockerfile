# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    TERRA_SERVE_FRONTEND=true \
    TERRA_FRONTEND_DIST=/app/frontend/dist \
    TERRA_TERRAIN_CACHE_DIR=/data/terrarium \
    TERRA_TERRAIN_CACHE_MAX_MB=1024 \
    TERRA_MAX_CONCURRENT_ANALYSES=1 \
    TERRA_MAX_CONCURRENT_CHATS=2 \
    TERRA_ALLOWED_ORIGINS="" \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libexpat1 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 terra \
    && useradd --uid 10001 --gid 10001 --create-home terra \
    && mkdir -p /data/terrarium \
    && chown 10001:10001 /data/terrarium
COPY requirements.txt requirements-ml.txt requirements-geo.txt constraints-runtime.txt ./
# The official CPU-only distribution provides the same pinned xgboost import,
# without the large CUDA/NCCL dependencies of the full Linux x86-64 wheel.
# This substitution affects the image's install inputs, not repository sources.
RUN sed -i 's/^xgboost>=2.0,<4$/xgboost-cpu==2.1.4/' requirements-ml.txt \
    && python -m pip install --only-binary=:all: -r requirements.txt -c constraints-runtime.txt \
    && python -c "import h3, h5py, pyproj, rasterio, scipy, sklearn, xgboost; assert xgboost.__version__ == '2.1.4'"
COPY backend/ ./backend/
COPY artifacts/physics_informed_models.pkl artifacts/runtime_manifest.json artifacts/MODEL_CARD.md ./artifacts/
COPY --from=frontend /build/frontend/dist/ ./frontend/dist/
COPY deployment/start.py ./deployment/start.py
RUN python -c "import hashlib; from pathlib import Path; p=Path('artifacts/physics_informed_models.pkl'); assert hashlib.sha256(p.read_bytes()).hexdigest() == 'ff70502e82cdd3cb57dbfc141478336c144234f6ed6eb1d059393aac377fcf0c'"
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "/app/deployment/start.py"]
