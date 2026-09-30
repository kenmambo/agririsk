# AgriRisk Kenya - serving image (FastAPI + Streamlit dashboard).
#
# The serving layer is read-only over pipeline artefacts, so the image ships
# the small bundle exported by scripts/export_deploy_bundle.py (a couple of MB)
# instead of the ~19 GB satellite-granule cache. Build will FAIL with a clear
# error if the bundle is missing - that is deliberate: no silent empty app.
#
#   python -m agrik                        # 1. generate artefacts (host, real feeds)
#   python scripts/export_deploy_bundle.py # 2. write deploy/seed/
#   docker build -t agrik:local .          # 3. build
#   docker compose up                      # 4. API :8000, dashboard :8502

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first (cached layer), then code - so rebuilding after a code
# change does not re-download streamlit/scikit-learn every time.
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir --no-deps .

# Precomputed artefacts only - api/dashboard never re-ingest or retrain.
COPY deploy/seed/data ./data
COPY deploy/seed/models ./models

# Non-root runtime user; logs/ must be writable by setup_logging().
RUN groupadd --system agrik \
    && useradd --system --gid agrik --home-dir /app --shell /usr/sbin/nologin agrik \
    && mkdir -p /app/logs \
    && chown -R agrik:agrik /app
USER agrik

# Bind on all interfaces inside the container (host publishing stays explicit
# in compose / your orchestrator). Override with AGRIK_API_HOST if desired.
ENV AGRIK_API_HOST=0.0.0.0 \
    AGRIK_ENVIRONMENT=production

EXPOSE 8000 8502

# Default role: the API. docker-compose overrides the command for Streamlit.
CMD ["python", "-m", "agrik.api"]
