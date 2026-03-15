# SwingRL — CPU-only Docker image for homelab CI and production
# Base: python:3.11-slim (Debian-based, minimal footprint)
# uv: copied from official uv image for fast, reproducible dep install
# PyTorch: CPU-only (via pytorch-cpu index in pyproject.toml linux marker)
# User: non-root trader (UID 1000) — follows principle of least privilege
#
# Targets:
#   ci         — includes dev deps (pytest, ruff, mypy, bandit) for CI pipeline
#   production — minimal image for homelab paper trading deployment

# ──────────────────────────────────────────────────────────
# CI target: all dependencies (prod + dev) for testing/linting
# Used by: ci-homelab.sh (docker build --target ci)
# ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS ci

# Copy uv binary from official image (avoids curl + install overhead)
COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /uvx /bin/

# Create non-root trader user (UID 1000) and pre-create /app with correct ownership.
# All subsequent COPY/RUN layers write into a trader-owned directory —
# no chown pass needed at the end (avoids layer duplication that doubles image size).
RUN useradd -m -u 1000 trader && mkdir -p /app && chown trader:trader /app

WORKDIR /app

# Install all dependencies (prod + dev for CI) using bind mounts.
# Running as root here so the uv cache mount target is accessible.
# The .venv is created inside /app which is owned by trader.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --group dev --no-install-project

# Copy source code (with correct ownership) and install the project itself.
# --chown ensures source files are trader-owned from the COPY instruction.
# Source changes only invalidate this layer, not the dep-install layer above.
COPY --chown=trader:trader . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group dev

USER trader

# Default entrypoint for CI use (overridden by docker compose for tests/lint).
ENTRYPOINT ["uv", "run", "python", "-m", "swingrl"]


# ──────────────────────────────────────────────────────────
# Production target: no dev deps, minimal image for deployment
# Used by: docker-compose.prod.yml (target: production)
# CMD: APScheduler entrypoint (scripts/main.py) with cron jobs
# ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS production

COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /uvx /bin/

RUN useradd -m -u 1000 trader && mkdir -p /app && chown trader:trader /app

WORKDIR /app

# Install production dependencies only (no dev group).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

# Copy only production-relevant files (exclude tests, docs, .planning).
COPY --chown=trader:trader src/ /app/src/
COPY --chown=trader:trader config/ /app/config/
COPY --chown=trader:trader scripts/ /app/scripts/
COPY --chown=trader:trader pyproject.toml uv.lock /app/

# Install the project itself (production deps already cached above).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

USER trader

# Docker HEALTHCHECK: verify process liveness and DB connectivity.
# Interval 60s, timeout 10s, 3 retries before marking unhealthy.
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD ["uv", "run", "python", "scripts/healthcheck.py"]

# Production entrypoint: APScheduler with cron jobs and stop-price polling.
CMD ["uv", "run", "python", "scripts/main.py"]
