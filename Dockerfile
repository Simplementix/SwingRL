# SwingRL — CPU-only Docker image for homelab CI and production
# Base: python:3.11-slim (Debian-based, minimal footprint)
# uv: copied from official uv image for fast, reproducible dep install
# PyTorch: CPU-only (via pytorch-cpu index in pyproject.toml linux marker)
# User: non-root trader (UID 1000) — follows principle of least privilege

FROM python:3.11-slim AS base

# Copy uv binary from official image (avoids curl + install overhead)
COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /uvx /bin/

# Create non-root trader user (UID 1000) and pre-create /app with correct ownership.
# All subsequent COPY/RUN layers write into a trader-owned directory —
# no chown pass needed at the end (avoids layer duplication that doubles image size).
RUN useradd -m -u 1000 trader && mkdir -p /app && chown trader:trader /app

WORKDIR /app

# Stage 1: Install all dependencies (prod + dev for CI) using bind mounts.
# Running as root here so the uv cache mount target is accessible.
# The .venv is created inside /app which is owned by trader.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --group dev --no-install-project

# Stage 2: Copy source code (with correct ownership) and install the project itself.
# --chown ensures source files are trader-owned from the COPY instruction.
# Source changes only invalidate this layer, not the dep-install layer above.
COPY --chown=trader:trader . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group dev

USER trader

# Default entrypoint for production use (overridden by docker compose for CI/tests).
ENTRYPOINT ["uv", "run", "python", "-m", "swingrl"]
