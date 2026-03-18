#!/usr/bin/env bash
# setup_ollama.sh — Pull Qwen models into the swingrl-ollama Docker volume.
#
# Usage (from repo root, after starting docker-compose.prod.yml):
#   bash scripts/setup_ollama.sh
#
# The 'ollama pull' command is idempotent: if the model is already present in
# the named volume (ollama_models:/root/.ollama), the pull is a no-op.
#
# Models pulled:
#   qwen2.5:3b   — sole local model for query agent and Ollama fallback (~2.0 GB)
#                  Consolidation primary uses NVIDIA kimi-k2.5 cloud API
#
# Requires: docker compose -f docker-compose.prod.yml up -d swingrl-ollama
# before running this script.

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
SERVICE="swingrl-ollama"

echo "[setup_ollama] Pulling models into ${SERVICE} container..."

echo "[setup_ollama] Pulling qwen2.5:3b ..."
docker compose -f "${COMPOSE_FILE}" exec "${SERVICE}" ollama pull qwen2.5:3b

echo "[setup_ollama] Done. Models available:"
docker compose -f "${COMPOSE_FILE}" exec "${SERVICE}" ollama list
