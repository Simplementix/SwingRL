#!/usr/bin/env bash
# deploy_model.sh — Deploy a trained model from M1 Mac to homelab
#
# Transfers model via SCP to models/shadow/{env_name}/, verifies SHA256
# integrity, and runs 6-point smoke test remotely.
#
# Usage:
#   bash scripts/deploy_model.sh <model_file> <env_name> [homelab_host]
#
# Arguments:
#   model_file   - Path to the .zip model file on local machine
#   env_name     - "equity" or "crypto"
#   homelab_host - Tailscale hostname (default: "homelab")
#
# Exit codes:
#   0 - Success (model deployed and smoke test passed)
#   1 - Failure (validation, transfer, or smoke test failed)

set -euo pipefail

# --- Configuration ---
# Use ~ so scp and ssh expand on the remote host, not locally.
REMOTE_PROJECT_DIR="~/swingrl"
EQUITY_OBS_DIM=$(uv run python -c "from swingrl.features.assembler import EQUITY_OBS_DIM; print(EQUITY_OBS_DIM)" 2>/dev/null || echo 164)
CRYPTO_OBS_DIM=$(uv run python -c "from swingrl.features.assembler import CRYPTO_OBS_DIM; print(CRYPTO_OBS_DIM)" 2>/dev/null || echo 47)

# --- Argument validation ---
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash scripts/deploy_model.sh <model_file> <env_name> [homelab_host]"
    echo ""
    echo "Arguments:"
    echo "  model_file   - Path to the .zip model file"
    echo "  env_name     - 'equity' or 'crypto'"
    echo "  homelab_host - Tailscale hostname (default: 'homelab')"
    exit 1
fi

MODEL_FILE="$1"
ENV_NAME="$2"
HOMELAB_HOST="${3:-homelab}"

# Validate model file exists
if [[ ! -f "$MODEL_FILE" ]]; then
    echo "ERROR: Model file not found: $MODEL_FILE"
    exit 1
fi

# Validate env_name
if [[ "$ENV_NAME" != "equity" && "$ENV_NAME" != "crypto" ]]; then
    echo "ERROR: env_name must be 'equity' or 'crypto', got: $ENV_NAME"
    exit 1
fi

# Determine obs_dim based on environment
if [[ "$ENV_NAME" == "equity" ]]; then
    OBS_DIM=$EQUITY_OBS_DIM
else
    OBS_DIM=$CRYPTO_OBS_DIM
fi

FILENAME=$(basename "$MODEL_FILE")
REMOTE_SHADOW_DIR="$REMOTE_PROJECT_DIR/models/shadow/$ENV_NAME"

echo "=== SwingRL Model Deployment ==="
echo "Model:    $MODEL_FILE"
echo "Env:      $ENV_NAME"
echo "Host:     $HOMELAB_HOST"
echo "Obs dim:  $OBS_DIM"
echo ""

# --- Step 1: Compute local SHA256 ---
echo "[1/4] Computing local SHA256 checksum..."
LOCAL_SHA=$(shasum -a 256 "$MODEL_FILE" | awk '{print $1}')
echo "  Local SHA256: $LOCAL_SHA"

# --- Step 2: SCP model to homelab shadow directory ---
echo "[2/4] Transferring model to $HOMELAB_HOST:$REMOTE_SHADOW_DIR/"
ssh "$HOMELAB_HOST" "mkdir -p $REMOTE_SHADOW_DIR"
scp "$MODEL_FILE" "$HOMELAB_HOST:$REMOTE_SHADOW_DIR/$FILENAME"
echo "  Transfer complete."

# --- Step 3: Verify SHA256 on remote ---
echo "[3/4] Verifying remote SHA256 checksum..."
REMOTE_SHA=$(ssh "$HOMELAB_HOST" "sha256sum $REMOTE_SHADOW_DIR/$FILENAME | awk '{print \$1}'")
echo "  Remote SHA256: $REMOTE_SHA"

if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
    echo "ERROR: SHA256 mismatch!"
    echo "  Local:  $LOCAL_SHA"
    echo "  Remote: $REMOTE_SHA"
    echo "  Removing corrupted remote file..."
    ssh "$HOMELAB_HOST" "rm -f $REMOTE_SHADOW_DIR/$FILENAME"
    exit 1
fi
echo "  Checksum verified."

# --- Step 4: Run 6-point smoke test on homelab ---
echo "[4/4] Running 6-point smoke test on $HOMELAB_HOST..."
SMOKE_TEST_CMD="cd $REMOTE_PROJECT_DIR && uv run python -c \"
from swingrl.shadow.lifecycle import smoke_test_model
from pathlib import Path
results = smoke_test_model(Path('models/shadow/$ENV_NAME/$FILENAME'), '$ENV_NAME', obs_dim=$OBS_DIM)
for check, passed in results.items():
    status = 'PASS' if passed else 'FAIL'
    print(f'  {check}: {status}')
if not all(results.values()):
    raise SystemExit(1)
\""

if ssh "$HOMELAB_HOST" "$SMOKE_TEST_CMD"; then
    echo ""
    echo "=== SUCCESS ==="
    echo "Model $FILENAME deployed to $HOMELAB_HOST:$REMOTE_SHADOW_DIR/"
    echo "All 6 smoke test checks passed."
    echo ""
    echo "Next steps:"
    echo "  1. Monitor shadow evaluation period"
    echo "  2. Promote when ready: ModelLifecycle(Path('models/')).promote('$ENV_NAME')"
    exit 0
else
    echo ""
    echo "=== FAILURE ==="
    echo "Smoke test failed for $FILENAME on $HOMELAB_HOST."
    echo "Model remains in shadow directory for investigation."
    exit 1
fi
