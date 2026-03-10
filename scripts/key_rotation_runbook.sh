#!/usr/bin/env bash
# ============================================================================
# SwingRL Key Rotation Runbook
# ============================================================================
#
# 90-Day Staggered Rotation Schedule:
#   Month 1: Alpaca API key rotation
#   Month 2: Binance.US API key rotation
#   Month 3: Buffer (no rotation — review logs, verify IP allowlist)
#
# This script serves as both documentation and executable helper.
# Run: bash scripts/key_rotation_runbook.sh check_key_age
# Run: bash scripts/key_rotation_runbook.sh show_schedule
#
# ============================================================================

set -euo pipefail

ENV_FILE="${ENV_FILE:-.env}"
MAX_KEY_AGE_DAYS=90

# ============================================================================
# SCHEDULE
# ============================================================================
show_schedule() {
    echo "============================================"
    echo "  90-Day Staggered Key Rotation Schedule"
    echo "============================================"
    echo ""
    echo "MONTH 1: Alpaca API Key Rotation"
    echo "  1. Log in to Alpaca dashboard (https://app.alpaca.markets)"
    echo "  2. Navigate to API Keys > Generate New Key"
    echo "  3. Copy the new API Key ID and Secret"
    echo "  4. Update .env:"
    echo "       ALPACA_API_KEY=<new-key-id>"
    echo "       ALPACA_API_SECRET=<new-secret>"
    echo "  5. Restart container:"
    echo "       docker compose restart swingrl"
    echo "  6. Verify paper order submission:"
    echo "       docker compose exec swingrl python -c \\"
    echo "         \"from swingrl.execution.alpaca_adapter import AlpacaAdapter; \\"
    echo "          a = AlpacaAdapter(); print(a.get_account_info())\""
    echo "  7. Delete old key in Alpaca dashboard"
    echo ""
    echo "MONTH 2: Binance.US API Key Rotation"
    echo "  1. Log in to Binance.US dashboard (https://www.binance.us)"
    echo "  2. Navigate to API Management > Create New API Key"
    echo "  3. CRITICAL security settings for new key:"
    echo "       - Withdrawals: OFF (never enable)"
    echo "       - IP allowlist: homelab IP only"
    echo "       - Permissions: Reading + Spot Trading only"
    echo "  4. Copy the new API Key and Secret"
    echo "  5. Update .env:"
    echo "       BINANCE_API_KEY=<new-key>"
    echo "       BINANCE_API_SECRET=<new-secret>"
    echo "  6. Restart container:"
    echo "       docker compose restart swingrl"
    echo "  7. Verify price fetch:"
    echo "       docker compose exec swingrl python -c \\"
    echo "         \"from swingrl.data.ingestors.binance_ingestor import BinanceIngestor; \\"
    echo "          b = BinanceIngestor(); print(b.fetch_latest('BTCUSDT'))\""
    echo "  8. Delete old key in Binance.US dashboard"
    echo ""
    echo "MONTH 3: Buffer (No Rotation)"
    echo "  1. Review access logs for anomalous activity"
    echo "  2. Verify IP allowlist is current (homelab IP unchanged)"
    echo "  3. Confirm .env file permissions are 600:"
    echo "       stat -c '%a' .env  # Should show 600"
    echo "  4. Run security checklist:"
    echo "       python scripts/security_checklist.py"
    echo ""
}

# ============================================================================
# KEY AGE CHECK
# ============================================================================
check_key_age() {
    if [ ! -f "$ENV_FILE" ]; then
        echo "ERROR: .env file not found at $ENV_FILE"
        exit 1
    fi

    # Get .env modification time (last key update indicator)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: use stat -f
        last_modified=$(stat -f %m "$ENV_FILE")
    else
        # Linux: use stat -c
        last_modified=$(stat -c %Y "$ENV_FILE")
    fi

    current_time=$(date +%s)
    age_seconds=$((current_time - last_modified))
    age_days=$((age_seconds / 86400))

    echo "Key Age Check"
    echo "============="
    echo "  .env last modified: $(date -r "$last_modified" 2>/dev/null || date -d "@$last_modified" 2>/dev/null || echo "unknown")"
    echo "  Age: ${age_days} days"
    echo "  Threshold: ${MAX_KEY_AGE_DAYS} days"
    echo ""

    if [ "$age_days" -gt "$MAX_KEY_AGE_DAYS" ]; then
        echo "WARNING: .env is ${age_days} days old (exceeds ${MAX_KEY_AGE_DAYS}-day rotation threshold)"
        echo "ACTION: Rotate API keys per the schedule above."
        exit 1
    else
        remaining=$((MAX_KEY_AGE_DAYS - age_days))
        echo "OK: Keys are within rotation window (${remaining} days remaining)"
        exit 0
    fi
}

# ============================================================================
# MAIN
# ============================================================================
main() {
    local command="${1:-show_schedule}"

    case "$command" in
        show_schedule)
            show_schedule
            ;;
        check_key_age)
            check_key_age
            ;;
        *)
            echo "Usage: $0 {show_schedule|check_key_age}"
            echo ""
            echo "  show_schedule  - Display the full 90-day rotation runbook"
            echo "  check_key_age  - Check if .env is older than ${MAX_KEY_AGE_DAYS} days"
            exit 1
            ;;
    esac
}

main "$@"
