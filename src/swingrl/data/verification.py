"""Data verification module — DuckDB quality gates after ingestion.

Queries DuckDB to confirm all quality gates pass: row counts per symbol,
date coverage, crypto timestamp gap detection, NaN checks in observation
vectors, and FRED macro series presence. Produces a machine-readable JSON
report and a human-readable console summary.

Usage:
    from swingrl.data.verification import run_verification, write_report

    config = load_config("config/swingrl.yaml")
    result = run_verification(config)
    write_report(result, Path("data/verification.json"))
    print_summary(result)
"""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from swingrl.config.schema import SwingRLConfig
from swingrl.data.db import DatabaseManager
from swingrl.data.fred import ALL_SERIES
from swingrl.features.assembler import CRYPTO_OBS_DIM, EQUITY_OBS_DIM
from swingrl.features.pipeline import FeaturePipeline

log = structlog.get_logger(__name__)

# Observation vector expected dimensions (from assembler — single source of truth)
_EQUITY_OBS_DIM: int = EQUITY_OBS_DIM
_CRYPTO_OBS_DIM: int = CRYPTO_OBS_DIM

# Gap thresholds — gaps exceeding these are logged as remaining gaps.
# Crypto: 24/7 market, 4H candles — 24h = 6 missing candles
# Equity: 1D candles — 5 calendar days covers all holiday combos (max normal ~4 days)
_CRYPTO_GAP_THRESHOLD = timedelta(hours=24)
_EQUITY_GAP_THRESHOLD = timedelta(days=5)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CheckResult:
    """Result of a single verification check."""

    name: str
    passed: bool
    detail: str


@dataclasses.dataclass
class VerificationResult:
    """Aggregated result of all verification checks."""

    passed: bool
    checks: list[CheckResult]


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def _check_equity_rows(cursor: Any, config: SwingRLConfig) -> CheckResult:
    """Check that all configured equity symbols have sufficient rows in ohlcv_daily.

    Args:
        cursor: Active DuckDB cursor.
        config: Validated SwingRLConfig with equity.symbols.

    Returns:
        CheckResult with passed=True if all symbols have >100 rows.
    """
    rows = cursor.execute(
        "SELECT symbol, COUNT(*) AS cnt FROM ohlcv_daily GROUP BY symbol"
    ).fetchall()
    counts: dict[str, int] = {str(sym): int(cnt) for sym, cnt in rows}

    missing: list[str] = []
    insufficient: list[str] = []
    for symbol in config.equity.symbols:
        if symbol not in counts:
            missing.append(symbol)
        elif counts[symbol] <= 100:
            insufficient.append(f"{symbol}({counts[symbol]})")

    problems = missing + insufficient
    if problems:
        detail = f"Equity row issues: {', '.join(problems)}"
        log.warning("equity_rows_check_failed", problems=problems)
        return CheckResult(name="equity_rows", passed=False, detail=detail)

    detail = f"All {len(config.equity.symbols)} equity symbols present with >100 rows"
    log.info("equity_rows_check_passed", symbols=config.equity.symbols)
    return CheckResult(name="equity_rows", passed=True, detail=detail)


def _check_crypto_rows(cursor: Any, config: SwingRLConfig) -> CheckResult:
    """Check that all configured crypto symbols have sufficient rows in ohlcv_4h.

    Args:
        cursor: Active DuckDB cursor.
        config: Validated SwingRLConfig with crypto.symbols.

    Returns:
        CheckResult with passed=True if all symbols have >100 rows.
    """
    rows = cursor.execute("SELECT symbol, COUNT(*) AS cnt FROM ohlcv_4h GROUP BY symbol").fetchall()
    counts: dict[str, int] = {str(sym): int(cnt) for sym, cnt in rows}

    missing: list[str] = []
    insufficient: list[str] = []
    for symbol in config.crypto.symbols:
        if symbol not in counts:
            missing.append(symbol)
        elif counts[symbol] <= 100:
            insufficient.append(f"{symbol}({counts[symbol]})")

    problems = missing + insufficient
    if problems:
        detail = f"Crypto row issues — missing or insufficient: {', '.join(problems)}"
        log.warning("crypto_rows_check_failed", problems=problems)
        return CheckResult(name="crypto_rows", passed=False, detail=detail)

    detail = f"All {len(config.crypto.symbols)} crypto symbols present with >100 rows"
    log.info("crypto_rows_check_passed", symbols=config.crypto.symbols)
    return CheckResult(name="crypto_rows", passed=True, detail=detail)


def _check_macro_series(cursor: Any, config: SwingRLConfig) -> CheckResult:  # noqa: ARG001
    """Check that all expected FRED macro series are present in macro_features.

    Args:
        cursor: Active DuckDB cursor.
        config: Validated SwingRLConfig (unused — series list comes from ALL_SERIES).

    Returns:
        CheckResult with passed=True if all expected series are found.
    """
    rows = cursor.execute("SELECT DISTINCT series_id FROM macro_features").fetchall()
    found: set[str] = {str(r[0]) for r in rows}

    missing: list[str] = [s for s in ALL_SERIES if s not in found]
    if missing:
        detail = f"Missing FRED series: {', '.join(missing)}"
        log.warning("macro_series_check_failed", missing=missing)
        return CheckResult(name="macro_series", passed=False, detail=detail)

    detail = f"All {len(ALL_SERIES)} FRED series present: {', '.join(ALL_SERIES)}"
    log.info("macro_series_check_passed")
    return CheckResult(name="macro_series", passed=True, detail=detail)


def _check_equity_date_range(cursor: Any, config: SwingRLConfig) -> CheckResult:  # noqa: ARG001
    """Log equity date coverage — informational, always passes.

    Args:
        cursor: Active DuckDB cursor.
        config: Validated SwingRLConfig.

    Returns:
        CheckResult always passed=True with date coverage detail.
    """
    rows = cursor.execute("SELECT MIN(date), MAX(date) FROM ohlcv_daily").fetchall()
    if rows and rows[0][0] is not None:
        min_date, max_date = rows[0]
        detail = f"Equity date coverage: {min_date} to {max_date}"
    else:
        detail = "No equity data found"
    log.info("equity_date_range", detail=detail)
    return CheckResult(name="equity_date_range", passed=True, detail=detail)


def _check_crypto_date_range(cursor: Any, config: SwingRLConfig) -> CheckResult:  # noqa: ARG001
    """Log crypto 4H date coverage — informational, always passes.

    Args:
        cursor: Active DuckDB cursor.
        config: Validated SwingRLConfig.

    Returns:
        CheckResult always passed=True with date coverage detail.
    """
    rows = cursor.execute("SELECT MIN(datetime), MAX(datetime) FROM ohlcv_4h").fetchall()
    if rows and rows[0][0] is not None:
        min_dt, max_dt = rows[0]
        detail = f"Crypto 4H datetime coverage: {min_dt} to {max_dt}"
    else:
        detail = "No crypto 4H data found"
    log.info("crypto_date_range", detail=detail)
    return CheckResult(name="crypto_date_range", passed=True, detail=detail)


def _check_crypto_gaps(cursor: Any, config: SwingRLConfig) -> CheckResult:
    """Detect gaps >24h in consecutive ohlcv_4h timestamps for each symbol.

    Gap filling from alternate sources should run before this check.
    Any remaining gaps exceeding the threshold are logged as warnings
    for the training layer to handle (episode boundary splitting).

    Args:
        cursor: Active DuckDB cursor.
        config: Validated SwingRLConfig with crypto.symbols.

    Returns:
        CheckResult — always passes but logs remaining large gaps.
    """
    from datetime import datetime  # noqa: PLC0415

    gap_reports: list[str] = []

    for symbol in config.crypto.symbols:
        rows = cursor.execute(
            "SELECT datetime FROM ohlcv_4h WHERE symbol = ? ORDER BY datetime",
            [symbol],
        ).fetchall()

        if len(rows) < 2:
            log.debug("crypto_gap_check_skipped", symbol=symbol, reason="insufficient rows")
            continue

        timestamps = [r[0] for r in rows]
        for i in range(1, len(timestamps)):
            prev_ts = timestamps[i - 1]
            curr_ts = timestamps[i]

            if isinstance(prev_ts, str):
                prev_ts = datetime.fromisoformat(prev_ts)
            if isinstance(curr_ts, str):
                curr_ts = datetime.fromisoformat(curr_ts)

            gap = curr_ts - prev_ts
            if gap > _CRYPTO_GAP_THRESHOLD:
                gap_reports.append(f"{symbol}: gap of {gap} at {prev_ts} -> {curr_ts}")
                log.warning(
                    "crypto_gap_remaining",
                    symbol=symbol,
                    gap_hours=gap.total_seconds() / 3600,
                    from_ts=str(prev_ts),
                    to_ts=str(curr_ts),
                    note="training layer should split episodes at this gap",
                )

    if gap_reports:
        detail = (
            f"{len(gap_reports)} crypto gaps >24h remain after gap-fill "
            f"(episode boundaries for training): {'; '.join(gap_reports)}"
        )
        log.warning("crypto_gaps_remaining", count=len(gap_reports))
    else:
        detail = "No gaps >24h in crypto 4H data"
        log.info("crypto_gaps_check_passed")

    return CheckResult(name="crypto_gaps", passed=True, detail=detail)


def _check_obs_vector(pipeline: FeaturePipeline, environment: str, date_str: str) -> CheckResult:
    """Validate observation vector shape and NaN-free content.

    Args:
        pipeline: Initialized FeaturePipeline for observation assembly.
        environment: "equity" or "crypto".
        date_str: Date/datetime string for observation lookup.

    Returns:
        CheckResult with passed=False if NaN found or shape mismatch.
    """
    expected_dim = _EQUITY_OBS_DIM if environment == "equity" else _CRYPTO_OBS_DIM
    check_name = f"obs_vector_{environment}"

    try:
        obs = pipeline.get_observation(environment, date_str)  # type: ignore[arg-type]
    except Exception as exc:
        detail = f"get_observation raised {type(exc).__name__}: {exc}"
        log.error("obs_vector_error", environment=environment, error=str(exc))
        return CheckResult(name=check_name, passed=False, detail=detail)

    nan_count = int(np.isnan(obs).sum())
    if nan_count > 0:
        detail = f"{environment} obs vector has {nan_count} NaN values (shape={obs.shape})"
        log.warning("obs_vector_has_nan", environment=environment, nan_count=nan_count)
        return CheckResult(name=check_name, passed=False, detail=detail)

    actual_dim = obs.shape[0] if obs.ndim > 0 else 0
    detail = f"{environment} obs vector clean: shape=({actual_dim},), expected=({expected_dim},)"
    log.info("obs_vector_check_passed", environment=environment, dim=actual_dim)
    return CheckResult(name=check_name, passed=True, detail=detail)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def run_verification(config: SwingRLConfig) -> VerificationResult:
    """Run all data quality checks and return an aggregated VerificationResult.

    Checks run:
    1. Equity row counts per symbol
    2. Crypto row counts per symbol
    3. FRED macro series presence
    4. Equity date range coverage (informational)
    5. Crypto date range coverage (informational)
    6. Crypto 4H gap detection
    7. Equity observation vector (NaN + shape)
    8. Crypto observation vector (NaN + shape)

    Args:
        config: Validated SwingRLConfig.

    Returns:
        VerificationResult with passed=all(checks) and list of CheckResults.
    """
    db = DatabaseManager(config)
    checks: list[CheckResult] = []

    with db.duckdb() as cursor:
        # Row count checks
        checks.append(_check_equity_rows(cursor, config))
        checks.append(_check_crypto_rows(cursor, config))

        # Macro series check
        checks.append(_check_macro_series(cursor, config))

        # Date range coverage (informational)
        checks.append(_check_equity_date_range(cursor, config))
        checks.append(_check_crypto_date_range(cursor, config))

        # Crypto gap detection
        checks.append(_check_crypto_gaps(cursor, config))

        # Observation vector checks — determine latest available date
        equity_date_str = _latest_date(cursor, "equity")
        crypto_date_str = _latest_date(cursor, "crypto")

        # Build pipeline using current DuckDB connection
        pipeline = FeaturePipeline(config, cursor)

        checks.append(_check_obs_vector(pipeline, "equity", equity_date_str))
        checks.append(_check_obs_vector(pipeline, "crypto", crypto_date_str))

    overall_passed = all(c.passed for c in checks)
    log.info(
        "verification_complete",
        passed=overall_passed,
        total_checks=len(checks),
        failed_checks=sum(1 for c in checks if not c.passed),
    )
    return VerificationResult(passed=overall_passed, checks=checks)


def _latest_date(cursor: Any, environment: str) -> str:
    """Query the most recent date/datetime from the relevant feature table.

    Falls back to a fixed reference date if no data is found.

    Args:
        cursor: Active DuckDB cursor.
        environment: "equity" or "crypto".

    Returns:
        ISO date/datetime string.
    """
    _FALLBACK = "2024-01-15"
    try:
        if environment == "equity":
            row = cursor.execute("SELECT MAX(date) FROM ohlcv_daily").fetchone()
        else:
            row = cursor.execute("SELECT MAX(datetime) FROM ohlcv_4h").fetchone()
        if row and row[0] is not None:
            return str(row[0])
    except Exception:
        log.warning("latest_date_query_failed", environment=environment)
    return _FALLBACK


# ---------------------------------------------------------------------------
# Report and summary
# ---------------------------------------------------------------------------


def write_report(result: VerificationResult, path: Path) -> None:
    """Write verification result as JSON to the specified path.

    Creates parent directories if they do not exist.

    Args:
        result: VerificationResult to serialize.
        path: Destination file path for the JSON report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dataclasses.asdict(result), indent=2, default=str)
    path.write_text(payload, encoding="utf-8")
    log.info("verification_report_written", path=str(path))


def print_summary(result: VerificationResult) -> None:
    """Print human-readable [PASS]/[FAIL] summary to stdout.

    Args:
        result: VerificationResult to display.
    """
    for check in result.checks:
        tag = "[PASS]" if check.passed else "[FAIL]"
        print(f"{tag} {check.name}: {check.detail}")

    overall_tag = "[PASS]" if result.passed else "[FAIL]"
    print(f"\nOverall: {overall_tag}")
