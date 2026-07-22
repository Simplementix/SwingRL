"""Record buy-and-hold benchmark baselines at epoch reset (spec D13).

At each epoch reset this one-shot recorder snapshots one ``benchmark_baselines`` row per
env symbol: the latest stored close as ``baseline_price``, today (UTC) as ``baseline_date``,
and the env's TOTAL configured capital (``config.capital.equity_usd`` / ``crypto_usd``) as
``capital_usd``. The daily digest's ``_benchmark_value()`` later equal-weights that capital
across the env's symbols and grows each slice by ``latest_close / baseline_price``, so
"agent vs passive buy-and-hold" is visible every day.

The env total is persisted AT RECORD TIME so later config drift never silently moves the
benchmark — the snapshot, not config, is the source of truth afterward.

``--dry-run`` is the DEFAULT: it prints the rows and writes nothing. ``--apply`` writes; the
write is an idempotent upsert on ``(environment, symbol)`` that REFUSES to overwrite an env
that already has baselines unless ``--force`` is given (so a mid-epoch re-run cannot silently
reset the benchmark).

Usage:
    python scripts/record_benchmark_baselines.py                  # dry-run (default)
    python scripts/record_benchmark_baselines.py --apply          # write (fresh env)
    python scripts/record_benchmark_baselines.py --apply --force  # overwrite existing
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Protocol

import structlog

from swingrl.config.schema import load_config
from swingrl.utils.exceptions import DataError
from swingrl.utils.logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import Sequence

    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Envs recorded, in report order. Each maps to its ohlcv close table (fixed SQL — no
# interpolation — so bandit stays quiet; ORDER BY column differs per env).
_ENVIRONMENTS: tuple[str, ...] = ("equity", "crypto")
_LATEST_CLOSE_SQL: dict[str, str] = {
    "equity": "SELECT close FROM ohlcv_daily WHERE symbol = %s ORDER BY date DESC LIMIT 1",
    "crypto": "SELECT close FROM ohlcv_4h WHERE symbol = %s ORDER BY datetime DESC LIMIT 1",
}


@dataclass
class BaselineRow:
    """One benchmark_baselines row: an env symbol's epoch-reset snapshot.

    ``capital_usd`` is the env TOTAL (not the per-symbol slice) — the digest divides by
    the symbol count when it values the benchmark.
    """

    environment: str
    symbol: str
    baseline_date: date
    baseline_price: float
    capital_usd: float


class BaselineGateway(Protocol):
    """Minimal DB access the recorder needs (keeps psycopg out of the unit tests)."""

    def latest_close(self, environment: str, symbol: str) -> float | None:
        """Return the latest stored close for ``symbol`` in ``environment``, or None."""
        ...

    def environments_with_baselines(self) -> set[str]:
        """Return the set of environments that already have benchmark_baselines rows."""
        ...

    def upsert(self, rows: Sequence[BaselineRow]) -> None:
        """Idempotently upsert baseline rows on (environment, symbol)."""
        ...


def build_baseline_rows(
    config: SwingRLConfig, gateway: BaselineGateway, today: date
) -> list[BaselineRow]:
    """Build one BaselineRow per env symbol from config capital + the latest stored close.

    Args:
        config: Validated SwingRLConfig (symbols + capital per env).
        gateway: Data access providing the latest close per symbol.
        today: Baseline date to stamp on every row (the epoch-reset day).

    Returns:
        Baseline rows for every symbol that has a stored close; symbols with no stored
        bar are skipped (logged) since there is no price to anchor their baseline.
    """
    rows: list[BaselineRow] = []
    for env in _ENVIRONMENTS:
        symbols: list[str] = list(getattr(config, env).symbols)
        capital = float(getattr(config.capital, f"{env}_usd"))
        for symbol in symbols:
            close = gateway.latest_close(env, symbol)
            if close is None:
                log.warning("benchmark_baseline_no_close", environment=env, symbol=symbol)
                continue
            rows.append(
                BaselineRow(
                    environment=env,
                    symbol=symbol,
                    baseline_date=today,
                    baseline_price=float(close),
                    capital_usd=capital,
                )
            )
    return rows


def record_baselines(
    config: SwingRLConfig,
    gateway: BaselineGateway,
    *,
    apply: bool,
    force: bool,
    today: date | None = None,
) -> list[BaselineRow]:
    """Compute baseline rows and, when ``apply`` is True, upsert them.

    Args:
        config: Validated SwingRLConfig.
        gateway: Data access (latest close, existing-env probe, upsert).
        apply: When False (default caller behavior), compute + return only (dry-run).
        force: When True, overwrite envs that already have baselines. When False, an
            env that already has baselines makes this refuse (raise) rather than clobber
            a live benchmark mid-epoch.
        today: Baseline date (defaults to today, UTC).

    Returns:
        The computed baseline rows (also the rows written when ``apply`` is True).

    Raises:
        DataError: ``apply`` is True, ``force`` is False, and some targeted env already
            has baselines.
    """
    today = today or datetime.now(UTC).date()
    rows = build_baseline_rows(config, gateway, today)
    if not apply:
        return rows
    conflicts = {row.environment for row in rows} & gateway.environments_with_baselines()
    if conflicts and not force:
        raise DataError(
            f"Baselines already exist for {sorted(conflicts)}; refusing to overwrite "
            "without --force."
        )
    gateway.upsert(rows)
    log.info("benchmark_baselines_recorded", rows=len(rows), force=force)
    return rows


class PostgresBaselineGateway:
    """Concrete BaselineGateway backed by the production PostgreSQL database."""

    _EXISTING = "SELECT DISTINCT environment FROM benchmark_baselines"
    _UPSERT = (
        "INSERT INTO benchmark_baselines"
        " (environment, symbol, baseline_date, baseline_price, capital_usd)"
        " VALUES (%s, %s, %s, %s, %s)"
        " ON CONFLICT (environment, symbol) DO UPDATE SET"
        " baseline_date = EXCLUDED.baseline_date,"
        " baseline_price = EXCLUDED.baseline_price,"
        " capital_usd = EXCLUDED.capital_usd"
    )

    def __init__(self, db: DatabaseManager) -> None:
        """Wrap a DatabaseManager for benchmark_baselines read/probe/upsert."""
        self._db = db

    def latest_close(self, environment: str, symbol: str) -> float | None:
        """Return the latest stored close for ``symbol`` in ``environment``, or None."""
        sql = _LATEST_CLOSE_SQL[environment]
        with self._db.connection() as conn:
            row = conn.execute(sql, (symbol,)).fetchone()
        if row is None or row["close"] is None:
            return None
        return float(row["close"])

    def environments_with_baselines(self) -> set[str]:
        """Return the set of environments that already have benchmark_baselines rows."""
        with self._db.connection() as conn:
            rows = conn.execute(self._EXISTING).fetchall()
        return {r["environment"] for r in rows}

    def upsert(self, rows: Sequence[BaselineRow]) -> None:
        """Idempotently upsert baseline rows on (environment, symbol)."""
        with self._db.connection() as conn:
            for row in rows:
                conn.execute(
                    self._UPSERT,
                    (
                        row.environment,
                        row.symbol,
                        row.baseline_date,
                        row.baseline_price,
                        row.capital_usd,
                    ),
                )


def _print_rows(rows: list[BaselineRow], *, apply: bool) -> None:
    """Print a human-readable per-env summary of the computed/written rows."""
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n=== Benchmark baselines ({mode}) ===")
    if not rows:
        print("No baseline rows (no stored closes for any configured symbol).")
        return
    verb = "wrote" if apply else "would write"
    for env in _ENVIRONMENTS:
        env_rows = [r for r in rows if r.environment == env]
        if not env_rows:
            continue
        capital = env_rows[0].capital_usd
        print(f"\n{env}: {verb} {len(env_rows)} rows (capital ${capital:,.2f}, equal-weight)")
        for row in env_rows:
            print(
                f"  {row.symbol:<10} baseline_price ${row.baseline_price:,.2f}  ({row.baseline_date})"
            )
    print(f"\nTotal rows: {len(rows)}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Record buy-and-hold benchmark baselines (spec D13).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write rows to benchmark_baselines (upsert). Default: dry-run (print only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; never write (the default). Wins over --apply if both are given.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite envs that already have baselines. Without it, --apply refuses them.",
    )
    parser.add_argument(
        "--config",
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the benchmark-baseline recorder.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success; 1 = refused overwrite without --force).
    """
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    apply = bool(args.apply) and not bool(args.dry_run)

    from swingrl.data.db import DatabaseManager  # noqa: PLC0415  # lazy: avoids DB import cost

    db = DatabaseManager(config)
    gateway = PostgresBaselineGateway(db)
    try:
        rows = record_baselines(config, gateway, apply=apply, force=bool(args.force))
    except DataError as exc:
        log.error("benchmark_baselines_refused", error=str(exc))
        print(f"REFUSED: {exc}")
        return 1

    _print_rows(rows, apply=apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
