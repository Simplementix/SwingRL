"""Re-anchor B&H benchmark baselines to the agent's real first-fill prices (spec D13-D17).

Model A (equal-weight passive index): for every instrument the agent traded at its env's
origin, set ``baseline_price`` = the agent's EARLIEST buy-fill price, ``baseline_date`` = the
env origin (crypto 2026-07-22, equity 2026-07-23), and ``capital_usd`` = the env's total
portfolio value at origin. The digest math is unchanged — this only corrects the rows.

``--dry-run`` is the DEFAULT (prints a diff, writes nothing). ``--apply`` first writes the
current rows as restore-SQL to a timestamped backup file, upserts, then asserts each env's
baseline symbol-set equals exactly the origin-fill instrument set (a wrong count corrupts the
equal-weight divisor). LIVE-DB write — run only with explicit approval.

Usage (run from the repo root — the script self-imports ``scripts.``):
    python -m scripts.reanchor_benchmark_baselines            # dry-run
    python -m scripts.reanchor_benchmark_baselines --apply    # write (gated)
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Protocol

import structlog

from scripts.record_benchmark_baselines import BaselineRow, PostgresBaselineGateway
from swingrl.config.schema import load_config
from swingrl.utils.exceptions import DataError
from swingrl.utils.logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)

_ENVIRONMENTS: tuple[str, ...] = ("equity", "crypto")
_DEFAULT_ORIGINS: dict[str, date] = {"equity": date(2026, 7, 23), "crypto": date(2026, 7, 22)}


class ReanchorGateway(Protocol):
    """DB access for the re-anchor (keeps psycopg out of unit tests)."""

    def first_buy_fill(
        self, environment: str, symbol: str, origin_date: date
    ) -> tuple[float, date, str] | None:
        """(price, origin_date, trade_type) of the earliest origin-day buy fill, or None."""
        ...

    def origin_capital(self, environment: str, origin_date: date) -> tuple[float, float] | None:
        """(total_value, cash_balance) of the earliest origin-day snapshot, or None."""
        ...

    def current_baselines(self, environment: str) -> list[BaselineRow]:
        """Current benchmark_baselines rows for an env (for the diff + backup)."""
        ...

    def baseline_symbols(self, environment: str) -> set[str]:
        """Symbols currently in benchmark_baselines for an env (post-write invariant check)."""
        ...

    def upsert(self, rows: Sequence[BaselineRow]) -> None:
        """Idempotently upsert baseline rows on (environment, symbol)."""
        ...


def build_reanchor_rows(
    config: SwingRLConfig, gateway: ReanchorGateway, origins: dict[str, date]
) -> list[BaselineRow]:
    """Compute one corrected BaselineRow per env symbol from first fills + origin capital.

    Raises:
        DataError: an env has no origin-day snapshot, or a symbol has no origin-day buy fill.
    """
    rows: list[BaselineRow] = []
    for env in _ENVIRONMENTS:
        origin = origins[env]
        capital = gateway.origin_capital(env, origin)
        if capital is None:
            raise DataError(f"No {env} portfolio_snapshots on origin {origin}; cannot set capital.")
        total_value, _cash = capital
        for symbol in getattr(config, env).symbols:
            fill = gateway.first_buy_fill(env, symbol, origin)
            if fill is None:
                raise DataError(
                    f"No first buy-fill for {env}/{symbol} on origin {origin} — refusing to guess."
                )
            price, _dt, _ttype = fill
            rows.append(
                BaselineRow(
                    environment=env,
                    symbol=symbol,
                    baseline_date=origin,
                    baseline_price=float(price),
                    capital_usd=float(total_value),
                )
            )
    return rows


def _write_backup(gateway: ReanchorGateway, backup_dir: Path, stamp: str) -> Path:
    """Write current rows as restore-SQL (idempotent UPDATEs) to a timestamped file."""
    path = backup_dir / f"reanchor_backup_{stamp}.sql"
    lines = ["-- benchmark_baselines restore point (re-anchor)", "BEGIN;"]
    for env in _ENVIRONMENTS:
        for r in gateway.current_baselines(env):
            stmt = (
                # nosec B608 attaches to the SQL string node below (bandit anchors B608 there,
                # not on `stmt = (`): restore-SQL from our own dataclass, written to a file,
                # never executed.
                "UPDATE benchmark_baselines SET "  # nosec B608
                f"baseline_date = '{r.baseline_date}', baseline_price = {r.baseline_price}, "
                f"capital_usd = {r.capital_usd} "
                f"WHERE environment = '{r.environment}' AND symbol = '{r.symbol}';"
            )
            lines.append(stmt)
    lines.append("COMMIT;")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def reanchor(
    config: SwingRLConfig,
    gateway: ReanchorGateway,
    *,
    apply: bool,
    origins: dict[str, date],
    backup_dir: Path,
    stamp: str | None = None,
) -> list[BaselineRow]:
    """Compute corrected rows and, when ``apply``, back up + upsert + verify the row-set.

    Raises:
        DataError: missing data (via build_reanchor_rows), or a post-write env symbol-set that
            does not equal the origin-fill set (a gap/extra corrupts the equal-weight divisor).
    """
    rows = build_reanchor_rows(config, gateway, origins)
    if not apply:
        return rows
    stamp = stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    _write_backup(gateway, backup_dir, stamp)
    gateway.upsert(rows)
    for env in _ENVIRONMENTS:
        want = {r.symbol for r in rows if r.environment == env}
        have = gateway.baseline_symbols(env)
        if have != want:
            raise DataError(
                f"{env} baseline row-set {sorted(have)} != origin-fill set {sorted(want)}; "
                "extras/gaps corrupt the equal-weight divisor. Restore from the backup file."
            )
    log.info("benchmark_baselines_reanchored", rows=len(rows))
    return rows


class PostgresReanchorGateway(PostgresBaselineGateway):
    """Concrete gateway: first fills + origin snapshots + baselines from production PostgreSQL."""

    _FIRST_FILL = (
        "SELECT price, timestamp, trade_type FROM trades "
        "WHERE environment = %s AND symbol = %s AND side = 'buy' "
        "AND (timestamp AT TIME ZONE 'America/New_York')::date = %s "
        "ORDER BY timestamp ASC LIMIT 1"
    )
    _ORIGIN_CAP = (
        "SELECT total_value, cash_balance FROM portfolio_snapshots "
        "WHERE environment = %s AND (timestamp AT TIME ZONE 'America/New_York')::date = %s "
        "ORDER BY timestamp ASC LIMIT 1"
    )
    _CURRENT = (
        "SELECT symbol, baseline_date, baseline_price, capital_usd "
        "FROM benchmark_baselines WHERE environment = %s ORDER BY symbol"
    )
    _SYMBOLS = "SELECT symbol FROM benchmark_baselines WHERE environment = %s"

    def first_buy_fill(
        self, environment: str, symbol: str, origin_date: date
    ) -> tuple[float, date, str] | None:
        with self._db.connection() as conn:
            row = conn.execute(self._FIRST_FILL, (environment, symbol, origin_date)).fetchone()
        return None if row is None else (float(row["price"]), origin_date, row["trade_type"])

    def origin_capital(self, environment: str, origin_date: date) -> tuple[float, float] | None:
        with self._db.connection() as conn:
            row = conn.execute(self._ORIGIN_CAP, (environment, origin_date)).fetchone()
        return None if row is None else (float(row["total_value"]), float(row["cash_balance"]))

    def current_baselines(self, environment: str) -> list[BaselineRow]:
        with self._db.connection() as conn:
            rows = conn.execute(self._CURRENT, (environment,)).fetchall()
        return [
            BaselineRow(
                environment,
                r["symbol"],
                r["baseline_date"],
                float(r["baseline_price"]),
                float(r["capital_usd"]),
            )
            for r in rows
        ]

    def baseline_symbols(self, environment: str) -> set[str]:
        with self._db.connection() as conn:
            rows = conn.execute(self._SYMBOLS, (environment,)).fetchall()
        return {r["symbol"] for r in rows}


def _print_diff(
    proposed: list[BaselineRow],
    current_by_env: dict[str, dict[str, BaselineRow]],
    *,
    apply: bool,
    total_by_env: dict[str, float | None],
    cash_by_env: dict[str, float | None],
) -> None:
    """Print current-vs-proposed rows so the operator can verify before/after any write.

    ``current_by_env`` is the caller's PRE-write snapshot (symbol -> row per env); using it
    rather than re-querying keeps the ``[was …]`` column truthful after an ``--apply`` upsert.
    ``total_by_env``/``cash_by_env`` are the origin ``total_value``/``cash_balance`` per env so
    the operator can eyeball the D16 relationship (``None`` prints as ``unknown``).
    """
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n=== Re-anchor benchmark baselines ({mode}) ===")
    for env in _ENVIRONMENTS:
        current = current_by_env.get(env, {})
        total = total_by_env.get(env)
        cash = cash_by_env.get(env)
        if total is None or cash is None:
            print(f"\n{env}: origin capital total_value unknown  cash_balance unknown")
        else:
            print(
                f"\n{env}: origin capital total_value ${total:,.2f}  "
                f"cash_balance ${cash:,.2f}   (Δ ${total - cash:,.2f})"
            )
        for r in [p for p in proposed if p.environment == env]:
            cur = current.get(r.symbol)
            was = (
                f"was ${cur.baseline_price:,.4f}/cap ${cur.capital_usd:,.2f}"
                if cur
                else "was (none)"
            )
            print(
                f"  {r.symbol:<10} price ${r.baseline_price:,.4f}  cap ${r.capital_usd:,.2f}  "
                f"date {r.baseline_date}   [{was}]"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-anchor B&H benchmark baselines (D13-D17).")
    parser.add_argument("--apply", action="store_true", help="Write rows (default: dry-run).")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report only (default; wins over --apply)."
    )
    parser.add_argument("--config", default="config/swingrl.yaml")
    parser.add_argument("--equity-origin", default=_DEFAULT_ORIGINS["equity"].isoformat())
    parser.add_argument("--crypto-origin", default=_DEFAULT_ORIGINS["crypto"].isoformat())
    parser.add_argument("--backup-dir", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (0 = ok; 1 = refused/aborted)."""
    from pathlib import Path

    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    origins = {
        "equity": date.fromisoformat(args.equity_origin),
        "crypto": date.fromisoformat(args.crypto_origin),
    }
    apply = bool(args.apply) and not bool(args.dry_run)

    from swingrl.data.db import DatabaseManager  # noqa: PLC0415

    gateway = PostgresReanchorGateway(DatabaseManager(config))
    # Snapshot current baselines + origin capital BEFORE reanchor upserts, so an --apply diff
    # reflects real before/after (post-write re-query would show the just-written rows).
    before_by_env = {
        env: {r.symbol: r for r in gateway.current_baselines(env)} for env in _ENVIRONMENTS
    }
    total_by_env: dict[str, float | None] = {}
    cash_by_env: dict[str, float | None] = {}
    for env in _ENVIRONMENTS:
        cap = gateway.origin_capital(env, origins[env])
        total_by_env[env] = None if cap is None else cap[0]
        cash_by_env[env] = None if cap is None else cap[1]
    try:
        rows = reanchor(
            config, gateway, apply=apply, origins=origins, backup_dir=Path(args.backup_dir)
        )
    except DataError as exc:
        log.error("benchmark_reanchor_refused", error=str(exc))
        print(f"REFUSED: {exc}")
        return 1
    _print_diff(
        rows,
        before_by_env,
        apply=apply,
        total_by_env=total_by_env,
        cash_by_env=cash_by_env,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
