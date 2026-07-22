"""Tests for scripts/record_benchmark_baselines.py — the D13 buy-and-hold recorder.

The recorder snapshots one benchmark_baselines row per env symbol at epoch reset:
the latest stored close as baseline_price, today as baseline_date, and the env's
TOTAL configured capital as capital_usd (the digest's ``_benchmark_value`` equal-weights
that total across the env's symbols). ``--dry-run`` is the default (prints, writes
nothing); a write is an idempotent upsert on (environment, symbol) that refuses to
overwrite an env that already has baselines unless ``--force`` is given.

All tests run WITHOUT a database: the DB is replaced by an in-memory fake gateway.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from scripts.record_benchmark_baselines import (
    BaselineRow,
    build_baseline_rows,
    record_baselines,
)
from swingrl.utils.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from swingrl.config.schema import SwingRLConfig

_TODAY = date(2026, 7, 22)


class _FakeGateway:
    """In-memory BaselineGateway: canned closes + recorded upsert (no psycopg)."""

    def __init__(
        self,
        closes: dict[tuple[str, str], float],
        existing: set[str] | None = None,
    ) -> None:
        self._closes = closes
        self._existing = set(existing or set())
        self.upserted: list[BaselineRow] | None = None

    def latest_close(self, environment: str, symbol: str) -> float | None:
        return self._closes.get((environment, symbol))

    def environments_with_baselines(self) -> set[str]:
        return set(self._existing)

    def upsert(self, rows: Sequence[BaselineRow]) -> None:
        self.upserted = list(rows)


def _all_closes(
    config: SwingRLConfig, *, equity: float, crypto: float
) -> dict[tuple[str, str], float]:
    """Canned latest close for every configured symbol in both envs."""
    closes: dict[tuple[str, str], float] = {("equity", s): equity for s in config.equity.symbols}
    closes.update({("crypto", s): crypto for s in config.crypto.symbols})
    return closes


def test_rows_snapshot_total_capital_and_latest_close(loaded_config: SwingRLConfig) -> None:
    """BENCH-D13: each row carries env TOTAL capital, latest close, and today's date."""
    gateway = _FakeGateway(_all_closes(loaded_config, equity=100.0, crypto=50.0))
    rows = build_baseline_rows(loaded_config, gateway, _TODAY)

    crypto = [r for r in rows if r.environment == "crypto"]
    assert {r.symbol for r in crypto} == set(loaded_config.crypto.symbols)
    for r in crypto:
        assert r.capital_usd == loaded_config.capital.crypto_usd  # env total, not the slice
        assert r.baseline_price == 50.0
        assert r.baseline_date == _TODAY

    equity = [r for r in rows if r.environment == "equity"]
    for r in equity:
        assert r.capital_usd == loaded_config.capital.equity_usd
        assert r.baseline_price == 100.0


def test_skips_symbol_with_no_stored_close(loaded_config: SwingRLConfig) -> None:
    """BENCH-D13: a symbol with no stored bar is skipped (no baseline_price to anchor)."""
    closes = _all_closes(loaded_config, equity=100.0, crypto=50.0)
    dropped = loaded_config.crypto.symbols[0]
    del closes[("crypto", dropped)]
    gateway = _FakeGateway(closes)

    rows = build_baseline_rows(loaded_config, gateway, _TODAY)
    assert dropped not in {r.symbol for r in rows if r.environment == "crypto"}


def test_dry_run_default_writes_nothing(loaded_config: SwingRLConfig) -> None:
    """BENCH-D13: dry-run (apply=False) returns the rows but never calls upsert."""
    gateway = _FakeGateway(_all_closes(loaded_config, equity=100.0, crypto=50.0))
    rows = record_baselines(loaded_config, gateway, apply=False, force=False, today=_TODAY)
    assert rows  # rows computed for the operator to eyeball
    assert gateway.upserted is None  # nothing written


def test_apply_writes_upsert_on_fresh_env(loaded_config: SwingRLConfig) -> None:
    """BENCH-D13: apply with no existing baselines writes every computed row."""
    gateway = _FakeGateway(_all_closes(loaded_config, equity=100.0, crypto=50.0))
    rows = record_baselines(loaded_config, gateway, apply=True, force=False, today=_TODAY)
    assert gateway.upserted == rows


def test_refuses_overwrite_without_force(loaded_config: SwingRLConfig) -> None:
    """BENCH-D13: existing baselines + no --force -> refuse (DataError), write nothing."""
    gateway = _FakeGateway(
        _all_closes(loaded_config, equity=100.0, crypto=50.0), existing={"crypto"}
    )
    with pytest.raises(DataError):
        record_baselines(loaded_config, gateway, apply=True, force=False, today=_TODAY)
    assert gateway.upserted is None


def test_force_overwrites_existing(loaded_config: SwingRLConfig) -> None:
    """BENCH-D13: --force upserts even when the env already has baselines."""
    gateway = _FakeGateway(
        _all_closes(loaded_config, equity=100.0, crypto=50.0), existing={"crypto"}
    )
    rows = record_baselines(loaded_config, gateway, apply=True, force=True, today=_TODAY)
    assert gateway.upserted == rows
