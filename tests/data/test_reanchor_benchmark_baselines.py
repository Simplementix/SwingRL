"""REQ D13-D17: re-anchor benchmark baselines to the agent's real first-fill prices."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.reanchor_benchmark_baselines import build_reanchor_rows, reanchor
from swingrl.config.schema import SwingRLConfig
from swingrl.utils.exceptions import DataError

_ORIGINS = {"equity": date(2026, 7, 23), "crypto": date(2026, 7, 22)}


class _FakeGateway:
    """In-memory ReanchorGateway (no psycopg), mirroring test_record_benchmark_baselines.py."""

    def __init__(
        self,
        fills: dict[tuple[str, str], tuple[float, str]],
        capitals: dict[str, tuple[float, float]],
        existing_symbols: dict[str, set[str]] | None = None,
    ) -> None:
        self.fills = fills
        self.capitals = capitals
        self._existing = existing_symbols or {}
        self.upserted: list = []

    def first_buy_fill(
        self, environment: str, symbol: str, origin_date: date
    ) -> tuple[float, date, str] | None:
        hit = self.fills.get((environment, symbol))
        return None if hit is None else (hit[0], origin_date, hit[1])

    def origin_capital(self, environment: str, origin_date: date) -> tuple[float, float] | None:
        return self.capitals.get(environment)

    def current_baselines(self, environment: str) -> list:
        return []

    def baseline_symbols(self, environment: str) -> set[str]:
        return set(self._existing.get(environment, set())) | {
            r.symbol for r in self.upserted if r.environment == environment
        }

    def upsert(self, rows) -> None:
        self.upserted.extend(rows)


def _all_fills(cfg: SwingRLConfig) -> dict[tuple[str, str], tuple[float, str]]:
    fills = {("equity", s): (100.0, "signal") for s in cfg.equity.symbols}
    fills.update({("crypto", s): (200.0, "signal") for s in cfg.crypto.symbols})
    return fills


def test_rows_use_first_fill_price_origin_date_and_origin_capital(
    loaded_config: SwingRLConfig,
) -> None:
    """D14/D15/D16: baseline_price = first fill, baseline_date = origin, capital = origin
    total_value — NOT cash_balance (capitals differ here to pin the D16 refinement)."""
    gw = _FakeGateway(
        _all_fills(loaded_config), {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)}
    )
    rows = build_reanchor_rows(loaded_config, gw, _ORIGINS)
    crypto = [r for r in rows if r.environment == "crypto"]
    assert all(r.baseline_price == 200.0 for r in crypto)
    assert all(r.baseline_date == date(2026, 7, 22) for r in crypto)
    assert all(r.capital_usd == 48.09 for r in crypto)  # total_value, not cash 40.0


def test_missing_fill_aborts(loaded_config: SwingRLConfig) -> None:
    """D14: a symbol with no origin-day buy fill aborts rather than guessing."""
    fills = _all_fills(loaded_config)
    del fills[("crypto", loaded_config.crypto.symbols[-1])]  # drop one crypto symbol's fill
    gw = _FakeGateway(fills, {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)})
    with pytest.raises(DataError, match="first buy-fill"):
        build_reanchor_rows(loaded_config, gw, _ORIGINS)


def test_dry_run_writes_nothing(loaded_config: SwingRLConfig, tmp_path: Path) -> None:
    gw = _FakeGateway(
        _all_fills(loaded_config), {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)}
    )
    reanchor(loaded_config, gw, apply=False, origins=_ORIGINS, backup_dir=tmp_path)
    assert gw.upserted == []


def test_apply_writes_backup_and_upserts(loaded_config: SwingRLConfig, tmp_path: Path) -> None:
    gw = _FakeGateway(
        _all_fills(loaded_config), {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)}
    )
    reanchor(loaded_config, gw, apply=True, origins=_ORIGINS, backup_dir=tmp_path)
    n = len(loaded_config.equity.symbols) + len(loaded_config.crypto.symbols)
    assert len(gw.upserted) == n
    assert list(tmp_path.glob("reanchor_backup_*.sql"))


def test_apply_aborts_on_stale_extra_row(loaded_config: SwingRLConfig, tmp_path: Path) -> None:
    """D17: a stale baseline row not in the origin-fill set corrupts the divisor → abort."""
    gw = _FakeGateway(
        _all_fills(loaded_config),
        {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)},
        existing_symbols={"crypto": {"GHOSTUSDT"}},
    )
    with pytest.raises(DataError, match="row-set"):
        reanchor(loaded_config, gw, apply=True, origins=_ORIGINS, backup_dir=tmp_path)
