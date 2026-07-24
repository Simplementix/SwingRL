"""REQ D13-D17: re-anchor benchmark baselines to the agent's real first-fill prices."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import scripts.reanchor_benchmark_baselines as reanchor_mod
import swingrl.data.db as db_mod
from scripts.reanchor_benchmark_baselines import build_reanchor_rows, main, reanchor
from scripts.record_benchmark_baselines import BaselineRow
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
        existing_baselines: dict[str, list[BaselineRow]] | None = None,
    ) -> None:
        self.fills = fills
        self.capitals = capitals
        self._existing = existing_symbols or {}
        self._baselines = existing_baselines or {}
        self.upserted: list = []

    def first_buy_fill(
        self, environment: str, symbol: str, origin_date: date
    ) -> tuple[float, date, str] | None:
        hit = self.fills.get((environment, symbol))
        return None if hit is None else (hit[0], origin_date, hit[1])

    def origin_capital(self, environment: str, origin_date: date) -> tuple[float, float] | None:
        return self.capitals.get(environment)

    def current_baselines(self, environment: str) -> list:
        # After an upsert, reflect the just-written rows (post-write state); otherwise return
        # the seeded pre-existing rows. Lets a test prove the diff snapshots PRE-write state.
        upserted = [r for r in self.upserted if r.environment == environment]
        if upserted:
            return upserted
        return list(self._baselines.get(environment, []))

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


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    gw: _FakeGateway,
    config: SwingRLConfig,
    argv: list[str],
) -> int:
    """Drive main() DB-free by swapping load_config + the Postgres gateway/DatabaseManager."""
    monkeypatch.setattr(reanchor_mod, "load_config", lambda _path: config)
    monkeypatch.setattr(reanchor_mod, "configure_logging", lambda **_kw: None)
    monkeypatch.setattr(reanchor_mod, "PostgresReanchorGateway", lambda _db: gw)
    monkeypatch.setattr(db_mod, "DatabaseManager", lambda _config: object())
    return main(argv)


def test_diff_prints_origin_cash(
    loaded_config: SwingRLConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#3: the diff shows the origin cash_balance alongside total_value so the operator can
    eyeball the D16 relationship — crypto cash 40.00 must appear distinct from total 48.09."""
    gw = _FakeGateway(
        _all_fills(loaded_config), {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)}
    )
    rc = _run_main(monkeypatch, gw, loaded_config, ["--dry-run", "--backup-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "total_value $48.09" in out
    assert "cash_balance $40.00" in out


def test_apply_diff_shows_pre_write_was(
    loaded_config: SwingRLConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """N1: an --apply diff must show the PRE-write baseline_price in [was …], not the row it
    just upserted. Seed distinct pre-write prices (999.0) and assert they survive into output."""
    pre = {
        env: [
            BaselineRow(env, s, _ORIGINS[env], 999.0, 111.0)
            for s in getattr(loaded_config, env).symbols
        ]
        for env in ("equity", "crypto")
    }
    gw = _FakeGateway(
        _all_fills(loaded_config),
        {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)},
        existing_baselines=pre,
    )
    rc = _run_main(monkeypatch, gw, loaded_config, ["--apply", "--backup-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert gw.upserted  # the write happened
    assert "was $999.0000" in out  # PRE-write price, not the just-upserted 100/200
