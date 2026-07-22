from __future__ import annotations

from unittest.mock import MagicMock

import scripts.collector_main as cm
from scripts.collector_main import all_job_ids, register_jobs
from swingrl.config.schema import SwingRLConfig


def _components(cfg: SwingRLConfig | None = None) -> dict:
    return {
        "config": cfg or SwingRLConfig(),
        "collector": MagicMock(),
        "store": MagicMock(),
        "alerter": MagicMock(),
        "db": MagicMock(),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_candle_jobs_registered_with_config_derived_cron() -> None:
    """CANDLE-1: both candle jobs register with cron fields read from NON-default config
    values — proves nothing (times, minute, grace) is hardcoded in register_jobs."""
    cfg = SwingRLConfig()
    cj = cfg.options_collector.candle_jobs
    cj.equity_time_et = "17:22"
    cj.crypto_minute = 3
    cj.equity_misfire_grace_s = 12345
    cj.crypto_misfire_grace_s = 6789
    scheduler = MagicMock()
    register_jobs(scheduler, _components(cfg))
    by_id = {c.kwargs["id"]: c.kwargs for c in scheduler.add_job.call_args_list}

    assert {"candles_equity", "candles_crypto"} <= set(by_id)

    eq = by_id["candles_equity"]
    assert (eq["hour"], eq["minute"]) == (17, 22)
    assert eq["day_of_week"] == "mon-fri"
    assert eq["timezone"] == "America/New_York"
    assert eq["misfire_grace_time"] == 12345
    assert eq["replace_existing"] is True

    cr = by_id["candles_crypto"]
    assert cr["hour"] == "0,4,8,12,16,20"
    assert cr["minute"] == 3
    assert cr["timezone"] == "UTC"
    assert cr["misfire_grace_time"] == 6789
    assert cr["replace_existing"] is True


def test_candle_jobs_absent_when_disabled() -> None:
    """CANDLE-2: candle_jobs.enabled=False registers neither job and drops both ids from the
    keep-set (so remove_stale_jobs prunes any previously persisted candle job)."""
    cfg = SwingRLConfig()
    cfg.options_collector.candle_jobs.enabled = False
    scheduler = MagicMock()
    register_jobs(scheduler, _components(cfg))
    registered = {c.kwargs["id"] for c in scheduler.add_job.call_args_list}
    assert not ({"candles_equity", "candles_crypto"} & registered)
    assert "candles_equity" not in all_job_ids(cfg)
    assert "candles_crypto" not in all_job_ids(cfg)


def test_candle_job_ids_in_all_job_ids_when_enabled() -> None:
    """CANDLE-3: both ids join all_job_ids() when enabled (stale-job-removal contract — a
    missing id would get the live job removed at boot)."""
    cfg = SwingRLConfig()  # candle_jobs.enabled defaults True
    assert {"candles_equity", "candles_crypto"} <= set(all_job_ids(cfg))


# ---------------------------------------------------------------------------
# Equity candle job
# ---------------------------------------------------------------------------


def test_candles_equity_job_calls_ingest_incremental(monkeypatch) -> None:
    """CANDLE-4: equity job calls run_equity(config, backfill=False) — never a backfill."""
    run_equity = MagicMock(return_value=0)
    monkeypatch.setattr(cm, "run_equity", run_equity)
    monkeypatch.setattr(cm, "run_features", MagicMock())
    cm.set_components(_components())
    cm.candles_equity_job()
    run_equity.assert_called_once()
    assert run_equity.call_args.kwargs.get("backfill") is False


def test_candles_equity_job_runs_features_when_rows_added(monkeypatch) -> None:
    """CANDLE-5: run_features runs only after run_equity reports rows added."""
    run_features = MagicMock()
    monkeypatch.setattr(cm, "run_equity", MagicMock(return_value=5))
    monkeypatch.setattr(cm, "run_features", run_features)
    cm.set_components(_components())
    cm.candles_equity_job()
    run_features.assert_called_once()


def test_candles_equity_job_skips_features_when_no_rows(monkeypatch) -> None:
    """CANDLE-6: zero rows added -> run_features NOT called (no wasted recompute)."""
    run_features = MagicMock()
    monkeypatch.setattr(cm, "run_equity", MagicMock(return_value=0))
    monkeypatch.setattr(cm, "run_features", run_features)
    cm.set_components(_components())
    cm.candles_equity_job()
    run_features.assert_not_called()


def test_candles_equity_job_failure_alerts_and_does_not_raise(monkeypatch) -> None:
    """CANDLE-7: an ingestion failure sends a warning alert and never propagates — the
    scheduler must survive (collector sends its own alerts)."""
    monkeypatch.setattr(cm, "run_equity", MagicMock(side_effect=RuntimeError("alpaca down")))
    run_features = MagicMock()
    monkeypatch.setattr(cm, "run_features", run_features)
    components = _components()
    cm.set_components(components)
    cm.candles_equity_job()  # must not raise
    run_features.assert_not_called()
    calls = components["alerter"].send_alert.call_args_list
    assert any(c.args[0] == "warning" for c in calls)
    assert not any(c.args[0] == "info" for c in calls)


def test_candles_equity_job_sends_info_on_success(monkeypatch) -> None:
    """CANDLE-12: USER RULING 2026-07-19 — success sends exactly one INFO alert mirroring
    the options snapshot "captured" pattern, titled for equity, with the rows count."""
    monkeypatch.setattr(cm, "run_equity", MagicMock(return_value=5))
    monkeypatch.setattr(cm, "run_features", MagicMock())
    components = _components()
    cm.set_components(components)
    cm.candles_equity_job()
    calls = components["alerter"].send_alert.call_args_list
    info_calls = [c for c in calls if c.args[0] == "info"]
    assert len(info_calls) == 1
    level, title, message = info_calls[0].args
    assert title == "Equity candles ingested"
    assert "5" in message


def test_candles_equity_job_info_alert_failure_does_not_raise(monkeypatch) -> None:
    """CANDLE-13: the Alerter itself is expected to swallow send failures (httpx errors
    caught in _post_webhook), but this proves the job is defensively isolated too — an
    INFO-send exception must not propagate and must not be mistaken for an ingest failure
    (no 'ingestion failed' warning gets sent)."""
    monkeypatch.setattr(cm, "run_equity", MagicMock(return_value=3))
    monkeypatch.setattr(cm, "run_features", MagicMock())
    components = _components()

    def _raise_on_info(level: str, *args: object, **kwargs: object) -> None:
        if level == "info":
            raise RuntimeError("discord webhook down")

    components["alerter"].send_alert.side_effect = _raise_on_info
    cm.set_components(components)
    cm.candles_equity_job()  # must not raise
    calls = components["alerter"].send_alert.call_args_list
    assert not any(c.args[0] == "warning" for c in calls)


# ---------------------------------------------------------------------------
# Crypto candle job
# ---------------------------------------------------------------------------


def test_candles_crypto_job_calls_ingest_incremental(monkeypatch) -> None:
    """CANDLE-8: crypto job calls run_crypto(config, backfill=False)."""
    run_crypto = MagicMock(return_value=0)
    monkeypatch.setattr(cm, "run_crypto", run_crypto)
    monkeypatch.setattr(cm, "detect_and_fill_crypto_gaps", MagicMock(return_value=[]))
    monkeypatch.setattr(cm, "run_features", MagicMock())
    cm.set_components(_components())
    cm.candles_crypto_job()
    run_crypto.assert_called_once()
    assert run_crypto.call_args.kwargs.get("backfill") is False


def test_candles_crypto_job_gapfill_runs_between_ingest_and_features(monkeypatch) -> None:
    """CANDLE-9: gap-fill runs AFTER ingest and BEFORE features, and features run when gaps
    were filled even with zero new rows."""
    order: list[str] = []
    monkeypatch.setattr(cm, "run_crypto", lambda config, backfill: (order.append("ingest"), 0)[1])
    gap = MagicMock(filled=True)
    monkeypatch.setattr(
        cm, "detect_and_fill_crypto_gaps", lambda config: (order.append("gapfill"), [gap])[1]
    )
    monkeypatch.setattr(cm, "run_features", lambda config: order.append("features"))
    cm.set_components(_components())
    cm.candles_crypto_job()
    assert order == ["ingest", "gapfill", "features"]


def test_candles_crypto_job_skips_features_when_no_rows_and_no_gaps(monkeypatch) -> None:
    """CANDLE-10: no new rows AND no gaps filled -> run_features NOT called."""
    monkeypatch.setattr(cm, "run_crypto", MagicMock(return_value=0))
    gap = MagicMock(filled=False)
    monkeypatch.setattr(cm, "detect_and_fill_crypto_gaps", MagicMock(return_value=[gap]))
    run_features = MagicMock()
    monkeypatch.setattr(cm, "run_features", run_features)
    cm.set_components(_components())
    cm.candles_crypto_job()
    run_features.assert_not_called()


def test_candles_crypto_job_failure_alerts_and_does_not_raise(monkeypatch) -> None:
    """CANDLE-11: a crypto ingestion failure sends a warning alert and never propagates."""
    monkeypatch.setattr(cm, "run_crypto", MagicMock(side_effect=RuntimeError("binance down")))
    monkeypatch.setattr(cm, "detect_and_fill_crypto_gaps", MagicMock())
    run_features = MagicMock()
    monkeypatch.setattr(cm, "run_features", run_features)
    components = _components()
    cm.set_components(components)
    cm.candles_crypto_job()  # must not raise
    run_features.assert_not_called()
    calls = components["alerter"].send_alert.call_args_list
    assert any(c.args[0] == "warning" for c in calls)
    assert not any(c.args[0] == "info" for c in calls)


def test_candles_crypto_job_sends_info_on_success(monkeypatch) -> None:
    """CANDLE-14: USER RULING 2026-07-19 — success sends exactly one INFO alert mirroring
    the options snapshot "captured" pattern, titled for crypto, with rows AND gaps filled."""
    monkeypatch.setattr(cm, "run_crypto", MagicMock(return_value=9))
    gaps = [MagicMock(filled=True), MagicMock(filled=True), MagicMock(filled=False)]
    monkeypatch.setattr(cm, "detect_and_fill_crypto_gaps", MagicMock(return_value=gaps))
    monkeypatch.setattr(cm, "run_features", MagicMock())
    components = _components()
    cm.set_components(components)
    cm.candles_crypto_job()
    calls = components["alerter"].send_alert.call_args_list
    info_calls = [c for c in calls if c.args[0] == "info"]
    assert len(info_calls) == 1
    level, title, message = info_calls[0].args
    assert title == "Crypto candles ingested"
    assert "9" in message
    assert "2" in message


def test_candles_crypto_job_info_alert_failure_does_not_raise(monkeypatch) -> None:
    """CANDLE-15: an INFO-send exception on the crypto job must not propagate and must not
    be mistaken for an ingest failure (no 'ingestion failed' warning gets sent)."""
    monkeypatch.setattr(cm, "run_crypto", MagicMock(return_value=4))
    monkeypatch.setattr(cm, "detect_and_fill_crypto_gaps", MagicMock(return_value=[]))
    monkeypatch.setattr(cm, "run_features", MagicMock())
    components = _components()

    def _raise_on_info(level: str, *args: object, **kwargs: object) -> None:
        if level == "info":
            raise RuntimeError("discord webhook down")

    components["alerter"].send_alert.side_effect = _raise_on_info
    cm.set_components(components)
    cm.candles_crypto_job()  # must not raise
    calls = components["alerter"].send_alert.call_args_list
    assert not any(c.args[0] == "warning" for c in calls)


# ---------------------------------------------------------------------------
# Schedule default
# ---------------------------------------------------------------------------


def test_equity_candle_default_lands_same_evening() -> None:
    """CANDLE-D1: default equity candle time is past 00:00 UTC year-round (20:15 ET),
    so day-D bars land day-D evening (Alpaca fetch end pins to 00:00 UTC)."""
    from swingrl.config.schema import CandleJobsConfig

    assert CandleJobsConfig().equity_time_et == "20:15"
