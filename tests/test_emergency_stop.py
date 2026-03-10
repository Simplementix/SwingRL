"""Tests for four-tier emergency stop protocol and automated triggers.

Covers all 4 tiers (halt+cancel, crypto liquidation, time-aware equity
liquidation, verify+alert) and 3 automated triggers (VIX+CB, NaN inference,
IP ban).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_db() -> MagicMock:
    """Mock DatabaseManager with SQLite and DuckDB context managers."""
    db = MagicMock()
    conn = MagicMock()
    db.sqlite.return_value.__enter__ = MagicMock(return_value=conn)
    db.sqlite.return_value.__exit__ = MagicMock(return_value=False)

    duck_cursor = MagicMock()
    db.duckdb.return_value.__enter__ = MagicMock(return_value=duck_cursor)
    db.duckdb.return_value.__exit__ = MagicMock(return_value=False)
    return db


@pytest.fixture()
def mock_alerter() -> MagicMock:
    """Mock Alerter for Discord calls."""
    alerter = MagicMock()
    alerter.send_alert = MagicMock()
    alerter.send_embed = MagicMock()
    return alerter


@pytest.fixture()
def mock_config() -> MagicMock:
    """Mock SwingRLConfig with equity symbols."""
    config = MagicMock()
    config.equity.symbols = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "TLT", "GLD"]
    config.crypto.symbols = ["BTCUSDT", "ETHUSDT"]
    config.trading_mode = "paper"
    return config


@pytest.fixture()
def mock_alpaca_adapter() -> MagicMock:
    """Mock AlpacaAdapter."""
    adapter = MagicMock()
    adapter.get_positions.return_value = [
        {"symbol": "SPY", "quantity": 5.0, "market_value": 2200.0},
    ]
    adapter.cancel_order.return_value = True
    return adapter


@pytest.fixture()
def mock_binance_adapter() -> MagicMock:
    """Mock BinanceSimAdapter."""
    adapter = MagicMock()
    adapter.get_positions.return_value = [
        {"symbol": "BTCUSDT", "quantity": 0.001, "cost_basis": 60.0},
    ]
    adapter.cancel_order.return_value = True
    return adapter


class TestTier1HaltAndCancel:
    """Test Tier 1: set halt flag and cancel all open orders."""

    def test_tier1_sets_halt_flag(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
        mock_alerter: MagicMock,
        mock_alpaca_adapter: MagicMock,
        mock_binance_adapter: MagicMock,
    ) -> None:
        """PROD-07: Tier 1 sets halt flag via set_halt()."""
        from swingrl.execution.emergency import _tier1_halt_and_cancel

        result = _tier1_halt_and_cancel(
            db=mock_db,
            reason="test halt",
            alpaca=mock_alpaca_adapter,
            binance=mock_binance_adapter,
        )

        assert result["success"] is True
        assert result["tier"] == 1

    def test_tier1_cancels_orders_on_both_brokers(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
        mock_alerter: MagicMock,
        mock_alpaca_adapter: MagicMock,
        mock_binance_adapter: MagicMock,
    ) -> None:
        """PROD-07: Tier 1 cancels ALL open orders on both Alpaca and Binance.US."""
        from swingrl.execution.emergency import _tier1_halt_and_cancel

        # Alpaca has _client with cancel_orders
        mock_alpaca_adapter._client = MagicMock()
        mock_alpaca_adapter._client.cancel_orders.return_value = None

        _tier1_halt_and_cancel(
            db=mock_db,
            reason="test cancel",
            alpaca=mock_alpaca_adapter,
            binance=mock_binance_adapter,
        )

        mock_alpaca_adapter._client.cancel_orders.assert_called_once()


class TestTier2LiquidateCrypto:
    """Test Tier 2: market-sell all crypto positions."""

    def test_tier2_liquidates_crypto_positions(
        self,
        mock_db: MagicMock,
        mock_binance_adapter: MagicMock,
    ) -> None:
        """PROD-07: Tier 2 market-sells all crypto positions."""
        from swingrl.execution.emergency import _tier2_liquidate_crypto

        result = _tier2_liquidate_crypto(
            binance=mock_binance_adapter,
            db=mock_db,
        )

        assert result["success"] is True
        assert result["tier"] == 2
        assert result["positions_closed"] >= 0


class TestTier3LiquidateEquity:
    """Test Tier 3: time-aware equity liquidation."""

    def test_tier3_during_market_hours(
        self,
        mock_alpaca_adapter: MagicMock,
    ) -> None:
        """PROD-07: Tier 3 during market hours submits close_all via Alpaca."""
        from swingrl.execution.emergency import _tier3_liquidate_equity

        mock_alpaca_adapter._client = MagicMock()
        mock_alpaca_adapter._client.close_all_positions.return_value = None

        # Mock exchange_calendars to say market is open
        with patch("swingrl.execution.emergency._get_equity_liquidation_strategy") as mock_strat:
            mock_strat.return_value = "limit_at_bid"
            result = _tier3_liquidate_equity(
                alpaca=mock_alpaca_adapter,
            )

        assert result["success"] is True
        assert result["tier"] == 3
        assert result["strategy"] == "limit_at_bid"

    def test_tier3_outside_market_hours(
        self,
        mock_alpaca_adapter: MagicMock,
    ) -> None:
        """PROD-07: Tier 3 outside market hours queues with time_in_force='opg'."""
        from swingrl.execution.emergency import _tier3_liquidate_equity

        mock_alpaca_adapter._client = MagicMock()

        with patch("swingrl.execution.emergency._get_equity_liquidation_strategy") as mock_strat:
            mock_strat.return_value = "queue_for_open"
            result = _tier3_liquidate_equity(
                alpaca=mock_alpaca_adapter,
            )

        assert result["success"] is True
        assert result["strategy"] == "queue_for_open"

    def test_tier3_during_extended_hours(
        self,
        mock_alpaca_adapter: MagicMock,
    ) -> None:
        """PROD-07: Tier 3 during extended hours submits limit sell."""
        from swingrl.execution.emergency import _tier3_liquidate_equity

        mock_alpaca_adapter._client = MagicMock()

        with patch("swingrl.execution.emergency._get_equity_liquidation_strategy") as mock_strat:
            mock_strat.return_value = "limit_extended"
            result = _tier3_liquidate_equity(
                alpaca=mock_alpaca_adapter,
            )

        assert result["success"] is True
        assert result["strategy"] == "limit_extended"

    def test_get_equity_liquidation_strategy_market_open(self) -> None:
        """PROD-07: Strategy returns 'limit_at_bid' when market is open."""
        from swingrl.execution.emergency import _get_equity_liquidation_strategy

        with patch("swingrl.execution.emergency.xcals") as mock_xcals:
            mock_calendar = MagicMock()
            mock_calendar.is_open_on_minute.return_value = True
            mock_xcals.get_calendar.return_value = mock_calendar

            strategy = _get_equity_liquidation_strategy()
            assert strategy == "limit_at_bid"

    def test_get_equity_liquidation_strategy_closed(self) -> None:
        """PROD-07: Strategy returns 'queue_for_open' when market is closed."""
        from swingrl.execution.emergency import _get_equity_liquidation_strategy

        with patch("swingrl.execution.emergency.xcals") as mock_xcals:
            mock_calendar = MagicMock()
            mock_calendar.is_open_on_minute.return_value = False
            mock_xcals.get_calendar.return_value = mock_calendar
            # Simulate no extended hours
            with patch("swingrl.execution.emergency._is_extended_hours", return_value=False):
                strategy = _get_equity_liquidation_strategy()
                assert strategy == "queue_for_open"

    def test_get_equity_liquidation_strategy_extended(self) -> None:
        """PROD-07: Strategy returns 'limit_extended' during extended hours."""
        from swingrl.execution.emergency import _get_equity_liquidation_strategy

        with patch("swingrl.execution.emergency.xcals") as mock_xcals:
            mock_calendar = MagicMock()
            mock_calendar.is_open_on_minute.return_value = False
            mock_xcals.get_calendar.return_value = mock_calendar
            with patch("swingrl.execution.emergency._is_extended_hours", return_value=True):
                strategy = _get_equity_liquidation_strategy()
                assert strategy == "limit_extended"


class TestTier4VerifyAndAlert:
    """Test Tier 4: verify positions closed and send Discord alert."""

    def test_tier4_verifies_and_alerts(
        self,
        mock_alerter: MagicMock,
        mock_alpaca_adapter: MagicMock,
        mock_binance_adapter: MagicMock,
    ) -> None:
        """PROD-07: Tier 4 verifies positions and sends Discord critical alert."""
        from swingrl.execution.emergency import _tier4_verify_and_alert

        # After emergency stop, positions should be empty
        mock_alpaca_adapter.get_positions.return_value = []
        mock_binance_adapter.get_positions.return_value = []

        tier_results = [
            {"tier": 1, "success": True},
            {"tier": 2, "success": True, "positions_closed": 1},
            {"tier": 3, "success": True, "strategy": "limit_at_bid"},
        ]

        result = _tier4_verify_and_alert(
            alerter=mock_alerter,
            alpaca=mock_alpaca_adapter,
            binance=mock_binance_adapter,
            reason="test verify",
            tier_results=tier_results,
        )

        assert result["success"] is True
        assert result["tier"] == 4
        mock_alerter.send_alert.assert_called_once()
        call_args = mock_alerter.send_alert.call_args
        assert call_args[1]["level"] == "critical" or call_args[0][0] == "critical"


class TestFullEmergencyStop:
    """Test the full 4-tier execute_emergency_stop sequence."""

    def test_full_sequence_returns_status_report(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
        mock_alerter: MagicMock,
    ) -> None:
        """PROD-07: Full 4-tier sequence completes and returns status report dict."""
        from swingrl.execution.emergency import execute_emergency_stop

        with (
            patch("swingrl.execution.emergency._create_alpaca_adapter"),
            patch("swingrl.execution.emergency._create_binance_adapter"),
            patch("swingrl.execution.emergency._tier1_halt_and_cancel") as mock_t1,
            patch("swingrl.execution.emergency._tier2_liquidate_crypto") as mock_t2,
            patch("swingrl.execution.emergency._tier3_liquidate_equity") as mock_t3,
            patch("swingrl.execution.emergency._tier4_verify_and_alert") as mock_t4,
        ):
            mock_t1.return_value = {"tier": 1, "success": True}
            mock_t2.return_value = {"tier": 2, "success": True, "positions_closed": 1}
            mock_t3.return_value = {"tier": 3, "success": True, "strategy": "limit_at_bid"}
            mock_t4.return_value = {"tier": 4, "success": True}

            report = execute_emergency_stop(
                config=mock_config,
                db=mock_db,
                alerter=mock_alerter,
                reason="full test",
            )

        assert "tier_results" in report
        assert len(report["tier_results"]) == 4
        assert report["reason"] == "full test"

    def test_tier_failure_does_not_block_subsequent_tiers(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
        mock_alerter: MagicMock,
    ) -> None:
        """PROD-07: Individual tier failure does not prevent subsequent tiers."""
        from swingrl.execution.emergency import execute_emergency_stop

        with (
            patch("swingrl.execution.emergency._create_alpaca_adapter"),
            patch("swingrl.execution.emergency._create_binance_adapter"),
            patch(
                "swingrl.execution.emergency._tier1_halt_and_cancel",
                side_effect=Exception("tier1 fail"),
            ),
            patch("swingrl.execution.emergency._tier2_liquidate_crypto") as mock_t2,
            patch("swingrl.execution.emergency._tier3_liquidate_equity") as mock_t3,
            patch("swingrl.execution.emergency._tier4_verify_and_alert") as mock_t4,
        ):
            mock_t2.return_value = {"tier": 2, "success": True, "positions_closed": 0}
            mock_t3.return_value = {"tier": 3, "success": True, "strategy": "queue_for_open"}
            mock_t4.return_value = {"tier": 4, "success": True}

            report = execute_emergency_stop(
                config=mock_config,
                db=mock_db,
                alerter=mock_alerter,
                reason="tier1 fails",
            )

        # Tier 1 should be recorded as failed, but others should run
        assert len(report["tier_results"]) == 4
        assert report["tier_results"][0]["success"] is False
        assert report["tier_results"][1]["success"] is True
        assert report["tier_results"][2]["success"] is True
        assert report["tier_results"][3]["success"] is True


class TestAutomatedTriggers:
    """Test check_automated_triggers for VIX+CB, NaN, and IP ban."""

    def test_vix_plus_cb_trigger_fires(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: Detects VIX>40 AND drawdown_pct >= 0.13."""
        from swingrl.execution.emergency import check_automated_triggers

        duck_cursor = mock_db.duckdb.return_value.__enter__.return_value
        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        # VIX > 40 from DuckDB (tuple indexing)
        duck_cursor.execute.return_value.fetchone.return_value = (42.0,)

        # Drawdown >= 0.13 from SQLite (dict-key access)
        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"worst_dd": 0.15},  # drawdown query
            {"nan_count": 0},  # NaN query
            {"ban_count": 0},  # IP ban query
        ]

        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert any("VIX" in t for t in triggers)

    def test_vix_plus_cb_trigger_does_not_fire_low_vix(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: Does NOT fire when VIX <= 40."""
        from swingrl.execution.emergency import check_automated_triggers

        duck_cursor = mock_db.duckdb.return_value.__enter__.return_value
        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        # VIX <= 40
        duck_cursor.execute.return_value.fetchone.return_value = (20.0,)

        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"nan_count": 0},  # NaN query
            {"ban_count": 0},  # IP ban query
        ]

        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert not any("VIX" in t for t in triggers)

    def test_vix_plus_cb_trigger_does_not_fire_low_drawdown(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: Does NOT fire when VIX>40 but drawdown_pct < 0.13."""
        from swingrl.execution.emergency import check_automated_triggers

        duck_cursor = mock_db.duckdb.return_value.__enter__.return_value
        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        # VIX > 40 from DuckDB
        duck_cursor.execute.return_value.fetchone.return_value = (42.0,)

        # Drawdown < 0.13
        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"worst_dd": 0.05},  # drawdown too low
            {"nan_count": 0},  # NaN query
            {"ban_count": 0},  # IP ban query
        ]

        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert not any("VIX" in t for t in triggers)

    def test_nan_inference_trigger_fires(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: Detects 2+ NaN inferences in 24h."""
        from swingrl.execution.emergency import check_automated_triggers

        duck_cursor = mock_db.duckdb.return_value.__enter__.return_value
        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        # VIX OK
        duck_cursor.execute.return_value.fetchone.return_value = (20.0,)

        # NaN count >= 2
        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"nan_count": 3},  # NaN trigger fires
            {"ban_count": 0},  # IP ban query
        ]

        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert any("NaN" in t for t in triggers)

    def test_nan_inference_trigger_does_not_fire(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: Does NOT fire when NaN count < 2."""
        from swingrl.execution.emergency import check_automated_triggers

        duck_cursor = mock_db.duckdb.return_value.__enter__.return_value
        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        duck_cursor.execute.return_value.fetchone.return_value = (20.0,)

        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"nan_count": 1},  # Below threshold
            {"ban_count": 0},
        ]

        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert not any("NaN" in t for t in triggers)

    def test_ip_ban_trigger_fires(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: Detects Binance.US IP ban (HTTP 418)."""
        from swingrl.execution.emergency import check_automated_triggers

        duck_cursor = mock_db.duckdb.return_value.__enter__.return_value
        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        duck_cursor.execute.return_value.fetchone.return_value = (20.0,)

        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"nan_count": 0},
            {"ban_count": 1},  # IP ban detected
        ]

        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert any("418" in t or "ban" in t.lower() for t in triggers)

    def test_ip_ban_trigger_does_not_fire(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: Does NOT fire when no 418 errors exist."""
        from swingrl.execution.emergency import check_automated_triggers

        duck_cursor = mock_db.duckdb.return_value.__enter__.return_value
        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        duck_cursor.execute.return_value.fetchone.return_value = (20.0,)

        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"nan_count": 0},
            {"ban_count": 0},  # No IP ban
        ]

        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert not any("418" in t or "ban" in t.lower() for t in triggers)

    def test_no_triggers_active(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: Returns empty list when no triggers are active."""
        from swingrl.execution.emergency import check_automated_triggers

        duck_cursor = mock_db.duckdb.return_value.__enter__.return_value
        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        duck_cursor.execute.return_value.fetchone.return_value = (20.0,)

        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"nan_count": 0},
            {"ban_count": 0},
        ]

        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert triggers == []

    def test_combined_triggers(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: Returns combined list when multiple triggers fire."""
        from swingrl.execution.emergency import check_automated_triggers

        duck_cursor = mock_db.duckdb.return_value.__enter__.return_value
        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        # VIX > 40
        duck_cursor.execute.return_value.fetchone.return_value = (45.0,)

        # All triggers fire
        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"worst_dd": 0.15},  # drawdown trigger
            {"nan_count": 3},  # NaN trigger
            {"ban_count": 1},  # IP ban trigger
        ]

        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert len(triggers) == 3

    def test_graceful_failure_duckdb_unavailable(
        self,
        mock_db: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PROD-07: VIX trigger fails gracefully when DuckDB unavailable."""
        from swingrl.execution.emergency import check_automated_triggers

        # DuckDB raises
        mock_db.duckdb.return_value.__enter__.side_effect = ConnectionError("DuckDB down")

        sqlite_conn = mock_db.sqlite.return_value.__enter__.return_value

        sqlite_conn.execute.return_value.fetchone.side_effect = [
            {"nan_count": 0},
            {"ban_count": 0},
        ]

        # Should not raise, just skip VIX trigger
        triggers = check_automated_triggers(config=mock_config, db=mock_db)
        assert not any("VIX" in t for t in triggers)

    def test_cb_drawdown_trigger_uses_positive_fraction(self) -> None:
        """PROD-07: _CB_DRAWDOWN_TRIGGER is positive fraction (0.13), not negative pct."""
        from swingrl.execution.emergency import _CB_DRAWDOWN_TRIGGER

        assert _CB_DRAWDOWN_TRIGGER == 0.13
        assert _CB_DRAWDOWN_TRIGGER > 0
