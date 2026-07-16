"""Model-loading hygiene tests (review H2, H3, M5, M7, M3 + hardcode sweep).

Covers the single on-disk layout source, mtime-keyed cache invalidation,
fail-closed VecNormalize loading, ensemble weight renormalization over the
actually-loaded algos, ghost-free emergency sells, and config-driven min order.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from swingrl.execution.model_paths import active_model_paths

from swingrl.execution.pipeline import ExecutionPipeline
from swingrl.shadow.lifecycle import ModelLifecycle
from swingrl.training.ensemble import DEFAULT_ENSEMBLE_WEIGHT, EnsembleBlender


def _write_active_algo(
    models_dir: Path, env: str, algo: str, *, with_vec: bool = True
) -> tuple[Path, Path]:
    """Create placeholder active model files for an algo at the canonical layout."""
    model_path, vec_path = active_model_paths(models_dir, env, algo)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    if with_vec:
        vec_path.write_bytes(b"vec")
    return model_path, vec_path


@pytest.fixture
def loading_pipeline(
    exec_config: object,
    mock_db: object,
    mock_alerter: MagicMock,
    tmp_path: Path,
) -> ExecutionPipeline:
    """ExecutionPipeline with a temp models_dir for loader/cache tests."""
    return ExecutionPipeline(
        config=exec_config,  # type: ignore[arg-type]
        db=mock_db,  # type: ignore[arg-type]
        feature_pipeline=MagicMock(),
        alerter=mock_alerter,
        models_dir=tmp_path / "models",
    )


class TestPromoteProducesLoaderLayout:
    """(a) Review H2: promotion writes the per-algo layout the loader reads."""

    def test_promote_writes_per_algo_active_layout_with_vec(self, tmp_path: Path) -> None:
        """REQ-2R-A-D: promote → loader finds model + vec_normalize at the per-algo path."""
        models_dir = tmp_path / "models"
        lifecycle = ModelLifecycle(models_dir)
        shadow_dir = models_dir / "shadow" / "crypto"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        (shadow_dir / "ppo_crypto_v1.zip").write_bytes(b"ppo-model")
        (shadow_dir / "ppo_crypto_v1.pkl").write_bytes(b"ppo-vec")

        lifecycle.promote("crypto")

        model_path, vec_path = active_model_paths(models_dir, "crypto", "ppo")
        assert model_path.exists()
        assert vec_path.exists()
        assert model_path.read_bytes() == b"ppo-model"
        assert vec_path.read_bytes() == b"ppo-vec"
        # Shadow candidate is consumed by the move.
        assert not (shadow_dir / "ppo_crypto_v1.zip").exists()


class TestModelCacheInvalidation:
    """(b) Review H3: cache busts on mtime change and never caches empty forever."""

    def test_touch_busts_cache_and_empty_first_load_retried(
        self, loading_pipeline: ExecutionPipeline, tmp_path: Path
    ) -> None:
        """REQ-2R-A-D: touching model.zip reloads; empty first load is retried next cycle."""
        models_dir = loading_pipeline._models_dir

        with patch.object(
            loading_pipeline, "_load_one_algo", return_value=(object(), object())
        ) as mock_load:
            # Empty first load — no files on disk — must not be cached permanently.
            assert loading_pipeline._load_models("crypto") == {}
            assert mock_load.call_count == 0

            # Files appear → retried, loads all three algos.
            for algo in ("ppo", "a2c", "sac"):
                _write_active_algo(models_dir, "crypto", algo)
            assert set(loading_pipeline._load_models("crypto")) == {"ppo", "a2c", "sac"}
            assert mock_load.call_count == 3

            # Second call with unchanged mtimes → cache hit, no new loads.
            loading_pipeline._load_models("crypto")
            assert mock_load.call_count == 3

            # Touch model.zip mtime → cache key changes → reload all three.
            model_path, _ = active_model_paths(models_dir, "crypto", "ppo")
            stat = model_path.stat()
            os.utime(model_path, (stat.st_mtime + 10, stat.st_mtime + 10))
            loading_pipeline._load_models("crypto")
            assert mock_load.call_count == 6


class TestEnsembleRenormalization:
    """(c) Review M5: blending renormalizes over loaded algos and never KeyErrors."""

    def test_two_of_three_models_renormalize_to_sum_one(self) -> None:
        """REQ-2R-A-D: 2-of-3 loaded → weights renormalize to sum 1.0."""
        blender = EnsembleBlender(MagicMock())
        actions = {"ppo": np.array([1.0, 0.0]), "a2c": np.array([0.0, 1.0])}
        # sac was not loaded; its metadata weight (0.5) must be dropped and the
        # remaining weights renormalized: 0.2/0.5=0.4, 0.3/0.5=0.6.
        weights = {"ppo": 0.2, "a2c": 0.3, "sac": 0.5}

        blended = blender.blend_actions(actions, weights)

        assert float(blended.sum()) == pytest.approx(1.0)
        np.testing.assert_allclose(blended, [0.4, 0.6])

    def test_missing_metadata_defaults_equal_share_no_keyerror(self) -> None:
        """REQ-2R-A-D: missing model_metadata row → equal share, never KeyError."""
        blender = EnsembleBlender(MagicMock())
        actions = {
            "ppo": np.array([1.0]),
            "a2c": np.array([1.0]),
            "sac": np.array([1.0]),
        }
        # Empty weights dict — every algo falls back to DEFAULT_ENSEMBLE_WEIGHT then
        # renormalizes to 1/3 each → weighted sum of identical actions == the action.
        blended = blender.blend_actions(actions, {})

        assert float(blended[0]) == pytest.approx(1.0)
        assert DEFAULT_ENSEMBLE_WEIGHT == pytest.approx(1.0 / 3)


class TestFailClosedVecNormalize:
    """(d) Review M7/A22: missing VecNormalize → skip that algo + alert, proceed."""

    def test_missing_vec_normalize_excludes_algo_and_alerts(
        self, loading_pipeline: ExecutionPipeline, mock_alerter: MagicMock
    ) -> None:
        """REQ-2R-A-D: algo with no vec_normalize.pkl is excluded + alerted, cycle proceeds."""
        models_dir = loading_pipeline._models_dir
        _write_active_algo(models_dir, "crypto", "ppo", with_vec=True)
        _write_active_algo(models_dir, "crypto", "a2c", with_vec=False)  # fail-closed target
        _write_active_algo(models_dir, "crypto", "sac", with_vec=True)

        with patch.object(loading_pipeline, "_load_one_algo", return_value=(object(), object())):
            models = loading_pipeline._load_models("crypto")

        assert set(models) == {"ppo", "sac"}
        assert "a2c" not in models
        assert mock_alerter.send_alert.called
        levels = [call.kwargs.get("level") for call in mock_alerter.send_alert.call_args_list]
        assert "warning" in levels


class TestMinOrderHonorsConfig:
    """(f) Hardcode sweep: equity min order comes from config, not a $5 literal."""

    def test_min_order_value_reads_config_not_hardcoded(
        self,
        exec_config: object,
        mock_db: object,
        mock_alerter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """REQ-2R-A-D: min order honors config.equity.min_order_usd / crypto.min_order_usd."""
        # A distinctive value that is NOT the retired $5 literal.
        exec_config.equity.min_order_usd = 7.5  # type: ignore[attr-defined]
        pipeline = ExecutionPipeline(
            config=exec_config,  # type: ignore[arg-type]
            db=mock_db,  # type: ignore[arg-type]
            feature_pipeline=MagicMock(),
            alerter=mock_alerter,
            models_dir=tmp_path / "models",
        )

        assert pipeline._min_order_value("equity") == pytest.approx(7.5)
        assert pipeline._min_order_value("crypto") == pytest.approx(
            exec_config.crypto.min_order_usd  # type: ignore[attr-defined]
        )


class TestEmergencySellNoGhostRows:
    """(e) Review M3: emergency sell writes a trade + deletes the position (no ghost)."""

    def test_emergency_sell_writes_trade_and_deletes_position(
        self, mock_db: object, exec_config: object
    ) -> None:
        """REQ-2R-A-D: emergency sell → trades row written, position row deleted."""
        from swingrl.execution.adapters.binance_sim import BinanceSimAdapter

        with mock_db.connection() as conn:  # type: ignore[attr-defined]
            conn.execute(
                "INSERT INTO positions (symbol, environment, quantity, cost_basis, "
                "last_price, unrealized_pnl, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("BTCUSDT", "crypto", 0.5, 50000.0, 50000.0, 0.0, "2026-03-09T10:00:00Z"),
            )

        adapter = BinanceSimAdapter(config=exec_config, db=mock_db)  # type: ignore[arg-type]
        with patch.object(adapter, "_get_mid_price", return_value=(50000.0, 49990.0, 50010.0)):
            fill = adapter.emergency_sell("BTCUSDT", 0.5)

        assert fill.status == "filled"
        with mock_db.connection() as conn:  # type: ignore[attr-defined]
            trade = conn.execute(
                "SELECT quantity FROM trades WHERE symbol = %s AND environment = %s",
                ("BTCUSDT", "crypto"),
            ).fetchone()
            pos = conn.execute(
                "SELECT quantity FROM positions WHERE symbol = %s AND environment = %s",
                ("BTCUSDT", "crypto"),
            ).fetchone()

        assert trade is not None  # a real trade is recorded (no silent liquidation)
        assert float(trade["quantity"]) == pytest.approx(0.5)
        assert pos is None  # position deleted, not left as a zero-qty ghost

    def test_tier4_counts_only_positive_quantity(
        self, mock_db: object, exec_config: object
    ) -> None:
        """REQ-2R-A-D: tier-4 verification counts only quantity > 0 (ghosts ignored)."""
        from swingrl.execution.adapters.binance_sim import BinanceSimAdapter
        from swingrl.execution.emergency import _tier4_verify_and_alert

        # A zero-quantity ghost row must NOT count as a remaining position.
        with mock_db.connection() as conn:  # type: ignore[attr-defined]
            conn.execute(
                "INSERT INTO positions (symbol, environment, quantity, cost_basis, "
                "last_price, unrealized_pnl, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("ETHUSDT", "crypto", 0.0, 3000.0, 3000.0, 0.0, "2026-03-09T10:00:00Z"),
            )

        binance = BinanceSimAdapter(config=exec_config, db=mock_db)  # type: ignore[arg-type]
        result = _tier4_verify_and_alert(
            alerter=MagicMock(),
            alpaca=None,
            binance=binance,
            reason="ghost-row test",
            tier_results=[],
        )

        assert result["remaining_crypto"] == 0
        assert result["all_closed"] is True
