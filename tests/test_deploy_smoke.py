"""Tests for 6-point model smoke test function.

Validates deserialization, output shape, non-degenerate actions,
inference speed, VecNormalize, and NaN detection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from swingrl.shadow.lifecycle import smoke_test_model


class TestSmokeTestDeserialization:
    """PROD-05: Smoke test validates model can be deserialized."""

    def test_deserialization_passes_with_valid_model(self, tmp_path: Path) -> None:
        """PROD-05: Deserialization check passes when model loads successfully."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.1, -0.2, 0.3]), None)

        with patch("swingrl.shadow.lifecycle._load_sb3_model", return_value=mock_model):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["deserialization"] is True

    def test_deserialization_fails_with_corrupt_model(self, tmp_path: Path) -> None:
        """PROD-05: Deserialization check fails when model cannot load."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"corrupt")

        with patch(
            "swingrl.shadow.lifecycle._load_sb3_model",
            side_effect=Exception("Load error"),
        ):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["deserialization"] is False


class TestSmokeTestOutputShape:
    """PROD-05: Smoke test validates output shape."""

    def test_output_shape_correct(self, tmp_path: Path) -> None:
        """PROD-05: Output shape check passes with correct action dimensions."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.1, -0.2]), None)

        with patch("swingrl.shadow.lifecycle._load_sb3_model", return_value=mock_model):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["output_shape"] is True


class TestSmokeTestNonDegenerate:
    """PROD-05: Smoke test detects degenerate (all identical) outputs."""

    def test_non_degenerate_passes_varied_actions(self, tmp_path: Path) -> None:
        """PROD-05: Non-degenerate check passes with varied predict outputs."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"fake")

        call_count = 0

        def varied_predict(obs: np.ndarray, deterministic: bool = True) -> tuple:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            return (np.array([0.1 * call_count, -0.2 * call_count]), None)

        mock_model = MagicMock()
        mock_model.predict.side_effect = varied_predict

        with patch("swingrl.shadow.lifecycle._load_sb3_model", return_value=mock_model):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["non_degenerate"] is True

    def test_non_degenerate_fails_identical_actions(self, tmp_path: Path) -> None:
        """PROD-05: Non-degenerate check fails when all outputs are identical."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.5, 0.5]), None)

        with patch("swingrl.shadow.lifecycle._load_sb3_model", return_value=mock_model):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["non_degenerate"] is False


class TestSmokeTestInferenceSpeed:
    """PROD-05: Smoke test checks inference time < 100ms."""

    def test_inference_speed_passes_fast_model(self, tmp_path: Path) -> None:
        """PROD-05: Inference speed check passes when predict is fast."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.1]), None)

        with patch("swingrl.shadow.lifecycle._load_sb3_model", return_value=mock_model):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["inference_speed"] is True


class TestSmokeTestVecNormalize:
    """PROD-05: Smoke test validates VecNormalize stats file if present."""

    def test_vec_normalize_passes_with_valid_stats(self, tmp_path: Path) -> None:
        """PROD-05: VecNormalize check passes when stats file loads."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"fake")
        stats_path = tmp_path / "ppo_equity.pkl"
        stats_path.write_bytes(b"fake_stats")

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.1]), None)

        with (
            patch("swingrl.shadow.lifecycle._load_sb3_model", return_value=mock_model),
            patch("swingrl.shadow.lifecycle._load_vec_normalize", return_value=MagicMock()),
        ):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["vec_normalize"] is True

    def test_vec_normalize_passes_when_no_stats_file(self, tmp_path: Path) -> None:
        """PROD-05: VecNormalize check passes when no stats file exists (optional)."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.1]), None)

        with patch("swingrl.shadow.lifecycle._load_sb3_model", return_value=mock_model):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["vec_normalize"] is True


class TestSmokeTestNoNaN:
    """PROD-05: Smoke test rejects model with NaN outputs."""

    def test_no_nan_passes_clean_outputs(self, tmp_path: Path) -> None:
        """PROD-05: No NaN check passes with clean numeric outputs."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"fake")

        call_count = 0

        def clean_predict(obs: np.ndarray, deterministic: bool = True) -> tuple:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            return (np.array([0.1 * call_count, 0.2]), None)

        mock_model = MagicMock()
        mock_model.predict.side_effect = clean_predict

        with patch("swingrl.shadow.lifecycle._load_sb3_model", return_value=mock_model):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["no_nan"] is True

    def test_no_nan_fails_with_nan_outputs(self, tmp_path: Path) -> None:
        """PROD-05: No NaN check fails when model outputs contain NaN."""
        model_path = tmp_path / "ppo_equity.zip"
        model_path.write_bytes(b"fake")

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([float("nan"), 0.1]), None)

        with patch("swingrl.shadow.lifecycle._load_sb3_model", return_value=mock_model):
            result = smoke_test_model(model_path, "equity", obs_dim=156)

        assert result["no_nan"] is False
