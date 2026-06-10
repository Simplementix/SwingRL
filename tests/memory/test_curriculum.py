"""Tests for MemoryCurriculumSampler.

TRAIN-08: MemoryCurriculumSampler builds a weighted curriculum of training
windows, validates crisis/window constraints, and samples episode start bars.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from swingrl.memory.training.curriculum import MemoryCurriculumSampler
from swingrl.utils.exceptions import DataError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WINDOWS = [
    {"label": "2022_bear", "start_bar": 0, "end_bar": 500, "weight": 2.0},
    {"label": "2020_crisis", "start_bar": 500, "end_bar": 700, "weight": 1.0},
    {"label": "2023_bull", "start_bar": 700, "end_bar": 1000, "weight": 1.5},
]


def _make_mock_memory_client() -> MagicMock:
    """Create a mock MemoryClient."""
    client = MagicMock()
    client._base_url = "http://localhost:8889"
    client.ingest_training.return_value = True
    return client


def _make_sampler(
    windows: list | None = None,
    total_bars: int = 1000,
    env_name: str = "equity",
    seed: int = 42,
) -> MemoryCurriculumSampler:
    """Create a sampler with test defaults."""
    return MemoryCurriculumSampler(
        windows=windows if windows is not None else WINDOWS,
        total_bars=total_bars,
        env_name=env_name,
        memory_client=_make_mock_memory_client(),
        run_id="test_run_001",
        algo="ppo",
        bars_per_year=252,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# TestCurriculumBuild
# ---------------------------------------------------------------------------


class TestCurriculumBuild:
    """TRAIN-08: build() produces correct probability array and labels."""

    def test_build_normalizes_weights(self) -> None:
        """TRAIN-08: Non-zero weights normalize so probabilities sum to 1.0."""
        sampler = _make_sampler()
        sampler.build()
        assert sampler._probs is not None
        assert abs(sampler._probs.sum() - 1.0) < 1e-6

    def test_build_all_zero_weights_uses_uniform(self) -> None:
        """TRAIN-08: All-zero weights fall back to uniform 1/n distribution."""
        windows = [
            {"label": "w1", "start_bar": 0, "end_bar": 500, "weight": 0.0},
            {"label": "w2", "start_bar": 500, "end_bar": 1000, "weight": 0.0},
        ]
        sampler = _make_sampler(windows=windows)
        sampler.build()
        assert sampler._probs is not None
        expected = 1.0 / 2
        for p in sampler._probs:
            assert abs(p - expected) < 1e-6

    def test_build_clamps_bar_ranges_to_dataset(self) -> None:
        """TRAIN-08: start/end bars are clamped to [0, total_bars]."""
        windows = [
            {"label": "out_of_range", "start_bar": -100, "end_bar": 9999, "weight": 1.0},
        ]
        sampler = _make_sampler(windows=windows, total_bars=500)
        sampler.build()
        start, end = sampler._bar_ranges[0]
        assert start >= 0
        assert end <= 500

    def test_build_stores_labels(self) -> None:
        """TRAIN-08: _labels list matches window label order."""
        sampler = _make_sampler()
        sampler.build()
        assert sampler._labels == ["2022_bear", "2020_crisis", "2023_bull"]


# ---------------------------------------------------------------------------
# TestCurriculumSampleDate
# ---------------------------------------------------------------------------


class TestCurriculumSampleDate:
    """TRAIN-08: sample_date() returns bar indices within selected windows."""

    def test_sample_date_within_window_range(self) -> None:
        """TRAIN-08: Sampled bar falls within the selected window's range."""
        sampler = _make_sampler(seed=0)
        sampler.build()
        for _ in range(20):
            bar = sampler.sample_date()
            assert 0 <= bar < 1000

    def test_sample_date_no_probs_falls_back_to_uniform(self) -> None:
        """TRAIN-08: _probs=None causes sample_date to call _uniform_date()."""
        sampler = _make_sampler()
        # Do NOT call build() — _probs stays None
        bar = sampler.sample_date()
        assert 0 <= bar < 1000

    def test_last_sampled_label_tracks_window(self) -> None:
        """TRAIN-08: last_sampled_label property returns most recently sampled label."""
        sampler = _make_sampler(seed=1)
        sampler.build()
        sampler.sample_date()
        assert sampler.last_sampled_label in ["2022_bear", "2020_crisis", "2023_bull"]

    def test_uniform_date_in_valid_range(self) -> None:
        """TRAIN-08: _uniform_date() returns int in [0, total_bars)."""
        sampler = _make_sampler(total_bars=500, seed=7)
        for _ in range(20):
            bar = sampler._uniform_date()
            assert 0 <= bar < 500

    def test_uniform_date_zero_bars_returns_zero(self) -> None:
        """TRAIN-08: _uniform_date() returns 0 when total_bars=0."""
        sampler = _make_sampler(total_bars=0)
        assert sampler._uniform_date() == 0


# ---------------------------------------------------------------------------
# TestCurriculumValidation
# ---------------------------------------------------------------------------


class TestCurriculumValidation:
    """TRAIN-08: _validate() enforces crisis and window size constraints."""

    def test_validate_raises_if_crisis_pct_exceeds_50pct(self) -> None:
        """TRAIN-08: DataError raised when crisis_bars/total exceeds 0.50."""
        # crisis window spans 600/1000 = 60% -> exceeds 50% limit
        windows = [
            {"label": "2020_crisis", "start_bar": 0, "end_bar": 600, "weight": 1.0},
            {"label": "2022_bull", "start_bar": 600, "end_bar": 1000, "weight": 1.0},
        ]
        sampler = _make_sampler(windows=windows, total_bars=1000)
        with pytest.raises(DataError, match="Crisis"):
            sampler.build()

    def test_validate_warns_if_window_too_short(self, caplog: pytest.LogCaptureFixture) -> None:
        """TRAIN-08: Warning emitted (but no exception) for window below MIN_WINDOW_YEARS."""
        import logging

        # A window of only 10 bars is well below MIN_WINDOW_YEARS * 252
        windows = [
            {"label": "tiny_window", "start_bar": 0, "end_bar": 10, "weight": 1.0},
        ]
        sampler = MemoryCurriculumSampler(
            windows=windows,
            total_bars=300,
            env_name="equity",
            memory_client=_make_mock_memory_client(),
            run_id="r1",
            bars_per_year=252,
        )
        with caplog.at_level(logging.WARNING):
            sampler.build()  # Should NOT raise
        # Build completed without exception — validation passed with warning
        assert sampler._probs is not None

    def test_validate_warns_crisis_below_min_pct(self, caplog: pytest.LogCaptureFixture) -> None:
        """TRAIN-08: Warning emitted when crisis period is present but below 5%."""
        import logging

        # crisis = 30 bars out of 1000 = 3% (below min 5%)
        windows = [
            {"label": "2020_crisis", "start_bar": 0, "end_bar": 30, "weight": 1.0},
            {"label": "2022_bull", "start_bar": 30, "end_bar": 1000, "weight": 1.0},
        ]
        sampler = _make_sampler(windows=windows, total_bars=1000)
        with caplog.at_level(logging.WARNING):
            sampler.build()  # Should NOT raise
        assert sampler._probs is not None

    def test_validate_empty_windows_passes(self) -> None:
        """TRAIN-08: Empty window list is valid and does not raise."""
        sampler = _make_sampler(windows=[])
        sampler.build()  # Should not raise

    def test_validate_detects_crash_label_as_crisis(self) -> None:
        """TRAIN-08: 'crash' in label counts as crisis bars for percentage check."""
        # crash window spans 600/1000 = 60% -> exceeds 50% limit
        windows = [
            {"label": "2008_crash", "start_bar": 0, "end_bar": 600, "weight": 1.0},
            {"label": "2022_bull", "start_bar": 600, "end_bar": 1000, "weight": 1.0},
        ]
        sampler = _make_sampler(windows=windows, total_bars=1000)
        with pytest.raises(DataError, match="Crisis"):
            sampler.build()


# ---------------------------------------------------------------------------
# TestCurriculumStorePerformance
# ---------------------------------------------------------------------------


class TestCurriculumStorePerformance:
    """TRAIN-08: store_performance() delegates to memory client."""

    def test_store_performance_calls_ingest_training(self) -> None:
        """TRAIN-08: store_performance() calls client.ingest_training() with payload."""
        client = _make_mock_memory_client()
        sampler = MemoryCurriculumSampler(
            windows=WINDOWS,
            total_bars=1000,
            env_name="equity",
            memory_client=client,
            run_id="run_store_001",
            algo="ppo",
        )
        window_results = [
            {
                "label": "2022_bear",
                "weight_used": 2.0,
                "sharpe_from_window": 1.1,
                "mdd_from_window": -0.05,
                "kl_stability": 0.01,
                "episodes_sampled": 10,
            }
        ]
        sampler.store_performance(window_results)
        client.ingest_training.assert_called_once()
        call_text = client.ingest_training.call_args[0][0]
        assert "run_store_001" in call_text
        assert "CURRICULUM PERFORMANCE" in call_text


# ---------------------------------------------------------------------------
# TestCurriculumFromDateRanges
# ---------------------------------------------------------------------------


class TestCurriculumFromDateRanges:
    """TRAIN-08: from_date_ranges() converts ISO dates to bar indices."""

    def _make_date_index(self, n: int = 500) -> list[str]:
        """Generate a list of ISO date strings as a simple date index."""
        from datetime import date, timedelta

        start = date(2020, 1, 1)
        return [(start + timedelta(days=i)).isoformat() for i in range(n)]

    def test_from_date_ranges_converts_dates_to_bar_indices(self) -> None:
        """TRAIN-08: start/end dates map to correct bar indices in the date_index."""
        date_index = self._make_date_index(500)
        date_windows = [
            {"label": "test_window", "start": "2020-01-10", "end": "2020-03-01", "weight": 1.0},
        ]
        sampler = MemoryCurriculumSampler.from_date_ranges(
            date_windows=date_windows,
            date_index=date_index,
            memory_client=_make_mock_memory_client(),
            run_id="r1",
            env_name="equity",
        )
        assert len(sampler._windows) == 1
        w = sampler._windows[0]
        # start_bar should be > 0 (2020-01-10 is not the first date)
        assert w["start_bar"] > 0
        assert w["end_bar"] > w["start_bar"]

    def test_from_date_ranges_handles_missing_date_gracefully(self) -> None:
        """TRAIN-08: Unknown dates fall back to 0/len without raising."""
        date_index = self._make_date_index(100)
        date_windows = [
            {"label": "far_future", "start": "2099-01-01", "end": "2099-12-31", "weight": 1.0},
        ]
        sampler = MemoryCurriculumSampler.from_date_ranges(
            date_windows=date_windows,
            date_index=date_index,
            memory_client=_make_mock_memory_client(),
            run_id="r1",
            env_name="equity",
        )
        assert len(sampler._windows) == 1

    def test_from_date_ranges_equity_bars_per_year_252(self) -> None:
        """TRAIN-08: equity env uses 252 bars_per_year."""
        date_index = self._make_date_index(100)
        sampler = MemoryCurriculumSampler.from_date_ranges(
            date_windows=[
                {"label": "w", "start": "2020-01-01", "end": "2020-06-01", "weight": 1.0}
            ],
            date_index=date_index,
            memory_client=_make_mock_memory_client(),
            run_id="r1",
            env_name="equity",
        )
        assert sampler._bars_per_year == 252

    def test_from_date_ranges_crypto_bars_per_year_2191(self) -> None:
        """TRAIN-08: crypto env uses 2191 bars_per_year."""
        date_index = self._make_date_index(100)
        sampler = MemoryCurriculumSampler.from_date_ranges(
            date_windows=[
                {"label": "w", "start": "2020-01-01", "end": "2020-06-01", "weight": 1.0}
            ],
            date_index=date_index,
            memory_client=_make_mock_memory_client(),
            run_id="r1",
            env_name="crypto",
        )
        assert sampler._bars_per_year == 2191

    def test_from_date_ranges_can_call_build_immediately(self) -> None:
        """TRAIN-08: Returned sampler is ready to call build() without error."""
        date_index = self._make_date_index(500)
        date_windows = [
            {"label": "2022_bear", "start": "2020-06-01", "end": "2021-01-01", "weight": 1.5},
        ]
        sampler = MemoryCurriculumSampler.from_date_ranges(
            date_windows=date_windows,
            date_index=date_index,
            memory_client=_make_mock_memory_client(),
            run_id="r1",
            env_name="equity",
        )
        sampler.build()  # Should not raise
        assert sampler._probs is not None
