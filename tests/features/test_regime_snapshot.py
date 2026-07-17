"""Tests for FeaturePipeline.regime_snapshot — the public capture wrapper.

``regime_snapshot`` composes the existing private ``_get_hmm_probs`` and
``_get_macro_array`` readers into the dict the CycleRecorder stamps onto each
inference cycle. The internals are patched so no live database is needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from swingrl.config.schema import SwingRLConfig
from swingrl.features.pipeline import FeaturePipeline


def _pipeline(config: SwingRLConfig) -> FeaturePipeline:
    """FeaturePipeline over a mock db (regime_snapshot never touches it directly)."""
    return FeaturePipeline(config, MagicMock())


def test_regime_snapshot_composes_hmm_and_vix(
    loaded_config: SwingRLConfig, monkeypatch: Any
) -> None:
    """regime_snapshot returns hmm_p_bull/p_bear from HMM and vix from macro[0]."""
    pipeline = _pipeline(loaded_config)
    monkeypatch.setattr(pipeline, "_get_hmm_probs", lambda env, d: np.array([0.72, 0.28]))
    monkeypatch.setattr(
        pipeline, "_get_macro_array", lambda env, d: np.array([19.5, 0.4, 1.0, 5.0, 300.0, 4.0])
    )

    snap = pipeline.regime_snapshot("equity", "2026-07-16")

    assert snap == {"hmm_p_bull": 0.72, "hmm_p_bear": 0.28, "vix": 19.5}


def test_regime_snapshot_passes_env_and_date_through(
    loaded_config: SwingRLConfig, monkeypatch: Any
) -> None:
    """The env name and date/datetime cutoff reach both underlying readers."""
    pipeline = _pipeline(loaded_config)
    seen: list[tuple[str, str]] = []

    def _hmm(env: str, d: str) -> np.ndarray:
        seen.append((env, d))
        return np.array([0.5, 0.5])

    def _macro(env: str, d: str) -> np.ndarray:
        seen.append((env, d))
        return np.zeros(6)

    monkeypatch.setattr(pipeline, "_get_hmm_probs", _hmm)
    monkeypatch.setattr(pipeline, "_get_macro_array", _macro)

    pipeline.regime_snapshot("crypto", "2026-07-16T20:00:00+00:00")

    assert seen == [
        ("crypto", "2026-07-16T20:00:00+00:00"),
        ("crypto", "2026-07-16T20:00:00+00:00"),
    ]
