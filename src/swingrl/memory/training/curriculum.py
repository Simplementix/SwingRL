"""LLM-guided curriculum sampler for training window selection.

MemoryCurriculumSampler builds a weighted curriculum of training windows,
where each window is labeled by its regime characteristics (e.g. "2022_bear",
"2020_crisis"). Weights come from LLM advice or uniform defaults. The sampler
selects start dates proportional to the window weights for each episode.

Usage:
    from swingrl.memory.training.curriculum import MemoryCurriculumSampler
    sampler = MemoryCurriculumSampler(
        windows=[{"label": "2022_bear", "start": "2022-01-01", "end": "2022-12-31", "weight": 1.5}],
        total_bars=5000,
        env_name="equity",
        memory_client=client,
        run_id="run_042",
    )
    sampler.build()
    start_idx = sampler.sample_date()
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from swingrl.memory.training.bounds import (
    CRISIS_PERIOD_PCT,
    MIN_WINDOW_YEARS,
)

if TYPE_CHECKING:
    from swingrl.memory.client import MemoryClient

log = structlog.get_logger(__name__)

# Minimum bars per window for the window to be used (avoids degenerate tiny windows)
_MIN_BARS_PER_WINDOW = 50


class MemoryCurriculumSampler:
    """LLM-weighted curriculum sampler for training episode start selection.

    Maintains a set of labeled regime windows with weights. Weights can be
    provided by the LLM memory agent or default to uniform. The sampler
    validates that crisis periods fall within [5%, 50%] of the total window
    set and that each window is at least MIN_WINDOW_YEARS wide.

    Args:
        windows: List of window dicts with keys: label, start (ISO date),
            end (ISO date), weight (float >= 0).
        total_bars: Total number of bars in the dataset.
        env_name: Environment name ("equity" or "crypto").
        memory_client: MemoryClient for ingesting curriculum performance.
        run_id: Training run identifier.
        algo: Algorithm name.
        bars_per_year: Bars per year for the environment (252 equity, 2191 crypto).
    """

    def __init__(
        self,
        windows: list[dict[str, Any]],
        total_bars: int,
        env_name: str,
        memory_client: MemoryClient,
        run_id: str,
        algo: str = "unknown",
        bars_per_year: int = 252,
        seed: int | None = None,
    ) -> None:
        """Initialize the curriculum sampler.

        Args:
            windows: Regime window definitions.
            total_bars: Total dataset bars.
            env_name: Environment name.
            memory_client: MemoryClient for performance ingestion.
            run_id: Training run identifier.
            algo: Algorithm name.
            bars_per_year: Used to convert window durations to years.
            seed: Optional RNG seed for reproducible sampling.
        """
        self._windows = windows
        self._total_bars = total_bars
        self._env_name = env_name
        self._client = memory_client
        self._run_id = run_id
        self._algo = algo
        self._bars_per_year = bars_per_year
        self._rng = np.random.default_rng(seed)

        # Built during build()
        self._bar_ranges: list[tuple[int, int]] = []
        self._weights_arr: np.ndarray | None = None
        self._labels: list[str] = []
        self._probs: np.ndarray | None = None

        # Track which window was last sampled
        self._last_sampled_label: str = "unknown"

    def build(self) -> None:
        """Build the weighted window distribution from the window list.

        Validates windows, computes bar ranges, normalizes weights to probabilities.
        Must be called before sample_date().

        Raises:
            ValueError: If window validation fails (MIN_WINDOW_YEARS or CRISIS_PERIOD_PCT).
        """
        self._validate()

        self._bar_ranges = []
        self._labels = []
        raw_weights: list[float] = []

        for w in self._windows:
            label = str(w.get("label", "unknown"))
            weight = max(0.0, float(w.get("weight", 1.0)))
            start_bar = int(w.get("start_bar", 0))
            end_bar = int(w.get("end_bar", self._total_bars))

            # Clamp to dataset range
            start_bar = max(0, min(start_bar, self._total_bars - _MIN_BARS_PER_WINDOW))
            end_bar = max(start_bar + _MIN_BARS_PER_WINDOW, min(end_bar, self._total_bars))

            self._bar_ranges.append((start_bar, end_bar))
            self._labels.append(label)
            raw_weights.append(weight)

        # Normalize to probability distribution
        total = sum(raw_weights)
        if total <= 0:
            log.warning("curriculum_all_zero_weights_using_uniform")
            n = len(raw_weights)
            self._probs = np.ones(n) / n if n > 0 else np.array([])
        else:
            self._probs = np.array([w / total for w in raw_weights])

        self._weights_arr = np.array(raw_weights)

        log.info(
            "curriculum_built",
            env_name=self._env_name,
            n_windows=len(self._windows),
            labels=self._labels,
        )

    def sample_date(self) -> int:
        """Sample a start bar index proportional to window weights.

        Returns:
            Bar index for episode start (int in [0, total_bars)).

        Raises:
            RuntimeError: If build() has not been called.
        """
        if self._probs is None or len(self._probs) == 0:
            log.warning("curriculum_not_built_or_empty_using_uniform")
            return self._uniform_date()

        # Sample a window by weight
        idx = int(self._rng.choice(len(self._probs), p=self._probs))
        start_bar, end_bar = self._bar_ranges[idx]
        self._last_sampled_label = self._labels[idx]

        # Uniform random start within that window
        if end_bar <= start_bar:
            return start_bar
        sampled_bar = int(self._rng.integers(start_bar, end_bar))

        log.debug(
            "curriculum_sample",
            window_label=self._last_sampled_label,
            sampled_bar=sampled_bar,
        )
        return sampled_bar

    def store_performance(
        self,
        window_results: list[dict[str, Any]],
    ) -> None:
        """Ingest per-window performance to memory agent.

        Called after training completes with per-window Sharpe/MDD breakdown.
        Stored as a single memory record for cycle_gate.

        Args:
            window_results: List of dicts with keys: label, weight_used,
                sharpe_from_window, mdd_from_window, kl_stability, episodes_sampled.
        """
        payload_text = (
            f"CURRICULUM PERFORMANCE: run_id={self._run_id} algo={self._algo} "
            f"env={self._env_name} "
            f"window_performance={json.dumps(window_results)}"
        )
        ok = self._client.ingest_training(payload_text, source="curriculum_performance:historical")
        log.info(
            "curriculum_performance_stored",
            run_id=self._run_id,
            n_windows=len(window_results),
            ok=ok,
        )

    @property
    def last_sampled_label(self) -> str:
        """Label of the most recently sampled window."""
        return self._last_sampled_label

    def _validate(self) -> None:
        """Validate windows against bounds constraints.

        Short windows (below MIN_WINDOW_YEARS) emit a warning but are not
        rejected, consistent with the fail-open design — the window still
        participates in sampling with its assigned weight.

        Raises:
            ValueError: If crisis period percentage exceeds CRISIS_PERIOD_PCT upper bound.
        """
        if not self._windows:
            return  # Empty curriculum is valid (build() will use uniform)

        crisis_bars = 0
        total_window_bars = 0

        for w in self._windows:
            start_bar = int(w.get("start_bar", 0))
            end_bar = int(w.get("end_bar", self._total_bars))
            bar_span = end_bar - start_bar

            # Check minimum window size
            min_bars = int(MIN_WINDOW_YEARS * self._bars_per_year)
            if bar_span < min_bars:
                log.warning(
                    "curriculum_window_too_short",
                    label=w.get("label"),
                    bar_span=bar_span,
                    min_bars=min_bars,
                )

            total_window_bars += bar_span
            label = str(w.get("label", "")).lower()
            if "crisis" in label or "crash" in label:
                crisis_bars += bar_span

        # Check CRISIS_PERIOD_PCT bounds
        if total_window_bars > 0:
            crisis_pct = crisis_bars / total_window_bars
            lo, hi = CRISIS_PERIOD_PCT
            if crisis_pct > hi:
                msg = (
                    f"Crisis periods ({crisis_pct:.1%}) exceed max allowed "
                    f"({hi:.1%}) per CRISIS_PERIOD_PCT bounds"
                )
                log.error("curriculum_crisis_pct_exceeded", crisis_pct=crisis_pct)
                raise ValueError(msg)
            # Warn if below minimum but don't block (may have no crisis windows)
            if crisis_pct > 0 and crisis_pct < lo:
                log.warning(
                    "curriculum_crisis_pct_below_min",
                    crisis_pct=crisis_pct,
                    min_pct=lo,
                )

    def _uniform_date(self) -> int:
        """Sample a uniform random start bar from the entire dataset.

        Returns:
            Bar index in [0, total_bars).
        """
        if self._total_bars <= 0:
            return 0
        return int(self._rng.integers(0, self._total_bars))

    @staticmethod
    def from_date_ranges(
        date_windows: list[dict[str, Any]],
        date_index: list[Any],
        memory_client: MemoryClient,
        run_id: str,
        env_name: str = "equity",
        algo: str = "unknown",
    ) -> MemoryCurriculumSampler:
        """Construct sampler from date-range windows (converts dates to bar indices).

        Args:
            date_windows: List of dicts with 'label', 'start' (ISO date), 'end' (ISO date), 'weight'.
            date_index: Sorted list of dates corresponding to dataset bars.
            memory_client: MemoryClient for ingestion.
            run_id: Training run identifier.
            env_name: Environment name.
            algo: Algorithm name.

        Returns:
            MemoryCurriculumSampler ready to call build().
        """
        windows: list[dict[str, Any]] = []
        date_strs = [str(d)[:10] for d in date_index]

        for w in date_windows:
            start_str = str(w.get("start", ""))[:10]
            end_str = str(w.get("end", ""))[:10]

            # Find bar indices for start/end dates
            try:
                start_bar = next((i for i, d in enumerate(date_strs) if d >= start_str), 0)
                end_bar = next(
                    (i for i, d in enumerate(reversed(date_strs)) if d <= end_str),
                    len(date_strs) - 1,
                )
                end_bar = len(date_strs) - 1 - end_bar
            except Exception:
                start_bar = 0
                end_bar = len(date_strs)

            windows.append(
                {
                    "label": w.get("label", "unknown"),
                    "start_bar": start_bar,
                    "end_bar": max(start_bar + 1, end_bar),
                    "weight": float(w.get("weight", 1.0)),
                }
            )

        bars_per_year = 252 if env_name == "equity" else 2191
        return MemoryCurriculumSampler(
            windows=windows,
            total_bars=len(date_index),
            env_name=env_name,
            memory_client=memory_client,
            run_id=run_id,
            algo=algo,
            bars_per_year=bars_per_year,
        )
