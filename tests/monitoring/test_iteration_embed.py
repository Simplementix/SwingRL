"""Tests for the iteration completion Discord embed builder.

REPORT-EMBED-01: build_iteration_completion_embed renders a fully-populated embed.
REPORT-EMBED-02: Color logic — green if no regression and CPS up, yellow if partial
                 regression with CPS up, red if CPS regressed.
REPORT-EMBED-03: Embed includes treatment-vs-control split when populated.
REPORT-EMBED-04: Iter 0 (no deltas) renders without "—" sigils erroring.
"""

from __future__ import annotations

from typing import Any

from swingrl.monitoring.embeds import (
    _COLOR_BUY,  # green
    _COLOR_CRITICAL,  # red
    _COLOR_WARNING,  # yellow/orange
    build_iteration_completion_embed,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic CPS summary dicts mirroring compute_and_persist_iteration_cps
# ---------------------------------------------------------------------------


def make_summary(
    *,
    iteration: int = 4,
    env: str = "equity",
    cps_v1: float = 0.01526,
    cps_v2: float = -0.422,
    cps_v3: float | None = 2.04,
    treatment_v1: float | None = 0.0148,
    control_v1: float | None = 0.0407,
    cps_v1_delta: float | None = 0.00188,
    return_delta: float | None = 0.00565,
    worst_mdd_delta: float | None = 0.0,
    median_return: float = 0.0662,
    mean_winner_sharpe: float = 4.00,
    winners_count: int = 13,
    chronic_failure_count: int = 5,
    worst_fold_number: int = 7,
    worst_fold_mdd: float = 0.3823,
    regression_flag: bool = False,
    regression_dimensions: list[str] | None = None,
    dedup_rows_dropped: int = 0,
) -> dict[str, Any]:
    return {
        "env": env,
        "iteration": iteration,
        "cps_v1_multiplicative": cps_v1,
        "cps_v2_additive": cps_v2,
        "cps_v3_sortino": cps_v3,
        "cps_v1_treatment_only": treatment_v1,
        "cps_v1_control_only": control_v1,
        "cps_v1_delta_vs_prev": cps_v1_delta,
        "return_delta_vs_prev": return_delta,
        "worst_mdd_delta_vs_prev": worst_mdd_delta,
        "median_return": median_return,
        "mean_winner_sharpe": mean_winner_sharpe,
        "winners_count": winners_count,
        "chronic_failure_count": chronic_failure_count,
        "worst_fold_number": worst_fold_number,
        "worst_fold_mdd": worst_fold_mdd,
        "regression_flag": regression_flag,
        "regression_dimensions": regression_dimensions or [],
        "dedup_rows_dropped": dedup_rows_dropped,
    }


# ---------------------------------------------------------------------------
# REPORT-EMBED-01: structural shape
# ---------------------------------------------------------------------------


class TestEmbedStructure:
    def test_returns_webhook_payload_with_one_embed(self) -> None:
        summary = make_summary()
        payload = build_iteration_completion_embed(summary)
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1

    def test_title_contains_iteration_and_env(self) -> None:
        summary = make_summary(iteration=4, env="equity")
        payload = build_iteration_completion_embed(summary)
        title = str(payload["embeds"][0]["title"])
        assert "4" in title
        assert "equity" in title.lower() or "Equity" in title

    def test_embed_has_required_fields(self) -> None:
        summary = make_summary()
        payload = build_iteration_completion_embed(summary)
        field_names = [str(f["name"]) for f in payload["embeds"][0]["fields"]]
        for required in (
            "CPS v1",
            "CPS v2",
            "CPS v3",
            "Median Return",
            "Worst Fold",
            "Winners",
            "Chronic Failures",
        ):
            assert any(required in name for name in field_names), (
                f"Missing field {required}; have {field_names}"
            )

    def test_footer_marks_swingrl(self) -> None:
        summary = make_summary()
        payload = build_iteration_completion_embed(summary)
        footer = payload["embeds"][0]["footer"]["text"]
        assert "SwingRL" in footer

    def test_timestamp_present(self) -> None:
        summary = make_summary()
        payload = build_iteration_completion_embed(summary)
        assert "timestamp" in payload["embeds"][0]


# ---------------------------------------------------------------------------
# REPORT-EMBED-02: color logic
# ---------------------------------------------------------------------------


class TestEmbedColor:
    def test_green_when_no_regression_and_cps_up(self) -> None:
        summary = make_summary(cps_v1_delta=0.005, regression_flag=False, regression_dimensions=[])
        payload = build_iteration_completion_embed(summary)
        assert payload["embeds"][0]["color"] == _COLOR_BUY  # green

    def test_yellow_when_cps_up_but_partial_regression(self) -> None:
        """Worst MDD jumped (one regression dim) but CPS still positive."""
        summary = make_summary(
            cps_v1_delta=0.001,
            regression_flag=True,
            regression_dimensions=["worst_fold_mdd"],
        )
        payload = build_iteration_completion_embed(summary)
        assert payload["embeds"][0]["color"] == _COLOR_WARNING  # yellow

    def test_red_when_cps_dropped(self) -> None:
        """Negative CPS delta = full regression = red."""
        summary = make_summary(
            cps_v1_delta=-0.003,
            regression_flag=True,
            regression_dimensions=["cps_v1"],
        )
        payload = build_iteration_completion_embed(summary)
        assert payload["embeds"][0]["color"] == _COLOR_CRITICAL  # red

    def test_neutral_when_no_delta_iter0(self) -> None:
        """Iter 0 with no baseline — render without crashing, neutral color."""
        summary = make_summary(
            iteration=0,
            cps_v1_delta=None,
            return_delta=None,
            worst_mdd_delta=None,
            cps_v3=None,
            regression_flag=False,
        )
        payload = build_iteration_completion_embed(summary)
        # Iter 0 without prior baseline = neutral (green is acceptable)
        assert payload["embeds"][0]["color"] == _COLOR_BUY


# ---------------------------------------------------------------------------
# REPORT-EMBED-03: treatment vs control display
# ---------------------------------------------------------------------------


class TestTreatmentControlDisplay:
    def test_treatment_control_fields_present_when_populated(self) -> None:
        summary = make_summary(treatment_v1=0.0148, control_v1=0.0407)
        payload = build_iteration_completion_embed(summary)
        field_names = [str(f["name"]) for f in payload["embeds"][0]["fields"]]
        assert any("Treatment" in n for n in field_names)
        assert any("Control" in n for n in field_names)

    def test_treatment_control_omitted_when_null(self) -> None:
        """Iter 0-2 had no control folds — treatment/control fields should be skipped."""
        summary = make_summary(treatment_v1=None, control_v1=None)
        payload = build_iteration_completion_embed(summary)
        field_names = [str(f["name"]) for f in payload["embeds"][0]["fields"]]
        assert not any("Treatment" in n for n in field_names)
        assert not any("Control" in n for n in field_names)

    def test_control_outperforming_treatment_visible(self) -> None:
        """The smoking gun finding: control > treatment should render with both values."""
        summary = make_summary(treatment_v1=0.0832, control_v1=0.4206)
        payload = build_iteration_completion_embed(summary)
        embed_str = str(payload)
        assert "0.0832" in embed_str or "0.08320" in embed_str
        assert "0.4206" in embed_str or "0.42063" in embed_str or "0.42060" in embed_str


# ---------------------------------------------------------------------------
# REPORT-EMBED-04: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_iter0_no_deltas_renders(self) -> None:
        """All delta fields None — should not crash."""
        summary = make_summary(
            iteration=0,
            cps_v1_delta=None,
            return_delta=None,
            worst_mdd_delta=None,
            cps_v3=None,
            treatment_v1=None,
            control_v1=None,
        )
        payload = build_iteration_completion_embed(summary)
        # Just check it built something
        assert len(payload["embeds"][0]["fields"]) > 0

    def test_regression_dimensions_listed_when_present(self) -> None:
        summary = make_summary(
            cps_v1_delta=-0.01,
            regression_flag=True,
            regression_dimensions=["cps_v1", "median_return"],
        )
        payload = build_iteration_completion_embed(summary)
        embed_str = str(payload)
        assert "cps_v1" in embed_str or "CPS" in embed_str
        # The dimensions list should appear somewhere
        assert "median_return" in embed_str or "regression" in embed_str.lower()

    def test_dedup_count_shown_when_nonzero(self) -> None:
        """Iter 1 equity had dedup_rows_dropped=9 — should be visible."""
        summary = make_summary(iteration=1, dedup_rows_dropped=9)
        payload = build_iteration_completion_embed(summary)
        embed_str = str(payload)
        assert "9" in embed_str

    def test_dedup_count_omitted_when_zero(self) -> None:
        """Clean iterations should NOT clutter the embed with 'Dedup: 0'."""
        summary = make_summary(dedup_rows_dropped=0)
        payload = build_iteration_completion_embed(summary)
        field_names = [str(f["name"]).lower() for f in payload["embeds"][0]["fields"]]
        # No "Dedup" field at all when count is 0
        assert not any("dedup" in n for n in field_names)
