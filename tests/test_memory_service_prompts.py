"""C3-PROMPT-01..03: goal / anti-pattern / fold-protection blocks present.

Tests verify that both builder functions embed the three required blocks so the LLM
advisor receives its CPS v1 objective, the empirical harm data, and fold protection
rules. Cross-checked against DIAGNOSIS_CORRECTIONS to catch prompt/module drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap — mirrors tests/test_memory_service.py exactly.
# Ensures services/memory/ is first on path so memory_agents can be imported
# without conflicting with any similarly-named module under scripts/.
# ---------------------------------------------------------------------------
_MEMORY_SERVICE_DIR = Path(__file__).parent.parent / "services" / "memory"
_MEMORY_MODULE_NAMES = [
    "app",
    "db",
    "auth",
    "memory_agents",
    "memory_agents.ingest",
    "memory_agents.consolidate",
    "memory_agents.query",
    "routers",
    "routers.core",
    "routers.training",
    "routers.debug",
]

if str(_MEMORY_SERVICE_DIR) in sys.path:
    sys.path.remove(str(_MEMORY_SERVICE_DIR))
sys.path.insert(0, str(_MEMORY_SERVICE_DIR))

# ---------------------------------------------------------------------------
# Remove any stale cached modules from previous imports so the path insert takes
# effect (matches conftest pattern used in test_memory_service.py).
# ---------------------------------------------------------------------------
for _mod in list(sys.modules.keys()):
    if any(_mod == name or _mod.startswith(name + ".") for name in _MEMORY_MODULE_NAMES):
        del sys.modules[_mod]

from memory_agents import query as q  # noqa: E402

from swingrl.memory.training.cps_diagnosis import DIAGNOSIS_CORRECTIONS  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test args — real _build_algo_system_prompt signature:
#   (hp_bounds, rw_bounds, algo_name)
# ---------------------------------------------------------------------------
_HP_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (1e-5, 1e-3),
    "clip_range": (0.1, 0.4),
    "n_epochs": (3, 20),
    "batch_size": (32, 512),
    "gamma": (0.95, 0.995),
    "gae_lambda": (0.85, 1.0),
    "target_kl": (0.01, 0.05),
}
_RW_BOUNDS: dict[str, tuple[float, float]] = {
    "profit": (0.10, 0.70),
    "sharpe": (0.10, 0.60),
    "drawdown": (0.05, 0.50),
    "turnover": (0.00, 0.20),
}
_ALGO = "ppo"


class TestGoalBlock:
    """C3-PROMPT-01: CPS v1 is stated as the single objective."""

    def test_epoch_prompt_states_cps_objective(self) -> None:
        """C3-PROMPT-01: epoch builder embeds goal block with CPS v1 objective."""
        p = q._build_epoch_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "CPS v1" in p
        assert "single objective" in p.lower()
        assert "pass rate is not your goal" in p.lower()
        assert "multiplicative" in p.lower()

    def test_run_config_prompt_states_cps_objective(self) -> None:
        """C3-PROMPT-01: run-config builder embeds goal block with CPS v1 objective."""
        p = q._build_algo_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "CPS v1" in p


class TestAntiPatternBlock:
    """C3-PROMPT-02: empirical harm numbers + diagnosis→correction map."""

    def test_epoch_prompt_cites_harm_numbers(self) -> None:
        """C3-PROMPT-02: epoch builder includes 2.7x and 5.1x harm ratios."""
        p = q._build_epoch_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "2.7" in p and "5.1" in p
        assert "trade_shy" in p
        assert "single_disaster" in p

    def test_corrections_match_diagnosis_module_verbatim(self) -> None:
        """C3-PROMPT-02: prompt correction strings mirror DIAGNOSIS_CORRECTIONS exactly.

        Cross-checked against the module so drift on EITHER side fails the test.
        """
        p = q._build_epoch_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        for label, correction in DIAGNOSIS_CORRECTIONS.items():
            assert correction in p, f"correction for {label!r} missing or drifted in prompt"
            assert label in p, f"label {label!r} missing from prompt"

    def test_mixed_confidence_guidance_present(self) -> None:
        """C3-PROMPT-02: mixed-confidence instruction is in the epoch prompt."""
        p = q._build_epoch_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "mixed" in p.lower()
        assert "conservative" in p.lower() or "prefer no change" in p.lower()


class TestFoldProtectionBlock:
    """C3-PROMPT-03: protected winners untouched; chronic failures bounded."""

    def test_epoch_prompt_has_fold_protection(self) -> None:
        """C3-PROMPT-03: epoch builder includes fold protection markers."""
        p = q._build_epoch_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "protected_winner" in p
        assert "unchanged" in p.lower()
        assert "chronic_failure" in p

    def test_blocks_present_in_both_builders(self) -> None:
        """C3-PROMPT-03: all three section headers present in both builders."""
        e = q._build_epoch_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        r = q._build_algo_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        for marker in ("YOUR OBJECTIVE", "EMPIRICAL ANTI-PATTERNS", "FOLD PROTECTION"):
            assert marker in e, f"epoch prompt missing section header: {marker!r}"
            assert marker in r, f"run-config prompt missing section header: {marker!r}"


class TestPayloadShapeAccuracy:
    """C3-PROMPT-04: each builder references only fields present in its own payload."""

    def test_run_config_prompt_references_prev_iter_diagnoses(self) -> None:
        """C3-PROMPT-04: run-config builder mentions prev_iter_diagnoses (its actual payload field)."""
        r = q._build_algo_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "prev_iter_diagnoses" in r

    def test_run_config_prompt_references_chronic_failure_folds(self) -> None:
        """C3-PROMPT-04: run-config builder mentions chronic_failure_folds list (its actual payload field)."""
        r = q._build_algo_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "chronic_failure_folds" in r

    def test_run_config_prompt_does_not_reference_epoch_only_diagnosis_field(self) -> None:
        """C3-PROMPT-04: run-config builder must NOT reference 'diagnosis' field — epoch-only construct."""
        r = q._build_algo_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "'diagnosis' field" not in r

    def test_run_config_prompt_does_not_reference_fold_role(self) -> None:
        """C3-PROMPT-04: run-config builder must NOT reference fold_role= — epoch-only construct."""
        r = q._build_algo_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "fold_role=" not in r

    def test_epoch_prompt_references_diagnosis_field(self) -> None:
        """C3-PROMPT-04: epoch builder still references 'diagnosis' field (its actual payload field)."""
        e = q._build_epoch_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "'diagnosis' field" in e

    def test_epoch_prompt_references_fold_role(self) -> None:
        """C3-PROMPT-04: epoch builder still references fold_role= (its actual payload field)."""
        e = q._build_epoch_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        assert "fold_role=" in e

    def test_run_config_corrections_match_diagnosis_module_verbatim(self) -> None:
        """C3-PROMPT-04: run-config correction strings mirror DIAGNOSIS_CORRECTIONS exactly.

        Byte-pinned against swingrl.memory.training.cps_diagnosis.DIAGNOSIS_CORRECTIONS
        so drift on either side fails this test.
        """
        r = q._build_algo_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        for label, correction in DIAGNOSIS_CORRECTIONS.items():
            assert correction in r, (
                f"correction for {label!r} missing or drifted in run-config prompt"
            )
            assert label in r, f"label {label!r} missing from run-config prompt"

    def test_epoch_corrections_match_diagnosis_module_verbatim(self) -> None:
        """C3-PROMPT-04: epoch correction strings mirror DIAGNOSIS_CORRECTIONS exactly (re-asserts C3-PROMPT-02)."""
        e = q._build_epoch_system_prompt(_HP_BOUNDS, _RW_BOUNDS, _ALGO)
        for label, correction in DIAGNOSIS_CORRECTIONS.items():
            assert correction in e, f"correction for {label!r} missing or drifted in epoch prompt"
            assert label in e, f"label {label!r} missing from epoch prompt"
