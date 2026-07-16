"""U2 (spec Section 2.2): cold-start / fallback weights must be identity with DEFAULT_WEIGHTS.

Variant used: AST-literal parsing, NOT importlib standalone module loading. Loading
query.py via importlib.util fails here — `from db import (...)` (query.py:35) resolves
against an empty top-level `db/` directory in the repo root, which Python treats as a
namespace package (no __init__.py), so the import succeeds but none of the expected
names (get_active_consolidations, etc.) exist on it, raising ImportError. Parsing the
module's literal assignments with `ast` avoids executing any of query.py's imports.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from swingrl.memory.training.reward_wrapper import DEFAULT_WEIGHTS

_QUERY_PY_PATH = Path(__file__).parents[2] / "services" / "memory" / "memory_agents" / "query.py"


def _extract_module_level_dict(source: str, target_name: str) -> dict[str, Any]:
    """Return the literal value assigned to a module-level name in `source`.

    Handles both plain assignments (`NAME = {...}`) and annotated assignments
    (`NAME: dict[str, Any] = {...}`), matching by the assignment target's name.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    return ast.literal_eval(node.value)  # type: ignore[no-any-return]
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == target_name:
                assert node.value is not None
                return ast.literal_eval(node.value)  # type: ignore[no-any-return]
    raise AssertionError(f"module-level assignment to {target_name!r} not found")


def test_safe_defaults_match_default_weights() -> None:
    """_SAFE_DEFAULTS['reward_weights'] must equal DEFAULT_WEIGHTS (identity fallback)."""
    source = _QUERY_PY_PATH.read_text()
    safe_defaults = _extract_module_level_dict(source, "_SAFE_DEFAULTS")
    assert safe_defaults["reward_weights"] == DEFAULT_WEIGHTS


def test_safe_epoch_defaults_match_default_weights() -> None:
    """_SAFE_EPOCH_DEFAULTS['reward_weights'] must equal DEFAULT_WEIGHTS (identity fallback)."""
    source = _QUERY_PY_PATH.read_text()
    safe_epoch_defaults = _extract_module_level_dict(source, "_SAFE_EPOCH_DEFAULTS")
    assert safe_epoch_defaults["reward_weights"] == DEFAULT_WEIGHTS
