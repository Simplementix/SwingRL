"""Pure helpers for the test-database safety guard.

Lives in its own module (not conftest) so both ``tests/conftest.py`` and
``tests/fixtures/db_cleanup.py`` can import it without a circular dependency.

A Postgres connection is bound to exactly one database, so a ``*_test`` connection
can never see or TRUNCATE the production ``swingrl`` database. These helpers add
two software checks on top of that backbone (a pre-suite guard and a pre-wipe
re-check), both resolving the target the *same way* ``DatabaseManager`` does:
env ``DATABASE_URL`` first, then ``config.system.database_url``.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from swingrl.config.schema import load_config

# Database names that are always safe to mutate/wipe.
SAFE_DB_NAMES = frozenset({"swingrl_test"})

# Extract the database name (last path segment) from a postgres URL, stopping at
# the query string (?) or fragment (#) so a fragment can't spoof a "_test" suffix:
#   postgresql://user:pw@host:5432/<name>[?...|#...]  ->  <name>  # pragma: allowlist secret
_DB_NAME_RE = re.compile(r"/([^/?#]+)(?:[?#]|$)")

# The YAML config DatabaseManager falls back to when DATABASE_URL is unset.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "swingrl.yaml"


@lru_cache(maxsize=1)
def _yaml_fallback_url() -> str:
    """``config.system.database_url`` from the default YAML, parsed once per process.

    The YAML file is static within a test session; re-parsing it per wipe charged
    up to ~1,900 Pydantic loads per DATABASE_URL-less run (practices review §0).
    """
    try:
        config = load_config(_DEFAULT_CONFIG_PATH)
    except Exception:  # noqa: BLE001 — guard must never crash the whole suite
        return ""
    return (config.system.database_url or "").strip()


def resolve_target_db_url() -> str:
    """Resolve the DB URL the same way ``DatabaseManager.__init__`` does.

    Order: ``DATABASE_URL`` env var (read fresh every call — tests monkeypatch
    it), then the memoized ``config.system.database_url`` from the default YAML.
    Returns ``""`` when neither is set (no DB -> tests skip). Whitespace-only
    DATABASE_URL is normalised to blank (treated as unset).
    """
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    return _yaml_fallback_url()


def classify_db_url(db_url: str) -> tuple[str, str | None]:
    """Classify a DB URL for guard decisions.

    Returns ``(verdict, db_name)`` where verdict is one of:
      - ``"blank"``       — no URL; no DB configured (safe; tests skip)
      - ``"unparseable"`` — URL present but no database name found (refuse)
      - ``"safe"``        — resolved name is a recognised test database
      - ``"unsafe"``      — resolved name is NOT a test database (refuse)
    """
    db_url = db_url.strip()
    if not db_url:
        return ("blank", None)
    match = _DB_NAME_RE.search(db_url)
    if not match:
        return ("unparseable", None)
    name = match.group(1)
    # case-sensitive: test databases must use a lowercase "_test" suffix
    if name in SAFE_DB_NAMES or name.endswith("_test"):
        return ("safe", name)
    return ("unsafe", name)
