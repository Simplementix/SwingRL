"""A30 trader/trainer deploy isolation — compose split + models/active write ban.

REQ: Plan A 2.R Task E. Training deploys must never interrupt running paper trading.

Two invariants are guarded here:

1. **Compose split** — the single ``swingrl`` service is split into ``swingrl-trader``
   (scheduler, pinned explicit non-``latest`` image tag, keeps container_name ``swingrl``)
   and ``swingrl-trainer`` (``profiles: ["training"]`` so a bare ``up -d`` never starts or
   recreates it; idle default command; training invoked via ``docker compose run``). The
   live ``swingrl-collector`` service must be left untouched.
2. **models/active write ban** — no trainer-path module (``src/swingrl/training/``,
   ``src/swingrl/memory/training/``, ``scripts/train*``) may WRITE to the live
   ``models/active/`` tree, because the trader loader hot-reloads on artifact mtime
   (Plan A Task D). The only sanctioned active-tree writers are the shadow
   promotion/lifecycle module and the Task 5 bootstrap deploy step (both out of the scan
   roots, both gated). The pre-existing pipeline ``deploy_best_models`` copy is recorded
   as a single documented exception, retired by the Plan B cutover (spec §4.14 point 5).
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Compose split
# ---------------------------------------------------------------------------

COMPOSE_FILES = ("docker-compose.yml", "docker-compose.prod.yml")


def _load_services(name: str) -> dict[str, dict]:
    """Parse a compose file and return its ``services`` mapping."""
    doc = yaml.safe_load((REPO_ROOT / name).read_text())
    assert isinstance(doc, dict), f"{name} did not parse to a mapping"
    services = doc.get("services")
    assert isinstance(services, dict), f"{name} has no services mapping"
    return services


def _env_files(svc: dict) -> list[str]:
    """Normalise a service ``env_file`` (str | list | None) to a list of strings."""
    raw = svc.get("env_file")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


@pytest.mark.parametrize("compose_name", COMPOSE_FILES)
def test_compose_parses(compose_name: str) -> None:
    """A30: both compose files are valid YAML with a services mapping."""
    services = _load_services(compose_name)
    assert services, f"{compose_name} has an empty services mapping"


@pytest.mark.parametrize("compose_name", COMPOSE_FILES)
def test_legacy_swingrl_service_removed(compose_name: str) -> None:
    """A30: the monolithic ``swingrl`` service is renamed away (split into trader/trainer)."""
    services = _load_services(compose_name)
    assert "swingrl" not in services, (
        f"{compose_name} still defines the monolithic `swingrl` service — it must be split "
        "into swingrl-trader + swingrl-trainer"
    )


@pytest.mark.parametrize("compose_name", COMPOSE_FILES)
def test_trader_service_present_and_pinned(compose_name: str) -> None:
    """A30: swingrl-trader exists, keeps container_name `swingrl`, pinned non-latest tag."""
    services = _load_services(compose_name)
    assert "swingrl-trader" in services, f"{compose_name} missing swingrl-trader service"
    trader = services["swingrl-trader"]

    # Container name kept so existing runbooks / `docker exec swingrl` keep working.
    assert trader.get("container_name") == "swingrl", (
        "swingrl-trader must keep container_name `swingrl`"
    )

    # Image pinned to an explicit, non-`latest` tag — bumped only by hand in a deploy window.
    image = trader.get("image")
    assert isinstance(image, str) and ":" in image, (
        "swingrl-trader must pin an explicit `image:` tag"
    )
    tag = image.rsplit(":", 1)[1]
    assert tag not in ("", "latest"), (
        f"swingrl-trader image tag must be explicit and non-latest (got {image!r})"
    )
    assert image.startswith("swingrl:trader-"), (
        f"swingrl-trader image should follow the swingrl:trader-<date> convention (got {image!r})"
    )

    # Secrets/env (incl. SWINGRL_ALERTING__* Discord webhook overrides) come from .env.
    assert ".env" in _env_files(trader), "swingrl-trader must load env_file: .env"


@pytest.mark.parametrize("compose_name", COMPOSE_FILES)
def test_trainer_service_profiled_and_idle(compose_name: str) -> None:
    """A30: swingrl-trainer is behind the `training` profile, same image build, idle default."""
    services = _load_services(compose_name)
    assert "swingrl-trainer" in services, f"{compose_name} missing swingrl-trainer service"
    trainer = services["swingrl-trainer"]

    # Profile-gated: a bare `docker compose up -d` never starts or recreates it.
    profiles = trainer.get("profiles") or []
    assert "training" in profiles, (
        "swingrl-trainer must declare profiles: ['training'] so `up -d` never starts it"
    )

    # Same image build as the trader (one image, one Dockerfile/target).
    build = trainer.get("build") or {}
    assert build.get("context") == ".", "swingrl-trainer must share build context '.'"
    assert build.get("target") == "production", "swingrl-trainer must build the production target"

    # Idle default command — NOT the scheduler; training is invoked via `compose run`.
    command = trainer.get("command")
    assert command is not None, "swingrl-trainer must set an idle default command (not the CMD)"
    command_str = " ".join(command) if isinstance(command, list) else str(command)
    assert "scripts/main.py" not in command_str, (
        "swingrl-trainer default command must NOT run the trading scheduler (scripts/main.py)"
    )

    # Discord webhook overrides ride the shared .env into the trainer too.
    assert ".env" in _env_files(trainer), "swingrl-trainer must load env_file: .env"


@pytest.mark.parametrize("compose_name", COMPOSE_FILES)
def test_dashboard_depends_on_trader(compose_name: str) -> None:
    """A30: the dashboard's dependency follows the rename to swingrl-trader."""
    services = _load_services(compose_name)
    dashboard = services.get("swingrl-dashboard")
    if dashboard is None:
        pytest.skip(f"{compose_name} has no swingrl-dashboard service")
    depends = dashboard.get("depends_on") or []
    depends_keys = list(depends.keys()) if isinstance(depends, dict) else list(depends)
    assert "swingrl" not in depends_keys, (
        "dashboard must not depend on the removed `swingrl` service"
    )
    assert "swingrl-trader" in depends_keys, "dashboard must depend on swingrl-trader"


def test_collector_service_untouched() -> None:
    """A30/D9: the live swingrl-collector service is left byte-identical (pinned tag + command)."""
    services = _load_services("docker-compose.yml")
    assert "swingrl-collector" in services, "swingrl-collector must remain in docker-compose.yml"
    collector = services["swingrl-collector"]
    # Its own pinned tag (unrelated image churn must never recreate it via bare `up -d`).
    assert collector.get("image") == "swingrl-collector:2026-07-22-1"
    assert collector.get("command") == ["python", "scripts/collector_main.py"]


def test_docker_compose_config_renders() -> None:
    """A30: `docker compose config` renders both default and training profiles cleanly.

    Skipped when docker or the host .env is unavailable (e.g. CI); exercised on homelab.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker binary not available")
    if not (REPO_ROOT / ".env").exists():
        pytest.skip(".env not present (compose env_file cannot be resolved)")
    for compose_name in COMPOSE_FILES:
        for extra in ([], ["--profile", "training"]):
            result = subprocess.run(  # noqa: S603
                ["docker", "compose", "-f", compose_name, *extra, "config", "-q"],  # noqa: S607
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert result.returncode == 0, (
                f"`docker compose -f {compose_name} {' '.join(extra)} config` failed:\n"
                f"{result.stderr}"
            )


# ---------------------------------------------------------------------------
# models/active write ban
# ---------------------------------------------------------------------------

# Trainer-path scan roots. Any module here that WRITES the live models/active tree
# would hot-reload straight into the running trader (Plan A Task D loader mtime cache).
_SCAN_DIRS = ("src/swingrl/training", "src/swingrl/memory/training")
_SCAN_SCRIPT_GLOB = "train*.py"

# A path-join to a top-level "active" segment — the LIVE production tree. Lines that
# also mention "iterations" are the per-iteration scratch tree (models/iterations/
# iter_N/active/) and are explicitly allowed.
_ACTIVE_JOIN = re.compile(r"""/\s*['"]active['"]|['"]models/active""")

# Filesystem-mutating verbs. Reads (`.exists()`) and plain path construction do not match.
_WRITE_VERB = re.compile(
    r"""\.mkdir\(|\.write_bytes\(|\.write_text\(|\.rename\(|\.unlink\(|\.rmdir\(|\.save\(|"""
    r"""shutil\.(?:copy2?|copyfile|move|rmtree)\(|pickle\.dump\(|torch\.save\(|"""
    r"""os\.(?:rename|replace|remove|removedirs|makedirs|mkdir)\(|open\([^)]*['"][waxb+]+['"]"""
)

# Single documented pre-existing exception: the training pipeline's end-of-run deploy
# still copies winners into models/active/. Retired by the Plan B cutover
# (spec §4.14 point 5 — "training writes models/iterations/ + shadow only"). Recorded by
# (relative_path, function_name) so the guard still fails on any NEW trainer-path write.
_SANCTIONED = frozenset({("scripts/train_pipeline.py", "deploy_best_models")})


def _has_prod_active(text: str) -> bool:
    """True if ``text`` constructs a LIVE models/active path (excluding iteration scratch)."""
    for line in text.splitlines():
        if "iterations" in line:
            continue
        if _ACTIVE_JOIN.search(line):
            return True
    return False


def _units(source: str) -> list[tuple[str, str]]:
    """Split source into (name, text) units: each function plus a `<module>` remainder.

    Splitting at function granularity lets the guard tie a write verb to the active-path
    construction in the same scope, so a read-only checkpoint helper that merely names the
    active tree is not mistaken for a writer.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    covered: set[int] = set()
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(source, node) or ""
            out.append((node.name, seg))
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                covered.add(ln)
    module_text = "\n".join(line for i, line in enumerate(lines, 1) if i not in covered)
    out.append(("<module>", module_text))
    return out


def _active_write_units(source: str) -> list[str]:
    """Return names of units in ``source`` that both build a live active path AND write."""
    return [
        name for name, text in _units(source) if _has_prod_active(text) and _WRITE_VERB.search(text)
    ]


def _scan_roots() -> list[Path]:
    """Collect every trainer-path python module under the scan roots."""
    files: list[Path] = []
    for d in _SCAN_DIRS:
        files.extend(sorted((REPO_ROOT / d).rglob("*.py")))
    files.extend(sorted((REPO_ROOT / "scripts").glob(_SCAN_SCRIPT_GLOB)))
    return files


def _all_active_write_violations() -> list[tuple[str, str]]:
    """Return (relpath, function) for every trainer-path unit writing the live active tree."""
    violations: list[tuple[str, str]] = []
    for path in _scan_roots():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for name in _active_write_units(path.read_text()):
            violations.append((rel, name))
    return violations


def test_no_unsanctioned_trainer_path_active_writes() -> None:
    """A30: no trainer-path module writes the live models/active tree (bar the 1 exception)."""
    unsanctioned = [v for v in _all_active_write_violations() if v not in _SANCTIONED]
    assert unsanctioned == [], (
        "Trainer-path modules must not write models/active/ (the trader hot-reloads it). "
        f"Unsanctioned writers found: {unsanctioned}. If a new active writer is legitimate "
        "it must be a gated promotion/bootstrap step outside the trainer path."
    )


def test_sanctioned_exception_still_present() -> None:
    """The one recorded exception is a real, detected writer — the allowlist is not dead code."""
    all_violations = set(_all_active_write_violations())
    assert _SANCTIONED <= all_violations, (
        "Sanctioned allowlist entry no longer matches a detected writer — update _SANCTIONED "
        f"(detected: {sorted(all_violations)})"
    )


def test_write_ban_detects_planted_violation() -> None:
    """The guard actually fires: a synthetic module that copies into models/active is flagged."""
    planted = (
        "import shutil\n"
        "from pathlib import Path\n"
        "def deploy(models_dir: Path) -> None:\n"
        "    dst = models_dir / 'active' / 'equity' / 'ppo'\n"
        "    dst.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copy2('src.zip', str(dst / 'model.zip'))\n"
    )
    assert _active_write_units(planted) == ["deploy"]


def test_write_ban_allows_reads_and_iteration_scratch() -> None:
    """Read-only checks and per-iteration writes are NOT flagged (no false positives)."""
    read_only = (
        "from pathlib import Path\n"
        "def check(models_dir: Path) -> bool:\n"
        "    return (models_dir / 'active' / 'equity' / 'ppo' / 'model.zip').exists()\n"
    )
    iteration_scratch = (
        "from pathlib import Path\n"
        "def save(models_dir: Path, i: int) -> None:\n"
        "    p = models_dir / 'iterations' / f'iter_{i}' / 'active' / 'equity' / 'ppo'\n"
        "    p.mkdir(parents=True, exist_ok=True)\n"
        "    (p / 'model.zip').write_bytes(b'x')\n"
    )
    assert _active_write_units(read_only) == []
    assert _active_write_units(iteration_scratch) == []
