# Phase 1: Dev Foundation - Research

**Researched:** 2026-03-06
**Domain:** Python dev environment, uv, Docker, pre-commit, GitHub repo scaffold
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Initial dependency scope**
- Include ALL project dependencies upfront in pyproject.toml (torch, FinRL, stable-baselines3, pandas_ta, hmmlearn, alpaca-py, etc.) — catches version conflicts early
- Use compatible version ranges (e.g., torch>=2.2,<2.3) with uv.lock for exact resolution
- uv.lock committed to repo for reproducibility

**Package structure**
- Full scaffold from day one: src/swingrl/{data/, envs/, agents/, training/, execution/, monitoring/, config/, utils/} with __init__.py
- Top-level project directories at repo root: config/, data/, db/, models/, tests/, scripts/, status/ — src/swingrl/ contains only Python source code
- tests/ mirrors src/swingrl/ subpackages: tests/data/, tests/envs/, tests/agents/, etc.
- py.typed marker included from day one — all code must have type annotations

**Linter and formatter configuration**
- Black line length: 100 characters (accommodates ML code with long parameter lists)
- bandit and detect-secrets are BLOCKING in pre-commit — security findings fail the commit
- .bandit config to exclude test files from security scanning

**CI test baseline**
- ci-homelab.sh runs full quality gate: Docker build + pytest + ruff/black/mypy checks inside container
- Docker cache used by default for fast builds; --no-cache available as flag for clean builds
- Phase 1 tests: import smoke test (torch, package versions) PLUS directory structure validation (all expected dirs, __init__.py files, py.typed marker)

### Claude's Discretion
- PyTorch platform handling (separate dependency groups vs. single install with platform detection at Docker build time)
- FinRL installation source (PyPI release vs. Git commit pin) — evaluate current package state
- mypy strictness level — account for third-party stub availability (FinRL, pandas_ta, etc.)
- ruff rule set selection — balance coverage with noise from RL/ML code patterns
- ci-homelab.sh cleanup strategy — balance disk usage with build speed on homelab

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENV-01 | Python 3.11 development environment with PyTorch MPS acceleration verified on M1 Mac | PyTorch MPS is included in standard macOS wheel; torch.backends.mps.is_available() used for verification |
| ENV-02 | uv package manager installed on M1 Mac and homelab (Ubuntu 24.04) | uv has single-binary install, works on both Darwin arm64 and Linux x86_64 |
| ENV-03 | GitHub repository with canonical directory structure (src/, config/, data/, db/, models/, tests/, scripts/, status/) | Standard src-layout with mkdir -p and touch commands at scaffold time |
| ENV-04 | pyproject.toml with pinned dependencies and tool configuration (pytest, ruff, black, mypy) | uv manages pyproject.toml + uv.lock; all tool config lives in [tool.*] sections |
| ENV-05 | Pre-commit hooks enforcing ruff, black, mypy, detect-secrets, and bandit on every commit | .pre-commit-config.yaml hooks; detect-secrets requires .secrets.baseline; bandit needs [tool.bandit.assert_used] to exclude tests |
| ENV-06 | Dockerfile using python:3.11-slim with CPU-only PyTorch and non-root trader user (UID 1000) | pip install torch --index-url https://download.pytorch.org/whl/cpu; useradd pattern verified |
| ENV-07 | docker-compose.yml with resource limits (2.5g mem, 1 cpu), bind mounts, env_file secrets, TZ=America/New_York | Standard compose v3 syntax; mem_limit + cpus; volumes for data/db/models/logs |
| ENV-08 | ci-homelab.sh script (git pull + docker compose build --no-cache + pytest + cleanup) validated via ssh homelab | bash script; ssh homelab "cd ~/swingrl && bash ci-homelab.sh"; cleanup = docker compose down + docker image prune |
</phase_requirements>

---

## Summary

Phase 1 establishes a reproducible Python 3.11 development environment that works natively on M1 Mac (with MPS acceleration) and inside Docker on the x86 homelab (CPU-only). The platform split is the central technical challenge: PyTorch has different installation paths for macOS (standard PyPI wheel includes MPS) and Linux/Docker (CPU-only wheel from `download.pytorch.org/whl/cpu`). uv handles this elegantly via `[tool.uv.sources]` with `sys_platform` markers, with a single `uv.lock` that resolves correctly on both platforms.

FinRL 0.3.7 is the latest PyPI release (April 2024) and is flagged as low-maintenance. However, for Phase 1 the concern is only whether it resolves correctly alongside stable-baselines3 >=2.0 and gymnasium — not whether it functions correctly (that's a Phase 6 concern). The recommended approach is to pin FinRL at `==0.3.7` from PyPI and verify the full dependency graph resolves. If it doesn't, git-pin to the latest master commit. The key known conflict: FinRL's requirements.txt pulls `alpaca_trade_api` and SB3 pulls `gymnasium>=0.29.1`; verify these don't fight.

Pre-commit is the other non-trivial setup area. bandit requires `bandit[toml]` as an additional dependency to read pyproject.toml config, and detect-secrets requires a committed `.secrets.baseline` file before the hook passes. Both of these are first-time-setup steps that must happen in a specific order.

**Primary recommendation:** Use uv's `[tool.uv.sources]` with `sys_platform` markers to serve CPU torch to Docker/Linux and standard torch (with MPS) to macOS — this is the officially documented pattern and produces a single uv.lock that works across both platforms.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.11.x | Runtime | Locked for FinRL/pyfolio compatibility |
| uv | >=0.4 (pin specific) | Package manager + venv | Replaces pip/poetry; single binary; uv.lock for reproducibility |
| torch | >=2.2,<2.4 | ML backend + MPS | MPS backend included in macOS wheel; CPU wheel for Docker |
| FinRL | ==0.3.7 | RL finance framework | Last PyPI release; wraps SB3 environments |
| stable-baselines3 | >=2.0,<3 | PPO/A2C/SAC implementations | Required by FinRL; Gymnasium backend |
| gymnasium | >=0.29.1 | RL env interface | SB3 v2 minimum; Farama standard |
| pre-commit | >=3.6 | Git hook manager | Industry standard; .pre-commit-config.yaml |
| ruff | >=0.4 | Linter + formatter (replaces isort) | Replaces flake8/isort; used alongside black |
| black | >=24.0 | Formatter | Enforced at line-length=100 |
| mypy | >=1.8 | Static type checker | py.typed project requires it |
| bandit | >=1.7.7 | Security linter | BLOCKING in pre-commit |
| detect-secrets | >=1.5.0 | Secret scanning | BLOCKING in pre-commit |
| pytest | >=8.0 | Test runner | Industry standard |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas-stubs | latest | mypy stubs for pandas | Installed in dev group to reduce mypy noise |
| types-requests | latest | mypy stubs for requests | If requests is a transitive dep |
| hmmlearn | >=0.3 | HMM regime detection | Included upfront for version conflict detection |
| pandas_ta | >=0.3.14b | Technical indicators | No native type stubs — ignore_missing_imports |
| alpaca-py | >=0.20 | Alpaca broker API | Replaces deprecated alpaca_trade_api |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| uv | poetry | uv is 10-100x faster; single binary; lockfile format is simpler |
| uv | pip + venv | No lockfile natively; no workspace support |
| ruff | flake8 + isort | ruff replaces both; faster; single config section |
| bandit (standalone) | ruff S rules | ruff S rules cover many bandit checks but not all; keep bandit for compliance |
| detect-secrets | git-secrets, gitleaks | detect-secrets has best pre-commit integration; .secrets.baseline audit workflow |

**Installation (local dev on M1 Mac):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group dev
source .venv/bin/activate
```

**Installation (homelab Ubuntu 24.04):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group dev
```

---

## Architecture Patterns

### Recommended Project Structure

```
SwingRL/                          # repo root
├── src/
│   └── swingrl/                  # Python package (src-layout)
│       ├── __init__.py
│       ├── py.typed              # PEP 561 marker — type annotations required
│       ├── data/                 # Ingestion, validation, storage
│       ├── envs/                 # Gymnasium environments
│       ├── agents/               # Agent wrappers
│       ├── training/             # Training loops, callbacks
│       ├── execution/            # Order execution middleware
│       ├── monitoring/           # Alerter, dashboard
│       ├── config/               # Config loading, Pydantic schemas
│       └── utils/                # Shared utilities
├── tests/
│   ├── conftest.py               # Shared fixtures (Phase 1: minimal)
│   ├── test_smoke.py             # Import + structure validation
│   ├── data/                     # Mirrors src/swingrl/data/
│   ├── envs/
│   ├── agents/
│   └── ...
├── config/                       # YAML config files
├── data/                         # Raw data (gitignored except .gitkeep)
├── db/                           # SQLite + DuckDB files (gitignored)
├── models/
│   ├── active/
│   ├── shadow/
│   └── archive/
├── scripts/                      # Operational scripts (ci-homelab.sh, etc.)
├── status/                       # Health status files
├── logs/                         # Bind-mounted in Docker (gitignored)
├── pyproject.toml
├── uv.lock                       # COMMITTED to repo
├── .pre-commit-config.yaml
├── .secrets.baseline             # detect-secrets baseline (committed)
├── Dockerfile
├── docker-compose.yml
├── .env.example                  # Template — actual .env gitignored
└── ci-homelab.sh                 # Homelab CI script
```

### Pattern 1: uv Platform-Specific PyTorch

**What:** Single pyproject.toml with platform-aware torch index routing via `[tool.uv.sources]`
**When to use:** Any project needing CPU torch in Docker/Linux and MPS/full torch on macOS

```toml
# Source: https://docs.astral.sh/uv/guides/integration/pytorch/
[project]
dependencies = [
    "torch>=2.2,<2.4",
    "torchvision>=0.17,<0.20",
]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]
torchvision = [
  { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

**Result:** macOS gets standard PyPI torch (includes MPS), Linux/Docker gets CPU-only wheel. A single `uv.lock` covers both.

### Pattern 2: Dockerfile with uv (Layered for Cache)

**What:** Two-stage uv sync to maximize Docker layer caching
**When to use:** Every Python/uv Dockerfile

```dockerfile
# Source: https://docs.astral.sh/uv/guides/integration/docker/
FROM python:3.11-slim AS base

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /bin/

WORKDIR /app

# Stage 1: Install deps only (cached as long as pyproject.toml/uv.lock unchanged)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

# Stage 2: Copy source and install project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Non-root user
RUN useradd -m -u 1000 trader
USER trader

ENTRYPOINT ["uv", "run", "python", "-m", "swingrl"]
```

**Note:** `--locked` asserts uv.lock is current — build fails if lockfile is stale, which is the correct behavior.

### Pattern 3: Pre-commit Configuration

**What:** .pre-commit-config.yaml with correct hook revisions and bandit pyproject.toml integration
**When to use:** This exact file for the project

```yaml
# Source: https://pre-commit.com/hooks.html + official hook repos
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
        # ruff-format replaces black in pre-commit; OR use black hook separately

  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pandas-stubs
          - types-requests

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.7
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ["--baseline", ".secrets.baseline"]
```

**Critical first-run setup sequence:**
1. `uv run detect-secrets scan > .secrets.baseline` — generate baseline first
2. `git add .secrets.baseline` — commit baseline before running pre-commit
3. `pre-commit install` — install hooks
4. `pre-commit run --all-files` — first full run

### Pattern 4: bandit pyproject.toml Config (Exclude Tests)

```toml
# Source: https://bandit.readthedocs.io/en/latest/config.html
[tool.bandit]
exclude_dirs = ["tests", ".venv"]

[tool.bandit.assert_used]
skips = ["*/*_test.py", "*/test_*.py", "tests/*"]
```

**Why:** `assert` statements in pytest tests trigger B101 (assert_used). This config skips that check in test files while keeping it active in production code.

### Pattern 5: mypy Configuration (Practical for ML)

```toml
# Source: https://mypy.readthedocs.io/en/stable/config_file.html
[tool.mypy]
python_version = "3.11"
strict = false          # Can't use strict — too many untyped deps
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true    # All new code MUST have annotations
check_untyped_defs = true

# Third-party libraries without stubs — ignore imports
[[tool.mypy.overrides]]
module = [
    "finrl.*",
    "pandas_ta.*",
    "hmmlearn.*",
    "stable_baselines3.*",   # SB3 has partial types; may not need this
    "gymnasium.*",
    "alpaca.*",
]
ignore_missing_imports = true
```

**Why not strict:** FinRL, pandas_ta, and hmmlearn have no py.typed markers or stub packages. `strict = true` would emit hundreds of errors on import. The override approach silences noise from third-party while enforcing types on project code.

### Pattern 6: Ruff Configuration (ML-Appropriate)

```toml
# Source: https://docs.astral.sh/ruff/configuration/
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # Pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "N",    # pep8-naming
]
ignore = [
    "E501",   # line too long (handled by formatter)
    "B008",   # do not perform function calls in argument defaults (common in ML)
    "N803",   # argument name should be lowercase (ML uses X, y conventions)
    "N806",   # variable in function should be lowercase (X, Y in ML)
]
```

**Why ignore N803/N806:** ML code conventionally uses `X` for feature matrices, `y` for labels. These pep8-naming rules fire constantly on legitimate ML patterns.

### Pattern 7: ci-homelab.sh Structure

```bash
#!/usr/bin/env bash
# Source: project-specific pattern from CONTEXT.md
set -euo pipefail

REPO_DIR="$HOME/swingrl"
NO_CACHE=${1:-""}  # Pass --no-cache as first arg for clean build

cd "$REPO_DIR"

echo "=== [1/4] Git pull ==="
git pull --ff-only

echo "=== [2/4] Docker build ==="
if [[ "$NO_CACHE" == "--no-cache" ]]; then
    docker compose build --no-cache
else
    docker compose build
fi

echo "=== [3/4] Run tests ==="
docker compose run --rm swingrl uv run pytest tests/ -v

echo "=== [4/4] Cleanup ==="
docker compose down
docker image prune -f   # Remove dangling images only (not all images)

echo "=== CI PASSED ==="
```

**Usage from M1 Mac:** `ssh homelab "bash ~/swingrl/ci-homelab.sh"`
**Clean build:** `ssh homelab "bash ~/swingrl/ci-homelab.sh --no-cache"`

### Anti-Patterns to Avoid

- **Committing uv.lock to .gitignore:** The lockfile IS the reproducibility guarantee. It must be committed.
- **Using `pip install` in Dockerfile directly:** Mix of uv and pip breaks dependency resolution. Use `uv sync` exclusively inside Docker.
- **Running detect-secrets without a baseline:** The hook will fail until `.secrets.baseline` exists and is committed. Generate baseline before `pre-commit install`.
- **Setting `strict = true` in mypy for a FinRL project:** FinRL, pandas_ta, hmmlearn have no type stubs. Strict mode generates hundreds of false positives. Use targeted `disallow_untyped_defs` instead.
- **Installing torch from default PyPI in Docker:** Default PyPI torch includes CUDA binaries (~2.5 GB). CPU-only wheel is ~200 MB. Always use `--index-url https://download.pytorch.org/whl/cpu` in Docker.
- **Using `detect-secrets` hook before generating baseline:** Pre-commit will fail with a cryptic error. The sequence is: generate baseline → commit baseline → install hooks.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dependency locking | Custom version pins in requirements.txt | uv + uv.lock | uv resolves the full transitive graph; manual pins miss transitive conflicts |
| Secret scanning | Grep for patterns | detect-secrets | Entropy analysis catches secrets that regex misses; auditable baseline |
| Import sorting | Manual ordering | ruff (I rules) | Isort-compatible; enforced consistently |
| Security scanning | Code review only | bandit | B-series rules catch subprocess injection, hardcoded passwords, assert misuse |
| Platform-specific deps | Separate requirements files per platform | uv `[tool.uv.sources]` markers | Single lockfile, platform resolved by uv at install time |
| Docker layer caching | Single `RUN pip install -r requirements.txt` | Two-stage uv sync | Separating deps from source maximizes Docker cache hit rate |

**Key insight:** The entire toolchain (uv, pre-commit, ruff, bandit, detect-secrets) is designed to compose. Attempting to replicate any of these with custom scripts adds maintenance burden with none of the ecosystem integration benefits.

---

## Common Pitfalls

### Pitfall 1: detect-secrets Baseline Missing on First Run

**What goes wrong:** `pre-commit run --all-files` fails with "Baseline file does not exist" or similar cryptic error.
**Why it happens:** The detect-secrets hook requires `.secrets.baseline` to exist before it runs. The file is generated separately, not automatically.
**How to avoid:** Always generate and commit the baseline before running pre-commit:
```bash
uv run detect-secrets scan > .secrets.baseline
git add .secrets.baseline
git commit -m "chore: add detect-secrets baseline"
pre-commit install
```
**Warning signs:** Any pre-commit error mentioning "baseline" or "secrets.baseline".

### Pitfall 2: bandit Cannot Read pyproject.toml Without `bandit[toml]`

**What goes wrong:** bandit ignores `[tool.bandit]` config, flags asserts in test files, or fails silently.
**Why it happens:** bandit requires the `toml` extra to parse pyproject.toml. The pre-commit hook needs `additional_dependencies: ["bandit[toml]"]`.
**How to avoid:** Always include `bandit[toml]` in pre-commit hook config (shown in Pattern 3 above).
**Warning signs:** bandit flags B101 (assert_used) in test files even though `[tool.bandit.assert_used] skips` is configured.

### Pitfall 3: FinRL Dependency Conflicts

**What goes wrong:** `uv sync` fails with conflicting gymnasium/gym/SB3 version requirements.
**Why it happens:** FinRL 0.3.7's pyproject.toml specifies `stable-baselines3 >= 2.0.0a5` which is fine, but its `requirements.txt` also lists `elegantrl`, `ray`, `alpaca_trade_api` (deprecated), and others that may conflict with newer packages.
**How to avoid:** Install FinRL with `pip install finrl` to inspect actual resolver behavior first. If conflicts occur, use FinRL from git:
```toml
# Fallback if PyPI version conflicts
finrl = { git = "https://github.com/AI4Finance-Foundation/FinRL.git" }
```
**Warning signs:** Resolver errors mentioning `gym` vs `gymnasium`, `alpaca_trade_api` vs `alpaca-py`.

### Pitfall 4: Docker Build Uses Wrong PyTorch (CUDA vs CPU)

**What goes wrong:** Docker image is 2-3 GB larger than expected; build fails due to missing CUDA libraries at runtime.
**Why it happens:** Without explicit `--index-url`, pip/uv resolves to the CUDA wheel from PyPI.
**How to avoid:** In Dockerfile, install torch explicitly from CPU index:
```dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Or use uv with the `[tool.uv.sources]` Linux marker so `uv sync --locked` in Docker picks the CPU wheel.
**Warning signs:** `docker images` shows image >1.5 GB; build mentions "cu" packages.

### Pitfall 5: M1 Mac MPS Not Available After Install

**What goes wrong:** `torch.backends.mps.is_available()` returns False on M1 Mac.
**Why it happens:** MPS requires macOS 12.3+ and the standard PyPI torch build. If installed from the CPU-only index URL (Linux-style), MPS is excluded.
**How to avoid:** On macOS, install torch from standard PyPI (no index override). The `[tool.uv.sources]` marker approach automatically handles this — only Linux gets the CPU index.
**Warning signs:** MPS returns False; `torch.__version__` shows `+cpu` suffix on macOS.

### Pitfall 6: uv.lock Drift Between Platforms

**What goes wrong:** uv.lock committed from macOS doesn't resolve correctly on Linux, or vice versa.
**Why it happens:** With platform-specific sources, uv generates a multi-platform lock. If `--locked` is used in Docker and the lock was generated only on macOS, it may have missing Linux entries.
**How to avoid:** After configuring `[tool.uv.sources]`, run `uv lock` once to regenerate the complete multi-platform lockfile, then commit.
**Warning signs:** `uv sync --locked` in Docker fails with "No solution found".

### Pitfall 7: bind mount Permission Errors with UID 1000

**What goes wrong:** Container's `trader` user (UID 1000) cannot write to bind-mounted volumes on the homelab host.
**Why it happens:** Host directories created as root don't allow writes by UID 1000.
**How to avoid:** Pre-create host directories with correct ownership before running docker compose:
```bash
mkdir -p ~/swingrl/{data,db,models,logs}
chown -R 1000:1000 ~/swingrl/{data,db,models,logs}
```
**Warning signs:** `PermissionError` in container logs when writing to data/ or db/ directories.

---

## Code Examples

Verified patterns from official sources and project decisions:

### pyproject.toml Full Template

```toml
# Source: https://docs.astral.sh/uv/concepts/projects/config/ + project decisions
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "swingrl"
version = "0.1.0"
description = "RL-based swing trading system"
requires-python = ">=3.11,<3.12"
dependencies = [
    # Core ML
    "torch>=2.2,<2.4",
    "torchvision>=0.17,<0.20",
    # RL framework
    "stable-baselines3[extra]>=2.0,<3",
    "gymnasium>=0.29.1",
    "finrl==0.3.7",
    # Data
    "pandas>=2.0,<3",
    "numpy>=1.26,<2",
    "pandas-ta>=0.3.14b",
    "hmmlearn>=0.3",
    # Broker APIs
    "alpaca-py>=0.20",
    # Config
    "pydantic>=2.5",
    "pyyaml>=6.0",
]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]
torchvision = [
  { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pre-commit>=3.6",
    "ruff>=0.4",
    "black>=24.0",
    "mypy>=1.8",
    "bandit[toml]>=1.7.7",
    "detect-secrets>=1.5.0",
    "pandas-stubs",
    "types-requests",
]

[tool.hatch.build.targets.wheel]
packages = ["src/swingrl"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "N"]
ignore = ["E501", "B008", "N803", "N806"]

[tool.mypy]
python_version = "3.11"
strict = false
warn_return_any = true
disallow_untyped_defs = true
check_untyped_defs = true

[[tool.mypy.overrides]]
module = ["finrl.*", "pandas_ta.*", "hmmlearn.*", "alpaca.*"]
ignore_missing_imports = true

[tool.bandit]
exclude_dirs = ["tests", ".venv"]

[tool.bandit.assert_used]
skips = ["*/test_*.py", "*_test.py"]
```

### tests/test_smoke.py (Phase 1 Baseline)

```python
# Phase 1 smoke test — import verification + structure validation
import importlib
import sys
from pathlib import Path

import pytest


def test_python_version() -> None:
    """ENV-01: Python 3.11 required."""
    assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version_info}"


def test_torch_importable() -> None:
    """ENV-01: torch must import without error."""
    import torch  # noqa: F401
    assert torch.__version__


def test_torch_mps_available_on_mac() -> None:
    """ENV-01: MPS acceleration on M1 Mac (skip in Docker/CI)."""
    import platform
    import torch

    if platform.system() != "Darwin":
        pytest.skip("MPS only available on macOS")
    # On macOS, MPS should be available (fails test if not)
    assert torch.backends.mps.is_available(), (
        "MPS not available — install standard macOS torch, not CPU-only wheel"
    )


def test_core_imports() -> None:
    """All project dependencies must import cleanly."""
    packages = [
        "stable_baselines3",
        "gymnasium",
        "pandas",
        "numpy",
        "pandas_ta",
        "hmmlearn",
        "pydantic",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_swingrl_package_importable() -> None:
    """src/swingrl package must be importable."""
    import swingrl  # noqa: F401


def test_py_typed_marker_exists() -> None:
    """py.typed marker must exist for mypy compliance."""
    import swingrl
    pkg_dir = Path(swingrl.__file__).parent  # type: ignore[arg-type]
    assert (pkg_dir / "py.typed").exists(), "py.typed marker missing from src/swingrl/"


def test_directory_structure() -> None:
    """ENV-03: Canonical directory structure must exist at repo root."""
    repo_root = Path(__file__).parent.parent
    required_dirs = [
        "src/swingrl",
        "config",
        "data",
        "db",
        "models",
        "tests",
        "scripts",
        "status",
    ]
    for d in required_dirs:
        assert (repo_root / d).is_dir(), f"Required directory missing: {d}"


def test_swingrl_subpackages_exist() -> None:
    """src/swingrl/ must have all subpackages with __init__.py."""
    import swingrl
    pkg_dir = Path(swingrl.__file__).parent  # type: ignore[arg-type]
    required_subpkgs = [
        "data", "envs", "agents", "training",
        "execution", "monitoring", "config", "utils",
    ]
    for subpkg in required_subpkgs:
        subpkg_dir = pkg_dir / subpkg
        assert subpkg_dir.is_dir(), f"Subpackage directory missing: src/swingrl/{subpkg}/"
        assert (subpkg_dir / "__init__.py").exists(), (
            f"__init__.py missing: src/swingrl/{subpkg}/__init__.py"
        )
```

### docker-compose.yml

```yaml
# Source: CONTEXT.md integration points + Docker Compose v3 spec
services:
  swingrl:
    build: .
    mem_limit: 2.5g
    cpus: 1.0
    env_file: .env
    environment:
      - TZ=America/New_York
    volumes:
      - ./data:/app/data
      - ./db:/app/db
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pip + requirements.txt | uv + uv.lock | 2024 | Single binary, 10-100x faster, cross-platform lockfile |
| poetry for ML projects | uv with source groups | 2024-2025 | uv natively handles PyPI + custom indexes in one lockfile |
| flake8 + isort | ruff (E, F, I rules) | 2023-2024 | Single tool, same rules, 100x faster |
| gym (OpenAI) | gymnasium (Farama) | SB3 v2.0 (2023) | SB3 2.x requires gymnasium; gym is deprecated |
| `pip install torch` (full) | CPU index URL for Docker | Ongoing | Saves ~2.3 GB per image; CUDA not needed for CPU homelab |
| `COPY . /app && RUN pip install` | Two-stage uv sync (deps then project) | 2024 | Docker cache hits on every code-only change |

**Deprecated/outdated:**
- `alpaca_trade_api`: Deprecated — use `alpaca-py` instead. FinRL's requirements.txt still lists it; install `alpaca-py` separately and let FinRL use it via compatibility.
- `gym` (OpenAI Gym): Replaced by `gymnasium`. shimmy package provides compatibility shim if needed.
- `setup.py` / `setup.cfg`: Replaced by `pyproject.toml` with hatchling/flit/setuptools as build backends.
- `pip install .[dev]` with extras for dev deps: Replaced by `uv sync --group dev` with `[dependency-groups]`.

---

## Open Questions

1. **FinRL 0.3.7 actual dependency resolution**
   - What we know: Last PyPI release April 2024; lists `stable-baselines3>=2.0.0a5`, `gymnasium`, `alpaca_trade_api`, `elegantrl`
   - What's unclear: Whether `elegantrl` and `alpaca_trade_api` cause resolver conflicts in practice with modern versions of other deps
   - Recommendation: Run `uv add finrl==0.3.7` in a fresh venv as Wave 0 task; if conflicts, use git pin to latest master. Mark as first task in Wave 1 to unblock all others.

2. **uv.lock multi-platform generation on M1 Mac only**
   - What we know: `[tool.uv.sources]` with Linux markers should generate multi-platform lockfile
   - What's unclear: Whether `uv lock` on macOS alone produces a valid lockfile that resolves the Linux CPU-torch path without ever running on Linux
   - Recommendation: After configuring sources, run `uv lock --python-platform linux` or verify Docker build succeeds. If lockfile is incomplete, run once on homelab to regenerate.

3. **FinRL PyPI vs. git pin decision**
   - What we know: FinRL 0.3.7 is last PyPI release (April 2024); git master has more recent commits
   - What's unclear: Git master stability vs. PyPI stability for dependency resolution
   - Recommendation: Start with PyPI `==0.3.7`. Only switch to git pin if resolver conflicts cannot be resolved with overrides. Note that for Phase 1, FinRL doesn't need to function — it just needs to import cleanly.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `uv run pytest tests/test_smoke.py -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-01 | Python 3.11 + MPS available | unit/smoke | `pytest tests/test_smoke.py::test_python_version tests/test_smoke.py::test_torch_mps_available_on_mac -v` | ❌ Wave 0 |
| ENV-02 | uv installed on both machines | manual | `uv --version` on M1 Mac + homelab | Manual verify |
| ENV-03 | Canonical directory structure exists | smoke | `pytest tests/test_smoke.py::test_directory_structure -v` | ❌ Wave 0 |
| ENV-04 | pyproject.toml + uv.lock + tool config | smoke | `uv sync --locked` (fails if broken) | ❌ Wave 0 |
| ENV-05 | pre-commit passes all hooks | integration | `pre-commit run --all-files` | ❌ Wave 0 |
| ENV-06 | Dockerfile builds + CPU torch works | integration | `docker build . && docker run --rm swingrl python -c "import torch; print(torch.__version__)"` | ❌ Wave 0 |
| ENV-07 | docker-compose.yml resource limits | integration | `docker compose config` (validates) + `docker compose up -d` | ❌ Wave 0 |
| ENV-08 | ci-homelab.sh runs end-to-end | e2e | `ssh homelab "bash ~/swingrl/ci-homelab.sh"` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_smoke.py -v`
- **Per wave merge:** `uv run pytest tests/ -v && pre-commit run --all-files`
- **Phase gate:** Full suite green + `ssh homelab "bash ~/swingrl/ci-homelab.sh"` before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_smoke.py` — covers ENV-01, ENV-03 (import smoke + directory structure)
- [ ] `tests/conftest.py` — shared fixtures (minimal for Phase 1; can be empty with just a comment)
- [ ] Framework install: `uv add --group dev pytest>=8.0` — if no existing setup
- [ ] `.secrets.baseline` — must be generated before pre-commit hooks work: `uv run detect-secrets scan > .secrets.baseline`

---

## Sources

### Primary (HIGH confidence)
- [uv PyTorch integration docs](https://docs.astral.sh/uv/guides/integration/pytorch/) — platform-specific source configuration
- [uv Docker integration docs](https://docs.astral.sh/uv/guides/integration/docker/) — Dockerfile patterns, two-stage sync
- [uv dependency management docs](https://docs.astral.sh/uv/concepts/projects/dependencies/) — dependency groups, markers
- [bandit configuration docs](https://bandit.readthedocs.io/en/latest/config.html) — pyproject.toml config, assert_used skips
- [detect-secrets GitHub](https://github.com/Yelp/detect-secrets) — baseline workflow, pre-commit args
- [mypy configuration docs](https://mypy.readthedocs.io/en/stable/config_file.html) — overrides, ignore_missing_imports
- [ruff configuration docs](https://docs.astral.sh/ruff/configuration/) — select/ignore rules
- [FinRL PyPI](https://pypi.org/project/FinRL/) — version 0.3.7, last release April 2024
- [FinRL GitHub requirements.txt](https://github.com/AI4Finance-Foundation/FinRL/blob/master/requirements.txt) — actual dependencies

### Secondary (MEDIUM confidence)
- [SB3 installation docs](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) — Python 3.10+ requirement, gymnasium 0.29.1 minimum
- [PyTorch CPU wheels index](https://download.pytorch.org/whl/cpu) — CPU-only wheel availability for Python 3.11
- [pre-commit hooks guide 2025](https://gatlenculp.medium.com/effortless-code-quality-the-ultimate-pre-commit-hooks-guide-for-2025-57ca501d9835) — modern hook selection

### Tertiary (LOW confidence)
- WebSearch results on FinRL Python 3.11 compatibility — not directly verified; treat as hypothesis until `uv sync` is run

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions verified against official docs and PyPI
- Architecture: HIGH — pyproject.toml/uv/Docker patterns from official uv documentation
- Pitfalls: HIGH — detect-secrets baseline and bandit[toml] pitfalls verified against official issue trackers
- FinRL compatibility: MEDIUM — last release April 2024, actual resolver behavior untested

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable toolchain; uv and ruff versions change fast — re-verify rev: pins)
