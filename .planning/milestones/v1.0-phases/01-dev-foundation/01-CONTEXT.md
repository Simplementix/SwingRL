# Phase 1: Dev Foundation - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Reproducible Python 3.11 environment with Docker validated on x86 homelab. Covers ENV-01 through ENV-08: local dev setup (M1 Mac + MPS), uv package manager, GitHub repo with canonical directory structure, pyproject.toml, pre-commit hooks, Dockerfile, docker-compose.yml, and ci-homelab.sh validation script.

</domain>

<decisions>
## Implementation Decisions

### Initial dependency scope
- Include ALL project dependencies upfront in pyproject.toml (torch, FinRL, stable-baselines3, pandas_ta, hmmlearn, alpaca-py, etc.) — catches version conflicts early
- Use compatible version ranges (e.g., torch>=2.2,<2.3) with uv.lock for exact resolution
- uv.lock committed to repo for reproducibility

### Package structure
- Full scaffold from day one: src/swingrl/{data/, envs/, agents/, training/, execution/, monitoring/, config/, utils/} with __init__.py
- Top-level project directories at repo root: config/, data/, db/, models/, tests/, scripts/, status/ — src/swingrl/ contains only Python source code
- tests/ mirrors src/swingrl/ subpackages: tests/data/, tests/envs/, tests/agents/, etc.
- py.typed marker included from day one — all code must have type annotations

### Linter and formatter configuration
- Black line length: 100 characters (accommodates ML code with long parameter lists)
- bandit and detect-secrets are BLOCKING in pre-commit — security findings fail the commit
- .bandit config to exclude test files from security scanning

### CI test baseline
- ci-homelab.sh runs full quality gate: Docker build + pytest + ruff/black/mypy checks inside container
- Docker cache used by default for fast builds; --no-cache available as flag for clean builds
- Phase 1 tests: import smoke test (torch, package versions) PLUS directory structure validation (all expected dirs, __init__.py files, py.typed marker)

### Claude's Discretion
- PyTorch platform handling (separate dependency groups vs. single install with platform detection at Docker build time)
- FinRL installation source (PyPI release vs. Git commit pin) — evaluate current package state
- mypy strictness level — account for third-party stub availability (FinRL, pandas_ta, etc.)
- ruff rule set selection — balance coverage with noise from RL/ML code patterns
- ci-homelab.sh cleanup strategy — balance disk usage with build speed on homelab

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Specs are detailed in Docs 01-15 (especially Doc 05 for architecture/hardware).

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield project (only .git and .planning exist)

### Established Patterns
- None yet — this phase establishes all foundational patterns

### Integration Points
- ci-homelab.sh connects to homelab via `ssh homelab` (must be preconfigured)
- Docker compose uses env_file for secrets, TZ=America/New_York
- Dockerfile: python:3.11-slim, CPU-only PyTorch, non-root trader user (UID 1000)
- docker-compose.yml: 2.5g memory limit, 1 CPU, bind mounts for data/db/models/logs

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-dev-foundation*
*Context gathered: 2026-03-06*
