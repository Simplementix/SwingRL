---
phase: 5
slug: feature-engineering
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/features/ -x -v` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/features/ -x -v`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 0 | FEAT-01 | unit | `uv run pytest tests/features/test_technical.py -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-02 | unit | `uv run pytest tests/features/test_technical.py::test_derived_features -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-03 | unit | `uv run pytest tests/features/test_fundamentals.py -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-04 | unit+integration | `uv run pytest tests/features/test_macro.py -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-05 | unit | `uv run pytest tests/features/test_hmm_regime.py -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-06 | unit | `uv run pytest tests/features/test_normalization.py -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-07 | unit | `uv run pytest tests/features/test_assembler.py -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-08 | unit | `uv run pytest tests/features/test_pipeline.py::test_feature_ab -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-09 | unit | `uv run pytest tests/features/test_correlation.py -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-10 | unit | `uv run pytest tests/features/test_technical.py::test_weekly_features -x` | ❌ W0 | ⬜ pending |
| TBD | 01 | 0 | FEAT-11 | integration | `uv run pytest tests/features/test_pipeline.py::test_duckdb_storage -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/features/__init__.py` — package init
- [ ] `tests/features/conftest.py` — shared fixtures (synthetic OHLCV, mock DuckDB)
- [ ] `tests/features/test_technical.py` — stubs for FEAT-01, FEAT-02, FEAT-10
- [ ] `tests/features/test_fundamentals.py` — stubs for FEAT-03
- [ ] `tests/features/test_hmm_regime.py` — stubs for FEAT-05
- [ ] `tests/features/test_macro.py` — stubs for FEAT-04
- [ ] `tests/features/test_normalization.py` — stubs for FEAT-06
- [ ] `tests/features/test_correlation.py` — stubs for FEAT-09
- [ ] `tests/features/test_turbulence.py` — stubs for turbulence calculator
- [ ] `tests/features/test_assembler.py` — stubs for FEAT-07
- [ ] `tests/features/test_pipeline.py` — stubs for FEAT-08, FEAT-11
- [ ] `uv add alpha-vantage` — new dependency

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| yfinance ETF data availability | FEAT-03 | Live API call to yfinance needed | Run `scripts/compute_features.py --check-fundamentals` and verify all 8 ETFs return expected fields |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
