---
phase: 8
slug: paper-trading-core
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/execution/ -v -x` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/execution/ -v -x`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | PAPER-01 | unit (mock SDK) | `uv run pytest tests/execution/test_alpaca_adapter.py -x` | ❌ W0 | ⬜ pending |
| 08-01-02 | 01 | 1 | PAPER-10 | unit | `uv run pytest tests/execution/test_alpaca_adapter.py::test_bracket -x` | ❌ W0 | ⬜ pending |
| 08-01-03 | 01 | 1 | PAPER-02 | unit (mock requests) | `uv run pytest tests/execution/test_binance_sim.py -x` | ❌ W0 | ⬜ pending |
| 08-02-01 | 02 | 1 | PAPER-03 | unit | `uv run pytest tests/execution/test_risk_manager.py -x` | ❌ W0 | ⬜ pending |
| 08-02-02 | 02 | 1 | PAPER-20 | unit | `uv run pytest tests/execution/test_risk_manager.py::test_turbulence -x` | ❌ W0 | ⬜ pending |
| 08-02-03 | 02 | 1 | PAPER-04 | unit | `uv run pytest tests/execution/test_circuit_breaker.py -x` | ❌ W0 | ⬜ pending |
| 08-02-04 | 02 | 1 | PAPER-05 | unit | `uv run pytest tests/execution/test_circuit_breaker.py::test_cooldown -x` | ❌ W0 | ⬜ pending |
| 08-02-05 | 02 | 1 | PAPER-06 | unit | `uv run pytest tests/execution/test_circuit_breaker.py::test_persistence -x` | ❌ W0 | ⬜ pending |
| 08-03-01 | 03 | 1 | PAPER-07 | unit | `uv run pytest tests/execution/test_position_sizer.py -x` | ❌ W0 | ⬜ pending |
| 08-03-02 | 03 | 1 | PAPER-08 | unit | `uv run pytest tests/execution/test_position_sizer.py::test_crypto_floor -x` | ❌ W0 | ⬜ pending |
| 08-03-03 | 03 | 1 | PAPER-11 | unit | `uv run pytest tests/execution/test_order_validator.py::test_cost_gate -x` | ❌ W0 | ⬜ pending |
| 08-04-01 | 04 | 2 | PAPER-09 | integration | `uv run pytest tests/execution/test_pipeline.py -x` | ❌ W0 | ⬜ pending |
| 08-05-01 | 05 | 2 | PAPER-18 | smoke | `uv run pytest tests/execution/test_seed_production.py -x` | ❌ W0 | ⬜ pending |
| 08-05-02 | 05 | 2 | PAPER-19 | smoke | `docker build --target production -t swingrl:test .` | manual | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/execution/__init__.py` — package init
- [ ] `tests/execution/conftest.py` — shared fixtures (mock broker, mock DB)
- [ ] `tests/execution/test_alpaca_adapter.py` — stubs for PAPER-01, PAPER-10
- [ ] `tests/execution/test_binance_sim.py` — stubs for PAPER-02
- [ ] `tests/execution/test_risk_manager.py` — stubs for PAPER-03, PAPER-20
- [ ] `tests/execution/test_circuit_breaker.py` — stubs for PAPER-04, PAPER-05, PAPER-06
- [ ] `tests/execution/test_position_sizer.py` — stubs for PAPER-07, PAPER-08
- [ ] `tests/execution/test_order_validator.py` — stubs for PAPER-11
- [ ] `tests/execution/test_pipeline.py` — stubs for PAPER-09
- [ ] `tests/execution/test_seed_production.py` — stubs for PAPER-18
- [ ] `tests/execution/test_signal_interpreter.py` — signal interpretation coverage
- [ ] `tests/execution/test_fill_processor.py` — DB recording coverage

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Docker multi-stage build succeeds | PAPER-19 | Requires Docker daemon and full image build | Run `docker build --target production -t swingrl:test .` and verify exit code 0 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
