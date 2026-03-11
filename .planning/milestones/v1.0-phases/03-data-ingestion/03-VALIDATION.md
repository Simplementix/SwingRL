---
phase: 3
slug: data-ingestion
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.0 (already installed) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/data/ -v -m "not integration" -x` |
| **Full suite command** | `uv run pytest tests/ -v -m "not integration"` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/data/ -v -m "not integration" -x`
- **After every plan wave:** Run `uv run pytest tests/ -v -m "not integration"`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | DATA-01 | unit | `uv run pytest tests/data/test_alpaca_ingestor.py -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 0 | DATA-02 | unit | `uv run pytest tests/data/test_binance_ingestor.py -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 0 | DATA-04 | unit | `uv run pytest tests/data/test_fred_ingestor.py -x` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 0 | DATA-05 | unit | `uv run pytest tests/data/test_validation.py -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | DATA-01 | unit | `uv run pytest tests/data/test_alpaca_ingestor.py -x` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 1 | DATA-02 | unit | `uv run pytest tests/data/test_binance_ingestor.py -x` | ❌ W0 | ⬜ pending |
| 03-03-02 | 03 | 1 | DATA-03 | unit | `uv run pytest tests/data/test_binance_ingestor.py::test_archive_parse_ms -x` | ❌ W0 | ⬜ pending |
| 03-04-01 | 04 | 1 | DATA-04 | unit | `uv run pytest tests/data/test_fred_ingestor.py -x` | ❌ W0 | ⬜ pending |
| 03-05-01 | 05 | 2 | DATA-05 | unit | `uv run pytest tests/data/test_validation.py -x` | ❌ W0 | ⬜ pending |
| 03-05-02 | 05 | 2 | DATA-03 | unit | `uv run pytest tests/data/test_binance_ingestor.py::test_stitch_validation -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/data/__init__.py` — package init
- [ ] `tests/data/test_alpaca_ingestor.py` — stubs for DATA-01
- [ ] `tests/data/test_binance_ingestor.py` — stubs for DATA-02, DATA-03
- [ ] `tests/data/test_fred_ingestor.py` — stubs for DATA-04
- [ ] `tests/data/test_validation.py` — stubs for DATA-05
- [ ] `tests/data/test_parquet_store.py` — stubs for upsert logic
- [ ] `tests/data/fixtures/alpaca_bars_spy.json` — Alpaca API mock response
- [ ] `tests/data/fixtures/binance_klines_btcusdt.json` — Binance live API mock response
- [ ] `tests/data/fixtures/binance_archive_btcusdt_4h.csv` — timestamp unit test data
- [ ] `tests/data/fixtures/fred_cpiaucsl_releases.json` — FRED all-releases mock response
- [ ] Framework install: `uv add "fredapi>=0.5" "exchange_calendars>=4.13" "responses>=0.25"` — not yet in pyproject.toml

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Binance.US 4H archive availability back to 2017 | DATA-03 | Requires live URL check against data.binance.vision | `curl -I https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-2017-08.zip` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
