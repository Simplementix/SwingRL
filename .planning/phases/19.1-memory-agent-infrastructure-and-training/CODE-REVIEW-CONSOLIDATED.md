# Phase 19.1 — Consolidated Code Review Findings

**Date:** 2026-03-18 (initial), 2026-03-19 (post-fix review)
**Branch:** `gsd/phase-19.1-memory-agent-infrastructure-and-training`
**Scope:** ~60K LOC, 92 Python files across 16 packages + memory service

---

## Quality Gate (Post-Fix)

| Check | Result |
|-------|--------|
| Tests | **1187 passed**, 0 failed |
| Ruff | **0 errors** |
| Mypy | **0 errors** (93 source files) |
| All 31 approved fixes | **Verified on disk** |

---

## Part A — Original Review (41 items, 2026-03-18)

### Disposition Summary

| ID | Severity | Description | Disposition |
|----|----------|-------------|-------------|
| 1 | CRITICAL | Train/live observation layout mismatch | **FIXED** (Wave 4) |
| 2 | CRITICAL | Expand to 4 per-asset features | **FIXED** (Wave 4) |
| 3 | CRITICAL | Live pipeline should use weight-based rebalancing | **FIXED** (Wave 5A) |
| 4 | CRITICAL | Pipeline cash calculation wrong | **RESOLVED** by #3 redesign |
| 5 | CRITICAL | Global CB uses initial capital instead of HWM | **FIXED** (Wave 1A) |
| 6 | CRITICAL | DuckDB persistent connection → short-lived | **FIXED** (Wave 6A) |
| 7 | CRITICAL | Missing auth header in meta-orchestrator | **FIXED** (Wave 1D) |
| 8 | CRITICAL | Blocking sync SQLite in async handlers | **FIXED** (Wave 6B) |
| 9 | HIGH | Agent cannot hold cash (softmax 100% invested) | **FIXED** (Wave 4) |
| 10 | HIGH | VecNormalize shared across algos in live | **FIXED** (Wave 5A) |
| 11 | HIGH | Shadow promotion overhaul | **FIXED** (Wave 5B) |
| 12 | HIGH | Shadow runner hardcoded prices | **FIXED** (Wave 5B) |
| 13 | HIGH | Alpaca fill assumes immediate fill | **FIXED** (Wave 1A) |
| 14 | HIGH | EST hardcoded in emergency stop | **FIXED** (Wave 1A) |
| 15 | HIGH | Equity env always starts at step 0 | **FIXED** (Wave 1D) |
| 16 | HIGH | BinanceIngestor/GapFill sessions never closed | **FIXED** (Wave 2A) |
| 17 | HIGH | Execution pipeline creates new adapter per cycle | **FIXED** (Wave 1A) |
| 18 | HIGH | Bare ValueError raises (17 occurrences) | **FIXED** (Wave 2B) |
| 19 | HIGH | retry(reraise=False) wraps exceptions | **FIXED** (Wave 1C) |
| 20 | HIGH | Duplicated bounds constants | **FIXED** (Wave 6B) |
| 21 | HIGH | IngestAgent XML not sanitized | **FIXED** (Wave 1C) |
| 22 | HIGH | stop_training signal ignored | **FIXED** (Wave 1C) |
| 23 | MEDIUM | Turbulence loads entire OHLCV table | **FIXED** (pre-review, Phase 19.1) |
| 24 | MEDIUM | Macro query dead if/else branches | **FIXED** (Wave 1D) |
| 25 | MEDIUM | Data pipeline overhaul | **DEFERRED** to Wave 7 / Phase 21 |
| 26 | MEDIUM | O(n²) turbulence compute_series | **FIXED** (Wave 2C) |
| 27 | MEDIUM | Sequential symbol ingestion | **SKIPPED** (not worth complexity) |
| 28 | MEDIUM | New requests.Session per gap-fill call | **RESOLVED** by #16 |
| 29 | MEDIUM | Feature pipeline reads SPY/BTC twice | **FIXED** (Wave 2C) |
| 30 | MEDIUM | Transaction costs not pre-deducted | **FIXED** (Wave 2C) |
| 31 | MEDIUM | Calmar ratio divides by cum[0] | **FIXED** (Wave 1B) |
| 31b | MEDIUM | Sortino downside deviation wrong denominator | **FIXED** (Wave 1B) |
| 31c | MEDIUM | Reward wrapper rolling Sharpe ddof=0 | **FIXED** (Wave 1B) |
| 32 | MEDIUM | Turbulence 90th pct uses ATR | **FIXED** (pre-review, Phase 19.1) |
| 33 | MEDIUM | CB event query counts all-time | **RESOLVED** by #11 |
| 34 | MEDIUM | EnsembleBlender dead code | **FIXED** (Wave 1B) |
| 35 | MEDIUM | PositionSizer static Kelly defaults | **RESOLVED** by #3 + #11 |
| 36 | MEDIUM | FeaturePipeline silently returns zeros | **FIXED** (Wave 3) |
| 37 | LOW | Dashboard port exposed | **SKIPPED** (behind firewall) |
| 38 | LOW | Data ingestors don't validate API keys | **FIXED** (Wave 1C) |
| 39 | LOW | API key fields use str not SecretStr | **SKIPPED** (not worth friction) |
| 40 | MEDIUM | Expanding equity covariance dilutes regime | **FIXED** (pre-review, Phase 19.1) |
| 41 | MEDIUM | 2-asset crypto Mahalanobis overkill | **FIXED** (pre-review, Phase 19.1) |

**Totals:** 31 fixed, 4 resolved by other fixes, 3 skipped, 2 pre-resolved, 1 deferred.

---

## Part B — Post-Fix Review Findings (2026-03-19)

### Issues Found and Fixed During Review

| ID | File | Problem | Fix |
|----|------|---------|-----|
| R1 | `execution/pipeline.py:185-189` | `_normalize_observation()` called before `_load_models()` — all algos got raw (unnormalized) obs, defeating fix #10 | Swapped order: models load first |
| R2 | `envs/portfolio.py:process_actions()` | Deadzone preserves old weights that can sum > 1.0, driving cash negative | Post-deadzone clamp: `if total > 1.0: target_weights *= 1/total` |
| R3 | `execution/risk/position_tracker.py:237` | weight_deviation always computed `weight - 1/n_assets` even at zero exposure; training returns 0.0 | Added `if exposure > 0` guard to match training |
| R4 | `execution/pipeline.py:223` | Zero portfolio value not guarded — delta orders proceed with all-zero values | Early return `if portfolio_value <= 0` |

### Remaining Findings — Not Blocking Merge

#### Finding 1: `bars_since_trade` vs `days_since_trade` semantic mismatch
- **Severity:** Important (fix before paper trading)
- **Files:** `envs/base.py:315` vs `execution/risk/position_tracker.py:246`
- **Problem:** Training env counts bars (steps) since trade. Live PositionTracker counts calendar days. For 4H crypto, 1 bar ≠ 1 day (6 bars/day). Equity is fine (1 bar = 1 day).
- **Impact:** Crypto model sees different magnitude of "time since trade" in training vs live. Magnitude difference ~6x.
- **Decision:** Use bars (timesteps) everywhere + `log(1 + bars)` normalization. RL literature universally uses timesteps (FinRL, Jiang et al. 2017). Calendar days vs bars is a distribution shift, not just rescaling. Live `_days_since_trade()` → `_bars_since_trade()`: equity = trading days, crypto = `hours_elapsed / 4`. Both sides apply `np.log1p()`. Implement in Phase 20.
- **Tracked:** Added to Phase 20 context (`20-CONTEXT.md` deferred section).

#### Finding 2: Shadow promoter compares different return types
- **Severity:** Important (fix before paper trading)
- **Files:** `shadow/promoter.py`
- **Problem:** Active model returns computed from portfolio_snapshots (period returns — daily or 4H). Shadow model returns computed from paired buy/sell trades (variable holding period returns). Annualized Sharpe comparison between these distributions is approximate.
- **Impact:** Shadow promotion decisions are directionally correct but not precisely comparable. Shadow must be "much better" to promote, so the approximation is conservative.
- **Decision:** Implement shadow NAV (virtual portfolio) tracking in Phase 20. Shadow runner will maintain a virtual portfolio (positions, cash, total value) and write `portfolio_snapshots` rows with `source='shadow'` at each cycle close using real market prices. Promoter then uses the same `_returns_from_portfolio_snapshots()` for both — eliminating the distribution mismatch entirely. This is the hedge fund "shadow book" standard (Ledoit-Wolf 2008, Bailey & Lopez de Prado 2012 PSR). Additionally consider: Minimum Track Record Length (PSR-derived) to prevent premature promotion, and paired statistical test on return differences instead of naive `shadow_sharpe > active_sharpe`.
- **Tracked:** Added to Phase 20 context.

#### Finding 3: `verification.py` and `deploy_model.sh` hardcode obs dims
- **Severity:** Important (fix before paper trading)
- **Files:** `data/verification.py:39-40`, `scripts/deploy_model.sh:23-24`
- **Problem:** Both hardcode `164`/`47` instead of importing `EQUITY_OBS_DIM`/`CRYPTO_OBS_DIM` from `features/assembler.py`. Violates project convention "Never hardcode 164 or 180."
- **Impact:** Low — dims rarely change. But will silently produce wrong verification if sentiment is enabled.
- **Decision:** FIXED in this phase. `verification.py` imports from `assembler.py`. `deploy_model.sh` queries Python at runtime with hardcoded fallback.

#### Finding 4: Emergency `_tier2_liquidate_crypto` is a stub
- **Severity:** Important (fix before live trading)
- **Files:** `execution/emergency.py:186-194`
- **Problem:** Logs crypto positions but never submits sell orders. Pre-existing — not introduced by these fixes.
- **Impact:** Emergency stop cannot liquidate crypto positions. Equity tier2 works via Alpaca API.
- **Decision:** Deferred to Phase 20 (production deployment). Requires Binance adapter wired for real order submission, which is Phase 20 scope. Added to Phase 20 context as a hard requirement — must be functional before live trading.

#### Finding 5: HMM not fitted if SPY/BTCUSDT absent from symbol list
- **Severity:** Important (low practical risk)
- **Files:** `features/pipeline.py`, `config/schema.py`
- **Problem:** HMM fitting was hardcoded to `symbol == "SPY"` / `symbol == "BTCUSDT"`. If config omitted these, HMM silently returned [0.5, 0.5] defaults.
- **Decision:** FIXED in this phase. Added `hmm_proxy_symbol` config field to `EquityConfig` (default "SPY") and `CryptoConfig` (default "BTCUSDT"). Pipeline now reads from config. Pydantic validator ensures proxy symbol is present in the symbols list.

#### Finding 6: Shadow runner price estimation and execution price gaps
- **Severity:** Important (elevated from Minor after research)
- **Files:** `shadow/shadow_runner.py`, `execution/pipeline.py`
- **Problem:** (A) Shadow runner uses hardcoded $100/$50K fallback prices with zero slippage/commission when DuckDB lookup fails — no warning logged. (B) Broader issue: no price deviation gate between observation price (last bar close) and execution price (real-time market). A 10% overnight gap or crypto flash crash means the model acts on stale observations.
- **Decision:** Fix in Phase 20 (production deployment). Three items:
  1. **Price deviation gate** (HIGH): Before order submission, compare real-time price to observation price. Skip/reduce trade if deviation exceeds threshold (5% equity, 10% crypto). Config-driven values.
  2. **Shadow fill realism** (MEDIUM): Add slippage estimates (5 bps equity, 10+10 bps crypto). Remove hardcoded fallback prices — skip shadow trade if price unavailable, log warning.
  3. **Training look-ahead bias** (LOW/FUTURE): Training uses same-bar close execution (industry standard shortcut). Consider shifting to "execute at bar t, evaluate at t+1 open" in a future phase. Not urgent — reward already uses next-bar prices.

#### Finding 7: Epoch callback and meta-orchestrator bypass MemoryClient for HTTP calls
- **Severity:** Minor
- **Files:** `memory/training/epoch_callback.py:89-90,338-341`, `memory/training/meta_orchestrator.py:241-242`
- **Problem:** Both files access private `_base_url` and `_api_key` attrs (with `# noqa: SLF001`) and build raw `urllib` requests — bypassing MemoryClient, duplicating HTTP logic, and creating fragile coupling to private attr names.
- **Decision:** Deferred to Phase 20. Add `epoch_advice()` and `query_run_config()` methods to MemoryClient (Option B). Callback and meta-orchestrator call client methods instead of raw HTTP. Eliminates private attr access, removes `# noqa` suppressions, and centralizes all HTTP logic in one place. Natural fit alongside the 5 new `/live/*` endpoint methods being added to MemoryClient in Phase 20.

#### Finding 8: Env `from_arrays` docstrings have stale dimensions
- **Severity:** Minor
- **Files:** `envs/equity.py:30,71`, `envs/crypto.py:31,79`
- **Problem:** Docstrings said 156 (equity) and 45 (crypto) after obs space expansion to 164/47.
- **Decision:** FIXED in this phase. Updated both equity and crypto docstrings.

---

## Part C — Pre-Review Fixes (already applied before this review)

These items from CODE-REVIEW-FINDINGS.md (the first 19.1 review) were fixed in earlier commits:

| ID | Description | Commit |
|----|-------------|--------|
| C1 | SubprocVecEnv never closed | `c66a44f` |
| C2 | DuckDB connections unused in tuning | `c66a44f` |
| C3 | Module-level config loading crashes | `c66a44f` |
| C4 | Memory Dockerfile no .dockerignore | `c66a44f` |
| C5 | Dockerfile layer ordering | `c66a44f` |
| I1 | Tuning round 1 wrong type to gate | `f77ac5f` |
| I2 | decide_final_timesteps always ESCALATED | `f77ac5f` |
| I3 | _train_final_algo missing api_key | `e990f5a` |
| I4 | epoch_callback missing API key header | `e990f5a` |
| I5 | rolling_sharpe hardcodes 252 | `f77ac5f` |
| I6 | epoch_advice overwrites pending | `f77ac5f` |
| I7 | archive_memories gated on created > 0 | `e990f5a` |
| I8 | No request body size limit | `e990f5a` |
| I10 | insert_consolidation_source loop | `e990f5a` |
| I11 | Dead merge/conflict prompts | `e990f5a` |
| I13 | from_date_ranges end_bar | `f77ac5f` |
| I14 | Memory service port bound to localhost | `e990f5a` |
| I15 | Memory service deps pinned | `966d693` |
| M1-M14 | Minor cleanup batch | `8623cf7` |

---

## Part D — Deferred Items (not in scope for 19.1)

| ID | Description | Deferred To |
|----|-------------|-------------|
| 25 | Data pipeline overhaul (Tiingo, raw prices, gap detection) | Wave 7 / Phase 21 |
| I9 | httpx connection pooling for LLM calls | Phase 20+ |
| I12 | Conflict detection heuristic (LLM-based) | Future |
| I16 | PathsConfig str→Path | Future refactor |

---

## Strengths Noted

1. **Train/live parity** — Interleaved 4-field observation layout + weight-based rebalancing via identical `process_actions()` in both training and live.
2. **Defense-in-depth bounds** — Hyperparameter bounds enforced at 3 layers (Pydantic, server-side, client-side).
3. **Fail-open design** — Memory service, epoch callback, meta-orchestrator all degrade gracefully.
4. **Shadow promoter overhaul** — Portfolio-snapshot-based annualized Sharpe with profit factor gate.
5. **FeatureHealthTracker** — Threshold-based blocking with staleness detection and auto-recovery.
6. **DuckDB short-lived connections** — Eliminates lock contention; `read_only` param ready for Phase 22.
7. **Comprehensive tests** — 1187 tests including 12 new error-path tests and 9 health tracker tests.
