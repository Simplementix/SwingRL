# Execution Alignment & Reporting Fixes — Design

**Date:** 2026-07-21 · **Status:** MERGED & DEPLOYED — PR #37 (execution-alignment); trader swingrl:trader-2026-07-22-1 (2026-07-22). · **Scope:** live/paper
execution side (era-0 compatible) + era-1 training-convention decision (recorded in the
training-redesign spec/plan, implemented in Plan B Phase 7).

## Glossary

| Term | Meaning |
|---|---|
| t−1 / t | Trading-day indices: t = today, t−1 = the previous trading day |
| Completed bar | An OHLCV candle whose time window has fully closed |
| Fill convention | Which price a trade executes at, relative to the last bar the agent observed |
| Deadzone cycle | A cycle where every weight delta is too small to order — nothing trades |
| Mark-to-market (MTM) | Valuing held positions at current prices rather than entry prices |
| Era | A generation of models sharing one observation layout + env convention; conventions change only at era boundaries |
| Decision snapshot | The options-chain capture taken at the equity decision time (for future training use) |
| MOC | Market-on-close (not used; listed only because it came up in review) |

## Problem statement (all verified live 2026-07-21)

1. **Equity candles land a day late.** The `candles_equity` job runs 16:50 ET but the
   Alpaca fetch window is pinned to 00:00 UTC (partial-bar immunity), so day-D bars are
   only fetchable after ~20:00 ET. Result: each run ingests day D−1, and the 15:45 ET
   cycle trades on bars through **t−2**.
2. **Equity fill convention diverges from training.** Training fills at the close of the
   last observed bar (`envs/base.py:187-195`). Live at 15:45 fills a **full session**
   after the last observed bar's close. Crypto (bar close → trade at :05) is aligned.
3. **Daily Summary reports fabricated zeros.** Trade counts are hardcoded 0
   (`scheduler/jobs.py:317-318`); portfolio value and P&L freeze on deadzone cycles
   because prices are only fetched for symbols that generate orders
   (`pipeline.py:348-360` skip precedes the fetch) — live snapshots have shown a
   bit-identical `total_value` and `daily_pnl=0` since 07-19.
4. **Deadzone cycles evade risk marks.** The frozen valuation above also means a market
   crash during a deadzone streak does not move recorded drawdown (drill finding #1
   adjacency; capital-preservation relevant).
5. **Sell fills carry no realized P&L** in the Discord trade embed (user request 07-21).

## Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | `candle_jobs.equity_time_et`: 16:50 → **20:15 ET** | Past 00:00 UTC year-round (EDT and EST); day-D bar lands day-D evening; pinning (partial-bar immunity) untouched |
| D2 | `equity.cycle_time_et`: 15:45 → **09:15 ET (pre-open)** | Decide on t−1 bars, fill AT t's open via D11's pre-open submission. Revised 2026-07-21 from the earlier 09:40 draft after the user ruled MOO-in-scope-now |
| D3 | Options **decision snapshot**: market 15:45→**09:30**, pull 16:00→**09:46** | Options quotes don't exist pre-open, so the snapshot targets the fill moment (the 09:30 auction), not the 09:15 decision; 16-min trail preserves the T6-measured 15m27s content-delay margin. **eod snapshot unchanged** |
| D4 | Ops quiet window becomes **two windows**: 09:00–10:15 and 16:00–16:45 ET | Morning: cycle 09:15 + auction fills + 09:46 decision pull + ~09:35 fill-confirmation; afternoon: eod snapshot chain remains |
| D5 | Digest trade counts = live SQL count of day's `trade_type='signal'` trades per env (ET day) | Excludes reconciliation `adjustment` rows; kills the hardcoded zeros |
| D6 | Every non-dry-run cycle marks ALL held positions to market at snapshot time — fetch prices for held symbols missing from `cycle_prices`, update `positions.last_price`, fail-open per symbol | Fixes frozen value/P&L; keeps "no extra broker calls" only when a symbol was already priced this cycle |
| D7 | Zero-order cycles run the breaker check post-snapshot with the fresh marks | Closes the deadzone risk-evasion gap (finding #1) using the same `check_and_update` the trade path uses |
| D8 | Sell fills compute realized P&L ((fill − cost basis) × qty − commission) in `fill_processor`, surfaced on `FillResult.realized_pnl`, shown in sell embeds | User-requested close-visibility |
| D9 | **Era-1 env convention: decide on bars through t, fill at open(t+1), both envs** (recorded in training-redesign spec §2.5 A28(d) + plan Task 25b; implemented in Plan B Phase 7, config-gated, default off) | Removes the fill-at-observed-close idealization; overnight gap becomes part of training. Era-0 models are never evaluated under the new convention (era-boundary rule) |
| D10 | **Risk-sweep job** (user-approved 2026-07-21): every `risk.sweep_interval_minutes` (default 30), mark all held positions to market (D6 machinery) + run breaker evaluation (D7 machinery), **no trading** | Shrinks the between-cycle risk blind window (equity ~24h, crypto 4h → sweep interval) with one generic code path instead of per-env special cases |
| D11 | **Opening-auction execution NOW (user ruling 2026-07-21 — era-1 deploys with zero trader changes):** the 09:15 cycle submits plain **DAY market orders before 09:28 ET** (buys as `notional` dollar amounts = the pipeline's `delta_value`; sells as `qty`), which Alpaca fills at the primary exchange's official opening print; a **fill-confirmation pass ~09:35** records fills, capture rows, and embeds. **OPG TIF is NOT used** — verified against Alpaca docs 2026-07-21: OPG is sales-enablement-gated for API accounts and incompatible with fractional/notional (DAY-only), which the $400 allocation requires (SPY > whole allocation). Pre-submission risk checks and sizing use t−1 close marks | Live equity fills at open(t) — the exact era-1 training convention (D9) — with no OPG dependency. Alpaca doc basis: "Any market orders received before 9:28 will be filled at the [official opening price]"; fractional/notional = DAY orders only |
| D12 | **Cycle-orders INFO ping, BOTH envs** (user-approved 2026-07-21; extended to crypto + explicit sells 2026-07-21): after EVERY cycle, one INFO alert "Cycle orders submitted — {env}" listing each order incl. sells — buys as "BUY SPY $25.00" (notional), sells as "SELL QQQ 0.0416 (~$25.00)" (qty + approx value) — or "no orders — deadzone held". Equity: fires at 09:15 submission (fills confirm 09:35); crypto: fires at cycle end (fills are synchronous, embeds follow as today). ~7 INFO/day total (1 equity + 6 crypto) — full-parity precedent | Heartbeat for every cycle: the user knows it ran and exactly what was placed, buys AND sells, without waiting for fills |
| D13 | **Buy-and-hold benchmark** (user request 2026-07-21): at paper-epoch reset, record per-env baselines (equal-weight across the env's symbols, same capital, epoch start date/prices) in a `benchmark_baselines` table; the daily digest gains per-env "Buy & Hold value" + "Agent vs B&H" delta, computed from baselines × latest stored candles | Answers "is the RL agent beating doing nothing?" continuously; deterministic (recomputable from candles), no phantom orders |
| D14 | **Paper epoch reset** (user-approved 2026-07-21, gated operational step BEFORE this plan deploys): close crypto positions via real sim sells (first live sell-path proof); user resets the Alpaca paper account (clears the phantom ETHUSD at the source); boot reconciliation trues the DB; verify 0 open positions + sane obs (`cash_ratio`, `exposure`); record D13 baselines; then deploy and collect fresh ≥2-cycles/env runbook evidence | Clean slate under the new rules WITHOUT deleting history — trades/capture rows are append-only evidence and FK-linked; old snapshots leave the HWM slightly conservative (breakers trip early = safe direction) |
| D15 | **Category-colored Discord alerts + formatting** (user request 2026-07-21): central category→(color, emoji) style map — ingests blue 📥, Daily Summary gold 📊, buys green 🟢 / sells red 🔴 (kept distinct), cycle pings + ops heartbeats purple 🔄, warning/critical unchanged; emoji title prefixes, order lists in monospace code blocks, digest ▲/▼ P&L arrows + ✅/❌ vs-B&H. No images (emoji only — hosted-URL maintenance rejected). Implemented as plan Task 10 | Every alert category identifiable at a glance from the phone notification; one style map to retheme |

## Out of scope (recorded, deliberate)

- **Stop-loss wiring into the live order path** (live orders carry no stops; poller inert;
  ATR-stop logic exists only in `PositionSizer`, used solely by shadow runner) and
  **auto-liquidation on stop breach** — deferred to the pre-live hardening pass, per the
  existing `stop_polling.py` risk statement. Tracked, not silently dropped.
- Crypto schedule (already aligned), INFO-on-0-rows cosmetics (final-review list),
  webhook rotation (user ruling: LATER).
- **Execution-venue fallback: REJECTED by user (2026-07-21)** — funds live on one venue;
  a backup execution venue makes no sense. Closed, not deferred.
- **Crypto data fallback via Alpaca** (candles/quotes only, explicit degraded-source
  mode, never silent venue-mixing per review H5) — refinement backlog, not this plan.

## Coupling & process notes

- **Paper-readiness runbook:** equity capture evidence to date was collected on the
  15:45 schedule. After D2 deploys, let **≥2 morning cycles** run and record them as the
  capture evidence before the go/no-go signs off the schedule actually being run.
- **Deploys required** (each needs explicit user approval, after PR merges): collector
  rebuild (D1, D3), trader rebuild (D2, D5–D8, D10, D11 — one unit: the 09:15 cycle
  must not deploy without the D11 async path). Until deployed, old times keep running
  (harmless).
- **Sim-mirror (T16 Step 5)** still needs a real equity fill; the morning schedule makes
  fills at t's open — the mirror comparison should use the same convention.

## Verified vs assumed

- **Verified:** items 1–5 of the problem statement (code + live DB + logs, session
  2026-07-21); training fill convention (`envs/base.py`); config field locations; test
  pin sites (`test_options_config.py:35`, `test_main.py:23`); Alpaca order rules for
  D11 (docs.alpaca.markets/docs/orders-at-alpaca fetched 2026-07-21: pre-9:28 market
  orders protected on the opening print; OPG sales-gated + fractional/notional DAY-only).
- **Assumed (low risk):** overnight feature recompute completes well before 09:15
  (features run within the 20:15 candle job — verify on first morning cycle);
  fractional/notional market orders queued pre-open participate at the opening-print
  price basis (Alpaca fills fractional internally — confirm on first paper morning
  cycle's `fill_quality` rows).

**Confidence:** high on defect mechanics and D1/D5–D8; medium-high on D2–D4 exact times
(the convention argument is verified; the specific minute choices are judgment).
