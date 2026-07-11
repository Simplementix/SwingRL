# Execution/Inference-Path Code Review (2026-07-07/08)

> **Status: COMPLETE — disposition approved by user 2026-07-11.** Findings-only review
> (review ≠ fix); every fix below is a *planned task*, not an applied change.
> **Method:** four parallel read-only review agents (execution pipeline core; adapters +
> fills + risk layer; training-side spec items; change-site register verification), run
> against branch `swingrl/2.R-training-redesign` @ `2d5b6f7`. All file:line references
> were read from code by the agents; nothing here is assumed unless explicitly marked.
> **Why it ran:** user-requested gate before Plan A Tasks 6–16
> (`docs/superpowers/plans/2026-07-07-capture-foundation-plan.md`, Task 0 Step 1);
> priority: "make sure it will work as expected without any surprises."

## Glossary

| Term | Meaning |
|---|---|
| C1/C2, H1–H5, M1–M11, L | Finding labels by severity: Critical / High / Medium / Low |
| CB | Circuit breaker — deterministic tripwire that halts trading (drawdown breach, turbulence spike) |
| F1 | Known bug: live turbulence-halt baseline query reads a nonexistent column → silent 0.0 → halt never fires |
| F1b | Known bug: models trained with the turbulence observation input frozen at 0.0, while live inference feeds real values |
| F2 | Known bug: training-epoch capture volume blowup (millions of rows) — root cause investigated here |
| A3 | Spec amendment: risk-penalty-discarded-under-reward-shaping bug; fix is a precondition of any L1 harness run |
| P-A1…P-A6 | Plan A's assumptions register (each with a written verification method) |
| VecNormalize | Stable Baselines3 wrapper holding per-dimension running mean/variance used to normalize observations |
| Snapshot | A `portfolio_snapshots` row — the recorded portfolio value/cash/P&L after a cycle |
| Phantom trade | A `trades` row recorded with quantity=0 / price=0 from an order that had not actually filled |
| Mark-to-market | Valuing current positions at current prices (as opposed to copying an old value forward) |
| Fire-and-forget | Submitting an order and never checking afterwards whether/how it filled |
| RTH | Regular trading hours (NYSE 09:30–16:00 ET) |
| IEX feed | Alpaca's free-plan market-data feed (single exchange; can diverge from consolidated market prices) |
| Sim / binance_sim | In-process crypto fill simulator (`execution/adapters/binance_sim.py`) — real Binance.US prices, simulated fills |
| Ramp-up | Post-halt staged capacity (25/50/75/100%) a circuit breaker is supposed to enforce while cooling down |
| HWM | High-water mark — peak portfolio value used for drawdown measurement |
| DDL | Database table-definition SQL (`CREATE TABLE …`) |

---

## 1. Critical findings

### C1 — Circular portfolio snapshots: drawdown & daily-loss breakers can never trip

**Defect.** The portfolio value recorded each cycle is the previous snapshot's value copied
forward. `PositionTracker.record_snapshot` (`position_tracker.py:162–196`) is the only
writer of `portfolio_snapshots`; it is fed `get_portfolio_value()` (`position_tracker.py:47–72`),
which just reads the latest snapshot's `total_value` (falling back to initial capital).
Nothing marks positions to market; no fill P&L or commissions ever enter `total_value`;
`daily_pnl` is read back from today's snapshot or defaults 0.0 (`:119–140`).

**Failure scenario.** Crypto book drops 30% → portfolio value on record never moves →
drawdown check (`risk_manager.py:123–143`), daily-loss check (`:146–161`),
`CircuitBreaker.check_and_update` (`circuit_breaker.py:69–107`),
`GlobalCircuitBreaker.check_combined` (`:298–345`) and the VIX+drawdown emergency trigger
(`emergency.py:456–470`) all see ~0% drawdown — no halt, no alert, trading continues.
Paper-trading results derived from snapshots are equally meaningless.

**Consequences for Plan A.** Task 16's circuit-breaker proof would fail today.
**Disposition: new Plan A Task A (real portfolio valuation).**

### C2 — Alpaca fills are fire-and-forget; unfilled submits record phantom trades and corrupt positions

**Defect.** `alpaca_adapter.submit_order` (`alpaca_adapter.py:124–140`) returns
`FillResult(quantity=0.0, fill_price=0.0)` whenever the submit response carries no
`filled_avg_price` — the normal case for asynchronously-filled market orders. There is no
polling, no websocket stream, no later re-query. The pipeline then processes it as a real
fill (`pipeline.py:292–295`): a qty=0/price=0 `trades` row is inserted
(`fill_processor.py:115–143`), the position's `last_price` is set to 0.0 (`:216`/`:258`),
and a Discord trade embed is sent (`jobs.py:113–119`).

**Compounding schedule defect.** The equity cycle cron fires at **16:15 ET — after market
close — seven days a week with no market-calendar/holiday check** (`main.py:73–81`), making
the async-fill case the norm, not the exception.

**Failure scenario.** Position `last_price=0` → `_get_current_weights` (`pipeline.py:566`)
values the holding at $0 → next cycle re-buys the full target weight → duplicated exposure
until the 17:00 ET reconciliation (`jobs.py:431–458`) corrects quantity — recording it as an
"adjustment" at average entry price, losing the real fill price and slippage forever.

**Verified vs assumed.** The zero-fill code branch, the 16:15 cron, and the phantom-recording
path are code-verified. That Alpaca queues after-close DAY market orders to the next open is
standard, documented API behavior but was not observed in this system's data — the live
`trades` table contains **exactly one row ever** (an equity adjustment; checked read-only on
pg16, 2026-07-11), i.e. the signal path has never actually traded, so there is no behavioral
history to confirm or refute against (see Honest gaps).

**Disposition: new Plan A Task B (fill lifecycle + schedule).** User decision 2026-07-11:
equity cycle moves to **~15:45 ET (before close, weekdays, market-calendar-gated)**; fill
confirmation by short post-submit polling.

## 2. High findings

| # | Finding | Failure scenario | Evidence | Disposition |
|---|---|---|---|---|
| H1 | Feature-health gate wired to an orphan tracker | Real macro/HMM fetch failures can never trip the gate (all `record_*` calls go to a tracker nobody assesses); meanwhile the assessed tracker's `last_success_ts` never updates from its creation default, so after **7 days of scheduler uptime every cycle is blocked** ("macro data stale") until restart | `pipeline.py:81,166` vs `scripts/main.py:257`, `scripts/run_cycle.py:108`; `health.py:34,124–135` | Plan A **Task D** (wiring) |
| H2 | Model promotion uses a different directory layout than the loader | Shadow promotion drops a flat `models/active/{env}/*.zip` while inference loads `models/active/{env}/{algo}/model.zip` (+`vec_normalize.pkl`) → promotion is a **silent no-op** and a false "promoted" Discord alert is sent | `lifecycle.py:102–186` vs `pipeline.py:364–365`; `promoter.py:123–135` | Plan A **Task D** |
| H3 | Model cache never invalidated + cached-empty poison | New model deploys are ignored until process restart (fresh DB ensemble weights applied to stale binaries); if the first cycle runs before models exist, the empty dict is cached forever and `blend_actions` raises `ModelError` every cycle thereafter (escapes `execute_cycle`'s try/except) | `pipeline.py:354–355,407`; `ensemble.py:121–123` | Plan A **Task D** |
| H4 | Post-halt ramp-up logged but never enforced | During RAMPING the capacity fraction is fetched and `order_scaled_by_ramp` logged, but the frozen `SizedOrder` proceeds at full size; combined with the HALTED-only gate, trading resumes at **100% size ~1 day into a 5-day cooldown** | `risk_manager.py:182–192`; `circuit_breaker.py:134–139`; `pipeline.py:150–153` | Plan A **Task C** |
| H5 | Crypto stop-loss is detection-only and watches the wrong book | On stop breach: one `log.warning`, no sell, no alert, no DB record ("Phase 10" TODO); and it polls the **BTCUSD** ticker while fills execute on the **BTCUSDT** book | `stop_polling.py:5–6,102–153,:128`; `binance_sim.py:232–235`; `config/swingrl.yaml:17` | Plan A **Task C** (alert + record + correct book); auto-sell execution stays deferred, documented |

**Re-confirmations of known bugs (not new):**
- **F1** (dead turbulence halt) — independently confirmed by three of the four agents:
  `pipeline.py:537` queries a `turbulence` column absent from `features_equity/crypto`
  (`postgres_schema.py:224–266`); broad except → 0.0 (`:541–544`); the `:517` gate then skips
  the halt. The halt has **never fired in production**. Already Plan A **Task 6**; re-triage
  as capture-quality blocker stands.
- **A3** (risk penalty discarded under shaping) — confirmed as a **live defect**:
  `reward_wrapper.py:172` replaces the env reward (`sharpe_reward − risk_penalty`,
  `envs/base.py:221–238`) with a weighted component sum that has no risk-penalty component —
  every shaped run trained penalty-free. Already a **Plan B precondition** (spec §2.11).

## 3. Medium findings

| # | Finding | Evidence | Disposition |
|---|---|---|---|
| M1 | CB trips persisted but never alerted; auto-resume silent | `circuit_breaker.py:189–208,347–354,129–132` | Plan A **Task C** |
| M2 | Sub-minimum rebalance deltas silently dropped, no record; equity min order hardcoded `$5` ignoring config | `pipeline.py:252–253,245`; `schema.py:57` | **Task 9** fold (capture skip records) + **Task D** sweep (config) |
| M3 | Emergency crypto sell: FillResult discarded, `UPDATE positions SET quantity=0` ghost rows, no trades row, tier-4 verification counts ghosts → reports failure after success | `emergency.py:194,286–287`; `binance_sim.py:143–190,166,169–170` | Plan A **Task D** |
| M4 | Snapshot `cash_balance` math wrong (abs() treats sells as buys; ignores prior cash and commissions) | `pipeline.py:334` | Plan A **Task A** (subsumed) |
| M5 | `blend_actions` KeyError if `model_metadata` lacks a loaded algo's row; no weight renormalization when only 1–2 of 3 models load → silent allocation drift toward uniform | `ensemble.py:114`; `pipeline.py:367–374,437,445` | Plan A **Task D** |
| M6 | Turbulence computed twice per cycle from divergent sources (consume-once cache vs recompute); obs-path recompute scans the **entire** price history every cycle, unbounded | `features/pipeline.py:557–559,565–572,607–613`; `pipeline.py:512` | **Task 6/9** folds (bounded lookback, single compute, value reused for capture) |
| M7 | Missing/failing VecNormalize silently feeds raw observations to models | `pipeline.py:379,492–496` | Plan A **Task D** (fail closed: skip algo + alert) |
| M8 | Duplicate memory-table DDL copies **diverge in type**: `last_confirmed_at` TEXT vs TIMESTAMPTZ; container start order decides a fresh DB's schema | `services/memory/db.py:115` vs `postgres_schema.py:583` | **Plan B** cutover-runbook input |
| M9 | Eval env unseeded → `ConvergenceCallback` early-stop is nondeterministic; breaks any seed-pinning scheme that ignores it | `trainer.py:450–487,360–361` | **Plan B** (seed-pinning task must seed the eval env) |
| M10 | Fill-processor failure after a real broker fill = executed-but-unrecorded trade; crypto has **no reconciliation job at all** | `pipeline.py:292–295`; `jobs.py:431–437` (equity-only) | Equity failure-alert → **Task B**; crypto reconciliation **documented + deferred** (sim positions *are* the DB — no external truth until real Binance) |
| M11 | `FillResult` carries no timestamps; trade timestamp = DB-write time → `time_to_fill_ms` unmeasurable | `types.py:50–61`; `fill_processor.py:121` | Plan A **Task B** (DTO gains `status`/`submitted_at`/`filled_at`) |

## 4. Low findings

- Third `trades` writer exists: `scripts/migrate_to_postgres.py:53,119,159` (one-off
  migration; column list built dynamically — a **nullable** `cycle_id` won't break it).
- Hardcoded values in the hot path: `$5` equity min (`pipeline.py:245`), `1/3` default
  ensemble weight (`:437,:445`), `models/active` path fragments duplicated across three
  modules (root cause of H2 going unnoticed). → **Task D** sweep.
- Multiple `datetime.now(UTC)` per cycle (`pipeline.py:161,188,511` + per fill/snapshot) —
  capture needs ONE canonical cycle timestamp. → **Task 9** fold.
- `GlobalCircuitBreaker` high-water mark is in-memory only; resets on restart
  (`circuit_breaker.py:296–297,316`). → **Task A** (derive from persisted snapshots).
- `backtest_results` has no natural-key uniqueness; 9 real duplicate rows exist (iter-1
  restart); every reader does `DISTINCT ON … created_at DESC`
  (`postgres_schema.py:124–170`; `iteration_report.py:114–144`). → **Plan B** (`fold_results`
  has UNIQUE `run_pk`).
- Open positions at fold end are excluded from `win_rate`/`profit_factor` (FIFO
  reconstruction never closes remaining lots); a buy-and-hold fold reports
  `total_trades=0`, which also feeds the validation gate (`backtest.py:114–199,431–436`).
  → **Plan B** input (backtest-trade semantics / gate design).
- SAC `_on_rollout_end` docstring claims lower frequency; the opposite is true
  (`epoch_callback.py:328–331` vs `:38–44`). → **Plan B** (F2 task fixes docs).
- `notable_event` storms are threshold-mitigated only — no rate limiting exists; the 2.8M-row
  incident mechanism is structurally still present (`epoch_callback.py:355–377`). → **Plan B**
  (rate-cap task; greenfield confirmed).
- `alpaca-py>=0.20` floating with no upper pin; no dependency CVE scanner anywhere. →
  already **Tasks 13–14** (premises confirmed).
- Alpaca `get_current_price` = **last IEX trade** (free-feed default, can be stale/off
  consolidated market) (`alpaca_adapter.py:75–78,202–213`). → **Task 13** audit input +
  P-A5 column comment.

## 5. Crypto sim-fidelity gap list (→ Task 13 audit input)

1. **[HIGH]** Slippage is a fixed constant off the mid (`binance_sim.py:35,84–93`): fill =
   mid × (1 ± 0.0003) while the fetched best bid/ask are discarded → every captured crypto
   `fill_quality` row will show slippage ≡ 0.03% **by construction** (zero informational
   value). Credible minimum: fill buys at ask / sells at bid (or walk the book).
2. **[HIGH]** Sim never rejects and always fills fully: no balance ledger, no
   LOT_SIZE/stepSize rounding, no MIN_NOTIONAL, no partial fills, no order lifecycle.
3. **[MEDIUM]** Fee model hardcoded 0.10%/side (`:36`), never deducted from any balance
   (zero P&L drag); computed on decision-price notional (`:92`) vs fill notional in
   `emergency_sell` (`:158`) — inconsistent.
4. **[MEDIUM]** Fills execute on the thin **USDT** books while the stop-poller watches USD
   (see H5); wide-spread events still fill at mid with only a warning (`:253–260`).
5. **[MEDIUM]** Time-to-fill ≡ 0 (synchronous in-process) — captured latency stats are
   meaningless as live predictors.
6. **[LOW]** Decision price and fill price are two depth calls milliseconds apart — the sim
   cannot express "price moved between decision and fill".
7. **[LOW]** Blocking retry sleeps inside the trading cycle (up to ~3s + timeouts per
   symbol; same pattern in the Alpaca adapter `_retry` `:215–243`).

**Task 13 outcome decision (explicit):** improve the sim fill model vs accept + document the
distortion — decided when Task 13 runs.

## 6. Training-side review items (spec §2.11 / §4.14 — Plan B inputs)

| Item | Verdict | Key evidence |
|---|---|---|
| F2 instrumentation | FEASIBLE, trivial — `_epoch` already counts rollout ends; SAC fires **every vec-step** (`train_freq=1`, ~167K/fold) vs PPO ~82/fold — root cause of volume asymmetry pinned; historical blowup = old `notable_event` thresholds (fired on ~80% of epochs) amplified by SAC cadence | `epoch_callback.py:325–353,455–477,38–44,73–78`; SB3 2.7.1 `off_policy_algorithm.py:606` |
| Backtest trade semantics | **Round-trip** via FIFO reconstruction; `win_rate`/`profit_factor` in `metrics.py:218–272` from FIFO-matched (partial) round trips; open lots at fold end excluded | `backtest.py:114–199,599–601` |
| `fold_results` single-writer collapse | STRAIGHTFORWARD — plain 22-col writer is dead in the training pipeline (only `scripts/backtest.py` uses it); rich 42-col writer has 3 call sites in `train_pipeline.py`; no natural-key constraint today | `backtest.py:646–702,772–927`; `train_pipeline.py:1997,2495,2570` |
| Wrapper MDD → equity-fraction | CONFIRMED reward-cumsum basis; change touches the `−25.0` notable-event threshold, historical comparability, and LLM context fields; `info["portfolio_value"]` already available per step; per-env vs pooled deque is a design decision for Plan B | `reward_wrapper.py:106–108,208–220`; `epoch_callback.py:78,375,411,471,507–565`; `envs/base.py:399–401` |
| Trend-window rate-cap | **GREENFIELD** — `_should_store` is pure thresholding; no rate limiting exists | `epoch_callback.py:355–377` |
| `learner_metrics` vs real SB3 keys | Current mapping **CORRECT** for SB3 2.7.1 (incl. `ent_coef_loss` under auto-tuning); realistic per-algo contract enumerated; gap: SAC `ent_coef` not captured (useful health signal); SAC keys absent (→0.0) during `learning_starts` | `epoch_callback.py:50–72,335–336`; SB3 `ppo.py:287–300`, `a2c.py:184–190`, `sac.py:297–302` |
| Per-fold seed pinning | **FEASIBLE** — seeds today are per-algo constants (42/43/44) identical across folds/iterations; threading a per-fold seed is mechanical — **IF** the eval env is also seeded (M9); advice-enabled folds are inherently irreproducible → **seed-pair replication fallback** (A25 pre-statement) for those | `trainer.py:71,265,272,421–431,450–487` |
| U3 fallback enumeration | All fail-open, single-shot, no retry: transport `{}`, callback timeout-count, validation silently ignores, meta-orchestrator cold-start/exception → baseline | `client.py:160–197`; `epoch_callback.py:687–690,715–762,806–814`; `meta_orchestrator.py:287–297,373–380,427–434,473–475` |
| Startup-guard placement | Guard exists only for `stop_training` (20%, `bounds.py:97`, `epoch_callback.py:696–705`); advice has NONE — natural seat: top of `_query_epoch_advice` (`epoch_callback.py:614–619`) |  |
| `_MAX_REWARD_DELTA` config surface | **NONE** — hardcoded `bounds.py:109–114` (unlike HP/reward bounds, which read `config.training.bounds`); `_ADJUSTMENT_COOLDOWN_STEPS` likewise (`:121–126`) |  |
| F1b + assembler layout | CONFIRMED line-by-line: training obs turbulence ≡ 0.0 (`data_loader.py:237,350`; no env override, `envs/base.py:359–365`); live obs real (`features/pipeline.py:363,407`); `turbulence_index` present for both envs; index derivable from layout constants exactly as Task 7 plans (equity idx 128 no-sentiment / crypto 34). Caveats: `CRYPTO_OBS_DIM=47` hardcoded for 2 symbols; partial sentiment dict shifts equity layout (shape check raises `DataError`) | `assembler.py:38–66,133,197–266,270,307,333` |

## 7. Plan A claim & assumption verification

- **Change-site register: 10/10 CONFIRMED** (one line-range nuance, one path nuance).
  Footnotes: M8 (DDL copies diverge in type, not just existence); third trades writer (LOW).
- **P-A5 (decision price): PARTIAL.** `get_current_price` (`pipeline.py:258`) is used only
  to convert dollars → quantity; the dollar sizing comes from stale snapshot values ×
  current weights (C1 makes this worse). Equity orders submit by **notional**; the crypto
  sim re-fetches a second mid-price for the fill. **Resolution (disposition):**
  `decision_price_usd` := the sizing-time `get_current_price()` value; nuances recorded in
  column comments (Task 10 amendment).
- **Model-loading claim: PARTIAL as originally stated.** Binaries load from the filesystem
  (`models/active/{env}/{algo}/`); `model_metadata` supplies **only ensemble weights**
  (`pipeline.py:410–445` = `_get_ensemble_weights`, newest-per-algo by TEXT
  `training_end_date` DESC). Legacy `model_id` version strings differ between writers
  (`train_pipeline.py:1635` v1.1.0 vs `train.py:483` v1.0.0).
- **P-A1/P-A2(A29)/P-A4/P-A6:** unaffected by review (P-A4/P-A6 remain Task 11
  external verifications; P-A3's empirical VecNormalize check remains Task 7 Step 3b).

## 8. Capture-hook feasibility (verified, for Tasks 8–10)

- **Clean insertion point:** after `target_weights` (`pipeline.py:223`) and before the order
  loop (`:247`) every needed value coexists un-mutated: `observation`, per-algo
  `normalized_obs` + `actions`, `weights`, `blended_actions`, `current_weights`,
  `target_weights`, `env_name`, `dry_run`. Add after `:238` for `portfolio_value`.
- HMM probabilities / VIX / turbulence are **not separate variables** — extract from the
  observation via the assembler layout helpers (never re-call `compute_turbulence`:
  consume-once cache makes a third computation inconsistent).
- Early-exit cycles (CB halt `:151–153`, turbulence halt `:156–158`, degraded features
  `:167–179`, NaN obs `:193–195`, zero portfolio `:240–242`) each need their own capture
  write (with halt reason) or those cycles won't exist in `inference_cycles`.
- Use ONE canonical cycle timestamp; follow the `inference_outcomes` insert pattern
  (`:183–191`: own try/except, warn on failure, never block the cycle).
- Concurrency: equity and crypto cycles can overlap on the same `ExecutionPipeline`
  instance — no shared mutable per-cycle capture state; `run_cycle.py` CLI is a second
  writer process (tag source); tag `dry_run` cycles.
- Callers of `execute_cycle`: exactly three — `scheduler/jobs.py:101` (equity), `:152`
  (crypto), `scripts/run_cycle.py:141` (manual CLI). The shadow runner does its own
  inference and bypasses `execute_cycle`.

## 9. Disposition (user-approved 2026-07-11)

| Destination | Findings |
|---|---|
| **New Plan A Task A — Real portfolio valuation** | C1, M4, GlobalCB HWM (L) |
| **New Plan A Task B — Fill lifecycle + schedule** | C2 (+15:45 ET decision, market calendar, polling), M11, M10-equity alert |
| **New Plan A Task C — Risk-layer honesty** | H4, M1, H5-minimal (auto-sell deferred, documented) |
| **New Plan A Task D — Model-loading hygiene** | H2, H3, M5, M7, M3, hardcode sweep (L) |
| **Existing Plan A task folds** | Task 6: M6; Task 9: M2-capture, canonical ts, early-exit capture; Task 10: P-A5 resolution, M11 consumption; Task 13: sim-fidelity list, IEX nuance, slippage-model outcome decision |
| **Plan B inputs** | M8, M9, A3 (precondition), all §6 verdicts, backtest natural key, open-position metric gap, SAC docstring, rate-cap |
| **Documented deferrals** | Crypto stop auto-sell execution (Task C records+alerts only); crypto reconciliation (meaningful only against real Binance) |
| **No action (footnotes)** | Third trades writer; phantom-check inconclusiveness |

Ordering: Tasks A/B complete **before** capture Tasks 9–10; Tasks C/D complete **before**
Task 16. Task 16's go/no-go then tests working breakers instead of proving a fiction.

## 10. Honest gaps (UNKNOWN at review close)

- **Alpaca after-close order behavior**: API-knowledge, not observed — the live `trades`
  table has 1 row ever (adjustment only), so the signal path has never traded and the
  phantom-trade query was inconclusive **for lack of data** (itself notable: paper trading
  has not actually been exercising the trade path).
- **OHLCV ingestion freshness at cycle time**: ingestion (`ingest_all.py`) is not in the
  trading scheduler; whether same-day bars exist when the cycle fires is UNVERIFIED —
  Task B adds a data-freshness guard before deciding.
- `DatabaseManager` pool semantics (whether the per-cycle F1 UndefinedColumn error poisons
  a pooled connection) — unreviewed; worth a targeted look before capture writes share the
  pool.
- Live `model_metadata` contents (whether every disk-loaded algo has a row — decides if the
  M5 KeyError is latent or live) — not queried.
- Actual Binance.US fee schedule (0.10% constant unverified externally) — Task 13.
- Alpaca-py installed-version behavior (IEX feed default) — Task 13.
- Startup-order winner of the M8 DDL divergence on live pg16 — Plan B cutover check.
- SB3 VecNormalize behavior on unbatched observations for the pinned version — Task 7
  implementation detail.
