# Paper-Readiness Runbook — End-to-End Verification & Go/No-Go

**Plan A Task 16 (Stage 2.R Wave 2)** · Executed 2026-07-15 → 2026-07-21 · All times ET.

## Purpose

Record the end-to-end verification that the paper-trading deployment on homelab is ready
for unattended operation: Discord alerting proven live on every path, both circuit
breakers proven to trip and recover, capture pipeline proven on real cycles, and the
user's explicit GO/NO-GO decision. Procedure text lives in
`docs/superpowers/plans/2026-07-07-capture-foundation-plan.md` (Task 16); this document
is the dated results record. Raw evidence trail:
`.superpowers/sdd/progress-wave2.md` (the Wave 2 ledger), BATCH 5–5f and the
2026-07-21 STOPPING POINT block.

## Glossary

| Term | Meaning |
|---|---|
| CB | Circuit breaker — halts a trading environment after a risk breach (e.g. drawdown) |
| F1 / turbulence halt | Market-turbulence breaker; halts the cycle when the turbulence index exceeds its threshold |
| INFO path | Discord alerts at `info` level (digest or immediate). Separate code path from critical/warning — proven separately by design |
| Capture | Writing `inference_cycles`, `cycle_algo_proposals`, `trades.cycle_id`, `fill_quality` rows during a live cycle |
| Fail-open | A capture/DB failure must never stop trading — the cycle proceeds and a Discord warning is sent |
| Sim mirror | Comparing a paper fill's `fill_quality` row against the simulator's for the same conditions (shape parity) |
| Vintage | A durably-marked (env, algo) model row (`era0-{env}-{algo}`) that the trader serves |
| Deadzone | Portfolio-weight band inside which the trader intentionally places no orders |

## System under test

| Component | Version | Notes |
|---|---|---|
| Integration head | `941c83b` (PRs #31–#35 merged) | candle feed, INFO parity, backup-job gating, hmm upsert, bar-boundary fix + repair |
| Trader | `swingrl:trader-2026-07-19-2` | schema fingerprint v8 asserted at boot; 9 jobs; persistent jobstore |
| Collector | `swingrl-collector:2026-07-19-4` | 9 jobs incl. `candles_equity` / `candles_crypto` |
| Memory | rebuilt `--no-cache` 2026-07-18 | floor semantics verified (`schema_version_ahead` = warn-and-run) |
| Database | pg16, schema_migrations = 8 (V001–V008 applied live 2026-07-18) | 68 tables, 6 views |
| Models | era-0 vintages, 6/6 rows, all sha256 match `models/active/` | crypto iter 0 (CPS 0.1531), equity iter 4 (CPS 0.0153); equity obs-dim 164 |

## Step results

### Step 1 — Discord live proof: ✅ PASS (digest embed eyeball pending)

| Path | Proven | Evidence |
|---|---|---|
| Critical / warning | ✅ 2026-07-15 (deploy) + live 2026-07-19 | Unplanned live proof: 02:00 'PostgreSQL Backup Failed' warning delivered (contained; led to backup-job gating fix, PR #33). Drill proof: 'Circuit Breaker Halted — crypto' ~13:23, **user confirmed the phone notification** |
| INFO | ✅ 2026-07-19 | First 'Crypto candles ingested' delivered 08:01; **user screenshot 12:01** of the INFO embed (rows_added=2) |
| Trade embeds | ✅ 2026-07-19 | BUY BTCUSDT / BUY ETHUSDT embeds 04:05; **user screenshot 12:04** — values match logs exactly |
| Daily digest | ✅ sent / 👁 eyeball open | 'Daily Summary' `alert_sent` observed in stream Sun 18:00 (and presumed Mon/Tue); user eyeball of the embed still wanted |

Note: the literal "startup smoke call" was superseded — no synthetic test alert was sent;
instead every alert class was proven with real deliveries (stronger evidence). No
automatic startup alert exists in the trader (observed at 2026-07-19 boot).

### Step 2 — Drawdown CB trip: ✅ PASS (drill 2026-07-19 ~13:2x)

Config-only routes dead-ended twice (schema guards reject drill values: daily<drawdown
relation, `min_order_usd` floor). Used the plan-sanctioned test-hook route: real
RiskManager + CB + Alerter against the live DB, synthetic $2 sell.

- Tripped: drawdown 0.00085 ≥ 0.0001 drill threshold ✅
- `circuit_breaker_events` row written ✅
- Discord 'Circuit Breaker Halted — crypto' `alert_sent` ✅ (user phone-notification confirmed)
- `risk_decisions` veto row ✅ (plus bonus exposure-veto proof)
- Real-config cycle halted: cycle 5 `halt_reason=circuit_breaker` ✅
- `resume()` → ACTIVE ✅; drill config deleted, real config untouched

### Step 3 — Turbulence halt, F1 first-ever fire: ✅ PASS (drill 2026-07-19 ~13:2x)

Drill percentile 0.01 → `turbulence_crash_protection` fired (turbulence 0.5811 vs
threshold 0.0654) → CB triggered → `cycle_halted_by_turbulence` → cycle 6
`halt_reason=turbulence_halt` with turbulence value stamped ✅. Resumed ACTIVE; drill
yaml deleted. This was the first fire in the breaker's history; findings below.

### Step 4 — Capture verification: ✅ PASS (live drill for fail-open not run — see open items)

- Capture bar: crypto ≥2 cycles exceeded 2026-07-19 (19 crypto cycles by 07-21); equity
  2 cycles ran Mon 07-20 + Tue 07-21 15:45 ✅
- First live cycle (crypto cycle 1, 2026-07-19 04:05): `inference_cycles`=1, 3
  `cycle_algo_proposals`, turbulence non-NULL, `trades.cycle_id` 2/2, `fill_quality`=2
  (signal fills only) — all Task 17 Step-4 queries pass ✅
- 48h autonomous window (07-19 → 07-21): 21 cycles, 0 halts, 0 trader criticals, 0
  collector job failures ✅
- Fail-open: proven at unit level
  (`tests/execution/test_cycle_recorder.py::test_fail_open_returns_none_and_alerts` —
  DB failure → 'Cycle capture failed' warning, cycle proceeds). The live forced-failure
  drill (revoke INSERT in a scratch-DB run) was **not** executed — user decision at
  go/no-go: accept unit proof or run the live drill.

### Step 5 — Sim mirror: ⏸ BLOCKED (zero equity fills so far)

Crypto-side context: cycle-1 fills show realistic slippage + commission (BUY BTCUSDT
$20.06, BUY ETHUSDT $19.90, comm $0.02 each). The equity `fill_quality`-vs-sim shape
parity check requires a real equity fill; both equity cycles to date held in the
deadzone (0 fills). **Check after every 15:45 equity cycle; complete on first fill.**
Known Task-13-adjacent note (final-review list): emergency-sell `fill_quality` rows
record `expected_cost_frac=0.0` (legacy fallback, benign); crypto expected-cost 0.0022
vs sim-realized 0.0013 is a training-assumption gap out of Task 13 scope.

### Step 6 — Go/No-Go: 🛑 PENDING USER (record below)

## Partial-fill & terminal-disposition policy (09:35 confirmation, 2026-07-22 rulings)

The 09:35 confirmation job (`equity_fill_confirmation_job` → `_confirm_one_pending_order`)
uses a **slice model** — partial auction fills are recorded, not merely warned about.

- **Slice recording (ruling #1).** Alpaca reports only a cumulative `filled_qty` and a
  cumulative `filled_avg_price`. Each run records the *increment* of shares filled since the
  last look as a real trade through the shared post-fill path (`pipeline.record_fill` —
  trades + positions + fill_quality + realized-P&L), so the books match the broker the same
  day and the between-cycle risk sweeps mark real positions. Earlier behaviour (partials
  warned loudly but were left unrecorded) is retired.
- **Slice pricing.** The increment is priced so recorded dollars reconcile to the broker's
  cumulative average exactly: `slice_dollars = filled_avg_price × cum_qty −
  already_recorded_dollars`; `slice_price = slice_dollars / slice_qty`. A degenerate
  non-positive result falls back to `filled_avg_price` and logs
  `pending_order_slice_price_fallback`.
- **Slice trade ids.** The first slice reuses the broker `order_id`; later slices are
  `{order_id}#<n>` (`#2`, `#3`, …) so one broker order maps to one-or-more `trades` rows
  without violating the `trades` TEXT primary key.
- **Terminal disposition (ruling #2).** A terminal broker state stamps `resolved_at` +
  `disposition` and fires ONE final alert, then the row leaves the worklist — a dead order is
  closed once, never re-warned daily forever. Mapping: `filled` → `'filled'`;
  `canceled`/`rejected`/`replaced` → `'canceled'`; `expired`/`done_for_day` → `'expired'`.
  Any unrecorded final increment is recorded before the row is stamped. A still-live order
  (`partially_filled`/`new`/`accepted`) gets a warning and stays open for the next run.

## Findings log (from drills — all on final-review list, none blocking by severity)

1. Zero-order cycles never reach breaker evaluation (deadzone-held cycles skip it;
   narrow — a real crash generates orders, and the turbulence gate runs at cycle start).
2. Exposure check adds SELL `dollar_amount` to exposure (a sell should reduce it).
3. CB reason string hardcodes the "90th_pct" label while the threshold is
   config-driven (cosmetic).
4. Drill B's Discord alert was likely suppressed by the 30-minute same-title cooldown
   after drill A's alert — alerter working as designed; operators should expect this
   during rapid repeat halts.

Known-permanent noise (refinement backlog): Binance 451 geo-block storm re-probing dead
gaps every 4H run (log-only); 26-day crypto hole 2026-03-10→04-06 unrecoverable
(training-data concern only).

## Attestations & operator hardening

| Item | Status |
|---|---|
| Alpaca key is PAPER-only | ✅ Attested by user 2026-07-19 (+ live API check: ACTIVE, paper, cash $100,241.00, equity $101,100.30) |
| Binance.US key is read-only + trade-only scope check | ⏳ User checking API Management |
| Key-rotation date recorded | ⏳ User supplying (initial ~07-11 guess was wrong) |
| `.env` chmod 600 | ✅ 2026-07-19 |
| `backups/` 700, dumps 600 | ✅ 2026-07-19 |
| Webhook rotation | Deferred by user ruling — old leaked-but-working webhook in use; rotate LATER |

## Open items at go/no-go

1. Daily-digest embed eyeball (digest is being sent; visual confirmation wanted).
2. Binance.US read-only attestation (user).
3. Key-rotation date (user).
4. Sim-mirror equity leg — blocked until first real equity fill (Step 5).
5. Live fail-open capture drill — not run; unit-proven. User to accept or require.

## Superseding events (2026-07-21)

The execution-alignment work approved 2026-07-21
(`docs/superpowers/specs/2026-07-21-execution-alignment-design.md`, D1–D15) changes what
this runbook is waiting for. Nothing above is rewritten — the step results stand as the
dated record of what was proven on the 15:45 schedule — but the go/no-go is **deferred**
until:

1. The exec-alignment branch deploys (collector + trader rebuilds, separately gated) and
   the **D14 paper-epoch reset** completes (crypto sim-sells → user resets Alpaca paper
   account → reconciliation → benchmark baselines recorded).
2. **≥2 fresh equity cycles run under the new 09:15 schedule** and are recorded here as
   the capture evidence (the 15:45-schedule evidence above certifies a configuration
   being replaced; spec coupling note).
3. **Step 5 (sim mirror) completes on an opening-auction fill** — the first equity fill
   will arrive via the new pre-open path, so the mirror comparison uses the open(t)
   convention, matching era-1 training (spec D9/D11).
4. The first morning cycle also verifies the spec's two flagged assumptions: feature
   freshness at 09:15, and fractional fills at the opening-print price basis.

The 5 open items below remain live and carry across unchanged.

## GO/NO-GO record

| Date (ET) | Decision | Decided by | Conditions / notes |
|---|---|---|---|
| _pending_ | _GO / NO-GO_ | user | |
