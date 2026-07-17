# Binance.US sim-fidelity audit (Task 13, Steps 2–4)

**Date:** 2026-07-17 · **Branch:** `swingrl/2.R-A-capture-foundation` · **Author:** Task 13 audit
**Status:** divergence inventory (Step 2) + the approved fix bundle (Steps 3–4). The user
approved **D1 + D4 + D8 + D9** (2026-07-17, AskUserQuestion); those four are now **FIXED**
(RED-first TDD — `tests/execution/test_binance_sim_fidelity.py`). The other nine divergences
remain **accept + document** by the same ruling.

This document compares what `src/swingrl/execution/adapters/binance_sim.py` models against the
current real Binance.US venue (fee schedule + exchange-info filters, fetched live 2026-07-17)
and against the Alpaca free-plan decision-price path. It is the input to the explicit Task 13
outcome decision: **improve the sim fill model vs accept + document the distortion.**

---

## Glossary

| Term | Meaning |
|---|---|
| `binance_sim` | Our simulated crypto broker. Binance.US has no real paper venue, so we fetch a real order book but simulate the fill locally. |
| `fill_quality` | DB table Task 10 populates from every fill: `slippage_frac`, `expected_cost_frac`, `realized_cost_frac`, `time_to_fill_ms`. Consumed by expected-vs-realized execution analysis. |
| `expected_cost_frac` | The *modeled* per-fill cost the analysis expects, snapshotted from config (`fill_processor._expected_cost_frac`). |
| `realized_cost_frac` | The *actual* per-fill cost the sim produced (commission + adverse slippage). |
| Taker / Maker | A market order that crosses the spread is a **taker**; a resting limit order is a **maker**. Our orders are market orders → taker fees apply. |
| `LOT_SIZE` / `stepSize` | Exchange filter: order quantity must be a multiple of `stepSize`. |
| `MIN_NOTIONAL` | Exchange filter: order value (price × qty) must be ≥ `minNotional`. |
| IEX / SIP | IEX = single exchange (~2–3% of US consolidated volume), the Alpaca free-feed default. SIP = the full consolidated tape (paid). |

---

## Real-world reference values (fetched live 2026-07-17)

### Binance.US spot fee schedule
Source: <https://www.binance.us/fees> (public fee page).

| Field | Value |
|---|---|
| Tier 0 maker fee | **0%** |
| Tier 0 taker fee | **0.01%** |
| BNB-payment discount | −5% on spot fees |
| Volume requirement for Tier 0 | none ("all new and existing customers") |

> **Honest gap (UNVERIFIED):** the 0% / 0.01% figures are the current *promotional* "Tier 0
> Pairs" schedule. Whether `BTCUSDT` / `ETHUSDT` are classified Tier 0 for *this account* (vs a
> standard fee tier) cannot be confirmed from public REST — it needs an authenticated
> `/sapi/v1/asset/tradeFee` call. Treated below as "real taker ≈ 0.01%" with that caveat.

### Exchange-info filters
Source: `GET https://api.binance.us/api/v3/exchangeInfo?symbols=["BTCUSDT","ETHUSDT"]`
(read-only public GET, fetched 2026-07-17). Configured crypto symbols read from
`config/swingrl.yaml:21` (`crypto.symbols: [BTCUSDT, ETHUSDT]`).

| Symbol | status | LOT_SIZE minQty / stepSize | MIN_NOTIONAL (applyToMarket) | PRICE tickSize |
|---|---|---|---|---|
| BTCUSDT | TRADING | 0.00001000 / 0.00001000 | 1.00 USDT (true) | 0.01 |
| ETHUSDT | TRADING | 0.00010000 / 0.00010000 | 1.00 USDT (true) | 0.01 |

### App-level floor (upstream of the sim)
`config/swingrl.yaml:26` `crypto.min_order_usd: 10.0`, enforced in
`position_sizer.py:112–114` **before** the order reaches the adapter. $10 ≫ the $1 exchange
MIN_NOTIONAL, so real orders always clear the exchange minimum (see D6).

### Measured sim cost constants (`binance_sim.py`)
| Constant | Value | Line |
|---|---|---|
| `_DEFAULT_SLIPPAGE` | 0.0003 (0.03% off mid, one side) | `binance_sim.py:36` |
| `_COMMISSION_RATE` | 0.001 (0.10% per side) | `binance_sim.py:37` |
| `_SPREAD_WARNING_THRESHOLD` | 0.005 (0.5%, warn-only) | `binance_sim.py:38` |
| **Per-fill realized cost** | **≈ 0.0013 (0.13%)** = 0.10% commission + 0.03% slippage | derived |
| Config `crypto_transaction_cost_pct` | 0.0022 (0.22%) | `config/swingrl.yaml:47` |

---

## Divergence inventory

Impact key: **High** = distorts captured `fill_quality` data or misstates P&L in a way that
misleads training/analysis · **Medium** = real distortion, bounded by the small-order BTC/ETH
regime or conservative direction · **Low** = cosmetic or already guarded upstream.

| # | Behavior | Sim value (file:line) | Real value (source) | Impact | Recommendation (fix now / accept + document) |
|---|---|---|---|---|---|
| **D1** | **Fill price = constant slippage off mid.** Buy = mid×(1+0.0003), sell = mid×(1−0.0003); the fetched best bid/ask are computed then **discarded**. | `binance_sim.py:36,88,91–94` (bid/ask fetched at `:257–258`, unused for the fill) | Real taker fills cross the spread: buy at best ask, sell at best bid, then walks the book. BTC/ETH USDT top-of-book spread is real and time-varying (typ. 1–5 bps, wider in stress). | **High** | **FIXED (Step 4).** `submit_order`/`emergency_sell` now fill buys at `best_ask`, sells at `best_bid`; recorded slippage is the real half-spread. Every `fill_quality.slippage_frac` now carries real information instead of a tautological 0.03%. |
| **D2** | **Commission rate hardcoded 0.10%/side.** | `binance_sim.py:37,96,167` | Binance.US Tier 0: **0% maker / 0.01% taker**; market orders are takers → ~0.01%/side (fee page, see honest gap). | **Medium** | **Accept + document** short-term — the sim errs *conservative* (overstates cost ~10×), so paper P&L is pessimistic, which is safe. But reconcile the number with config as part of D9 so expected == realized. Do **not** silently lower it without confirming the account's real fee tier. |
| **D3** | **Fee never deducted from a virtual balance.** Docstrings claim "virtual balance tracking" but there is **no balance/cash ledger** in the adapter — commission is returned on `FillResult` and recorded to `fill_quality`, but nothing is subtracted from a cash/equity balance (zero P&L drag). | `binance_sim.py:1,42,61` (docstring-only "balance"); no ledger code (grep) — review §5.3 | Real fees reduce spendable USD every fill. | **Medium** | **Accept + document now** (correct the misleading docstring), **defer** a real cash ledger. Commission *is* captured in `realized_cost_frac`, so it is not fully invisible to analysis — the gap is P&L accounting, not data capture. A ledger is a larger change; scope it separately. |
| **D4** | **Commission notional-basis is inconsistent.** `submit_order` charges commission on `dollar_amount` (decision-time notional); `emergency_sell` charges on `quantity × fill_price` (fill notional). | `binance_sim.py:96` vs `binance_sim.py:167` | A real venue always charges on the executed fill notional. | **Low** | **FIXED (Step 4).** Both `submit_order` and `emergency_sell` now charge commission on the executed fill notional (`fill_price × quantity × _COMMISSION_RATE`). |
| **D5** | **No LOT_SIZE / stepSize rounding.** Sim fills the raw `quantity = dollar/price`; `position_sizer` never snaps to `stepSize` either. | `binance_sim.py` (no rounding anywhere); `position_sizer.py:133` | BTCUSDT step 0.00001, ETHUSDT step 0.0001 (exchange-info). Real orders are rounded/rejected on sub-step qty. | **Low** | **Accept + document.** Fractional-dust difference on a $10–$50 order is negligible P&L. Note it; snap-to-step only if capital scales up. |
| **D6** | **No MIN_NOTIONAL rejection.** Sim always accepts. | `binance_sim.py:75–126` (no filter check) | MIN_NOTIONAL = $1.00, `applyToMarket=true` (exchange-info). | **Low** | **Accept + document.** Guarded upstream: the $10 app floor (`position_sizer.py:112`) is 10× the exchange minimum, so a real order can never fall below MIN_NOTIONAL. Documentation, not a fix. |
| **D7** | **Never rejects, always fills fully, no partial fills, no order lifecycle.** Every order returns `status="filled"` for the full quantity. | `binance_sim.py:113–126` (`status="filled"`, full qty) — review §5.2 | Real market orders can partial-fill on thin books, or reject (insufficient balance / filters). | **Medium** | **Accept + document** for the current small-order BTC/ETH regime, where full immediate fill is realistic. Revisit (partial-fill modeling) if per-order size grows relative to top-of-book depth. |
| **D8** | **Fills on the USDT book while the stop-poller watches USD; wide spreads only warn, still fill at mid.** | `binance_sim.py:245–248` (USDT depth), `:264–273` (warn-only at 0.5%) — review §5.4 / H5 | USDT and USD books are distinct, with different (thinner USDT) depth; a real fill at a 0.5%+ spread pays that spread, it is not a free mid-fill. | **Medium** | **FIXED (Step 4).** `submit_order` now raises `BrokerError` when the spread exceeds `_SPREAD_REJECT_THRESHOLD = 0.01` (1.0% — a named constant, 2× the 0.5% warn band; kept out of config as an execution-safety guardrail, not a tunable). `emergency_sell` is deliberately **exempt** — a forced exit must never be blocked by a wide book. |
| **D9** | **`expected_cost_frac` snapshots config 0.0022 while the sim realizes ≈0.0013 → systematic ~0.09% expected-minus-realized artifact on every crypto `fill_quality` row.** Compounded: config comments 0.0022 as "round-trip" but `fill_processor` applies it **per fill**. | `fill_processor.py:236,242,246,295` (snapshots `crypto_transaction_cost_pct`); sim constants `binance_sim.py:36–37`; config `swingrl.yaml:47` | N/A — this is an internal self-consistency defect, not a venue divergence. | **High** | **FIXED (Step 4).** `fill_processor._expected_cost_frac` now reads `binance_sim.modeled_crypto_cost_frac()` (= `_COMMISSION_RATE + _DEFAULT_SLIPPAGE` = 0.0013), the single source of truth, instead of the config figure. `expected_cost_frac` is now unambiguously per-fill (round-trip-vs-per-side resolved). Equity stays 0.0 (Task 10 contract). Config `crypto_transaction_cost_pct = 0.0022` is **unchanged** (training reward still uses it) — the D2 caveat holds: we aligned the expectation to the sim, not the sim to Binance.US. |
| **D10** | **Time-to-fill ≡ 0.** `submitted_at` and `filled_at` are stamped in the same synchronous call. | `binance_sim.py:100,124–125,170` | Real fills take tens–hundreds of ms of venue + network latency. | **Medium** | **Accept + document.** The sim genuinely cannot know venue latency; faking a number is worse than none. Document that crypto `time_to_fill_ms ≈ 0` is a sim artifact, not a live latency estimate (consider recording NULL rather than 0). |
| **D11** | **Decision price ≠ fill price by construction** — two separate depth calls milliseconds apart; the sim can't express "price moved between decision and fill." | `binance_sim.py:88` (fill-time depth) vs sizing-time depth call (`pipeline.py`) — review §5.6 | Real slippage between decision and fill is a genuine, single-tape phenomenon. | **Low** | **Accept + document.** Two-call jitter adds noise to `slippage_frac`; after D1 (fill at bid/ask) the dominant signal will be the real spread, so this residual noise is acceptable. |
| **D12** | **Blocking retry sleeps inside the trading cycle** (up to ~1+2s backoff per symbol on depth-fetch failure). | `binance_sim.py:34–35,285–287` (same pattern in `alpaca_adapter.py:_retry`) — review §5.7 | N/A — an availability/latency concern, not a fill-price divergence. | **Low** | **Accept + document.** Bounded (max ~3 attempts). Note as a known cycle-latency cost; async retry is a hardening-phase item, not a fidelity fix. |
| **D13** | **Alpaca decision price = last IEX trade (free feed).** `StockHistoricalDataClient` defaults to the free IEX feed; `get_current_price` returns the last IEX trade. Equity decision prices can be **stale or off-consolidated**. | `alpaca_adapter.py:77–80,263–274` — review §4 | IEX is ~2–3% of US consolidated volume; the SIP (paid) tape is the true NBBO. | **Medium** | **Accept + document** (decision-price caveat on the P-A5 column). Paper-only today; upgrading to the SIP feed is a paid-plan change. Flag that captured equity decision prices carry an IEX-staleness caveat when analyzing fill quality. |

---

## Summary counts

- **High impact: 2** — D1 (constant-slippage tautology), D9 (expected-vs-realized 0.09% artifact).
- **Medium impact: 6** — D2, D3, D7, D8, D10, D13.
- **Low impact: 5** — D4, D5, D6, D11, D12.

## Resolution (2026-07-17) — approved fix bundle implemented

The user approved the recommended bundle. Implemented via RED-first TDD
(`tests/execution/test_binance_sim_fidelity.py`):

1. **D1 — fills at best ask (buy) / best bid (sell)** instead of mid ± constant. The bid/ask were
   already fetched and discarded; this is the change that makes `fill_quality.slippage_frac` carry
   real information (recorded slippage is now the real half-spread).
2. **D8 — hard-rejects wide spreads**: `submit_order` raises `BrokerError` above
   `_SPREAD_REJECT_THRESHOLD = 0.01` (1.0%, a named constant = 2× the 0.5% warn band).
   `emergency_sell` is exempt — forced exits must never be blocked.
3. **D9 — unifies the cost source**: `expected_cost_frac` is derived from
   `binance_sim.modeled_crypto_cost_frac()` (0.0013), killing the systematic 0.09% offset and
   resolving the round-trip-vs-per-side semantics. Equity stays 0.0.
4. **D4 — commission on fill notional** on both paths.

The other nine (D2 rate, D3 ledger, D5 stepSize, D6 min-notional, D7 partials, D10 latency,
D11 two-call jitter, D12 blocking retries, D13 IEX feed) remain **accept + document** per the same
ruling: each is either conservative (safe direction), guarded upstream, bounded by the small-order
regime, or a paid-plan / larger-refactor item that doesn't belong in a fidelity pass.

> **Residual note (D9):** the config `crypto_transaction_cost_pct = 0.0022` that the training reward
> still uses now differs from the sim's realized cost (0.0013). That expected-vs-realized gap has
> moved out of `fill_quality` (fixed) but persists as a *training-assumption* vs *sim-reality*
> mismatch — a Plan B concern, out of Task 13 scope.
>
> **Reviewer note:** `FillProcessor`'s `config` param is now only a presence-gate for the crypto
> cost (it no longer supplies the value). Retained for API compatibility / the legacy no-config
> fallback (0.0); a later cleanup could drop it. Kept here to keep the fix bundle focused.

---

## Alpaca-py currency (Task 13 Step 1 — for reference)

Committed separately (`chore(2.R-A): pin alpaca-py after changelog review`). Locked **0.43.2** →
latest **0.43.5**; the three intervening releases are patch-only (pytz dependency, DocumentType
enum value, deprecated PDT/DTBP account-field tolerance, docstring fixes) with **no breaking
changes** to the adapter's calls (`submit_order`, `get_order_by_id`, `cancel_order_by_id`,
`get_all_positions`, `get_clock`, `get_stock_latest_trade`). Pinned `alpaca-py>=0.20,<0.44`;
`uv lock` retained 0.43.2; adapter tests 9 passed.

## Honest gaps (could not verify)

- **Binance.US fee tier for BTCUSDT / ETHUSDT on this account** — the 0% / 0.01% figures are the
  public promotional Tier 0 schedule; per-account tier classification needs an authenticated
  `/sapi/v1/asset/tradeFee` call (not attempted — read-only public REST only in this phase).
- **Exact IEX-vs-SIP staleness magnitude** for the configured equity symbols — asserted
  qualitatively (IEX ≈ 2–3% of consolidated volume), not measured against a live SIP tape.
