# Binance.US sim-fidelity audit (Task 13, Step 2)

**Date:** 2026-07-17 · **Branch:** `swingrl/2.R-A-capture-foundation` · **Author:** Task 13 audit
**Scope:** divergence inventory only. No `binance_sim.py` behavior was changed. The
`disposition` column is a *recommendation*; the user decides fix-now vs accept + document
before Step 3 (failing tests) and Step 4 (fixes) run in a later dispatch.

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
| **D1** | **Fill price = constant slippage off mid.** Buy = mid×(1+0.0003), sell = mid×(1−0.0003); the fetched best bid/ask are computed then **discarded**. | `binance_sim.py:36,88,91–94` (bid/ask fetched at `:257–258`, unused for the fill) | Real taker fills cross the spread: buy at best ask, sell at best bid, then walks the book. BTC/ETH USDT top-of-book spread is real and time-varying (typ. 1–5 bps, wider in stress). | **High** | **Fix now.** Fill buys at `best_ask`, sells at `best_bid` (both already in hand at `:275`). Today every `fill_quality.slippage_frac` ≈ 0.03% *by construction* → zero informational value for the exact analysis Task 10 exists to enable. Cheapest high-value fix in the list. |
| **D2** | **Commission rate hardcoded 0.10%/side.** | `binance_sim.py:37,96,167` | Binance.US Tier 0: **0% maker / 0.01% taker**; market orders are takers → ~0.01%/side (fee page, see honest gap). | **Medium** | **Accept + document** short-term — the sim errs *conservative* (overstates cost ~10×), so paper P&L is pessimistic, which is safe. But reconcile the number with config as part of D9 so expected == realized. Do **not** silently lower it without confirming the account's real fee tier. |
| **D3** | **Fee never deducted from a virtual balance.** Docstrings claim "virtual balance tracking" but there is **no balance/cash ledger** in the adapter — commission is returned on `FillResult` and recorded to `fill_quality`, but nothing is subtracted from a cash/equity balance (zero P&L drag). | `binance_sim.py:1,42,61` (docstring-only "balance"); no ledger code (grep) — review §5.3 | Real fees reduce spendable USD every fill. | **Medium** | **Accept + document now** (correct the misleading docstring), **defer** a real cash ledger. Commission *is* captured in `realized_cost_frac`, so it is not fully invisible to analysis — the gap is P&L accounting, not data capture. A ledger is a larger change; scope it separately. |
| **D4** | **Commission notional-basis is inconsistent.** `submit_order` charges commission on `dollar_amount` (decision-time notional); `emergency_sell` charges on `quantity × fill_price` (fill notional). | `binance_sim.py:96` vs `binance_sim.py:167` | A real venue always charges on the executed fill notional. | **Low** | **Fix now** (trivial): both paths should use fill notional. One-line consistency fix; folds naturally into the D1/D9 pass. |
| **D5** | **No LOT_SIZE / stepSize rounding.** Sim fills the raw `quantity = dollar/price`; `position_sizer` never snaps to `stepSize` either. | `binance_sim.py` (no rounding anywhere); `position_sizer.py:133` | BTCUSDT step 0.00001, ETHUSDT step 0.0001 (exchange-info). Real orders are rounded/rejected on sub-step qty. | **Low** | **Accept + document.** Fractional-dust difference on a $10–$50 order is negligible P&L. Note it; snap-to-step only if capital scales up. |
| **D6** | **No MIN_NOTIONAL rejection.** Sim always accepts. | `binance_sim.py:75–126` (no filter check) | MIN_NOTIONAL = $1.00, `applyToMarket=true` (exchange-info). | **Low** | **Accept + document.** Guarded upstream: the $10 app floor (`position_sizer.py:112`) is 10× the exchange minimum, so a real order can never fall below MIN_NOTIONAL. Documentation, not a fix. |
| **D7** | **Never rejects, always fills fully, no partial fills, no order lifecycle.** Every order returns `status="filled"` for the full quantity. | `binance_sim.py:113–126` (`status="filled"`, full qty) — review §5.2 | Real market orders can partial-fill on thin books, or reject (insufficient balance / filters). | **Medium** | **Accept + document** for the current small-order BTC/ETH regime, where full immediate fill is realistic. Revisit (partial-fill modeling) if per-order size grows relative to top-of-book depth. |
| **D8** | **Fills on the USDT book while the stop-poller watches USD; wide spreads only warn, still fill at mid.** | `binance_sim.py:245–248` (USDT depth), `:264–273` (warn-only at 0.5%) — review §5.4 / H5 | USDT and USD books are distinct, with different (thinner USDT) depth; a real fill at a 0.5%+ spread pays that spread, it is not a free mid-fill. | **Medium** | **Accept + document**, but pair with D1: once fills cross the real spread, add a **hard spread reject** (raise `BrokerError` above the current warn-only threshold) so wide-spread events don't silently fill at an unrealistic mid. |
| **D9** | **`expected_cost_frac` snapshots config 0.0022 while the sim realizes ≈0.0013 → systematic ~0.09% expected-minus-realized artifact on every crypto `fill_quality` row.** Compounded: config comments 0.0022 as "round-trip" but `fill_processor` applies it **per fill**. | `fill_processor.py:236,242,246,295` (snapshots `crypto_transaction_cost_pct`); sim constants `binance_sim.py:36–37`; config `swingrl.yaml:47` | N/A — this is an internal self-consistency defect, not a venue divergence. | **High** | **Fix now.** Make one source of truth: derive `expected_cost_frac` from the same commission+slippage constants the sim applies (or set the config figure to match), so expected−realized reflects *real* execution surprise, not a 0.09% modeling offset. Also resolve the round-trip-vs-per-side semantic mismatch. This is the artifact that biases Task 10's core analysis. |
| **D10** | **Time-to-fill ≡ 0.** `submitted_at` and `filled_at` are stamped in the same synchronous call. | `binance_sim.py:100,124–125,170` | Real fills take tens–hundreds of ms of venue + network latency. | **Medium** | **Accept + document.** The sim genuinely cannot know venue latency; faking a number is worse than none. Document that crypto `time_to_fill_ms ≈ 0` is a sim artifact, not a live latency estimate (consider recording NULL rather than 0). |
| **D11** | **Decision price ≠ fill price by construction** — two separate depth calls milliseconds apart; the sim can't express "price moved between decision and fill." | `binance_sim.py:88` (fill-time depth) vs sizing-time depth call (`pipeline.py`) — review §5.6 | Real slippage between decision and fill is a genuine, single-tape phenomenon. | **Low** | **Accept + document.** Two-call jitter adds noise to `slippage_frac`; after D1 (fill at bid/ask) the dominant signal will be the real spread, so this residual noise is acceptable. |
| **D12** | **Blocking retry sleeps inside the trading cycle** (up to ~1+2s backoff per symbol on depth-fetch failure). | `binance_sim.py:34–35,285–287` (same pattern in `alpaca_adapter.py:_retry`) — review §5.7 | N/A — an availability/latency concern, not a fill-price divergence. | **Low** | **Accept + document.** Bounded (max ~3 attempts). Note as a known cycle-latency cost; async retry is a hardening-phase item, not a fidelity fix. |
| **D13** | **Alpaca decision price = last IEX trade (free feed).** `StockHistoricalDataClient` defaults to the free IEX feed; `get_current_price` returns the last IEX trade. Equity decision prices can be **stale or off-consolidated**. | `alpaca_adapter.py:77–80,263–274` — review §4 | IEX is ~2–3% of US consolidated volume; the SIP (paid) tape is the true NBBO. | **Medium** | **Accept + document** (decision-price caveat on the P-A5 column). Paper-only today; upgrading to the SIP feed is a paid-plan change. Flag that captured equity decision prices carry an IEX-staleness caveat when analyzing fill quality. |

---

## Summary counts

- **High impact: 2** — D1 (constant-slippage tautology), D9 (expected-vs-realized 0.09% artifact).
- **Medium impact: 6** — D2, D3, D7, D8, D10, D13.
- **Low impact: 5** — D4, D5, D6, D11, D12.

## Opinionated recommendation

If the user greenlights a fill-model improvement, the highest-leverage minimal change is a
single pass over `binance_sim.submit_order` / `emergency_sell` that:

1. **D1 — fills at best ask (buy) / best bid (sell)** instead of mid ± constant. The bid/ask are
   already fetched and discarded, so this is nearly free and it is the *only* change that makes
   `fill_quality.slippage_frac` carry real information.
2. **D8 — hard-rejects wide spreads** (raise `BrokerError` above the 0.5% threshold instead of
   warn-and-fill-at-mid), so stress events don't fill at fantasy prices.
3. **D9 — unifies the cost source** so `expected_cost_frac` is derived from the same constants
   the sim charges, killing the systematic 0.09% offset in Task 10's analysis.
4. **D4 — fixes the commission notional basis** to fill notional on both paths (one-liner).

Everything else (D2 rate, D3 ledger, D5 stepSize, D6 min-notional, D7 partials, D10 latency,
D11 two-call jitter, D12 blocking retries, D13 IEX feed) I would **accept + document**: each is
either conservative (safe direction), guarded upstream, bounded by the small-order regime, or a
paid-plan / larger-refactor item that doesn't belong in a fidelity pass.

**Net:** D1 + D8 + D9 + D4 together are a small, well-contained edit that removes both High-impact
distortions and the worst Medium one, and they are exactly the fixes that make the *captured data*
trustworthy. That is the recommended "improve the fill model" option. The alternative — accept +
document all of it — is defensible only if the crypto capture is treated as telemetry-only and no
analysis will lean on `slippage_frac` or `expected_cost_frac` until the fill model is revised.

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
