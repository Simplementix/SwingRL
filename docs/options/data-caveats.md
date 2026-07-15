# Options Data Caveats

What this collector captures is real CBOE market data, but it has a specific shape — read
this before treating a value as ground truth, and before training anything on it. Companion
to `docs/options/ops.md` (operations) and `docs/options/week1-data-quality-audit.md`
(the manual sanity checks that catch violations of what's documented here).

**Last verified against code:** 2026-07-15

## Delay convention: pull time ≠ market time

The CBOE delayed-quotes feed is assumed ~15 minutes behind the live market (spec §17 C3, D8).
Four separate timestamps exist per snapshot — never assume they're interchangeable:

| Field | Where | Meaning |
|---|---|---|
| `market_time_et` (config) → `snapshot_time_utc` (stored) | `options_collector.snapshots[].market_time_et` in YAML; `options_snapshots.snapshot_time_utc` | The market moment this label is **asserted to represent** — 15:45 for `decision`, 16:15 for `eod`. Derived from config, not from the payload. |
| `pulled_at_utc` | `options_snapshots.pulled_at_utc`, `options_chains.pulled_at_utc` | Wall-clock time the collector actually fetched the chain. |
| `raw_header.payload_timestamp` | inside `options_snapshots.raw_header` (jsonb) | CBOE's own top-level `timestamp` field from the response — the feed's self-reported data-as-of time. |
| `raw_header.late_by_s` | same | How late the job fired past its *scheduled* pull time (not the delay itself) — `0.0` unless the container was down/slow. Non-zero on a `decision` snapshot triggers a WARNING (lookahead-bias guard): a late-fired decision snapshot must never silently pass as genuine 15:45 data. |

**`trade_time_utc` (from CBOE `last_trade_time`) is localized from ET, not UTC.** The feed's
per-contract `last_trade_time` is a naive wall-clock string that the fixtures show clustering in
market hours (e.g. a 16:00:00 last trade = the ET close). The parser
(`chain_parser.parse_cboe_ts`) therefore treats it as `America/New_York` and converts to UTC —
so a 15:59:07 last trade is stored as 19:59:07Z in summer. This ET convention is **empirically
confirmed by the T6 probe (2026-07-15)**: at 16:05:02 ET wall clock the header
`last_trade_time` read 15:49:35 — exactly a 15-min-delayed ET view of a continuously trading
SPX; a UTC reading would imply a 4 h 16 m delay, which is impossible. The ET→UTC localization
is correct as written — do not revert it.

**Measured offset (T6 probe, 2026-07-15): content delay 15 m 27 s**, identical at both
mid-session probes (15:50:02 ET → header `last_trade_time` 15:34:35; 16:05:02 ET → 15:49:35).
The delay applies to the quote **content** (`last_trade_time`), NOT to the top-level
`timestamp` — that field is the feed's UTC *generation* time and tracked wall clock within
~23–34 s all day (08:25:53 ET morning pull: 34 s behind; both afternoon probes: 23 s).
Computing the delay as `wall_clock − payload_timestamp` measures nothing. Pull times and
misfire-grace windows are now anchored on a *measured* delay, and the shipped values are
confirmed: `decision` pulled 16:00 → content ≈15:44:33 ET, 27 s *before* the asserted 15:45
label — the safe, no-lookahead side; `eod` pulled 16:35 → content ≈16:19:33 ET, past the
16:15 freeze with ~4.5 min margin. **Do not shift the decision pull later** — content would
land *after* the asserted label (lookahead bias). Probe log:
`.superpowers/sdd/t6-probe-2026-07-15.log` (dev checkout).

**Timestamp-convention evidence trail (2026-07-15, morning + off-hours):**

- **08:24 ET browser check:** cboe.com quote page showed `timestamp` `2026-07-15 12:23:19` —
  only sensible as UTC (12:23 UTC = 08:23 EDT). First direct evidence the top-level
  `timestamp` is UTC.
- **08:25:53 ET live pull:** `timestamp` `2026-07-15 12:25:19` — ~34 s behind wall clock,
  i.e. the field is a near-real-time *generation* stamp, NOT itself 15-min delayed. This is
  why the 15-minute delay must be judged against quote content, never against `timestamp`.
- **00:40 ET off-hours pull (informational only — not the T6 result):** `payload_timestamp`
  roughly 47 minutes behind wall clock. The market was closed, so the feed was showing
  Tuesday's (2026-07-14) close state — overnight feed staleness, not the intraday delay
  convention. Off-hours observations mislead; only the mid-session probe counts.

## `iv` is a decimal fraction, not a percent

CBOE reports implied volatility as a **decimal fraction**: `0.1164` means 11.64% — the
opposite convention from Schwab's percent-based `volatility` field the original design was
written against (spec §6.3 lists Schwab's convention as `12.34 = 12.34%`; that table does not
apply to the live CBOE data). Read `iv` values directly as fractions when computing anything
downstream (e.g. `iv * 100` for a human-readable percent).

**Illiquid-contract convention:** CBOE reports `iv = 0.0` for contracts it can't price (deep
wings, no recent trade). The parser (`chain_parser.clean_sentinel(..., zero_is_missing=True)`)
maps that `0.0` to real `NaN` on capture — the stored value is never a literal `0.0` for a
genuinely-unpriced contract. A true `iv` of exactly zero is not a value CBOE actually emits for
a liquid contract, so this rule is unambiguous in practice.

## Open interest is T-1 / once-daily

OI updates once overnight (OCC) and is stable intraday — the `decision` and `eod` snapshots on
the same trading day report **identical** OI, and that value is technically one day in arrears
(the correct no-lookahead value; it is *not* "as of this morning's open"). The monthly audit
(`audit.py::oi_stability_failures`) flags any contract whose OI **differs** across same-day
snapshots as a bug signal — it should never happen.

**Splice-time note:** if purchased historical options data is ever spliced in ahead of this
collector's start date, confirm its OI date convention (as-of prior close vs as-of the
session's open) matches this collector's before joining the two series — a one-day
misalignment at the splice point silently corrupts the combined history.

## 16:35 = close mark, not settlement

The `eod` snapshot captures the frozen quote state after options stop trading (~16:15 ET) —
it is a **close mark**, not an official settlement value. Settlement/exercise pricing (needed
for e.g. AM-settled index options) is a separate sourcing concern that belongs to the future
premium-trading project, not this collector.

## Greeks/IV are CBOE's vendor values

`delta`/`gamma`/`theta`/`vega`/`rho`/`iv`/`theoretical_value` are whatever CBOE's own pricing
model computed — not independently recomputed here. If cross-source consistency is ever
needed, recompute from the captured quotes (`bid`/`ask`/`last`) plus a FRED risk-free rate
(CBOE's payload carries no `interest_rate`/`dividend_yield` header — those columns are stored
`NULL` by design, see `chain_parser.parse_chain`). The fields worth trusting outright are
**quotes + contract identity** (`bid`, `ask`, `strike`, `expiration`, `option_right`), not the
vendor greeks.

## Endpoint has no SLA — fallback ladder

`cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json` is an undocumented public feed
that powers cboe.com itself; there is no published uptime/rate-limit guarantee. Breakage looks
like a **health-check CRITICAL** (missed run) or a schema-drift **WARNING** (field renamed) —
never silence, per the guards in `docs/options/ops.md`.

| Rank | Provider | Status | Notes |
|---|---|---|---|
| Primary | **CBOE delayed-quotes** | Live (this collector) | No auth; SPX is CBOE's own product. |
| Fallback #1 | **Schwab** | Design complete, app registered, **no active token** — zero standing maintenance while shelved | Would need `cboe_client.py` swapped for the researched `schwab_client.py` design (spec §4/§7) — a 7-day refresh-token ritual comes back online if activated. |
| Fallback #2 | **moomoo OpenAPI** | Researched, not implemented | Free real-time OPRA LV1 with a funded account; SPX in scope. OpenD gateway auth can require SMS/CAPTCHA re-verification on a whitelist expiry — not unattended-safe as a primary. |
| Rejected | E*TRADE | Researched | Daily hard token expiry at midnight ET, no refresh mechanism. |
| Rejected | Headless-browser auth (any provider) | — | Credentials on disk + ToS risk — not worth it for a capital-preservation system. |

**Provider-quarantine seam:** a swap only ever touches `src/swingrl/data/options/cboe_client.py`
(`CboeChainClient.get_option_chain`) — `chain_parser`, `store`, `collector`, and the scheduler
are provider-agnostic and unaffected (spec D7). There is no automatic failover between
providers; a swap is a manual code change plus redeploy.

## CBOE historical-candles endpoint (secondary source, not this collector's main job)

`https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json` returns daily
OHLCV — verified live 2026-07-14: `SPY` back to 2004, `_SPX` back to **1975** (12,990 rows,
current same-day). This is documented here, not implemented as a scheduled job:

- Future use: the underlying-history source for the premium-selling environment once it's
  built, and a cross-check for the trader's existing equity daily bars.
- **Never a replacement for the trader's Alpaca/Binance.US sources.** The trader's ingestion
  path is untouched by this collector; Plan B's `data_fingerprint` guard exists specifically to
  keep training data provenance from silently drifting across sources — do not point any
  trader-facing ingestion at this endpoint.

## Real-time chains for the future live premium trader — source-seam note

This collector is explicitly **not** the data source for a future live premium-selling
strategy's real-time decisions — that trader will get real-time chains from its own executing
broker (candidates ranked in the premium-trader spec: Schwab / moomoo / Tradier-IBKR). The
standing plan for that boundary:

- **Training history** for the premium env = this collector's CBOE-delayed captures.
- **Live decisions** at go-live = the executing broker's real-time feed.
- **At go-live, run an overlap capture** — both feeds simultaneously for a period — to measure
  the CBOE-delayed-vs-broker-real-time offset empirically before trusting the trained model's
  calibration against live prices.
- Features built on z-scores/ratios/spreads (spec §6.5) wash out most of the level-shift effect
  from this gap; raw-level features would not.

## Source of truth

| Concern | File |
|---|---|
| Delay/timestamp fields, `iv`/sentinel handling, OSI parsing | `src/swingrl/data/options/chain_parser.py` |
| Late-fire stamping, lookahead-bias guard | `src/swingrl/data/options/collector.py` |
| Postgres/Parquet schema (`raw_header`, `raw_json`) | `src/swingrl/data/options/schema.py`, `src/swingrl/data/options/store.py` |
| Provider config, fallback ladder rationale | `docs/superpowers/specs/2026-07-14-schwab-options-collector-design.md` §17 C1–C3 |
| Provider quarantine seam | `src/swingrl/data/options/cboe_client.py` |

## Changelog

- **2026-07-15** — Initial version (Task 15). Delay convention marked PENDING T6 probe.
- **2026-07-15** — Document `trade_time_utc` ET-localization convention (C2): `last_trade_time`
  is parsed as ET and converted to UTC, inferred from fixtures, pending T6 confirmation.
