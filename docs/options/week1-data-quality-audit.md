# Week-1 Data-Quality Audit Runbook

Manual sanity checks to run once the collector has a real trading week of captures (spec
§10.6). "It didn't crash" is not "it's good" — this data is un-backfillable, so the checks
here catch the quality problems a crash never would, before they cost a year of history.

**Last verified against code:** 2026-07-15

## When to run this

Once, during Task 16 Step 7 (first live week), after both `decision` and `eod` snapshots have
landed for at least a few trading days across all 9 symbols. Re-run informally any time the
automated monthly audit (`options_data_audit`, 1st of month) flags something and you need to
inspect the raw rows behind the alert.

**Relationship to the automated monthly audit:** `src/swingrl/data/options/audit.py` already
runs checks 2–4 below (delta range, crossed markets, OI-null, OI same-day stability)
automatically every month and CRITICAL-alerts on failure. This runbook adds the checks that
are **not** automated — the IV-surface shape (needs a human eyeball), the decision→eod drift
plausibility check, and the delay-offset spot-check — plus gives you the SQL to look at the
automated checks' underlying rows directly.

All queries below run against the live `pg16` container:

```bash
docker exec pg16 psql -U swingrl -d swingrl -c "<query>"
```

Substitute today's date / a recent trading date for `2026-07-15` in every query.

## 1. Reconstruct an SPX IV surface

Pull one full SPX chain and eyeball the smile: IV should be higher away from at-the-money on
both sides, roughly symmetric-ish for near-dated expirations, with no discontinuous jumps or
clusters of missing values in the middle of the strike range.

```sql
SELECT expiration, strike, option_right, iv, delta, open_interest, volume
FROM options_chains
WHERE underlying_symbol = '_SPX'
  AND quote_date = '2026-07-15'
  AND snapshot_label = 'eod'
ORDER BY expiration, strike, option_right;
```

**Pass:** IV forms a recognizable smile/skew per expiration, non-`NULL` for the bulk of
near-the-money strikes; no zero-strike or negative-IV rows (the parser already maps CBOE's
"no IV" sentinel `iv=0.0` to `NULL` — see `docs/options/data-caveats.md` — so a `NULL` here is
expected on illiquid wings, a `0` is not).

## 2. Confirm delta bounds and monotonicity across strikes

```sql
-- Bounds: nothing outside [-1, 1]
SELECT COUNT(*) AS out_of_range
FROM options_chains
WHERE underlying_symbol = '_SPX' AND quote_date = '2026-07-15' AND snapshot_label = 'eod'
  AND (delta < -1.0 OR delta > 1.0);

-- Monotonicity: eyeball one expiration's delta ordered by strike
SELECT strike, option_right, delta
FROM options_chains
WHERE underlying_symbol = '_SPX' AND quote_date = '2026-07-15' AND snapshot_label = 'eod'
  AND expiration = '2026-07-17'
ORDER BY option_right, strike;
```

**Pass:** `out_of_range = 0`. For a fixed expiration, call delta should decrease monotonically
(toward 0) as strike increases; put delta should increase monotonically (toward 0, from
negative) as strike increases. (This exact bound check also runs automatically every month via
`audit.audit_dataframe` — this is the manual first look at the raw numbers behind it.)

## 3. OI populated on liquid names

```sql
SELECT underlying_symbol,
       COUNT(*) AS rows,
       COUNT(open_interest) AS oi_populated,
       ROUND(100.0 * COUNT(open_interest) / NULLIF(COUNT(*), 0), 1) AS pct_populated
FROM options_chains
WHERE quote_date = '2026-07-15' AND snapshot_label = 'eod'
GROUP BY underlying_symbol
ORDER BY underlying_symbol;
```

**Pass:** high-volume names (`_SPX`, `SPY`, `QQQ`) show OI populated on the large majority of
rows, concentrated near the money; thin ETFs (e.g. `VTI`) can legitimately have more `NULL` OI
on deep/far-dated strikes — a near-0% populated rate for a liquid name is the red flag, not a
low rate on a thin one.

## 4. `bid ≤ ask` with plausible spreads

```sql
-- Crossed markets (should be zero or explainable — e.g. stale illiquid quote)
SELECT underlying_symbol, contract_symbol, bid, ask
FROM options_chains
WHERE quote_date = '2026-07-15' AND snapshot_label = 'eod'
  AND bid IS NOT NULL AND ask IS NOT NULL AND ask < bid;

-- Spread distribution (sanity, not a hard pass/fail)
SELECT underlying_symbol,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask - bid)::numeric, 4) AS median_spread,
       ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ask - bid)::numeric, 4) AS p95_spread
FROM options_chains
WHERE quote_date = '2026-07-15' AND snapshot_label = 'eod'
  AND bid IS NOT NULL AND ask IS NOT NULL
GROUP BY underlying_symbol
ORDER BY underlying_symbol;
```

**Pass:** zero (or near-zero, explainable) crossed rows; median spreads tight for `_SPX`/`SPY`
near-the-money, widening sensibly for thin/far strikes.

## 5. Decision → EOD drift is non-trivial and plausible

Same-day `decision` (15:45-labeled) vs `eod` (16:15-labeled) quotes should differ — a feed
that's frozen/stuck would show zero drift everywhere, which is itself a bug signal.

CBOE's payload has no `mark` field (the `options_chains.mark` column is carried by the DDL
for schema compatibility but is never populated by the parser — see
`docs/options/data-caveats.md`), so drift here is computed from the mid-price
`(bid + ask) / 2.0` instead. Both sides of the join require non-`NULL` `bid` and `ask`.

```sql
SELECT d.underlying_symbol, d.contract_symbol, d.strike, d.option_right,
       (d.bid + d.ask) / 2.0 AS decision_mid,
       (e.bid + e.ask) / 2.0 AS eod_mid,
       ((e.bid + e.ask) / 2.0 - (d.bid + d.ask) / 2.0) AS drift
FROM options_chains d
JOIN options_chains e
  ON d.underlying_symbol = e.underlying_symbol
 AND d.quote_date = e.quote_date
 AND d.contract_symbol = e.contract_symbol
WHERE d.quote_date = '2026-07-15'
  AND d.snapshot_label = 'decision' AND e.snapshot_label = 'eod'
  AND d.underlying_symbol = '_SPX'
  AND d.bid IS NOT NULL AND d.ask IS NOT NULL
  AND e.bid IS NOT NULL AND e.ask IS NOT NULL
ORDER BY ABS((e.bid + e.ask) / 2.0 - (d.bid + d.ask) / 2.0) DESC
LIMIT 25;
```

**Pass:** a real, plausible spread of non-zero mid-price drift values (bigger on more volatile
days, smaller on quiet ones) — not all zeros, and not implausibly large jumps that would
suggest a mislabeled snapshot.

## 6. Delay-offset spot-check against the T6 finding

Compares the feed's self-reported timestamp against wall-clock pull time for real captured
days. **This does not replace T6** (`docs/options/data-caveats.md` — the formal delay-convention
measurement is still `PENDING`); it's a week-of-data cross-check once T6 lands, and in the
meantime a place to see whether the pattern looks consistent day to day.

```sql
SELECT quote_date, snapshot_label, pulled_at_utc,
       raw_header->>'payload_timestamp' AS payload_timestamp,
       raw_header->>'late_by_s' AS late_by_s
FROM options_snapshots
WHERE underlying_symbol = '_SPX'
ORDER BY quote_date, snapshot_label;
```

**Pass:** `late_by_s` is `0` (or small) on every row — a non-zero value means the job fired
late, not that the delay convention is wrong. Once T6's measured offset is recorded in
`docs/options/data-caveats.md`, confirm `pulled_at_utc − payload_timestamp` here lands in the
same ballpark across the week; a day that's wildly different from the rest is worth
investigating before trusting that day's `decision` label.

## Source of truth

| Concern | File |
|---|---|
| Automated equivalent of checks 2–4 | `src/swingrl/data/options/audit.py` (`audit_dataframe`, `oi_stability_failures`) |
| `options_chains` / `options_snapshots` DDL | `src/swingrl/data/options/schema.py` |
| Delay/timestamp field definitions | `docs/options/data-caveats.md` |
| Monthly audit schedule + alerting | `docs/options/ops.md` |

## Changelog

- **2026-07-15** — Initial version (Task 15).
- **2026-07-15** — Task-15 review fix: Check 5 used `mark`, which CBOE never sends and the
  parser never populates (permanently `NULL`), so drift could never be non-zero. Rewrote to
  compute mid-price `(bid + ask) / 2.0` instead, with a `bid`/`ask` non-`NULL` filter on both
  sides of the join. Also fixed a stale "checks 2–3" cross-reference (should be 2–4) in two
  places.
