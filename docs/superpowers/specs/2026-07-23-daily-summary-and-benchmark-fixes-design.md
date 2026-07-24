# Daily Summary & Benchmark Accuracy — Fixes Design

**Date:** 2026-07-23 · **Status:** DRAFT — Fable-reviewed (APPROVE-WITH-CHANGES, all findings folded in), pending user approval · **Scope:** live/paper
reporting + alerting + benchmark baseline **data** (era-0 compatible — no training-side or
digest-math changes), bundled with a test/tooling/docstring cleanup batch. One branch, one
PR to `swingrl/2.R-training-redesign`.

## Glossary

| Term | Meaning |
|---|---|
| Daily digest / Daily Summary | The 18:00 ET Discord embed summarizing each env's value, P&L, trades, and Buy & Hold comparison |
| `portfolio_snapshots` | Table with one row per env per cycle: `total_value`, `cash_balance`, `daily_pnl`, `drawdown_pct`, `timestamp` |
| Latest-per-env | The newest snapshot row *for each environment*, not the newest rows overall |
| `DISTINCT ON (col)` | PostgreSQL clause returning the first row per distinct value of `col` under the `ORDER BY` — the repo's idiom for latest-per-group (already used in `V008`) |
| Cooldown key | The string the alerter uses to dedupe repeat alerts within a 30-min window; currently `f"{level}:{title}"` |
| `bypass_suppression` | A `send_alert` flag that skips **only** the consecutive-warnings gate — it does **not** skip the cooldown dedup |
| Partial fill | An auction order that filled for fewer shares than requested |
| `order_id` | The broker's unique order id (Alpaca UUID); primary key of `pending_orders` |
| B&H (Buy & Hold) benchmark | A passive baseline: at each env's start, split the env's cash equally across all its instruments, buy once, hold forever — the yardstick the agent is measured against |
| Origin | The moment an env first deployed capital: **crypto = 2026-07-22**, **equity = 2026-07-23** |
| `baseline_price` | Per-instrument entry price stored in `benchmark_baselines`, used to grow the B&H value as prices move |
| `capital_usd` | Per-env total capital stored on each `benchmark_baselines` row; the digest divides it by the instrument count for equal weighting |
| Epoch reset | The paper-account reset that established the current post-reset cash balances |
| `filterwarnings` | A pytest/pyproject setting that silences a named warning category |
| `LIKE ESCAPE` | A SQL clause that lets `%`/`_` be matched literally instead of as wildcards |
| Dry-run-first | A script whose default mode computes and prints proposed changes but writes nothing; a `--apply` flag is required to write |

## Problem statement

All six items verified against the current tree on 2026-07-23 by three parallel code traces
(exact file:line below). Item numbers match the backlog the user triaged.

1. **[#1 — prod bug] Daily digest drops the equity section and shows the *older* crypto
   snapshot.** `daily_summary_job` selects the last **2** snapshots *overall*
   (`ORDER BY timestamp DESC LIMIT 2`, `scheduler/jobs.py:452-456`) despite its own docstring
   promising "latest per environment." Under the new morning schedule (equity ~09:15 ET;
   crypto 12:05 + 16:05 ET) the two newest rows are **both crypto**, so the equity section is
   silently dropped. Second facet: the build loop (`jobs.py:477-486`) overwrites `crypto_snap`
   on each matching row without breaking, so with two crypto rows ordered newest-first it keeps
   `rows[1]` — the **older** one. Live evidence (per session hand-off): a recent 18:00 ET digest was crypto-only
   and showed `$47.42` (the 12:05 snapshot), not the 16:05 value. Report-only, no trading impact.
2. **[#2 — prod bug] The second same-symbol partial-fill alert is silently suppressed.**
   The cooldown key is `f"{level}:{title}"` (`monitoring/alerter.py:178`) and the cooldown check
   runs **unconditionally**; `bypass_suppression=True` (`jobs.py:1013`) skips only the
   *consecutive* gate. The partial-fill title carries the **symbol only**
   (`jobs.py:1001`), so two partials on the same symbol in one 09:35 run (necessarily two
   distinct `order_id`s — the `pending_orders` PK) collide on the key: the first posts, the
   second is dropped within the 30-min window.
3. **[#5 — benchmark accuracy] The B&H benchmark is anchored on wrong prices and wrong
   capital, so "agent vs B&H" is not a fair measure.** The recorder stamped crypto
   `capital_usd = $47.00` (config value) while the real post-reset crypto cash was **$48.09** —
   a structural **+$1.09** head-start that flatters the agent (ruling
   `progress-rulings.md:42`). More broadly, `baseline_price` was whatever "latest close" the
   recorder happened to grab at record time, not the price at which trading actually started.
   The user requires the benchmark to enter **at the agent's actual first-fill price per
   instrument**, with each env anchored to its true origin date.
4. **[#3 — PR #40 deferred minors]** (a) `"$None notional"` renders when qty *and* notional are
   both None (`jobs.py:997`; unreachable with real Alpaca payloads); (b) a test couples to the
   bare float repr `"$61.1 notional"` (`tests/scheduler/test_jobs.py:1231`); (c) a repo-level
   `websockets.legacy` `DeprecationWarning` (from `alpaca-py`→`websockets`) prints in every run.
5. **[#4 — rulings-branch minors]** (a) `slippage_frac` asserted only non-null
   (`test_jobs.py:1281`); (b) rows-4/6 of the fill-status table have no dedicated tests
   (`jobs.py:1015-1028` else-branch); (d) the row-4 alert title says "unfilled" for a
   partially-filled-no-new-shares case (`jobs.py:1018`); (e) a `%%` printf artifact in a test
   docstring (`tests/execution/test_risk_manager.py:343`); (f) `test_v010_...` is now a misnomer
   after the V011 rename (`tests/data/test_migrations_content.py:1487`); (g) `trade_id LIKE`
   has no `ESCAPE` clause (`jobs.py:1043-1044`; safe today, Alpaca ids are UUIDs).
6. **[#6 — test-infra deferred minors]** 11 test/tooling/config items from
   `progress-test-infra.md` (listed under D18) — none touch production `src/` behavior.

## Decisions

### Group A — production alert/report fixes (drive the trader deploy)

| # | Decision | Rationale |
|---|---|---|
| **D1** (#1) | Replace the LIMIT-2 query with `SELECT DISTINCT ON (environment) environment, total_value, cash_balance, daily_pnl, drawdown_pct, timestamp FROM portfolio_snapshots ORDER BY environment, timestamp DESC` | Returns exactly one **newest** row per env — fixes both facets (equity reappears; newest crypto wins) with the repo's own latest-per-group idiom (`V008`). No build-loop change needed. `timestamp` is now selected to feed D19 staleness marking |
| **D2** (#2) | Append the `order_id` to the partial-fill alert title (`jobs.py:1001`) **and to the sibling close/disposition alert title (`jobs.py:969`), which has the identical same-symbol collision**, so the cooldown key `level:title` becomes unique per broker order | De-collides two same-symbol partials (or closes) in one run; `order_id` is already in scope and shown in the body. Intended side effect: per-order titles disable the 30-min cooldown for these alerts (distinct orders always post; a same-order re-run stays deduped). *(Alternative considered: an explicit `cooldown_key` param on `send_alert` — rejected as a larger change for low-frequency alerts; see Out-of-Scope)* |
| **D3** (#3a) | Only build the `"$X notional requested"` clause when notional is not None; otherwise emit a neutral string (no `"$None"`) | Removes a latent cosmetic edge; behavior-preserving for real payloads |
| **D4** (#4d) | Reword the row-4 else-branch title from "unfilled" to a status-aware phrase (e.g. "still working") matching the spec table; message body already carries true status | Title should not contradict the body; cosmetic, alert-text only |
| **D5** (#4g) | Add `ESCAPE '\'` to the `trade_id LIKE` clause (`jobs.py:1043-1044`); escape `%`/`_`/`\` **in the `order_id` value only**, then append the literal `#%` wildcard (which must stay live to match slice suffixes) | Future-proofs the slice-suffix match against ids containing wildcard chars; inert today |
| **D19** (#1, staleness) | Select each section's snapshot `timestamp` (D1); if its **ET date ≠ today (ET)**, render the section label with an "(as of YYYY-MM-DD)" suffix in `build_daily_summary_embed`. Same-date-but-hours-old rows (e.g. equity's ~09:15 snapshot at the 18:00 digest) are **not** marked | `DISTINCT ON` resurrects an env's newest row regardless of age; a missed env-day would otherwise present stale numbers as today's. Low-noise — only genuinely missed days get flagged |

### Group B — test / tooling / docstring cleanup (merge only, no deploy)

| # | Decision | Rationale |
|---|---|---|
| **D6** (#3b) | Assert on a formatted value (`"$61.10"` via the prod formatter) instead of the bare float repr `"$61.1"` (`test_jobs.py:1231`) | Decouples the test from float-repr quirks |
| **D7** (#3c) | Add a `filterwarnings` ignore for `websockets.legacy` `DeprecationWarning` to `[tool.pytest.ini_options]` in `pyproject.toml` | One entry clears a warning cited ~8× across all three ledgers; dependency noise, not our code |
| **D8** (#4a) | Pin `slippage_frac == pytest.approx(0.001)` (`test_jobs.py:1281`) | Locks sign convention + math instead of mere non-null |
| **D9** (#4b) | Add two tests exercising the unfilled else-branch for fill-status rows 4 and 6 | Closes an incidental-only coverage gap |
| **D10** (#4e) | `%%` → `%` in the test docstring (`test_risk_manager.py:343`) | Removes a printf-escape artifact |
| **D11** (#4f) | Rename `test_v010_schema_version_is_10` to a superseded-convention name (e.g. `test_v010_in_migration_ledger`) | Name became a misnomer after the newest-version invariant moved to the V011 test |
| **D12** (#6) | Apply the 11 test-infra minors under D18 | House-keeping the test harness the prior branch deferred |

### Group C — B&H benchmark re-anchor (data-only, gated live apply, **no trader deploy**)

| # | Decision | Rationale |
|---|---|---|
| **D13** (#5, model) | **Model A — equal-weight passive index.** Each env's B&H holds *every* instrument in the env, equal $ weight, entered once and held. **Rejected: Model B** (mirror the agent's exact first book) — it would hand the agent's *instrument selection* to the benchmark for free and understate the agent's edge; a benchmark must owe nothing to the agent's choices | A benchmark is a dumb neutral baseline; Model A measures the full skill (selection + timing + management). Keeps the digest math untouched → data-only fix |
| **D14** (#5, prices) | `baseline_price[instrument]` = the price of the **earliest** `trades` row (by `timestamp`) for that `(environment, symbol)` with `side='buy'` on the env's origin ET date — the real opening print, not a later derived-reconciliation slice. Every instrument was traded on day one (user-confirmed), so **no market-close fallback is needed**; if the dry-run finds any instrument without such a fill, it **stops and reports** rather than guessing. Exact values confirmed at the read-only dry-run | Enters the benchmark at the exact price/time trading started — the user's fairness requirement; earliest slice = the market entry, later slices carry confirmation-derived prices |
| **D15** (#5, dates) | `baseline_date` = crypto **2026-07-22**, equity **2026-07-23** (per-env origin). `baseline_date` is recorded but not used by the digest math, so mixed dates are safe | Each env is anchored to when *it* started |
| **D16** (#5, capital) | `capital_usd` = the env's real starting capital at origin: crypto **$48.09** (reconciled against the reset snapshot at dry-run); equity = `total_value` of the **earliest** `portfolio_snapshots` row for `environment='equity'` on origin ET date **2026-07-23** (should equal its `cash_balance` pre-deployment — the dry-run aborts if they differ by > $0.01). Same value on every row of an env (the digest divides by instrument count) | Removes the +$1.09 anchor bias and sets the equity anchor to the actual capital being benchmarked |
| **D17** (#5, delivery) | Deliver as a **tested, dry-run-first maintenance script** (`scripts/reanchor_benchmark_baselines.py`) that: (1) reads first-fill prices + origin snapshots from `trades`/`portfolio_snapshots`; (2) default dry-run prints a current-vs-proposed diff, writing nothing; (3) `--apply` first **writes the current rows as restore-SQL to a timestamped backup file** (not just stdout), then **upserts** on `(environment, symbol)`, then **asserts the env's final row set equals exactly the origin-fill instrument set** — aborting on any gap or extra row, since a wrong row count silently corrupts the equal-weight divisor. Tests are DB-free via a fake gateway, mirroring `tests/data/test_record_benchmark_baselines.py`. The **live `--apply` is a separate, explicitly-approved step** run after merge; rollback = replay the backup file | Auditable, reversible, TDD-compliant live-DB write; the row-set invariant protects the very divisor this fix exists to correct |

### Group D — test-infra minors (D18 detail)

Actionable items (each becomes a plan task; all test/tooling/config unless noted):

1. Extract `is_db_test(item)` helper into `tests/db_marker.py`; call it from the conftest hook (`tests/conftest.py:98-100`).
2. Add a test exercising the marker-derivation `@cache` hit/miss.
3. Add a session finalizer that closes the persistent cleanup connection (`tests/fixtures/db_cleanup.py`).
4. Add a drift-guard comment (or dedup) for the inner-conftest glue duplicated in `tests/test_wipe_conditionality.py`.
5. Catch `psycopg.ProgrammingError` (not only `OperationalError`) in the preflight (`tests/conftest.py:75`) → clean `pytest.exit`.
6. Catch psycopg errors from `activate_isolated_db()` (`tests/conftest.py:53`) → `pytest.exit(returncode=2)` instead of a raw traceback.
7. Harden the schema-preflight PK detection to parse an actual `PRIMARY KEY` constraint rather than a substring match (`tests/fixtures/schema_preflight.py:33`).
8. Add a test for the create-from-absent canonical-DDL path (`tests/fixtures/schema_preflight.py`).
9. Use explicit named columns in the `test_model_metadata_table_created` INSERT (`tests/agents/test_validation.py:148`).
10. Don't cache empty-string in `_yaml_fallback_url` on transient `load_config` failure (`tests/db_guard.py`).
11. **[prod src, behaviorally inert]** Guard the numpy `RuntimeWarning` at `src/swingrl/features/technical.py:76`.

Documented-only (no code): the force-OUT marker asymmetry (per-spec) and the "stale safe URL exits 2 at configure" behavior note.

## Testing strategy

- **TDD throughout:** every behavioral change (D1, D2, D3, D4, D5, D9, D14–D17) gets a
  **RED** test committed before the GREEN implementation, per project convention.
- **D1 RED tests** (`TestDailySummaryJob`, `tests/scheduler/test_jobs.py`): (a) seed three
  snapshots — one equity **older** than two crypto rows — run `daily_summary_job`, assert the
  embed **contains the Equity section** *and* that the crypto figures are the **newest** crypto
  row; (b) crypto-only DB → equity section correctly absent, no crash; (c) empty table → the
  `if not rows` early return. These are `db`-marked (real DB); under `env -u DATABASE_URL` they
  hit an explicit `pytest.skip` (visible, not silent), so they run against a scratch DB.
- **D19 RED test**: seed an equity snapshot dated *yesterday* (ET) + crypto today → assert the
  equity label carries the "(as of …)" suffix and the crypto label does not.
- **D2 RED test** (`TestEquityFillConfirmationJob`, after the existing
  `test_partial_fill_alert_is_per_symbol_and_unsuppressed`): seed two **same-symbol** pending
  orders with distinct `order_id`s, run the job, collect the two "PARTIALLY filled" calls, and
  assert the two **titles are distinct** (each carries its `order_id`). RED today (titles
  identical). *(Optional companion at the alerter level: two identical-title
  `bypass_suppression=True` sends → assert `mock_post.call_count == 2`.)*
- **D17 tests**: assert the script computes the correct `(baseline_price, baseline_date,
  capital_usd)` rows from fake `trades` (D14 earliest-slice rule, D16 capital rule); that dry-run
  writes nothing; that `--apply` writes the backup file *then* upserts; that the **row-set
  invariant** aborts on a gap/extra row; and that a missing origin-day fill aborts with a clear
  error (D14 guard).
- **Gate before push** (production `src/` changed): FAST lane (<2 min) → lockfile ruff
  (`uv run ruff check src/ tests/`) → mypy → full suite `-n 4` against a scratch DB → homelab CI
  literal `=== CI PASSED ===`. Preflight a cheap targeted run before the ~18-min suite; after any
  failure, `--lf` first; 0 failures required before push.

## Out of scope (recorded, deliberate)

- **#4 `pending_order_slice_price_fallback` "unreachable log"** — already resolved by commit
  **b72f3e5** (explicit `if slice_dollars <= 0:` branch). Excluded, not deferred.
- **#5 Model B** (mirror the agent's actual first book) — rejected per D13; would contaminate
  the benchmark with the agent's selection.
- **Redesigning `record_benchmark_baselines.py`** to support `--capital`/per-env/per-date — not
  worth it for a one-time correction; the targeted D17 script does the job.
- **The alerter `cooldown_key` parameter** alternative to D2 — larger surface, not justified for
  a once-daily low-volume alert.
- **Crypto B&H structural-bias "leave-as-is" ruling (2026-07-22)** — explicitly reversed by the
  user this session; now fixed via D13–D17.

## Packaging & deploy

- **Branch:** `swingrl/2.R-H-daily-summary-and-benchmark` off `origin/swingrl/2.R-training-redesign`.
- **PR:** to `swingrl/2.R-training-redesign` (never `main`).
- **Trader deploy** (path B, pin-first to preserve rollback; market-safe window outside
  15:30–16:45 ET, between crypto cycles) is driven **only by the Group A ✅ items** (D1–D5, D19)
  plus the inert `technical.py:76` guard. Group B/D are merge-only (no runtime change). Deploy needs
  its **own explicit user approval**.
- **#5 live apply** is a **separate gated operational step** after merge: run the D17 script
  `--apply` (crypto and equity — both origins are now in the past; the 09:15 ET equity cycle
  already ran today), with the printed backup captured first. A `benchmark_baselines` write is
  low-risk (affects only the digest, never trading), so its timing is flexible.

## Verified vs assumed

- **Verified (code read 2026-07-23, three parallel traces):** the D1 defect mechanics
  (`jobs.py:452-486`); the D2 cooldown-key derivation (`alerter.py:178`) and
  `bypass_suppression` scope; every file:line in the problem statement still exists in the tree;
  `DISTINCT ON` is repo-idiomatic (`V008`); the digest math uses `baseline_price` + `capital_usd`
  + latest close only, so D13–D16 need **no** digest-code change; the recorder's inability to
  produce per-env dates / fill-price baselines (hence the D17 script); the already-fixed status
  of the excluded #4 item (b72f3e5).
- **Assumed / to confirm at the #5 dry-run (read-only prod queries, user-gated):** the exact
  crypto starting cash **$48.09** (against the reset snapshot); the equity starting cash and the
  per-instrument first-fill prices (from `trades`); that every instrument has an origin-day buy
  fill (user-recalled "bought some of everything" — the D14 guard aborts if not). Live evidence
  that today's equity morning cycle produced fills is expected present but confirmed at dry-run.

**Confidence:** high on D1–D12, D18–D19 (defect mechanics + fixes verified against the tree and
adversarially reviewed by a Fable pass — APPROVE-WITH-CHANGES, all findings folded in); high on
the D13 model choice (user-decided, sound benchmarking rationale); medium-high on D14–D17 exact
numbers, which are read-only-verified at the gated dry-run before any write.
