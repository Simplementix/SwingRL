# EOD Option-Chain Data Collector — Implementation Plan (CBOE primary)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status: RESTRUCTURED 2026-07-14 (user-approved, master-sequence reconciliation session).**
> Primary provider switched **Schwab → CBOE** (spec amendments C1–C4). Consequences: Tasks 3–4
> (TokenManager, re-auth CLI) **REMOVED** (tombstoned below — no auth exists to manage); Task 5
> is a CBOE HTTP client; Task 6's manual gate shrinks to fixture capture + delay-convention
> measurement (no OAuth); Task 7's parser maps CBOE fields + parses the OSI symbol; Tasks
> 13–16 lose all token machinery and gain the approved restart-resilience set (per-label
> misfire grace, lookback health check, boot self-check, pinned image tag). Container renamed
> **`swingrl-collector`** (it will also host Plan A Task 11's calendar-ingest jobs, and later
> a scheduled OHLCV refresh). Original task numbers are KEPT (tombstones preserve
> cross-references). Filename keeps "schwab" for link stability; the provider history lives in
> the spec's §17.

**Goal:** Reliably capture the full option chain for `$SPX`-class index options + the 8 equity ETFs twice per trading day (the ~15:45 ET decision state, pulled ~16:00; the frozen close, pulled 16:35) into durable, idempotent, resumable storage (Parquet → Postgres), running in a standalone always-on container that never touches the live trader — and that itself survives the many restarts Plans A/B will impose on this host.

**Architecture:** Three layers. **Layer 1** is a reusable chain-source library (`src/swingrl/data/options/`): a thin CBOE endpoint client, chain parser, and store, with the provider quarantined behind the client interface (Schwab = shelved fallback #1, moomoo = fallback #2 — spec §17 C2). **Layer 2** is the EOD collector orchestration (`collector.py`) — per-symbol fetch→parse→store with isolation, silent-corruption guards, and alert routing. **Layer 3** is the container entrypoint (`scripts/collector_main.py`) — its own APScheduler, its own jobstore, config-driven jobs, boot-time self-checks.

**Tech Stack:** Python 3.11, `httpx` (already a dependency — CBOE endpoint needs no SDK and no auth), pandas + pyarrow (Parquet), psycopg 3 + `psycopg_pool` (Postgres 16, JSONB + monthly range partitions), APScheduler `BackgroundScheduler` (SQLAlchemy sqlite jobstore), `exchange_calendars` (XNYS), Pydantic v2 (config), structlog (logging), `swingrl_retry` (retry), the existing `Alerter` (Discord), Docker Compose.

---

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from CLAUDE.md and the spec.

- **Python 3.11 only.** `from __future__ import annotations` as the first line of every module.
- **Type hints on all function signatures** (`disallow_untyped_defs = true`). snake_case for all identifiers.
- **`pathlib.Path` for all file operations.** Never `os.path` or raw string paths.
- **Line length 100** (ruff + black). Imports: `from __future__` first, then stdlib, third-party, first-party — **absolute imports only** (never relative in `src/swingrl/`).
- **UTC internally.** All timestamps stored/computed in UTC. Convert to ET (`America/New_York`) only for display output and Discord alerts.
- **structlog only.** `configure_logging(...)` once at entrypoint; `log = structlog.get_logger(__name__)` per module; context as kwargs, never f-strings. Log the error **before** raising.
- **Typed errors.** Raise `DataError` for fetch/parse/empty failures; `ConfigError` for bad config. Never bare `Exception`/`ValueError`. (`src/swingrl/utils/exceptions.py`.)
- **Config-driven, nothing hardcoded.** No hardcoded ticker symbols, API keys, file paths, or dollar amounts. Obtain config via `load_config(path)`; never call `yaml.safe_load()` in business logic. The 8 ETFs come from `config.equity.symbols`; index underlyings from `config.options_collector.index_symbols`.
- **No secrets in this subsystem (C2).** The CBOE endpoint is unauthenticated — no API keys, no token file, no `secrets/` mount. (If the Schwab fallback is ever activated, its shelved secrets rules re-apply.)
- **TDD, tests-first.** Write the failing test, commit RED, then GREEN implementation. Test files `tests/test_<module>.py`, functions `test_<behavior>`, docstring `"""OPT-<id>: what is tested (spec §N)."""`, fixtures from `tests/conftest.py`. Run `uv run pytest tests/ -v`.
- **Never skip pre-commit.** Fix the hook failure; never pass `--no-verify`.
- **A30 compliance.** The collector writes only to its own tables (`options_snapshots`, `options_chains`) and `data/options_eod/`. It never writes to `models/active/` or any shared trader table. Schema changes are **additive only** (new tables). The container is a separate always-on service — rebuilding/restarting it never touches the trader.
- **Quiet window (C4).** No recreation of `swingrl-collector`, and no homelab CI runs spanning **15:30–16:45 ET** on trading days. `ci-homelab.sh`'s cleanup must be dev-compose-scoped **before** Task 16's deploy (Wave-0 prerequisite in the master sequence).

---

## Glossary (no undefined shorthand)

Carried from spec §1, plus plan-specific terms. Read this before the tasks.

| Term | Plain meaning |
|---|---|
| **Option chain** | Full list of option contracts for one underlying — every strike × expiration × call/put — with prices and analytics. |
| **Underlying** | The thing the option is on (`SPY`, `$SPX`). |
| **`_SPX`** | CBOE's URL symbol for S&P 500 **index** options (European, cash-settled): `…/options/_SPX.json`. Verified live 2026-07-14 — one file carries both SPX and SPXW roots. Stored as `$SPX`-style `underlying_symbol`? No — stored verbatim as configured (`_SPX`), directory name strips the underscore. |
| **Greeks** | Risk sensitivities: delta, gamma, theta, vega, rho. |
| **IV** | Implied volatility. **CBOE returns it as a decimal fraction** (0.1164 = 11.64%) — opposite of Schwab's percent convention; the column is documented as a fraction (units-in-names discipline). |
| **OI** | Open interest — contracts currently outstanding. |
| **DTE** | Days to expiration — **derived** from the OSI expiry vs quote_date (CBOE sends no dte field). |
| **OSI id** | The standardized option contract symbol (e.g. `SPXW260724P07840000`) — the natural key part, and on CBOE the **only** source of root/expiry/right/strike (parsed, not provided as fields). |
| **EOD** | End of day. |
| **Decision snapshot** | The ~**15:45 ET market state** — the moment the future premium agent trades. **Pulled at ~16:00 ET** because the feed is 15-min delayed (C3). **Un-backfillable.** |
| **EOD snapshot** | The frozen post-close chain, pulled at **16:35 ET** (options freeze by 16:15, so the delayed view at 16:35 IS the close; decision D3 survives the provider swap). |
| **Market time vs pull time** | `market_time_et` = the moment the data represents (from config + payload timestamp); `pulled_at_utc` = fetch wall clock. Stored separately, never conflated (C3 provenance honesty). |
| **`late_by_s`** | Provenance field: how late past its scheduled pull a snapshot job fired. Non-zero on a decision snapshot ⇒ WARNING (lookahead-bias guard). |
| **Snapshot** | One pull of one symbol's full chain at one scheduled time on one trading day. |
| **Grain** | One stored row per `(underlying_symbol, quote_date, snapshot_label, contract_symbol)`. |
| **Resume unit** | One Parquet file per `(symbol, date, snapshot_label)` — the smallest thing we skip-if-present and reconcile. |
| **Parquet** | Compact columnar **file format** (no locks). The durable first write. |
| **Postgres / pg16** | The project's PostgreSQL 16 DB. MVCC — concurrent-safe. |
| **JSONB** | Postgres binary-JSON column type. Stores raw payload; queryable. psycopg 3 writes it by wrapping a dict in `psycopg.types.json.Jsonb(...)`. |
| **Range partition** | Splitting one logical table into physical child tables by a range of a key (here: monthly on `quote_date`). Child = `options_chains_YYYY_MM`. |
| **`PARTITION OF`** | Postgres DDL that attaches a child table to a partitioned parent for a value range. |
| **`ON CONFLICT DO NOTHING`** | Postgres idempotent insert — a row whose primary key already exists is silently skipped. |
| **Reconcile** | At run start, load any Parquet file whose `(symbol, date, snapshot)` has no parent DB row → self-heals DB outages. |
| **CBOE delayed-quotes endpoint** | `https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json` — unauthenticated CDN feed powering cboe.com's quote pages. Full chain, one GET. **No SLA** (fallback ladder covers it, spec §17 C2). |
| **Delay convention** | The feed is ~15-min delayed; the exact offset (payload `timestamp` vs wall clock) is measured at T6 before the decision label is trusted. |
| **GTH** | Global trading hours — SPX options trade overnight; the endpoint serves live-ish quotes even at night (verified). Irrelevant to our two snapshots; documented for future use. |
| **Contract-count drift** | CBOE has no truncation flag; the partial-chain guard is a row-count comparison against the previous same-label snapshot (a >configured-fraction drop ⇒ WARNING). |
| **Schema drift** | The provider silently adding/renaming/removing response fields; can null out our typed columns with no error. Guarded by `EXPECTED_CONTRACT_FIELDS` against the raw payload. |
| **NBBO** | National Best Bid and Offer. What we capture — **not** your fill price. |
| **APScheduler** | The Python job scheduler the live trader already uses. `BackgroundScheduler` = non-blocking. |
| **Jobstore** | Where APScheduler persists scheduled jobs (a sqlite file). The collector has its **own**, separate from the trader's. |
| **XNYS** | The NYSE calendar in `exchange_calendars` — trading days, holidays, early closes. |
| **Early close** | NYSE half-day (13:00 ET close). Recorded as `is_early_close` in provenance. |
| **A30** | Deploy-isolation rule: never rebuild/restart the live trader; additive-only DB migrations while it runs. |
| **TDD / RED / GREEN** | Test-Driven Development: write a failing test (RED), commit it, then write minimal code to pass (GREEN). |
| **3-2-1 backup** | 3 copies, 2 media types, 1 offsite. |

---

## Requirement IDs (for test docstrings)

Tests use `"""OPT-<id>: <behavior> (spec §N)."""`. IDs group by module:

| ID prefix | Module / concern | Spec §|
|---|---|---|
| `OPT-CFG` | `OptionsCollectorConfig` schema | §5, §17 C3/C4 |
| `OPT-CLIENT` | `cboe_client.get_option_chain` | §17 C1, §10.5 |
| `OPT-PARSE` | `chain_parser` (CBOE payload + OSI parsing) | §6.3, §17 C1 |
| `OPT-STORE` | `store` Parquet + Postgres + reconcile | §8 |
| `OPT-SCHEMA` | Postgres tables + partitions + migration | §8.2 |
| `OPT-COLLECT` | `collector` orchestration + guards | §6, §10, §17 C3 |
| `OPT-AUDIT` | `audit` data-quality | §10.6 |
| `OPT-SCHED` | scheduler + calendar guards + boot self-check | §9, §17 C4 |

(`OPT-AUTH` / `OPT-CLI` retired with Tasks 3–4 — no auth exists in the CBOE design.)

---

## File Structure

**New package** `src/swingrl/data/options/` (one responsibility per file):

| File | Responsibility |
|---|---|
| `__init__.py` | Package marker; re-exports the public surface. |
| `market_calendar.py` | `is_trading_day(date)`, `is_early_close(date)` over XNYS. |
| `cboe_client.py` | `CboeChainClient.get_option_chain(symbol)`: unauthenticated GET, retry, payload validation, raw dict out. Provider-quarantine seam (fallbacks swap this module only). |
| `chain_parser.py` | `parse_chain(...) -> ParsedChain`: raw dict → typed DataFrame + `raw_json` + header; OSI-symbol parsing. |
| `schema.py` | `ensure_options_schema(conn)`, `ensure_monthly_partition(conn, quote_date)`; DDL constants. |
| `store.py` | `OptionsStore`: atomic Parquet write, skip-existing, Postgres JSONB sync, reconcile. |
| `collector.py` | `OptionsCollector.run_snapshot(label)`: per-symbol loop, isolation, guards, alert routing, late-fire stamping. |
| `audit.py` | `run_data_quality_audit(...)`: greeks/OI/spread sanity over a trailing window. |

**New scripts:**

| File | Responsibility |
|---|---|
| `scripts/capture_chain_fixture.py` | One-shot: pull one real chain, truncate to a small sample, save as the pinning fixture (T6 — no sanitization needed, the payload is public data with no account fields). |
| `scripts/collector_main.py` | Container entrypoint: configure logging, build Alerter, own scheduler, register config-driven jobs, run boot self-check, block. (Plan A Task 11's calendar jobs register here too, per its amendment.) |
| `scripts/migrations/add_options_capture_tables.py` | Standalone additive migration for the one-time live apply (outside the Stage-2.R ledger by decision D1/R-3). |

**Modified:** `src/swingrl/config/schema.py` (add `OptionsCollectorConfig`), `config/swingrl.yaml` (add `options_collector:` block), `docker-compose.yml` (new `swingrl-collector` service), `scripts/ci-homelab.sh` (Wave-0 prerequisite: dev-compose-scoped cleanup). No new dependency (`httpx` already present), no `.env`/secrets changes, no `.gitignore` secrets rule (only `data/options_eod/` coverage confirmed).

**Design decisions locked with the user (deviations from a literal reading of the spec):**
- **D1 — "V011" migration is self-contained.** There is no numbered migration ledger in the repo. We implement V011 as (a) a standalone additive script `scripts/migrations/add_options_capture_tables.py` (mirrors the existing `scripts/migrations/add_cps_columns.py`) for the one-time live apply, **and** (b) `schema.ensure_options_schema(conn)` called at collector startup (idempotent `CREATE TABLE IF NOT EXISTS`). The trader's `postgres_schema.py` is **not** touched — options tables live only in this subsystem. This keeps A30 isolation clean.
- **D2 (amended 2026-07-14) — parser is TDD'd against REAL fixtures from day one.** The CBOE endpoint needs no auth, and full real payloads for `_SPX`/`SPY` were already pulled live during the 2026-07-14 planning session — T6 commits bounded samples of them as fixtures before T7 starts. The hand-authored-fixture workaround (a Schwab-era necessity) is dead; T16 keeps only the re-verification against a fresh trading-day capture.
- **D3 — EOD snapshot is 16:35, not 16:30.** The spec says 16:30, but 16:30 sits *exactly* on the 15-min-delay boundary (options freeze at 16:15; a delayed feed first shows that close at 16:30:00 sharp). Firing on the edge risks catching a pre-close quote if clock skew / fetch time / an occasional >15-min delay pushes it early. 16:35 clears the boundary with a 5-minute buffer at zero cost (values are frozen after ~16:15). Chosen from day one so the forward-captured history has no self-inflicted time-of-day seam. The 15:45 `decision` snapshot is unaffected (intraday, market open). We deliberately did **not** add a 16:00 "close" snapshot — the splice-alignment case for it largely washes out because the premium pipeline feeds z-scores/ratios/spreads, not raw levels, and its only independent benefit (simultaneous underlying+options for IV inversion) is small and helps only under real-time entitlement.
- **D4 — snapshot jobs are config-driven.** The scheduler registers **one job per entry in `options_collector.snapshots`** (T13), and the label validator accepts `{open, decision, close, eod}` — so revising the number/times of snapshots (e.g. adding 16:00 later) is a YAML edit, not a code change ("nothing hardcoded").
- **D5 — `postgres_store_raw_json` flag.** `raw_json` is always kept in Parquet (the durable archive). Its copy in Postgres JSONB — the fastest-growing, least-queried storage — is behind a config flag (default **on**, spec-compliant), so it can be flipped off at first-run once real GB/day is known. `options_chains.raw_json` is therefore nullable. If flipped off later, reconcile can re-load from Parquet.
- **D6 — OI is captured but understood as T-1/once-daily.** Open interest updates once overnight (OCC) and is stable intraday, so it is identical across same-day snapshots and one day in arrears — which is the correct no-lookahead value. The audit adds an invariant that flags any same-day OI *difference* (a bug signal), and a splice-time note to verify the OI date-convention matches the purchased history on the overlap.
- **D7 (2026-07-14) — provider = CBOE; fallback ladder documented.** Spec §17 C1/C2: CBOE delayed-quotes primary (verified live: full SPX+SPXW + all ETF chains, greeks/IV/OI/sizes, no auth); Schwab = shelved fallback #1 (design retained, app registered, no token); moomoo = fallback #2; E*TRADE and headless-browser auth rejected. The client module is the only swap point.
- **D8 (2026-07-14) — pull time ≠ market time; decision pulled ~16:00.** Spec §17 C3: the feed is ~15-min delayed, so the decision snapshot is pulled at ~16:00 ET to capture the ~15:45 market state. `market_time_et` (per snapshot config) and `pulled_at_utc` are stored separately; the payload's own `timestamp` is recorded; a late decision fire past its grace stamps `late_by_s` and WARNs — never a silent mislabel (lookahead-bias guard). Exact delay convention measured at T6 before the label is trusted.
- **D9 (2026-07-14) — restart-resilience set.** Spec §17 C4: per-label config-driven misfire grace (decision short ~600–900 s, eod long ~4–6 h); health check with lookback over the last N trading days; boot-time self-check (reconcile + lookback health check); pinned `swingrl-collector` image tag; 15:30–16:45 ET quiet window; `ci-homelab.sh` cleanup fix as a Wave-0 prerequisite.
- **D10 (2026-07-14) — the container is the market-data plane.** Named `swingrl-collector` because it will host more than options: Plan A Task 11's FRED calendar jobs register in its scheduler (per that task's amendment), and a scheduled OHLCV refresh (existing Alpaca/Binance ingestors, sources unchanged) is a named follow-up. The CBOE historical-candles endpoint (`…/charts/historical/{symbol}.json`, `_SPX` daily to 1975) is documented in data-caveats as the future premium-env underlying-history source + a dailies cross-check — zero tasks now (backfillable, no urgency).

---

## Task Dependency Order

```
T1 (setup) → T2 (config) → T5 (CBOE client)
   → T6 (commit real fixtures; 🛑 delay-convention measurement needs one trading day)
   → T7 (parser) → T8 (store: Parquet) → T9 (schema) → T10 (store: Postgres+reconcile)
   → T11 (collector) → T12 (audit) → T13 (scheduler) → T14 (docker) → T15 (docs)
   → T16 🛑 manual: CI-cleanup fix verified + homelab CI + live migration + deploy
        + first live run + offsite backup

(T3, T4 removed — tombstones below. Numbering preserved for cross-references.)
```

T6's fixture-commit step can happen immediately (payloads already captured live 2026-07-14);
only its delay-convention measurement waits for a trading-day window, and T7+ need not wait
for that measurement (it gates T13's final pull-time config, not the parser). 🛑 =
human-in-the-loop gate.

---

### Task 1: Project setup — package scaffold (RESTRUCTURED 2026-07-14: no dependency, no secrets)

**Files:**
- Create: `data/options_eod/.gitkeep`
- Create: `src/swingrl/data/options/__init__.py`

**Interfaces:**
- Consumes: nothing. (`httpx>=0.27` is already a project dependency — verified in
  `pyproject.toml:45`; the CBOE endpoint needs no SDK, no keys, no token, no `secrets/`.)
- Produces: the `swingrl.data.options` package namespace.

- [ ] **Step 1: Scaffold**

```bash
mkdir -p data/options_eod
touch data/options_eod/.gitkeep src/swingrl/data/options/__init__.py
```

- [ ] **Step 2: Confirm the data dir is gitignored and httpx imports**

Run: `git check-ignore data/options_eod/SPX && uv run python -c "import httpx; print(httpx.__version__)"`
Expected: the path prints (ignored) and an httpx version ≥ 0.27 prints.

- [ ] **Step 3: Commit**

```bash
git add data/options_eod/.gitkeep src/swingrl/data/options/__init__.py
git commit -m "chore(options): package scaffold for CBOE chain collector (no new deps)"
```

---

### Task 2: `OptionsCollectorConfig` schema + YAML block (RESTRUCTURED 2026-07-14)

**Files:**
- Modify: `src/swingrl/config/schema.py` (add the config models + attach to `SwingRLConfig`)
- Modify: `config/swingrl.yaml` (add the `options_collector:` block)
- Test: `tests/test_options_config.py`

**Interfaces:**
- Consumes: `load_config()`, `SwingRLConfig`, `EquityConfig.symbols` (existing, `src/swingrl/config/schema.py`).
- Produces:
  - `OptionsCollectorConfig(BaseModel)` with nested `OptionsSnapshotConfig`, `OptionsIntegrityConfig`, `OptionsBackupConfig`. (No auth model — C2; no chain-request model — the CBOE endpoint takes no parameters, full chain always.)
  - Attribute path `config.options_collector.*`.
  - Fields relied on downstream: `enabled: bool`, `provider: str = "cboe"`,
    `endpoint_url_template: str` (default `https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json`),
    `index_symbols: list[str] = ["_SPX"]`, `include_equity_symbols: bool`,
    `output_dir: str = "data/options_eod/cboe"`, `schema_version: str`,
    `snapshots: list[OptionsSnapshotConfig]` — each **`label`, `market_time_et`
    (the moment the data represents), `pull_time_et` (when the job fires), and
    `misfire_grace_s` (D8/D9 per-label grace)** — defaults:
    `{decision, 15:45, 16:00, 900}` and `{eod, 16:15, 16:35, 18000}`,
    `request_timeout_s: float`, `rate_limit_per_sec: float`,
    `health_check_time_et: str`, `health_lookback_days: int = 3` (D9),
    `apscheduler_db_path: str`, `postgres_store_raw_json: bool`,
    `integrity.contract_count_drop_warn_frac: float = 0.5` (contract-count drift guard) /
    `integrity.audit_day_of_month` / `integrity.audit_time_et`,
    `backup.enabled/rclone_remote/time_et`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options_config.py
from __future__ import annotations

import pytest

from swingrl.config.schema import (
    OptionsCollectorConfig,
    OptionsSnapshotConfig,
    load_config,
)


def test_options_collector_defaults_present() -> None:
    """OPT-CFG-1: OptionsCollectorConfig has spec §17 C1 defaults."""
    cfg = OptionsCollectorConfig()
    assert cfg.enabled is True
    assert cfg.provider == "cboe"
    assert cfg.endpoint_url_template == (
        "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
    )
    assert cfg.index_symbols == ["_SPX"]
    assert cfg.include_equity_symbols is True
    assert cfg.output_dir == "data/options_eod/cboe"
    assert cfg.schema_version == "v1"
    assert cfg.apscheduler_db_path == "db/apscheduler_options.sqlite"
    assert cfg.postgres_store_raw_json is True
    assert cfg.health_lookback_days == 3


def test_options_snapshots_pull_vs_market_time() -> None:
    """OPT-CFG-2: decision pulls 16:00 for the 15:45 state; eod pulls 16:35 (D8)."""
    cfg = OptionsCollectorConfig()
    rows = [(s.label, s.market_time_et, s.pull_time_et, s.misfire_grace_s) for s in cfg.snapshots]
    assert rows == [("decision", "15:45", "16:00", 900), ("eod", "16:15", "16:35", 18000)]


def test_options_config_attached_to_root() -> None:
    """OPT-CFG-4: load_config exposes options_collector (spec §5)."""
    cfg = load_config("config/swingrl.yaml")
    assert cfg.options_collector.enabled is True
    assert cfg.options_collector.integrity.contract_count_drop_warn_frac == 0.5


def test_options_config_env_override() -> None:
    """OPT-CFG-5: nested env override works (spec §5)."""
    import os

    os.environ["SWINGRL_OPTIONS_COLLECTOR__ENABLED"] = "false"
    try:
        cfg = load_config("config/swingrl.yaml")
        assert cfg.options_collector.enabled is False
    finally:
        del os.environ["SWINGRL_OPTIONS_COLLECTOR__ENABLED"]


def test_snapshot_label_must_be_known() -> None:
    """OPT-CFG-6: snapshot label validated against the known set (spec §6.1, D4)."""
    from swingrl.utils.exceptions import ConfigError

    with pytest.raises(ConfigError):
        OptionsSnapshotConfig(label="lunchtime", market_time_et="12:00", pull_time_et="12:15")


def test_snapshot_grace_positive_and_pull_not_before_market_time() -> None:
    """OPT-CFG-7: misfire grace > 0; pull_time_et >= market_time_et (delayed feed, D8)."""
    from swingrl.utils.exceptions import ConfigError

    with pytest.raises(ConfigError):
        OptionsSnapshotConfig(label="decision", market_time_et="15:45",
                              pull_time_et="15:30", misfire_grace_s=900)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'OptionsCollectorConfig'`.

- [ ] **Step 3: Add the config models**

In `src/swingrl/config/schema.py`, near the other sub-config models (e.g. after `SchedulerConfig`), add:

```python
class OptionsSnapshotConfig(BaseModel):
    """One scheduled snapshot: label, the market moment it represents, the pull time,
    and its misfire grace (D8/D9 — pull time trails market time on a delayed feed)."""

    label: str = Field(default="decision")
    market_time_et: str = Field(default="15:45")   # the moment the data represents
    pull_time_et: str = Field(default="16:00")     # when the cron fires (delay-adjusted)
    misfire_grace_s: int = Field(default=900, gt=0)

    @field_validator("label")
    @classmethod
    def label_known(cls, v: str) -> str:
        """Recognized snapshot labels (config-driven; add times via YAML, not code)."""
        allowed = {"open", "decision", "close", "eod"}
        if v not in allowed:
            raise ConfigError(f"options snapshot label must be one of {sorted(allowed)}, got {v!r}")
        return v

    @model_validator(mode="after")
    def pull_not_before_market(self) -> OptionsSnapshotConfig:
        """A delayed feed can never show the market time before it happened."""
        if self.pull_time_et < self.market_time_et:  # HH:MM strings compare lexically
            raise ConfigError(
                f"snapshot {self.label!r}: pull_time_et {self.pull_time_et} precedes "
                f"market_time_et {self.market_time_et}"
            )
        return self


class OptionsIntegrityConfig(BaseModel):
    """Silent-corruption guards + audit schedule (spec §10.5/§10.6, §17 C1)."""

    # CBOE has no truncation flag; the partial-chain guard is a contract-count drop
    # vs the previous same-label snapshot (fraction; 0.5 = warn on a >50% drop).
    contract_count_drop_warn_frac: float = Field(default=0.5, gt=0.0, le=1.0)
    audit_day_of_month: int = Field(default=1, ge=1, le=28)
    audit_time_et: str = Field(default="18:00")


class OptionsBackupConfig(BaseModel):
    """Offsite 3-2-1 backup of the un-backfillable capture (spec §13)."""

    enabled: bool = Field(default=True)
    rclone_remote: str = Field(default="b2:swingrl-options")
    time_et: str = Field(default="02:30")


class OptionsCollectorConfig(BaseModel):
    """EOD option-chain collector configuration (spec §5 as amended §17)."""

    enabled: bool = Field(default=True)
    provider: str = Field(default="cboe")
    endpoint_url_template: str = Field(
        default="https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
    )
    index_symbols: list[str] = Field(default_factory=lambda: ["_SPX"])
    include_equity_symbols: bool = Field(default=True)
    output_dir: str = Field(default="data/options_eod/cboe")
    schema_version: str = Field(default="v1")
    snapshots: list[OptionsSnapshotConfig] = Field(
        default_factory=lambda: [
            OptionsSnapshotConfig(label="decision", market_time_et="15:45",
                                  pull_time_et="16:00", misfire_grace_s=900),
            OptionsSnapshotConfig(label="eod", market_time_et="16:15",
                                  pull_time_et="16:35", misfire_grace_s=18000),
        ]
    )
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    rate_limit_per_sec: float = Field(default=1.0, gt=0.0)
    health_check_time_et: str = Field(default="17:15")
    health_lookback_days: int = Field(default=3, ge=1)   # D9 lookback window
    apscheduler_db_path: str = Field(default="db/apscheduler_options.sqlite")
    # Keep raw_json in Postgres JSONB too (bulky). Default on; revisit at first-run (decision D5).
    postgres_store_raw_json: bool = Field(default=True)
    integrity: OptionsIntegrityConfig = Field(default_factory=OptionsIntegrityConfig)
    backup: OptionsBackupConfig = Field(default_factory=OptionsBackupConfig)
```

Confirm `model_validator` is imported in `schema.py` alongside `field_validator` (add it if
only `field_validator` is present).

Then attach it to `SwingRLConfig` (add one line alongside the other `Field(default_factory=...)` sub-configs, e.g. after `scheduler: SchedulerConfig = ...`):

```python
    options_collector: OptionsCollectorConfig = Field(default_factory=OptionsCollectorConfig)
```

Confirm `field_validator` and `ConfigError` are already imported at the top of `schema.py` (they are — used by `EquityConfig`).

- [ ] **Step 4: Add the YAML block**

Append to `config/swingrl.yaml` a new top-level `options_collector:` block (mirroring the `scheduler:` block's comment style):

```yaml
options_collector:
  enabled: true
  provider: cboe                    # primary (spec §17 C1); schwab/moomoo = fallbacks (C2)
  endpoint_url_template: "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
  index_symbols: ["_SPX"]           # CBOE URL symbol; verified live 2026-07-14
  include_equity_symbols: true      # also capture chains for config.equity.symbols
  output_dir: "data/options_eod/cboe"
  schema_version: "v1"
  apscheduler_db_path: db/apscheduler_options.sqlite   # SEPARATE from the trader's jobstore
  postgres_store_raw_json: true     # also store raw_json JSONB in Postgres; revisit at first-run
  request_timeout_s: 30.0
  rate_limit_per_sec: 1.0           # courtesy throttle; 9 GETs per run
  health_check_time_et: "17:15"
  health_lookback_days: 3           # D9: health check scans the last N trading days
  snapshots:                        # pull trails market time by the feed delay (D8; verify at T6)
    - { label: decision, market_time_et: "15:45", pull_time_et: "16:00", misfire_grace_s: 900 }
    - { label: eod,      market_time_et: "16:15", pull_time_et: "16:35", misfire_grace_s: 18000 }
  integrity:
    contract_count_drop_warn_frac: 0.5   # partial-chain guard (CBOE has no truncation flag)
    audit_day_of_month: 1
    audit_time_et: "18:00"
  backup:
    enabled: true
    rclone_remote: "b2:swingrl-options"
    time_et: "02:30"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_config.py -v`
Expected: all 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/config/schema.py config/swingrl.yaml tests/test_options_config.py
git commit -m "feat(options): OptionsCollectorConfig schema + YAML block (spec §5)"
```

---

### Task 3: REMOVED — `schwab_auth.py` token manager (2026-07-14, spec §17 C1/C2)

Tombstone. The CBOE endpoint is unauthenticated — there is no token to manage, no reminder
jobs, no `invalid_client` path. The full `TokenManager` design (8 tests + implementation)
lives in this file's git history and re-activates only if the Schwab fallback is promoted
(spec §17 C2). `OPT-AUTH-*` requirement IDs retired.

---

### Task 4: REMOVED — `scripts/schwab_reauth.py` manual-OAuth CLI (2026-07-14, spec §17 C1/C2)

Tombstone. No OAuth exists in the CBOE design. The manual-flow CLI design lives in git
history; re-activates only with the Schwab fallback. `OPT-CLI-*` requirement IDs retired.

---

### Task 5: `cboe_client.py` — chain fetch (RESTRUCTURED 2026-07-14)

**Files:**
- Create: `src/swingrl/data/options/cboe_client.py`
- Test: `tests/test_cboe_client.py`

**Interfaces:**
- Consumes: `OptionsCollectorConfig` (T2: `endpoint_url_template`, `request_timeout_s`, `rate_limit_per_sec`), `DataError`, `swingrl_retry` (`src/swingrl/utils/retry.py`), `httpx` (existing dependency).
- Produces: `CboeChainClient` with:
  - `__init__(self, config: OptionsCollectorConfig) -> None`
  - `chain_url(self, symbol: str) -> str` — template + symbol (symbols are config-controlled, no escaping needed).
  - `get_option_chain(self, symbol: str) -> dict` — raw payload dict; throttled; retried on transport errors; raises `DataError` on non-200, non-JSON, or a payload without `data.options`.
- **This module is the provider-quarantine seam** (D7): a fallback provider replaces this file only; parser/store/collector interfaces are provider-agnostic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cboe_client.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from swingrl.config.schema import OptionsCollectorConfig
from swingrl.data.options.cboe_client import CboeChainClient
from swingrl.utils.exceptions import DataError


def _client() -> CboeChainClient:
    cfg = OptionsCollectorConfig()
    cfg.rate_limit_per_sec = 1000.0  # no real sleeping in tests
    return CboeChainClient(cfg)


def _response(payload: dict | None, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    if payload is None:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = payload
    return resp


def test_chain_url_from_template() -> None:
    """OPT-CLIENT-1: URL built from the config template (spec §17 C1)."""
    assert _client().chain_url("_SPX") == (
        "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
    )


def test_get_option_chain_returns_payload() -> None:
    """OPT-CLIENT-2: happy path returns the parsed JSON dict (spec §17 C1)."""
    ok = {"timestamp": "2026-07-14 19:45:10", "symbol": "^SPX",
          "data": {"current_price": 7543.59,
                   "options": [{"option": "SPXW260724C07500000"}]}}
    with patch("swingrl.data.options.cboe_client.httpx.get",
               return_value=_response(ok)) as fake_get:
        out = _client().get_option_chain("_SPX")
    assert out["data"]["current_price"] == 7543.59
    fake_get.assert_called_once()


def test_http_error_raises_dataerror() -> None:
    """OPT-CLIENT-3: non-200 -> DataError (spec §10.3)."""
    with patch("swingrl.data.options.cboe_client.httpx.get",
               return_value=_response({}, status=503)):
        with pytest.raises(DataError):
            _client().get_option_chain("SPY")


def test_missing_options_key_raises_dataerror() -> None:
    """OPT-CLIENT-4: payload without data.options -> DataError (spec §10.5)."""
    with patch("swingrl.data.options.cboe_client.httpx.get",
               return_value=_response({"data": {}})):
        with pytest.raises(DataError):
            _client().get_option_chain("SPY")


def test_non_json_raises_dataerror() -> None:
    """OPT-CLIENT-5: non-JSON body -> DataError (spec §10.3)."""
    with patch("swingrl.data.options.cboe_client.httpx.get", return_value=_response(None)):
        with pytest.raises(DataError):
            _client().get_option_chain("SPY")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cboe_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swingrl.data.options.cboe_client'`.

- [ ] **Step 3: Write the implementation**

```python
# src/swingrl/data/options/cboe_client.py
"""Thin, swappable client for CBOE's delayed-quotes chain endpoint (spec §17 C1).

Provider-quarantine seam (D7): fallback providers (Schwab, moomoo — spec §17 C2)
replace THIS module only; parser/store/collector stay provider-agnostic.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from swingrl.utils.exceptions import DataError
from swingrl.utils.retry import swingrl_retry

if TYPE_CHECKING:
    from swingrl.config.schema import OptionsCollectorConfig

log = structlog.get_logger(__name__)


class CboeChainClient:
    """Fetch a full option chain as a raw dict — unauthenticated, throttled, retried."""

    def __init__(self, config: OptionsCollectorConfig) -> None:
        self._config = config
        self._min_interval_s = 1.0 / config.rate_limit_per_sec
        self._last_call_ts = 0.0

    def chain_url(self, symbol: str) -> str:
        """Endpoint URL for one underlying (symbols are config-controlled)."""
        return self._config.endpoint_url_template.format(symbol=symbol)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)
        self._last_call_ts = time.monotonic()

    @swingrl_retry(
        max_attempts=4, retryable_exceptions=(httpx.TransportError, TimeoutError, OSError)
    )
    def _fetch(self, url: str) -> httpx.Response:
        self._throttle()
        return httpx.get(url, timeout=self._config.request_timeout_s)

    def get_option_chain(self, symbol: str) -> dict[str, Any]:
        """Fetch the full chain payload for one underlying (spec §17 C1)."""
        resp = self._fetch(self.chain_url(symbol))
        if resp.status_code != 200:
            log.error("cboe_chain_http_error", symbol=symbol, status=resp.status_code)
            raise DataError(f"CBOE chain HTTP {resp.status_code} for {symbol}")
        try:
            payload: dict[str, Any] = resp.json()
        except ValueError as exc:
            log.error("cboe_chain_bad_json", symbol=symbol, error=str(exc))
            raise DataError(f"CBOE chain returned non-JSON for {symbol}") from exc
        if not isinstance(payload.get("data"), dict) or "options" not in payload["data"]:
            log.error("cboe_chain_bad_shape", symbol=symbol, keys=sorted(payload)[:8])
            raise DataError(f"CBOE chain payload missing data.options for {symbol}")
        return payload
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cboe_client.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/options/cboe_client.py tests/test_cboe_client.py
git commit -m "feat(options): CboeChainClient — unauthenticated fetch, throttle, retry (spec §17 C1)"
```

---

### Task 6: Commit real fixtures + 🛑 delay-convention measurement (RESTRUCTURED 2026-07-14)

Decision D2 (amended): fixtures are REAL from day one — full `_SPX`/`SPY` payloads were
already pulled live during the 2026-07-14 planning session. The only human-gated part is
the trading-day delay measurement (D8), and it gates **T13's final pull-time config only**
— T7+ do not wait on it.

**Files:**
- Create: `scripts/capture_chain_fixture.py`
- Create (committed): `tests/fixtures/cboe_chain_spx.json`, `tests/fixtures/cboe_chain_spy.json`

**Interfaces:**
- Consumes: `CboeChainClient` (T5), `load_config`.
- Produces: `bound_chain(raw, max_options) -> dict` (full header, representative slice of
  contracts — bounding for repo size, NOT sanitizing: the payload is public market data
  with no account fields), plus a `--probe` mode printing wall clock vs payload timestamps.

- [ ] **Step 1: Write `scripts/capture_chain_fixture.py`**

```python
# scripts/capture_chain_fixture.py
"""One-shot: pull one real CBOE chain; save a bounded fixture OR probe the delay (D8).

Usage:
    uv run python scripts/capture_chain_fixture.py --symbol _SPX --out tests/fixtures/cboe_chain_spx.json
    uv run python scripts/capture_chain_fixture.py --symbol _SPX --probe
Public, unauthenticated data — fixtures are bounded (size), not sanitized (no account fields).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import structlog

from swingrl.config.schema import load_config
from swingrl.data.options.cboe_client import CboeChainClient
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)
_MAX_OPTIONS = 40


def bound_chain(raw: dict, max_options: int = _MAX_OPTIONS) -> dict:
    """Keep the full header but only an evenly-spaced slice of contracts."""
    out = {k: v for k, v in raw.items() if k != "data"}
    data = dict(raw["data"])
    opts = data.get("options", [])
    step = max(1, len(opts) // max_options)
    data["options"] = opts[::step][:max_options]
    out["data"] = data
    return out


def main() -> int:
    """Capture a fixture or print the delay-probe readings."""
    parser = argparse.ArgumentParser(description="Capture a CBOE chain fixture / probe delay")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--out")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    config = load_config("config/swingrl.yaml")
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    client = CboeChainClient(config.options_collector)
    raw = client.get_option_chain(args.symbol)
    if args.probe:
        print(f"wall_clock_utc={datetime.now(UTC).isoformat()}")
        print(f"payload_timestamp={raw.get('timestamp')}")
        print(f"header_last_trade_time={raw['data'].get('last_trade_time')}")
        print(f"contracts={len(raw['data'].get('options', []))}")
        return 0
    if not args.out:
        print("ERROR: --out required unless --probe", file=sys.stderr)
        return 2
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bound_chain(raw), indent=1))
    log.info("fixture_written", symbol=args.symbol, path=str(out_path), rows=_MAX_OPTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Capture and commit the two fixtures** (works any time — even after hours;
  SPX quotes overnight via global trading hours, verified 2026-07-14)

```bash
uv run python scripts/capture_chain_fixture.py --symbol _SPX --out tests/fixtures/cboe_chain_spx.json
uv run python scripts/capture_chain_fixture.py --symbol SPY  --out tests/fixtures/cboe_chain_spy.json
```

Eyeball both: contract objects must carry the T7 mapping fields (`option`, `bid`, `ask`,
`bid_size`, `ask_size`, `iv`, `open_interest`, `volume`, `delta`, `gamma`, `theta`, `vega`,
`rho`, `theo`, `last_trade_price`, `last_trade_time`, `prev_day_close`). Note any deviation
— T7's mapping and T11's `EXPECTED_CONTRACT_FIELDS` must use the real names.

```bash
git add scripts/capture_chain_fixture.py tests/fixtures/cboe_chain_spx.json tests/fixtures/cboe_chain_spy.json
git commit -m "chore(options): capture-fixture helper + real CBOE SPX/SPY chain fixtures (D2)"
```

- [x] **Step 3: 🛑 Delay-convention measurement (human; needs one trading day, 15:45–16:10 ET window)** — DONE 2026-07-15

Run `--probe` for `_SPX` at ~**15:50 ET** and again at ~**16:05 ET**. Compute
`wall_clock − payload_timestamp` (and sanity-check `header_last_trade_time`). Record in
this plan (below) and in spec §17 C1's table:

- Measured offset at 15:50 ET: **content delay 15 m 27 s** (wall 15:50:02 ET vs header
  `last_trade_time` 15:34:35 ET). Payload `timestamp` was only ~23 s behind wall clock —
  it is the UTC *generation* time, not the quote time; the 15-min delay lives in the
  quote CONTENT, so the delay is measured against `last_trade_time`, not `timestamp`.
- Measured offset at 16:05 ET: **content delay 15 m 27 s** (wall 16:05:02 ET vs header
  `last_trade_time` 15:49:35 ET; payload `timestamp` again ~23 s behind).
- Timestamp timezone convention confirmed: top-level `timestamp` = **UTC generation time**
  (2026-07-14 observation confirmed); per-contract/header `last_trade_time` = **ET**
  (read as UTC the 16:05 probe would imply a 4 h 16 m delay — impossible; the parser's
  ET→UTC localization in `parse_cboe_ts` is empirically confirmed).
- **Verdict: offset ≈ assumed 15 min → `snapshots[].pull_time_et` UNCHANGED** (decision
  16:00 → content ≈15:44:33 ET, at-or-just-before the 15:45 label = zero lookahead; eod
  16:35 → content ≈16:19:33 ET, safely past the 16:15 freeze). Probe log:
  `.superpowers/sdd/t6-probe-2026-07-15.log` (dev checkout).

**If the offset ≠ ~15 min, adjust `snapshots[].pull_time_et` in `config/swingrl.yaml`
(T2) so the decision pull captures the 15:45 ET state — before T13's deploy.** A grossly
different convention (e.g. no delay at all) is good news; just re-derive pull times.
*(Resolved: offset matched the assumption; no adjustment needed.)*

---

### Task 7: `chain_parser.py` — raw dict → typed rows + raw_json (RESTRUCTURED 2026-07-14)

Maps the **verified CBOE payload** (fixtures from T6). Key differences vs the Schwab-era
design: contract identity (root / expiry / right / strike) is **parsed from the OSI
symbol** (CBOE sends no per-field identity); `iv` is a **decimal fraction** (0.1164 =
11.64% — documented; column name `iv` kept for DDL stability, fraction convention recorded
in data-caveats + a column comment in T9's DDL); illiquid contracts show `iv == 0.0`
(observed 2026-07-14) which maps to NaN; there is no per-contract quote time (column stays
NULL) and no independent header contract count (`number_of_contracts` is computed as
`len(options)`); snapshot-level extras (`late_by_s`, `payload_timestamp`) ride in
`raw_header` JSONB — **no DDL change needed** (T9 unchanged).

**Files:**
- Create: `src/swingrl/data/options/chain_parser.py`
- Test: `tests/test_chain_parser.py`

**Interfaces:**
- Consumes: `DataError`.
- Produces:
  - `ParsedChain` — frozen dataclass: `header: dict[str, Any]`, `contracts: pd.DataFrame`.
  - `parse_chain(raw, *, underlying_symbol, snapshot_label, quote_date, snapshot_time_utc,
    pulled_at_utc, schema_version, is_early_close, late_by_s=0.0, source="cboe") ->
    ParsedChain` — `snapshot_time_utc` = the MARKET moment represented (D8), not the pull
    time.
  - `parse_osi(symbol: str) -> tuple[str, date, str, float]` — `(root, expiration,
    right, strike)` from e.g. `SPXW260724P07840000`.
  - `parse_cboe_ts(value: str | None) -> datetime | None` — payload timestamp strings →
    tz-aware UTC (convention verified at T6).
  - `clean_sentinel(value, *, zero_is_missing=False) -> float` — `-999`/`NaN`/`None` (and
    optionally `0.0`, used for `iv`) → `float("nan")`.
  - Module constant `CONTRACT_COLUMNS: list[str]` — the exact ordered typed columns.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chain_parser.py
from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pytest

from swingrl.data.options.chain_parser import (
    CONTRACT_COLUMNS,
    clean_sentinel,
    parse_chain,
    parse_osi,
)


def _raw() -> dict:
    """Representative CBOE payload (shape verified live 2026-07-14)."""
    liquid_call = {
        "option": "SPXW260724C07500000",
        "bid": 12.3, "bid_size": 10.0, "ask": 12.7, "ask_size": 8.0,
        "iv": 0.1234, "open_interest": 15000.0, "volume": 4200.0,
        "delta": 0.55, "gamma": 0.01, "vega": 1.2, "theta": -0.9, "rho": 0.3,
        "theo": 12.45, "change": 0.1, "open": 12.0, "high": 13.0, "low": 11.8,
        "tick": "up", "last_trade_price": 12.5,
        "last_trade_time": "2026-07-14T15:59:07", "percent_change": 0.8,
        "prev_day_close": 12.4,
    }
    illiquid_put = {**liquid_call, "option": "SPX260918P04000000",
                    "iv": 0.0, "delta": -999.0, "open_interest": 0.0,
                    "last_trade_price": 0.0, "last_trade_time": None}
    return {
        "timestamp": "2026-07-14 19:45:10",
        "symbol": "^SPX",
        "data": {
            "symbol": "^SPX", "security_type": "index",
            "current_price": 7543.59, "bid": 7543.0, "ask": 7544.0,
            "iv30": 13.2, "seqno": 12345,
            "last_trade_time": "2026-07-14T15:59:59",
            "options": [liquid_call, illiquid_put],
        },
    }


def _parse():
    return parse_chain(
        _raw(),
        underlying_symbol="_SPX",
        snapshot_label="decision",
        quote_date=date(2026, 7, 14),
        snapshot_time_utc=datetime(2026, 7, 14, 19, 45, tzinfo=UTC),  # 15:45 ET market state
        pulled_at_utc=datetime(2026, 7, 14, 20, 0, 3, tzinfo=UTC),    # pulled 16:00 ET (D8)
        schema_version="v1",
        is_early_close=False,
        late_by_s=0.0,
    )


def test_parse_osi_call_and_put() -> None:
    """OPT-PARSE-1: root/expiry/right/strike parsed from the OSI symbol (§17 C1)."""
    assert parse_osi("SPXW260724C07500000") == ("SPXW", date(2026, 7, 24), "CALL", 7500.0)
    assert parse_osi("SPX260918P04000000") == ("SPX", date(2026, 9, 18), "PUT", 4000.0)
    assert parse_osi("SPY260821C00650000") == ("SPY", date(2026, 8, 21), "CALL", 650.0)


def test_clean_sentinel_maps_to_nan() -> None:
    """OPT-PARSE-2: -999 / NaN / None -> NaN; iv zero-is-missing rule (§6.3, §17 C1)."""
    assert math.isnan(clean_sentinel(-999.0))
    assert math.isnan(clean_sentinel(None))
    assert math.isnan(clean_sentinel(0.0, zero_is_missing=True))
    assert clean_sentinel(0.0) == 0.0
    assert clean_sentinel(0.55) == 0.55


def test_contracts_flattened_one_row_per_contract() -> None:
    """OPT-PARSE-3: grain = one row per contract (spec §6.3)."""
    df = _parse().contracts
    assert len(df) == 2
    assert list(df.columns) == CONTRACT_COLUMNS


def test_identity_columns_derived() -> None:
    """OPT-PARSE-4: right/strike/expiration/dte derived from OSI + quote_date."""
    df = _parse().contracts.set_index("contract_symbol")
    row = df.loc["SPXW260724C07500000"]
    assert row["option_right"] == "CALL"
    assert row["strike"] == 7500.0
    assert row["expiration"] == date(2026, 7, 24)
    assert int(row["dte"]) == 10  # 2026-07-24 minus quote_date 2026-07-14


def test_iv_fraction_preserved_and_zero_becomes_nan() -> None:
    """OPT-PARSE-5: iv stored as the CBOE decimal fraction; 0.0 -> NaN (illiquid)."""
    df = _parse().contracts.set_index("contract_symbol")
    assert df.loc["SPXW260724C07500000", "iv"] == pytest.approx(0.1234)
    assert math.isnan(df.loc["SPX260918P04000000", "iv"])


def test_sentinel_greeks_become_nan() -> None:
    """OPT-PARSE-6: -999 greeks stored as NaN, never -999 (spec §6.3)."""
    df = _parse().contracts.set_index("contract_symbol")
    assert math.isnan(df.loc["SPX260918P04000000", "delta"])


def test_raw_json_populated_per_row() -> None:
    """OPT-PARSE-7: full original contract dict kept in raw_json (spec §6.2)."""
    df = _parse().contracts.set_index("contract_symbol")
    raw = df.loc["SPXW260724C07500000", "raw_json"]
    assert isinstance(raw, dict) and raw["theo"] == 12.45


def test_header_denormalized_context() -> None:
    """OPT-PARSE-8: header carries market context + D8 provenance (spec §6.4, §17 C3)."""
    parsed = _parse()
    assert parsed.header["underlying_price"] == 7543.59
    assert parsed.header["is_delayed"] is True          # constant: delayed feed (§17 C1)
    assert parsed.header["number_of_contracts"] == 2    # computed = len(options)
    assert parsed.header["raw_header"]["payload_timestamp"] == "2026-07-14 19:45:10"
    assert parsed.header["raw_header"]["late_by_s"] == 0.0
    assert "options" not in parsed.header["raw_header"]
    assert (parsed.contracts["underlying_price"] == 7543.59).all()


def test_empty_chain_raises_dataerror() -> None:
    """OPT-PARSE-9: no contracts -> DataError (spec §10.3)."""
    from swingrl.utils.exceptions import DataError

    empty = _raw()
    empty["data"]["options"] = []
    with pytest.raises(DataError):
        parse_chain(
            empty, underlying_symbol="_SPX", snapshot_label="eod",
            quote_date=date(2026, 7, 14),
            snapshot_time_utc=datetime(2026, 7, 14, 20, 15, tzinfo=UTC),
            pulled_at_utc=datetime(2026, 7, 14, 20, 35, tzinfo=UTC),
            schema_version="v1", is_early_close=False,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chain_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swingrl.data.options.chain_parser'`.

- [ ] **Step 3: Write the implementation**

```python
# src/swingrl/data/options/chain_parser.py
"""Parse a raw CBOE chain dict into typed contract rows + raw_json (spec §6, §17 C1)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import structlog

from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

_SENTINELS = {-999.0, -999}
_OSI_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

# Ordered typed columns — the grain (spec §6.3). Columns absent from CBOE payloads
# (quote_time_utc, settlement/exercise/multiplier fields) stay in the T9 DDL as NULLs.
CONTRACT_COLUMNS: list[str] = [
    "underlying_symbol", "quote_date", "snapshot_label",
    "underlying_price", "is_delayed", "trade_time_utc", "pulled_at_utc",
    "source", "schema_version",
    "contract_symbol", "option_root", "expiration", "dte", "strike", "option_right",
    "bid", "ask", "bid_size", "ask_size", "last", "volume", "open_interest",
    "net_change", "prev_day_close", "open", "high", "low",
    "delta", "gamma", "theta", "vega", "rho", "iv", "theoretical_value",
    "raw_json",
]


@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain: snapshot-level header + one DataFrame row per contract."""

    header: dict[str, Any]
    contracts: pd.DataFrame


def parse_osi(symbol: str) -> tuple[str, date, str, float]:
    """Split an OSI id into (root, expiration, CALL|PUT, strike) — CBOE sends no fields."""
    m = _OSI_RE.match(symbol)
    if not m:
        raise DataError(f"Unparseable OSI option symbol: {symbol!r}")
    root, yymmdd, right, strike_milli = m.groups()
    expiration = datetime.strptime(yymmdd, "%y%m%d").date()
    return root, expiration, ("CALL" if right == "C" else "PUT"), int(strike_milli) / 1000.0


def parse_cboe_ts(value: str | None) -> datetime | None:
    """CBOE timestamp strings -> tz-aware UTC (convention verified at T6)."""
    if not value:
        return None
    normalized = value.replace("T", " ")
    try:
        return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        log.warning("cboe_ts_unparsed", value=value)
        return None


def clean_sentinel(value: float | int | None, *, zero_is_missing: bool = False) -> float:
    """Map -999 / NaN / None (and optionally 0.0, for iv) to real NaN (spec §6.3)."""
    if value is None:
        return float("nan")
    if isinstance(value, float) and math.isnan(value):
        return float("nan")
    if value in _SENTINELS:
        return float("nan")
    if zero_is_missing and float(value) == 0.0:
        return float("nan")
    return float(value)


def _f(value: Any) -> float | None:
    return None if value is None else float(value)


def _i(value: Any) -> int | None:
    return None if value is None else int(value)


def _row(contract: dict, *, quote_date: date, base: dict) -> dict:
    symbol = contract.get("option", "")
    root, expiration, right, strike = parse_osi(symbol)
    row = dict(base)
    row.update(
        contract_symbol=symbol,
        option_root=root,
        expiration=expiration,
        dte=(expiration - quote_date).days,
        strike=strike,
        option_right=right,
        bid=_f(contract.get("bid")), ask=_f(contract.get("ask")),
        bid_size=_i(contract.get("bid_size")), ask_size=_i(contract.get("ask_size")),
        last=_f(contract.get("last_trade_price")),
        volume=_i(contract.get("volume")),
        open_interest=_i(contract.get("open_interest")),
        net_change=_f(contract.get("change")),
        prev_day_close=_f(contract.get("prev_day_close")),
        open=_f(contract.get("open")), high=_f(contract.get("high")),
        low=_f(contract.get("low")),
        delta=clean_sentinel(contract.get("delta")),
        gamma=clean_sentinel(contract.get("gamma")),
        theta=clean_sentinel(contract.get("theta")),
        vega=clean_sentinel(contract.get("vega")),
        rho=clean_sentinel(contract.get("rho")),
        # CBOE illiquid convention observed 2026-07-14: iv == 0.0 means "no IV".
        iv=clean_sentinel(contract.get("iv"), zero_is_missing=True),
        theoretical_value=_f(contract.get("theo")),
        trade_time_utc=parse_cboe_ts(contract.get("last_trade_time")),
        raw_json=contract,
    )
    return row


def parse_chain(
    raw: dict[str, Any],
    *,
    underlying_symbol: str,
    snapshot_label: str,
    quote_date: date,
    snapshot_time_utc: datetime,
    pulled_at_utc: datetime,
    schema_version: str,
    is_early_close: bool,
    late_by_s: float = 0.0,
    source: str = "cboe",
) -> ParsedChain:
    """Flatten a raw CBOE chain to typed rows + raw_json and build the header (spec §6).

    snapshot_time_utc = the MARKET moment the data represents (D8), never the pull time.
    """
    data = raw.get("data", {})
    options = data.get("options", [])
    base = {
        "underlying_symbol": underlying_symbol,
        "quote_date": quote_date,
        "snapshot_label": snapshot_label,
        "underlying_price": _f(data.get("current_price")),
        "is_delayed": True,  # constant: this IS the delayed feed (spec §17 C1)
        "pulled_at_utc": pulled_at_utc,
        "source": source,
        "schema_version": schema_version,
    }
    rows = [_row(c, quote_date=quote_date, base=base) for c in options]
    if not rows:
        log.error("options_empty_chain", underlying_symbol=underlying_symbol)
        raise DataError(f"Empty option chain for {underlying_symbol}")
    contracts_df = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)

    raw_header = {k: v for k, v in data.items() if k != "options"}
    raw_header["payload_timestamp"] = raw.get("timestamp")
    raw_header["late_by_s"] = late_by_s
    header = {
        "underlying_symbol": underlying_symbol,
        "quote_date": quote_date,
        "snapshot_label": snapshot_label,
        "snapshot_time_utc": snapshot_time_utc,
        "pulled_at_utc": pulled_at_utc,
        "underlying_price": _f(data.get("current_price")),
        "is_delayed": True,
        "is_early_close": is_early_close,
        "interest_rate": None,       # not provided by CBOE; FRED covers recomputation
        "dividend_yield": None,      # not provided by CBOE
        "underlying_volatility": _f(data.get("iv30")),
        "number_of_contracts": len(rows),   # computed — CBOE has no header count
        "status": "SUCCESS",
        "source": source,
        "schema_version": schema_version,
        "raw_header": raw_header,
    }
    return ParsedChain(header=header, contracts=contracts_df)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chain_parser.py -v`
Expected: all 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/options/chain_parser.py tests/test_chain_parser.py
git commit -m "feat(options): chain_parser — CBOE payload + OSI parsing -> typed rows + raw_json (spec §6, §17 C1)"
```

---

### Task 8: `store.py` (Parquet half) — atomic write + skip-existing

**Files:**
- Create: `src/swingrl/data/options/store.py`
- Test: `tests/test_options_store.py`

**Interfaces:**
- Consumes: `OptionsCollectorConfig` (T2), `ParsedChain` (T7). Reuses the existing atomic-write **idiom** from `ParquetStore` (temp file → `Path.replace`), not the class (its index-dedup semantics don't fit a write-once snapshot).
- Produces: `OptionsStore` with (Parquet half — the DB half is added in T10):
  - `__init__(self, config: OptionsCollectorConfig, db: DatabaseManager | None = None) -> None`
  - static `symbol_to_dir(symbol: str) -> str` — strips `_`/`$` prefixes (`_SPX` → `SPX`).
  - `parquet_path(symbol, quote_date, snapshot_label) -> Path`
  - `header_path(symbol, quote_date, snapshot_label) -> Path`
  - `snapshot_exists_parquet(symbol, quote_date, snapshot_label) -> bool`
  - `write_snapshot(parsed: ParsedChain, symbol, quote_date, snapshot_label) -> Path` — atomic Parquet (`raw_json` serialized to JSON string) + atomic header sidecar; returns the Parquet path.
  - `read_snapshot(symbol, quote_date, snapshot_label) -> ParsedChain` — inverse; `raw_json`/`raw_header` parsed back to dicts, datetime fields restored.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options_store.py
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from swingrl.config.schema import OptionsCollectorConfig
from swingrl.data.options.chain_parser import ParsedChain
from swingrl.data.options.store import OptionsStore


def _store(tmp_path: Path) -> OptionsStore:
    cfg = OptionsCollectorConfig()
    cfg.output_dir = str(tmp_path / "options_eod" / "cboe")
    return OptionsStore(cfg)


def _parsed() -> ParsedChain:
    df = pd.DataFrame(
        [{"contract_symbol": "SPXW260718C05000000", "strike": 5000.0, "iv": 12.3,
          "raw_json": {"symbol": "SPXW260718C05000000", "strikePrice": 5000.0}}]
    )
    header = {
        "underlying_symbol": "_SPX", "quote_date": date(2026, 7, 14), "snapshot_label": "decision",
        "snapshot_time_utc": datetime(2026, 7, 14, 19, 45, tzinfo=UTC),
        "pulled_at_utc": datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        "number_of_contracts": 1, "is_early_close": False,
        "raw_header": {"symbol": "_SPX", "status": "SUCCESS"},
    }
    return ParsedChain(header=header, contracts=df)


def test_symbol_to_dir_strips_prefixes() -> None:
    """OPT-STORE-1: _SPX/$SPX -> dir SPX (spec §5, §17 C1)."""
    assert OptionsStore.symbol_to_dir("_SPX") == "SPX"
    assert OptionsStore.symbol_to_dir("$SPX") == "SPX"
    assert OptionsStore.symbol_to_dir("SPY") == "SPY"


def test_parquet_path_layout(tmp_path: Path) -> None:
    """OPT-STORE-2: one file per (symbol,date,label) (spec §8.1)."""
    p = _store(tmp_path).parquet_path("_SPX", date(2026, 7, 14), "decision")
    assert p.name == "2026-07-14_decision.parquet"
    assert p.parent.name == "SPX"


def test_write_then_exists(tmp_path: Path) -> None:
    """OPT-STORE-3: write makes snapshot_exists_parquet true (spec §10.1)."""
    store = _store(tmp_path)
    assert store.snapshot_exists_parquet("_SPX", date(2026, 7, 14), "decision") is False
    store.write_snapshot(_parsed(), "_SPX", date(2026, 7, 14), "decision")
    assert store.snapshot_exists_parquet("_SPX", date(2026, 7, 14), "decision") is True


def test_write_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    """OPT-STORE-4: no .tmp file remains after write (spec §8.1)."""
    store = _store(tmp_path)
    path = store.write_snapshot(_parsed(), "_SPX", date(2026, 7, 14), "decision")
    assert not path.with_suffix(".parquet.tmp").exists()
    assert list(path.parent.glob("*.tmp")) == []


def test_roundtrip_restores_dicts_and_datetimes(tmp_path: Path) -> None:
    """OPT-STORE-5: read_snapshot restores raw_json dict + header datetimes (spec §8.1)."""
    store = _store(tmp_path)
    store.write_snapshot(_parsed(), "_SPX", date(2026, 7, 14), "decision")
    back = store.read_snapshot("_SPX", date(2026, 7, 14), "decision")
    assert isinstance(back.contracts.iloc[0]["raw_json"], dict)
    assert back.contracts.iloc[0]["raw_json"]["strikePrice"] == 5000.0
    assert back.header["snapshot_time_utc"] == datetime(2026, 7, 14, 19, 45, tzinfo=UTC)
    assert back.header["raw_header"]["status"] == "SUCCESS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swingrl.data.options.store'`.

- [ ] **Step 3: Write the implementation**

```python
# src/swingrl/data/options/store.py
"""Durable options-snapshot storage: Parquet-first, then Postgres (spec §8)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

from swingrl.data.options.chain_parser import CONTRACT_COLUMNS, ParsedChain

if TYPE_CHECKING:
    from swingrl.config.schema import OptionsCollectorConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

_HEADER_DT_FIELDS = ("snapshot_time_utc", "pulled_at_utc")
_HEADER_DATE_FIELDS = ("quote_date",)


class OptionsStore:
    """Writes each snapshot as an atomic Parquet + header sidecar (spec §8.1)."""

    def __init__(self, config: OptionsCollectorConfig, db: DatabaseManager | None = None) -> None:
        self._config = config
        self._db = db
        self._root = Path(config.output_dir)

    @staticmethod
    def symbol_to_dir(symbol: str) -> str:
        """Filesystem-safe directory name for a symbol (_SPX/$SPX -> SPX)."""
        return symbol.lstrip("$_")

    def parquet_path(self, symbol: str, quote_date: date, snapshot_label: str) -> Path:
        """Path to the contract Parquet for one (symbol, date, snapshot)."""
        return self._root / self.symbol_to_dir(symbol) / f"{quote_date.isoformat()}_{snapshot_label}.parquet"

    def header_path(self, symbol: str, quote_date: date, snapshot_label: str) -> Path:
        """Path to the header sidecar for one (symbol, date, snapshot)."""
        return self.parquet_path(symbol, quote_date, snapshot_label).with_suffix(".header.json")

    def snapshot_exists_parquet(self, symbol: str, quote_date: date, snapshot_label: str) -> bool:
        """True if the Parquet file for this snapshot already exists (skip unit)."""
        return self.parquet_path(symbol, quote_date, snapshot_label).exists()

    def write_snapshot(
        self, parsed: ParsedChain, symbol: str, quote_date: date, snapshot_label: str
    ) -> Path:
        """Atomically write the header sidecar and the contract Parquet (spec §8.1)."""
        pq_path = self.parquet_path(symbol, quote_date, snapshot_label)
        pq_path.parent.mkdir(parents=True, exist_ok=True)

        # Header sidecar (atomic).
        hdr_path = self.header_path(symbol, quote_date, snapshot_label)
        hdr_tmp = hdr_path.with_suffix(".json.tmp")
        hdr_tmp.write_text(json.dumps(parsed.header, default=str, indent=2))
        hdr_tmp.replace(hdr_path)

        # Contracts Parquet (atomic); raw_json -> JSON string for a stable columnar type.
        df = parsed.contracts.copy()
        df["raw_json"] = df["raw_json"].map(lambda d: json.dumps(d, default=str))
        pq_tmp = pq_path.with_suffix(".parquet.tmp")
        df.to_parquet(pq_tmp, index=False, compression="snappy")
        pq_tmp.replace(pq_path)
        log.info(
            "options_snapshot_written", symbol=symbol, quote_date=quote_date.isoformat(),
            snapshot_label=snapshot_label, rows=len(df),
        )
        return pq_path

    def read_snapshot(self, symbol: str, quote_date: date, snapshot_label: str) -> ParsedChain:
        """Read a snapshot back, restoring raw_json dicts and header datetimes."""
        df = pd.read_parquet(self.parquet_path(symbol, quote_date, snapshot_label))
        df["raw_json"] = df["raw_json"].map(json.loads)
        header = self._read_header(self.header_path(symbol, quote_date, snapshot_label))
        return ParsedChain(header=header, contracts=df[CONTRACT_COLUMNS])

    @staticmethod
    def _read_header(path: Path) -> dict[str, Any]:
        header = json.loads(path.read_text())
        for field in _HEADER_DT_FIELDS:
            if header.get(field):
                header[field] = datetime.fromisoformat(header[field])
        for field in _HEADER_DATE_FIELDS:
            if header.get(field):
                header[field] = date.fromisoformat(header[field])
        return header
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_store.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/options/store.py tests/test_options_store.py
git commit -m "feat(options): OptionsStore Parquet half — atomic write, skip-existing, roundtrip (spec §8.1)"
```

---

### Task 9: `schema.py` + migration — options tables & monthly partitions

Implements decision **D1**: self-contained additive schema (idempotent ensure at startup + a standalone one-time migration script). The trader's `postgres_schema.py` is untouched.

**Files:**
- Create: `src/swingrl/data/options/schema.py`
- Create: `scripts/migrations/add_options_capture_tables.py`
- Test: `tests/test_options_schema.py`

**Interfaces:**
- Consumes: a `psycopg.Connection`.
- Produces:
  - `OPTIONS_SNAPSHOTS_DDL: str`, `OPTIONS_CHAINS_DDL: str` (verbatim from spec §8.2).
  - `monthly_partition_bounds(quote_date: date) -> tuple[str, date, date]` — pure: `(partition_name, lo_inclusive, hi_exclusive)`.
  - `ensure_options_schema(conn) -> None` — idempotent `CREATE TABLE IF NOT EXISTS` for both tables.
  - `ensure_monthly_partition(conn, quote_date: date) -> str` — creates `options_chains_YYYY_MM` if absent; returns the name.
  - migration `apply_migration(conn) -> None`, `main() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options_schema.py
from __future__ import annotations

import os
from datetime import date

import psycopg
import pytest

from swingrl.data.options.schema import (
    ensure_monthly_partition,
    ensure_options_schema,
    monthly_partition_bounds,
)

_DB_URL = os.environ.get("DATABASE_URL")
_needs_db = pytest.mark.skipif(not _DB_URL, reason="DATABASE_URL not set")


def test_monthly_partition_bounds_normal_month() -> None:
    """OPT-SCHEMA-1: partition name + [lo, hi) bounds (spec §8.2)."""
    name, lo, hi = monthly_partition_bounds(date(2026, 7, 14))
    assert name == "options_chains_2026_07"
    assert lo == date(2026, 7, 1)
    assert hi == date(2026, 8, 1)


def test_monthly_partition_bounds_december_rolls_year() -> None:
    """OPT-SCHEMA-2: December -> next Jan (spec §8.2)."""
    name, lo, hi = monthly_partition_bounds(date(2026, 12, 3))
    assert name == "options_chains_2026_12"
    assert lo == date(2026, 12, 1)
    assert hi == date(2027, 1, 1)


@_needs_db
def test_ensure_schema_is_idempotent() -> None:
    """OPT-SCHEMA-3: ensure_options_schema runs twice cleanly (spec §8.2)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        ensure_options_schema(conn)  # second call must not error
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.options_snapshots')")
            assert cur.fetchone()[0] is not None
            cur.execute("SELECT to_regclass('public.options_chains')")
            assert cur.fetchone()[0] is not None
        conn.rollback()


@_needs_db
def test_ensure_monthly_partition_creates_child() -> None:
    """OPT-SCHEMA-4: monthly partition auto-created + idempotent (spec §8.2)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        name = ensure_monthly_partition(conn, date(2026, 7, 14))
        assert name == "options_chains_2026_07"
        ensure_monthly_partition(conn, date(2026, 7, 20))  # same month, no error
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"public.{name}",))
            assert cur.fetchone()[0] is not None
        conn.rollback()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swingrl.data.options.schema'` (the two pure tests fail on import; DB tests skip if no `DATABASE_URL`).

- [ ] **Step 3: Write the schema module**

```python
# src/swingrl/data/options/schema.py
"""Additive Postgres schema for options capture (spec §8.2, decision D1)."""

from __future__ import annotations

from datetime import date
from typing import Any

import psycopg
import structlog

log = structlog.get_logger(__name__)

OPTIONS_SNAPSHOTS_DDL = """
CREATE TABLE IF NOT EXISTS options_snapshots (
  underlying_symbol   text NOT NULL,
  quote_date          date NOT NULL,
  snapshot_label      text NOT NULL,
  snapshot_time_utc   timestamptz NOT NULL,
  pulled_at_utc       timestamptz NOT NULL,
  underlying_price    double precision,
  is_delayed          boolean,
  is_early_close      boolean,
  interest_rate       double precision,
  dividend_yield      double precision,
  underlying_volatility double precision,
  number_of_contracts integer,
  status              text,
  source              text NOT NULL DEFAULT 'cboe',
  schema_version      text NOT NULL,
  raw_header          jsonb NOT NULL,
  PRIMARY KEY (underlying_symbol, quote_date, snapshot_label)
)
"""

OPTIONS_CHAINS_DDL = """
CREATE TABLE IF NOT EXISTS options_chains (
  underlying_symbol text NOT NULL,
  quote_date        date NOT NULL,
  snapshot_label    text NOT NULL,
  contract_symbol   text NOT NULL,
  option_root text, expiration date, dte integer, strike double precision,
  option_right text, expiration_type text, settlement_type text, exercise_type text,
  multiplier double precision, in_the_money boolean,
  bid double precision, ask double precision, last double precision, mark double precision,
  bid_size integer, ask_size integer, last_size integer,
  open double precision, high double precision, low double precision, close double precision,
  volume bigint, open_interest bigint, net_change double precision,
  delta double precision, gamma double precision, theta double precision,
  vega double precision, rho double precision,
  iv double precision, theoretical_value double precision,
  time_value double precision, intrinsic_value double precision, extrinsic_value double precision,
  underlying_price double precision, is_delayed boolean,
  quote_time_utc timestamptz, trade_time_utc timestamptz,
  pulled_at_utc timestamptz NOT NULL,
  source text NOT NULL DEFAULT 'cboe', schema_version text NOT NULL,
  raw_json jsonb,   -- nullable: NULL when postgres_store_raw_json=false (decision D5; Parquet always keeps it)
  PRIMARY KEY (underlying_symbol, quote_date, snapshot_label, contract_symbol)
) PARTITION BY RANGE (quote_date)
"""


def monthly_partition_bounds(quote_date: date) -> tuple[str, date, date]:
    """Return (partition_name, lo_inclusive, hi_exclusive) for quote_date's month."""
    lo = quote_date.replace(day=1)
    hi = date(lo.year + 1, 1, 1) if lo.month == 12 else date(lo.year, lo.month + 1, 1)
    name = f"options_chains_{lo.year:04d}_{lo.month:02d}"
    return name, lo, hi


def ensure_options_schema(conn: psycopg.Connection[Any]) -> None:
    """Idempotently create both options tables (additive; A30-safe, spec §8.2)."""
    with conn.cursor() as cur:
        cur.execute(OPTIONS_SNAPSHOTS_DDL)
        cur.execute(OPTIONS_CHAINS_DDL)
    log.info("options_schema_ensured")


def ensure_monthly_partition(conn: psycopg.Connection[Any], quote_date: date) -> str:
    """Create the monthly partition of options_chains if absent; return its name."""
    name, lo, hi = monthly_partition_bounds(quote_date)
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {name} "
            f"PARTITION OF options_chains FOR VALUES FROM (%s) TO (%s)",
            (lo, hi),
        )
    log.info("options_partition_ensured", partition=name)
    return name
```

(The `{name}` interpolation is safe — it is derived from a date, never user input.)

- [ ] **Step 4: Write the migration script**

```python
# scripts/migrations/add_options_capture_tables.py
"""One-time additive migration: create options_snapshots + options_chains (spec §8.2).

This is the "V011_options_capture" migration. It is ADDITIVE ONLY (new tables),
safe to run against live pg16 while the trader runs (A30). Idempotent.

Usage:
    DATABASE_URL=... uv run python scripts/migrations/add_options_capture_tables.py
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

import psycopg
import structlog

from swingrl.data.options.schema import ensure_monthly_partition, ensure_options_schema

log = structlog.get_logger(__name__)


def get_database_url() -> str:
    """Read DATABASE_URL from the environment (raises KeyError if unset)."""
    return os.environ["DATABASE_URL"]


def apply_migration(conn: psycopg.Connection) -> None:
    """Create both tables + the current-month partition (additive, idempotent)."""
    ensure_options_schema(conn)
    ensure_monthly_partition(conn, datetime.now(UTC).date())


def main() -> int:
    """Apply the migration against DATABASE_URL and commit."""
    with psycopg.connect(get_database_url()) as conn:
        apply_migration(conn)
        conn.commit()
    log.info("options_capture_migration_applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_schema.py -v`
Expected: the 2 pure tests PASS; the 2 DB tests PASS if `DATABASE_URL` is set, else SKIP. (Homelab CI at T16 runs them against pg16.)

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/data/options/schema.py scripts/migrations/add_options_capture_tables.py tests/test_options_schema.py
git commit -m "feat(options): V011 additive schema — options_snapshots + options_chains + monthly partitions (spec §8.2)"
```

---

### Task 10: `store.py` (Postgres half) — JSONB sync + reconcile

**Files:**
- Modify: `src/swingrl/data/options/store.py` (add DB methods + column constants)
- Test: `tests/test_options_store_postgres.py`

**Interfaces:**
- Consumes: `DatabaseManager.connection()` (`src/swingrl/data/db.py`), `ensure_options_schema`/`ensure_monthly_partition` (T9), `psycopg.types.json.Jsonb`.
- Produces (added to `OptionsStore`):
  - `snapshot_exists_db(conn, symbol, quote_date, snapshot_label) -> bool`
  - `sync_to_postgres(parsed: ParsedChain) -> None` — ensures partition, upserts parent + children (`ON CONFLICT DO NOTHING`, JSONB); keys derived from `parsed.header`.
  - `reconcile() -> int` — loads any Parquet snapshot with no parent DB row; returns count.
  - `DB_SNAPSHOT_COLUMNS: list[str]`, `DB_CHAIN_COLUMNS: list[str]`.

- [ ] **Step 1: Write the failing test** (DB-gated, self-cleaning via rollback)

```python
# tests/test_options_store_postgres.py
from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import psycopg
import pytest

from swingrl.config.schema import OptionsCollectorConfig
from swingrl.data.options.chain_parser import CONTRACT_COLUMNS, ParsedChain
from swingrl.data.options.schema import ensure_options_schema
from swingrl.data.options.store import OptionsStore

_DB_URL = os.environ.get("DATABASE_URL")
_needs_db = pytest.mark.skipif(not _DB_URL, reason="DATABASE_URL not set")


class _FakeDB:
    """Hands out one connection whose commits are swallowed for test isolation."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def connection(self):  # mimics DatabaseManager.connection() contextmanager
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield self._conn  # no commit — test rolls back at the end

        return _cm()


def _parsed(symbol: str = "_SPX") -> ParsedChain:
    row = {c: None for c in CONTRACT_COLUMNS}
    row.update(
        underlying_symbol=symbol, quote_date=date(2026, 7, 14), snapshot_label="decision",
        contract_symbol="SPXW260718C05000000", strike=5000.0, dte=4, option_right="CALL",
        delta=0.55, iv=12.3, underlying_price=5001.2, is_delayed=False,
        pulled_at_utc=datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        expiration=date(2026, 7, 18), source="cboe", schema_version="v1",
        raw_json={"symbol": "SPXW260718C05000000", "strikePrice": 5000.0},
    )
    header = {
        "underlying_symbol": symbol, "quote_date": date(2026, 7, 14), "snapshot_label": "decision",
        "snapshot_time_utc": datetime(2026, 7, 14, 19, 45, tzinfo=UTC),
        "pulled_at_utc": datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        "underlying_price": 5001.2, "is_delayed": False, "is_early_close": False,
        "interest_rate": 5.0, "dividend_yield": 1.3, "underlying_volatility": 13.0,
        "number_of_contracts": 1, "status": "SUCCESS", "source": "cboe", "schema_version": "v1",
        "raw_header": {"symbol": symbol, "status": "SUCCESS"},
    }
    return ParsedChain(header=header, contracts=pd.DataFrame([row])[CONTRACT_COLUMNS])


def _store(tmp_path: Path, conn: psycopg.Connection) -> OptionsStore:
    cfg = OptionsCollectorConfig()
    cfg.output_dir = str(tmp_path / "options_eod" / "cboe")
    return OptionsStore(cfg, db=_FakeDB(conn))


@_needs_db
def test_sync_inserts_parent_and_child(tmp_path: Path) -> None:
    """OPT-STORE-6: sync writes parent snapshot + child contract rows (spec §8.2)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        store = _store(tmp_path, conn)
        store.sync_to_postgres(_parsed())
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM options_snapshots")
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM options_chains")
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT raw_json->>'strikePrice' FROM options_chains")
            assert cur.fetchone()[0] == "5000.0"
        conn.rollback()


@_needs_db
def test_sync_is_idempotent(tmp_path: Path) -> None:
    """OPT-STORE-7: re-sync -> ON CONFLICT DO NOTHING, no duplicate (spec §10.1)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        store = _store(tmp_path, conn)
        store.sync_to_postgres(_parsed())
        store.sync_to_postgres(_parsed())
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM options_chains")
            assert cur.fetchone()[0] == 1
        conn.rollback()


@_needs_db
def test_sync_respects_raw_json_flag(tmp_path: Path) -> None:
    """OPT-STORE-9: postgres_store_raw_json=False stores NULL raw_json (decision D5)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        cfg = OptionsCollectorConfig()
        cfg.output_dir = str(tmp_path / "options_eod" / "cboe")
        cfg.postgres_store_raw_json = False
        store = OptionsStore(cfg, db=_FakeDB(conn))
        store.sync_to_postgres(_parsed())
        with conn.cursor() as cur:
            cur.execute("SELECT raw_json FROM options_chains")
            assert cur.fetchone()[0] is None
        conn.rollback()


@_needs_db
def test_reconcile_loads_unsynced_parquet(tmp_path: Path) -> None:
    """OPT-STORE-8: reconcile loads a Parquet with no parent row (spec §8.2)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        store = _store(tmp_path, conn)
        store.write_snapshot(_parsed(), "_SPX", date(2026, 7, 14), "decision")  # Parquet only
        loaded = store.reconcile()
        assert loaded == 1
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM options_snapshots")
            assert cur.fetchone()[0] == 1
        conn.rollback()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_store_postgres.py -v`
Expected: FAIL (import of `sync_to_postgres` succeeds but `AttributeError`), or SKIP if no `DATABASE_URL`. To force a real fail locally, add `assert hasattr(OptionsStore, "sync_to_postgres")` — expected FAIL until Step 3.

- [ ] **Step 3: Extend `store.py`**

Add imports at the top of `store.py`:

```python
import math

from psycopg.types.json import Jsonb

from swingrl.data.options.schema import ensure_monthly_partition, ensure_options_schema
```

Add the column constants at module level (after `_HEADER_DATE_FIELDS`):

```python
DB_SNAPSHOT_COLUMNS = [
    "underlying_symbol", "quote_date", "snapshot_label", "snapshot_time_utc", "pulled_at_utc",
    "underlying_price", "is_delayed", "is_early_close", "interest_rate", "dividend_yield",
    "underlying_volatility", "number_of_contracts", "status", "source", "schema_version",
    "raw_header",
]
DB_CHAIN_COLUMNS = [
    "underlying_symbol", "quote_date", "snapshot_label", "contract_symbol",
    "option_root", "expiration", "dte", "strike", "option_right",
    "expiration_type", "settlement_type", "exercise_type", "multiplier", "in_the_money",
    "bid", "ask", "last", "mark", "bid_size", "ask_size", "last_size",
    "open", "high", "low", "close", "volume", "open_interest", "net_change",
    "delta", "gamma", "theta", "vega", "rho", "iv",
    "theoretical_value", "time_value", "intrinsic_value", "extrinsic_value",
    "underlying_price", "is_delayed", "quote_time_utc", "trade_time_utc", "pulled_at_utc",
    "source", "schema_version", "raw_json",
]
_DATE_DB_COLS = {"quote_date", "expiration"}
```

Add these methods to `OptionsStore`:

```python
    def snapshot_exists_db(self, conn, symbol: str, quote_date: date, snapshot_label: str) -> bool:
        """True if the parent snapshot row already exists in Postgres."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM options_snapshots "
                "WHERE underlying_symbol=%s AND quote_date=%s AND snapshot_label=%s",
                (symbol, quote_date, snapshot_label),
            )
            return cur.fetchone() is not None

    def sync_to_postgres(self, parsed: ParsedChain) -> None:
        """Upsert parent + child rows for one snapshot (idempotent, JSONB; spec §8.2)."""
        if self._db is None:
            return
        with self._db.connection() as conn:
            self._write_db(conn, parsed)

    def _write_db(self, conn, parsed: ParsedChain) -> None:
        hdr = parsed.header
        ensure_options_schema(conn)
        ensure_monthly_partition(conn, hdr["quote_date"])
        with conn.cursor() as cur:
            parent = tuple(self._db_value(k, hdr.get(k)) for k in DB_SNAPSHOT_COLUMNS)
            placeholders = ", ".join(["%s"] * len(DB_SNAPSHOT_COLUMNS))
            cur.execute(
                f"INSERT INTO options_snapshots ({', '.join(DB_SNAPSHOT_COLUMNS)}) "
                f"VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                parent,
            )
            df = parsed.contracts
            store_raw = self._config.postgres_store_raw_json
            records = [
                tuple(
                    None if (col == "raw_json" and not store_raw) else self._db_value(col, row[col])
                    for col in DB_CHAIN_COLUMNS
                )
                for row in df.to_dict("records")
            ]
            child_placeholders = ", ".join(["%s"] * len(DB_CHAIN_COLUMNS))
            cur.executemany(
                f"INSERT INTO options_chains ({', '.join(DB_CHAIN_COLUMNS)}) "
                f"VALUES ({child_placeholders}) ON CONFLICT DO NOTHING",
                records,
            )
        log.info(
            "options_snapshot_synced", underlying_symbol=hdr["underlying_symbol"],
            quote_date=hdr["quote_date"].isoformat(), snapshot_label=hdr["snapshot_label"],
            rows=len(parsed.contracts),
        )

    @staticmethod
    def _db_value(column: str, value):
        """Adapt a Python/pandas value for psycopg (JSONB, NaN->NULL, date coercion)."""
        if column in ("raw_json", "raw_header"):
            return Jsonb(value if isinstance(value, dict) else json.loads(value))
        if isinstance(value, float) and math.isnan(value):
            return None
        if value is None:
            return None
        if column in _DATE_DB_COLS and isinstance(value, datetime):
            return value.date()
        if column in _DATE_DB_COLS and isinstance(value, pd.Timestamp):
            return value.date()
        return value

    def reconcile(self) -> int:
        """Load any Parquet snapshot with no parent DB row; self-heals outages (spec §8.2)."""
        if self._db is None:
            return 0
        loaded = 0
        with self._db.connection() as conn:
            for pq in sorted(self._root.glob("*/*.parquet")):
                hdr = self._read_header(pq.with_suffix(".header.json"))
                sym, qdate, label = hdr["underlying_symbol"], hdr["quote_date"], hdr["snapshot_label"]
                if self.snapshot_exists_db(conn, sym, qdate, label):
                    continue
                self._write_db(conn, self.read_snapshot(sym, qdate, label))
                loaded += 1
        log.info("options_reconcile_done", loaded=loaded)
        return loaded
```

> **Perf note (not a placeholder):** `executemany` is fine for the twice-daily cadence up to ~50k contracts/symbol (per `pg_helpers` guidance). If T16 shows a single symbol exceeds that, switch the child insert to `cursor.copy()`; the interface is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_store_postgres.py -v`
Expected: 3 PASS with `DATABASE_URL` set (homelab/CI); else SKIP.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/options/store.py tests/test_options_store_postgres.py
git commit -m "feat(options): OptionsStore Postgres half — JSONB sync + reconcile self-heal (spec §8.2)"
```

---

### Task 11: `market_calendar.py` + `collector.py` — orchestration & guards (RESTRUCTURED 2026-07-14)

Deltas vs the Schwab-era design: no auth preflight (nothing to authenticate — C1); the
truncation re-fetch is replaced by the **contract-count drift guard** (CBOE has no
truncation flag); the collector computes `snapshot_time_utc` from the label's
`market_time_et` (not the pull time — D8) and stamps **`late_by_s`** when a job fired past
its scheduled pull time, WARNING on any late `decision` capture (lookahead-bias guard).

**Files:**
- Create: `src/swingrl/data/options/market_calendar.py`
- Create: `src/swingrl/data/options/collector.py`
- Modify: `src/swingrl/data/options/store.py` (add `last_snapshot_row_count`)
- Test: `tests/test_options_market_calendar.py`, `tests/test_options_collector.py`

**Interfaces:**
- Consumes: `CboeChainClient` (T5), `parse_chain` (T7), `OptionsStore` (T8/T10), `Alerter.send_alert` (existing), `SwingRLConfig` (both `options_collector` and `equity.symbols`), `DataError`.
- Produces:
  - `market_calendar.is_trading_day(quote_date: date) -> bool`, `market_calendar.is_early_close(quote_date: date) -> bool`.
  - `SnapshotResult` — dataclass `label, succeeded, failed, skipped, warnings`.
  - `OptionsCollector.__init__(self, config: SwingRLConfig, client, store, alerter=None) -> None`
  - `OptionsCollector.symbols() -> list[str]`
  - `OptionsCollector.run_snapshot(self, snapshot_label: str, now: datetime | None = None, scheduled_pull_utc: datetime | None = None) -> SnapshotResult` — `late_by_s = max(0, now − scheduled_pull_utc)`.
  - `OptionsStore.last_snapshot_row_count(symbol, snapshot_label) -> int | None` — row count of the most recent stored Parquet for that (symbol, label), via pyarrow metadata (cheap, no full read).
  - module-level `EXPECTED_CONTRACT_FIELDS: set[str]`, `check_schema_drift(raw) -> list[str]`.

- [ ] **Step 1: Write the calendar test**

```python
# tests/test_options_market_calendar.py
from __future__ import annotations

from datetime import date

from swingrl.data.options import market_calendar as mc


def test_weekend_is_not_trading_day() -> None:
    """OPT-COLLECT-1: Saturday is not a trading day (spec §9.2)."""
    assert mc.is_trading_day(date(2026, 7, 18)) is False  # Saturday


def test_regular_weekday_is_trading_day() -> None:
    """OPT-COLLECT-2: a normal Tuesday is a trading day (spec §9.2)."""
    assert mc.is_trading_day(date(2026, 7, 14)) is True


def test_christmas_is_not_trading_day() -> None:
    """OPT-COLLECT-3: NYSE holiday skipped (spec §9.2)."""
    assert mc.is_trading_day(date(2026, 12, 25)) is False


def test_black_friday_is_early_close() -> None:
    """OPT-COLLECT-4: half-day detected as early close (spec §6.1)."""
    # 2026-11-27 (day after Thanksgiving) is a 13:00 ET early close.
    assert mc.is_early_close(date(2026, 11, 27)) is True
    assert mc.is_early_close(date(2026, 7, 14)) is False
```

- [ ] **Step 2: Run → fail; then write `market_calendar.py`**

```python
# src/swingrl/data/options/market_calendar.py
"""NYSE (XNYS) trading-day and early-close helpers (spec §6.1, §9.2)."""

from __future__ import annotations

from datetime import date

import exchange_calendars as xcals
import pandas as pd

_CALENDAR_NAME = "XNYS"
_REGULAR_CLOSE_HOUR_ET = 16
_calendar: xcals.ExchangeCalendar | None = None


def _cal() -> xcals.ExchangeCalendar:
    global _calendar
    if _calendar is None:
        _calendar = xcals.get_calendar(_CALENDAR_NAME)
    return _calendar


def is_trading_day(quote_date: date) -> bool:
    """True if quote_date is an NYSE session (excludes weekends + holidays)."""
    return bool(_cal().is_session(pd.Timestamp(quote_date)))


def is_early_close(quote_date: date) -> bool:
    """True if quote_date is an NYSE half-day (regular close is 16:00 ET)."""
    session = pd.Timestamp(quote_date)
    if not _cal().is_session(session):
        return False
    close_et = _cal().session_close(session).tz_convert("America/New_York")
    return close_et.hour < _REGULAR_CLOSE_HOUR_ET


def recent_sessions(as_of: date, n: int) -> list[date]:
    """The last n NYSE sessions ending at (and including, if a session) as_of."""
    ts = pd.Timestamp(as_of)
    sessions = _cal().sessions_in_range(ts - pd.Timedelta(days=n * 3 + 10), ts)
    return [s.date() for s in sessions[-n:]]
```

Run: `uv run pytest tests/test_options_market_calendar.py -v` → 4 PASS. (If a hardcoded early-close date drifts in a future `exchange_calendars` release, adjust the fixture date — the logic is date-agnostic. `recent_sessions` is consumed by T13's lookback health check.)

- [ ] **Step 3: Write the collector test**

```python
# tests/test_options_collector.py
from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pandas as pd

from swingrl.config.schema import SwingRLConfig
from swingrl.data.options.chain_parser import CONTRACT_COLUMNS, ParsedChain
from swingrl.data.options.collector import OptionsCollector, check_schema_drift
from swingrl.utils.exceptions import DataError


def _cfg() -> SwingRLConfig:
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY", "QQQ"]
    cfg.options_collector.index_symbols = ["_SPX"]
    cfg.options_collector.include_equity_symbols = True
    return cfg


def _raw(symbol: str, n: int = 3) -> dict:
    contract = {"option": "SPXW260724C07500000", "bid": 1.0, "ask": 1.1, "bid_size": 1,
                "ask_size": 1, "iv": 0.2, "open_interest": 5, "volume": 1, "delta": 0.5,
                "gamma": 0.0, "theta": -0.1, "vega": 0.2, "rho": 0.1}
    return {"timestamp": "2026-07-14 20:00:00", "symbol": symbol,
            "data": {"current_price": 100.0, "options": [dict(contract) for _ in range(n)]}}


def _collector(client, store) -> tuple[OptionsCollector, MagicMock]:
    alerter = MagicMock()
    return OptionsCollector(_cfg(), client, store, alerter=alerter), alerter


def _store_mock() -> MagicMock:
    store = MagicMock()
    store.snapshot_exists_parquet.return_value = False
    store.last_snapshot_row_count.return_value = None
    return store


def test_symbols_combines_index_and_equity() -> None:
    """OPT-COLLECT-5: symbols = index + equity when enabled (spec §5)."""
    c, _ = _collector(MagicMock(), _store_mock())
    assert c.symbols() == ["_SPX", "SPY", "QQQ"]


def test_per_symbol_isolation_one_fails_others_succeed() -> None:
    """OPT-COLLECT-6: one symbol failing does not abort the rest (spec §10.2)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: (
        (_ for _ in ()).throw(DataError("boom")) if s == "SPY" else _raw(s)
    )
    c, _ = _collector(client, _store_mock())
    result = c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
    assert "SPY" in result.failed
    assert set(result.succeeded) == {"_SPX", "QQQ"}


def test_skip_already_captured() -> None:
    """OPT-COLLECT-7: existing Parquet snapshot is skipped (spec §10.1)."""
    store = _store_mock()
    store.snapshot_exists_parquet.return_value = True
    client = MagicMock()
    c, _ = _collector(client, store)
    result = c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
    client.get_option_chain.assert_not_called()
    assert set(result.skipped) == {"_SPX", "SPY", "QQQ"}


def test_all_symbols_fail_is_critical() -> None:
    """OPT-COLLECT-8: every symbol failing -> CRITICAL summary (spec §10.4)."""
    client = MagicMock()
    client.get_option_chain.side_effect = DataError("boom")
    c, alerter = _collector(client, _store_mock())
    c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
    assert any(call.args[0] == "critical" for call in alerter.send_alert.call_args_list)


def test_schema_drift_detected() -> None:
    """OPT-COLLECT-9: missing expected field flagged (spec §10.5)."""
    raw = {"data": {"options": [{"option": "A", "bid": 1.0}]}}
    missing = check_schema_drift(raw)
    assert "delta" in missing and "open_interest" in missing


def test_contract_count_drop_warns() -> None:
    """OPT-COLLECT-10: >50% contract-count drop vs previous snapshot -> WARNING (§17 C1)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: _raw(s, n=2)
    store = _store_mock()
    store.last_snapshot_row_count.return_value = 100   # previous run had 100 rows
    c, _ = _collector(client, store)
    result = c.run_snapshot("eod", now=datetime(2026, 7, 14, 20, 35, tzinfo=UTC))
    assert any("count" in w for w in result.warnings)


def test_late_decision_fire_warns_and_stamps() -> None:
    """OPT-COLLECT-11: decision fired past schedule -> late_by_s stamped + WARNING (D8)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: _raw(s)
    store = _store_mock()
    c, _ = _collector(client, store)
    result = c.run_snapshot(
        "decision",
        now=datetime(2026, 7, 14, 20, 10, tzinfo=UTC),                 # fired 16:10 ET
        scheduled_pull_utc=datetime(2026, 7, 14, 20, 0, tzinfo=UTC),   # scheduled 16:00 ET
    )
    assert any("late" in w for w in result.warnings)
    # late_by_s reaches the stored header via parse_chain(late_by_s=600)
    _, kwargs = store.write_snapshot.call_args
    parsed: ParsedChain = store.write_snapshot.call_args.args[0]
    assert parsed.header["raw_header"]["late_by_s"] == 600.0
```

- [ ] **Step 4: Run → fail; then write `collector.py`**

```python
# src/swingrl/data/options/collector.py
"""EOD collector orchestration: per-symbol fetch->parse->store with guards (spec §6, §10, §17)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import structlog

from swingrl.data.options import market_calendar
from swingrl.data.options.chain_parser import parse_chain
from swingrl.utils.exceptions import DataError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.options.cboe_client import CboeChainClient
    from swingrl.data.options.store import OptionsStore
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)
_ET = ZoneInfo("America/New_York")

EXPECTED_CONTRACT_FIELDS = {
    "option", "bid", "ask", "bid_size", "ask_size", "iv", "open_interest",
    "volume", "delta", "gamma", "theta", "vega", "rho",
}


@dataclass
class SnapshotResult:
    """Outcome of one snapshot run across all symbols."""

    label: str
    succeeded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_schema_drift(raw: dict[str, Any]) -> list[str]:
    """Return expected contract fields missing from the first contract (spec §10.5)."""
    options = raw.get("data", {}).get("options", [])
    if not options:
        return []
    return sorted(EXPECTED_CONTRACT_FIELDS - set(options[0]))


class OptionsCollector:
    """Runs one snapshot across all configured symbols with per-symbol isolation."""

    def __init__(
        self,
        config: SwingRLConfig,
        client: CboeChainClient,
        store: OptionsStore,
        alerter: Alerter | None = None,
    ) -> None:
        self._config = config
        self._oc = config.options_collector
        self._client = client
        self._store = store
        self._alerter = alerter

    def symbols(self) -> list[str]:
        """Index symbols first, then equity symbols if enabled (spec §5)."""
        symbols = list(self._oc.index_symbols)
        if self._oc.include_equity_symbols:
            symbols.extend(self._config.equity.symbols)
        return symbols

    def _market_time_utc(self, snapshot_label: str, quote_date: date) -> datetime:
        """The MARKET moment this label represents (D8) — never the pull time."""
        time_et = next(
            s.market_time_et for s in self._oc.snapshots if s.label == snapshot_label
        )
        hh, mm = (int(x) for x in time_et.split(":"))
        return datetime(
            quote_date.year, quote_date.month, quote_date.day, hh, mm, tzinfo=_ET
        ).astimezone(UTC)

    def run_snapshot(
        self,
        snapshot_label: str,
        now: datetime | None = None,
        scheduled_pull_utc: datetime | None = None,
    ) -> SnapshotResult:
        """Fetch+store every symbol's chain for one snapshot; alert on the summary."""
        now = now or datetime.now(UTC)
        quote_date = now.astimezone(_ET).date()
        result = SnapshotResult(label=snapshot_label)

        late_by_s = 0.0
        if scheduled_pull_utc is not None:
            late_by_s = max(0.0, (now - scheduled_pull_utc).total_seconds())
        if late_by_s > 0 and snapshot_label == "decision":
            result.warnings.append(
                f"decision snapshot fired late by {late_by_s:.0f}s — market state is "
                f"NOT the {snapshot_label} moment (lookahead guard, D8)"
            )

        early_close = market_calendar.is_early_close(quote_date)
        market_time_utc = self._market_time_utc(snapshot_label, quote_date)

        for symbol in self.symbols():
            if self._store.snapshot_exists_parquet(symbol, quote_date, snapshot_label):
                result.skipped.append(symbol)
                continue
            try:
                self._capture_one(
                    symbol, snapshot_label, quote_date, market_time_utc,
                    early_close, late_by_s, result,
                )
                result.succeeded.append(symbol)
            except DataError as exc:
                log.error("options_symbol_failed", symbol=symbol, error=str(exc))
                result.failed.append(symbol)

        self._route_summary_alert(result)
        return result

    def _capture_one(
        self, symbol: str, snapshot_label: str, quote_date: date,
        market_time_utc: datetime, early_close: bool, late_by_s: float,
        result: SnapshotResult,
    ) -> None:
        raw = self._client.get_option_chain(symbol)
        missing = check_schema_drift(raw)
        if missing:
            result.warnings.append(f"{symbol}: schema drift, missing {missing}")
        parsed = parse_chain(
            raw, underlying_symbol=symbol, snapshot_label=snapshot_label,
            quote_date=quote_date, snapshot_time_utc=market_time_utc,
            pulled_at_utc=datetime.now(UTC), schema_version=self._oc.schema_version,
            is_early_close=early_close, late_by_s=late_by_s,
        )
        previous = self._store.last_snapshot_row_count(symbol, snapshot_label)
        threshold = self._oc.integrity.contract_count_drop_warn_frac
        if previous and len(parsed.contracts) < previous * (1.0 - threshold):
            result.warnings.append(
                f"{symbol}: contract count dropped {previous} -> {len(parsed.contracts)} "
                f"(possible partial chain — CBOE has no truncation flag)"
            )
        self._store.write_snapshot(parsed, symbol, quote_date, snapshot_label)
        self._store.sync_to_postgres(parsed)

    def _route_summary_alert(self, result: SnapshotResult) -> None:
        attempted = len(result.succeeded) + len(result.failed)
        if attempted > 0 and not result.succeeded:
            self._alert("critical", f"Options {result.label}: ALL symbols failed",
                        f"failed={result.failed}")
        elif result.failed or result.warnings:
            self._alert("warning", f"Options {result.label} completed with issues",
                        f"failed={result.failed} warnings={result.warnings}")
        else:
            self._alert("info", f"Options {result.label} captured",
                        f"succeeded={result.succeeded} skipped={result.skipped}")

    def _alert(self, level: str, title: str, message: str) -> None:
        if self._alerter is not None:
            self._alerter.send_alert(level, title, message)  # type: ignore[arg-type]
```

Add to `OptionsStore` (T8's module — pyarrow metadata read, no full deserialization):

```python
    def last_snapshot_row_count(self, symbol: str, snapshot_label: str) -> int | None:
        """Row count of the most recent stored Parquet for (symbol, label), else None."""
        import pyarrow.parquet as pq

        directory = self._root / self.symbol_to_dir(symbol)
        candidates = sorted(directory.glob(f"*_{snapshot_label}.parquet"))
        if not candidates:
            return None
        return int(pq.ParquetFile(candidates[-1]).metadata.num_rows)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_market_calendar.py tests/test_options_collector.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/data/options/market_calendar.py src/swingrl/data/options/collector.py src/swingrl/data/options/store.py tests/test_options_market_calendar.py tests/test_options_collector.py
git commit -m "feat(options): collector orchestration — isolation, drift + count guards, late-fire stamping (spec §6, §10, §17)"
```

---

### Task 12: `audit.py` — data-quality audit

**Files:**
- Create: `src/swingrl/data/options/audit.py`
- Test: `tests/test_options_audit.py`

**Interfaces:**
- Consumes: `SwingRLConfig`, `DatabaseManager.connection()`, `pg_helpers.fetchdf` (`src/swingrl/data/pg_helpers.py`), `Alerter`.
- Produces:
  - `AuditResult` — dataclass `passed, failures, notes, symbols_checked`.
  - `audit_dataframe(df: pd.DataFrame) -> list[str]` — pure check list (delta ∈ [-1,1], bid ≤ ask, OI populated).
  - `oi_stability_failures(df: pd.DataFrame) -> list[str]` — flags OI differing across same-day snapshots (decision D6).
  - `descriptive_stats(df: pd.DataFrame) -> dict` — lightweight monthly digest (rows, median IV, median spread).
  - `run_data_quality_audit(config, db, *, since_days=30, now=None, alerter=None) -> AuditResult` — CRITICAL on failure, else an INFO digest of the stats.
  - `audit_symbols(config: SwingRLConfig) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options_audit.py
from __future__ import annotations

import pandas as pd

from swingrl.data.options.audit import (
    AuditResult,
    audit_dataframe,
    audit_symbols,
    descriptive_stats,
    oi_stability_failures,
)
from swingrl.config.schema import SwingRLConfig


def test_clean_frame_has_no_failures() -> None:
    """OPT-AUDIT-1: sane greeks/spreads/OI pass (spec §10.6)."""
    df = pd.DataFrame(
        {"delta": [0.5, -0.4], "bid": [1.0, 2.0], "ask": [1.1, 2.2], "open_interest": [10, 20]}
    )
    assert audit_dataframe(df) == []


def test_delta_out_of_range_fails() -> None:
    """OPT-AUDIT-2: |delta| > 1 flagged (spec §10.6)."""
    df = pd.DataFrame({"delta": [1.5], "bid": [1.0], "ask": [1.1], "open_interest": [10]})
    assert any("delta" in f for f in audit_dataframe(df))


def test_crossed_market_fails() -> None:
    """OPT-AUDIT-3: ask < bid flagged (spec §10.6)."""
    df = pd.DataFrame({"delta": [0.5], "bid": [2.0], "ask": [1.0], "open_interest": [10]})
    assert any("bid" in f.lower() or "ask" in f.lower() for f in audit_dataframe(df))


def test_all_oi_null_fails() -> None:
    """OPT-AUDIT-4: open_interest entirely null flagged (spec §10.6)."""
    df = pd.DataFrame({"delta": [0.5], "bid": [1.0], "ask": [1.1], "open_interest": [None]})
    assert any("open_interest" in f for f in audit_dataframe(df))


def test_audit_symbols_combines_index_and_equity() -> None:
    """OPT-AUDIT-5: audit covers index + equity symbols (spec §5)."""
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["_SPX"]
    assert audit_symbols(cfg) == ["_SPX", "SPY"]


def test_audit_result_passed_flag() -> None:
    """OPT-AUDIT-6: passed is True iff no failures (spec §10.6)."""
    assert AuditResult(failures=[]).passed is True
    assert AuditResult(failures=["x: delta"]).passed is False


def test_oi_stability_passes_when_identical() -> None:
    """OPT-AUDIT-7: identical OI across same-day snapshots passes (decision D6)."""
    df = pd.DataFrame({
        "quote_date": ["2026-07-14", "2026-07-14"],
        "contract_symbol": ["C", "C"],
        "snapshot_label": ["decision", "eod"],
        "open_interest": [100, 100],
    })
    assert oi_stability_failures(df) == []


def test_oi_stability_flags_intraday_change() -> None:
    """OPT-AUDIT-8: OI differing across same-day snapshots is flagged (decision D6)."""
    df = pd.DataFrame({
        "quote_date": ["2026-07-14", "2026-07-14"],
        "contract_symbol": ["C", "C"],
        "snapshot_label": ["decision", "eod"],
        "open_interest": [100, 120],
    })
    assert oi_stability_failures(df) != []


def test_descriptive_stats_shape() -> None:
    """OPT-AUDIT-9: monthly digest stats computed (spec §10.6)."""
    df = pd.DataFrame({"iv": [10.0, 20.0], "bid": [1.0, 2.0], "ask": [1.2, 2.3]})
    stats = descriptive_stats(df)
    assert stats["rows"] == 2
    assert stats["median_iv"] == 15.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_audit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swingrl.data.options.audit'`.

- [ ] **Step 3: Write the implementation**

```python
# src/swingrl/data/options/audit.py
"""Data-quality audit over captured options (the slow-rot net; spec §10.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
import structlog

from swingrl.data.pg_helpers import fetchdf

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)
_ET = ZoneInfo("America/New_York")


@dataclass
class AuditResult:
    """Result of a data-quality audit run."""

    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    symbols_checked: list[str] = field(default_factory=list)
    stats: dict[str, dict[str, float | int | None]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when no hard quality failures were found."""
        return not self.failures


def audit_dataframe(df: pd.DataFrame) -> list[str]:
    """Return quality failures for one symbol's recent contracts (spec §10.6)."""
    failures: list[str] = []
    delta = pd.to_numeric(df.get("delta"), errors="coerce").dropna()
    if len(delta) and bool(((delta < -1.0) | (delta > 1.0)).any()):
        failures.append("delta outside [-1, 1]")
    both = df.dropna(subset=["bid", "ask"])
    if len(both) and bool((both["ask"] < both["bid"]).any()):
        failures.append("ask < bid (crossed market)")
    if "open_interest" in df and int(df["open_interest"].notna().sum()) == 0:
        failures.append("open_interest entirely null")
    return failures


def oi_stability_failures(df: pd.DataFrame) -> list[str]:
    """OI must match across same-day snapshots for a contract (T-1/once-daily; decision D6)."""
    required = {"quote_date", "contract_symbol", "snapshot_label", "open_interest"}
    if not required <= set(df.columns):
        return []
    oi = df.dropna(subset=["open_interest"])
    per_contract_day = oi.groupby(["quote_date", "contract_symbol"])["open_interest"].nunique()
    bad = int((per_contract_day > 1).sum())
    return [f"OI differs across same-day snapshots on {bad} contract-days"] if bad else []


def descriptive_stats(df: pd.DataFrame) -> dict[str, float | int | None]:
    """Lightweight monthly digest stats for one symbol (spec §10.6)."""
    iv = pd.to_numeric(df.get("iv"), errors="coerce").dropna() if "iv" in df else pd.Series(dtype=float)
    spread = (df["ask"] - df["bid"]).dropna() if {"ask", "bid"} <= set(df.columns) else pd.Series(dtype=float)
    return {
        "rows": int(len(df)),
        "median_iv": round(float(iv.median()), 2) if len(iv) else None,
        "median_spread": round(float(spread.median()), 4) if len(spread) else None,
    }


def audit_symbols(config: SwingRLConfig) -> list[str]:
    """Index symbols + equity symbols (when enabled) covered by the audit."""
    symbols = list(config.options_collector.index_symbols)
    if config.options_collector.include_equity_symbols:
        symbols.extend(config.equity.symbols)
    return symbols


def _load_recent(conn, symbol: str, cutoff) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT quote_date, snapshot_label, contract_symbol, delta, bid, ask, iv, "
            "open_interest, mark FROM options_chains "
            "WHERE underlying_symbol = %s AND quote_date >= %s",
            (symbol, cutoff),
        )
        return fetchdf(cur)


def run_data_quality_audit(
    config: SwingRLConfig,
    db: DatabaseManager,
    *,
    since_days: int = 30,
    now: datetime | None = None,
    alerter: Alerter | None = None,
) -> AuditResult:
    """Audit the trailing window per symbol; CRITICAL-alert on any failure (spec §10.6)."""
    now = now or datetime.now(UTC)
    cutoff = now.astimezone(_ET).date() - timedelta(days=since_days)
    result = AuditResult()
    with db.connection() as conn:
        for symbol in audit_symbols(config):
            df = _load_recent(conn, symbol, cutoff)
            if df.empty:
                result.notes.append(f"{symbol}: no data in trailing {since_days}d")
                continue
            result.symbols_checked.append(symbol)
            for message in audit_dataframe(df) + oi_stability_failures(df):
                result.failures.append(f"{symbol}: {message}")
            result.stats[symbol] = descriptive_stats(df)
    log.info(
        "options_audit_complete", passed=result.passed,
        failures=len(result.failures), symbols=len(result.symbols_checked),
    )
    if alerter is not None:
        if not result.passed:
            alerter.send_alert(
                "critical", "Options data-quality audit FAILED",
                "; ".join(result.failures[:20]),
            )
        else:
            digest = "; ".join(
                f"{s}: rows={st['rows']} iv~{st['median_iv']}" for s, st in result.stats.items()
            )
            alerter.send_alert("info", "Options monthly audit summary", digest or "no data")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_audit.py -v`
Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/options/audit.py tests/test_options_audit.py
git commit -m "feat(options): data-quality audit — greeks/OI/spread sanity (spec §10.6)"
```

---

### Task 13: `scripts/collector_main.py` — scheduler entrypoint (RESTRUCTURED 2026-07-14)

Deltas vs the Schwab-era design: renamed (`collector_main.py` — the container is the
market-data plane, D10; Plan A Task 11's calendar jobs register here later); **no token
jobs** (fixed jobs drop from 4 to 3); snapshot jobs fire at **`pull_time_et`** with
**per-label `misfire_grace_s`** (D8/D9); a **boot-time self-check** (reconcile + lookback
health check) runs before the scheduler starts; the health check scans the **last
`health_lookback_days` trading days**, not just today (D9 — the watchdog survives being
down at its own check time).

**Files:**
- Create: `scripts/collector_main.py`
- Test: `tests/test_options_scheduler.py`

**Interfaces:**
- Consumes: everything from Layers 1–2, `configure_logging`, `DatabaseManager`, `Alerter`, APScheduler (`BackgroundScheduler`, `SQLAlchemyJobStore`, `ThreadPoolExecutor`), `run_data_quality_audit` (T12), `market_calendar.recent_sessions` (T11).
- Produces:
  - `build_app(config_path: str) -> dict` — wires all components (mirrors `scripts/main.py` construction; no TokenManager).
  - `register_jobs(scheduler, components) -> None` — one job per configured snapshot (at `pull_time_et`, with that snapshot's `misfire_grace_s`) + the fixed jobs.
  - `all_job_ids(config) -> list[str]`; `FIXED_JOB_IDS = ["options_health_check", "options_data_audit", "options_offsite_backup"]`.
  - `guarded_snapshot(collector, label, scheduled_pull_utc=None, now=None) -> None` — trading-day guard; passes `scheduled_pull_utc` through for late-fire stamping.
  - `run_health_check(config, collector, store, alerter, now=None) -> None` — for each of the last `health_lookback_days` NYSE sessions × each configured snapshot: all symbols missing → CRITICAL; some missing → WARNING.
  - `boot_self_check(components) -> None` — reconcile + `run_health_check` once at startup (D9).
  - `run_offsite_backup(config, alerter=None) -> None` — rclone sync (subprocess).
  - `main() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options_scheduler.py
from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

from scripts.collector_main import (
    all_job_ids,
    boot_self_check,
    guarded_snapshot,
    register_jobs,
    run_health_check,
)
from swingrl.config.schema import SwingRLConfig


def _components(cfg: SwingRLConfig | None = None) -> dict:
    return {
        "config": cfg or SwingRLConfig(), "collector": MagicMock(), "store": MagicMock(),
        "alerter": MagicMock(), "db": MagicMock(),
    }


def test_register_jobs_registers_snapshots_plus_fixed() -> None:
    """OPT-SCHED-1: one job per snapshot + 3 fixed jobs, stable ids (D4, §17 C2)."""
    cfg = SwingRLConfig()
    scheduler = MagicMock()
    register_jobs(scheduler, _components(cfg))
    registered = {call.kwargs["id"] for call in scheduler.add_job.call_args_list}
    assert registered == set(all_job_ids(cfg))
    assert {"options_decision_snapshot", "options_eod_snapshot"} <= registered
    assert len(registered) == 5  # 2 default snapshots + 3 fixed (no token jobs)


def test_snapshot_jobs_use_pull_time_and_per_label_grace() -> None:
    """OPT-SCHED-2: decision fires at 16:00 with 900s grace; eod 16:35 with 18000s (D8/D9)."""
    scheduler = MagicMock()
    register_jobs(scheduler, _components())
    by_id = {c.kwargs["id"]: c.kwargs for c in scheduler.add_job.call_args_list}
    dec = by_id["options_decision_snapshot"]
    assert (dec["hour"], dec["minute"], dec["misfire_grace_time"]) == (16, 0, 900)
    eod = by_id["options_eod_snapshot"]
    assert (eod["hour"], eod["minute"], eod["misfire_grace_time"]) == (16, 35, 18000)


def test_guarded_snapshot_skips_non_trading_day(monkeypatch) -> None:
    """OPT-SCHED-3: holiday/weekend -> run_snapshot NOT called (spec §9.2)."""
    monkeypatch.setattr(
        "scripts.collector_main.market_calendar.is_trading_day", lambda d: False
    )
    collector = MagicMock()
    guarded_snapshot(collector, "decision", now=datetime(2026, 12, 25, 21, 0, tzinfo=UTC))
    collector.run_snapshot.assert_not_called()


def test_guarded_snapshot_passes_schedule_through(monkeypatch) -> None:
    """OPT-SCHED-4: trading day -> run_snapshot called with scheduled_pull_utc (D8)."""
    monkeypatch.setattr(
        "scripts.collector_main.market_calendar.is_trading_day", lambda d: True
    )
    collector = MagicMock()
    sched = datetime(2026, 7, 14, 20, 0, tzinfo=UTC)
    guarded_snapshot(collector, "decision", scheduled_pull_utc=sched,
                     now=datetime(2026, 7, 14, 20, 1, tzinfo=UTC))
    collector.run_snapshot.assert_called_once()
    assert collector.run_snapshot.call_args.kwargs["scheduled_pull_utc"] == sched


def test_health_check_scans_lookback_days(monkeypatch) -> None:
    """OPT-SCHED-5: a hole YESTERDAY is caught today -> CRITICAL (D9 lookback)."""
    monkeypatch.setattr(
        "scripts.collector_main.market_calendar.recent_sessions",
        lambda as_of, n: [date(2026, 7, 13), date(2026, 7, 14)],
    )
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["_SPX"]
    collector = MagicMock()
    collector.symbols.return_value = ["_SPX", "SPY"]
    store = MagicMock()
    # Everything present on the 14th; NOTHING on the 13th.
    store.snapshot_exists_parquet.side_effect = lambda s, d, label: d == date(2026, 7, 14)
    alerter = MagicMock()
    run_health_check(cfg, collector, store, alerter,
                     now=datetime(2026, 7, 14, 21, 15, tzinfo=UTC))
    assert any(c.args[0] == "critical" for c in alerter.send_alert.call_args_list)


def test_health_check_partial_is_warning(monkeypatch) -> None:
    """OPT-SCHED-6: some symbols missing -> WARNING, not CRITICAL (spec §10.4)."""
    monkeypatch.setattr(
        "scripts.collector_main.market_calendar.recent_sessions",
        lambda as_of, n: [date(2026, 7, 14)],
    )
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["_SPX"]
    collector = MagicMock()
    collector.symbols.return_value = ["_SPX", "SPY"]
    store = MagicMock()
    store.snapshot_exists_parquet.side_effect = lambda s, d, label: s == "_SPX"
    alerter = MagicMock()
    run_health_check(cfg, collector, store, alerter,
                     now=datetime(2026, 7, 14, 21, 15, tzinfo=UTC))
    levels = [c.args[0] for c in alerter.send_alert.call_args_list]
    assert "warning" in levels and "critical" not in levels


def test_boot_self_check_runs_reconcile_and_health(monkeypatch) -> None:
    """OPT-SCHED-7: boot self-check = reconcile + lookback health check (D9)."""
    called = {"health": 0}
    monkeypatch.setattr(
        "scripts.collector_main.run_health_check",
        lambda *a, **k: called.__setitem__("health", called["health"] + 1),
    )
    components = _components()
    boot_self_check(components)
    components["store"].reconcile.assert_called_once()
    assert called["health"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.collector_main'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/collector_main.py
"""Standalone market-data collector container entrypoint (spec §9, §17 C4).

Its OWN scheduler + jobstore. Never touches the trader (A30). No auth (C1).
Plan A Task 11's calendar-ingest jobs register here when that task lands (D10).
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.options import market_calendar
from swingrl.data.options.audit import run_data_quality_audit
from swingrl.data.options.cboe_client import CboeChainClient
from swingrl.data.options.collector import OptionsCollector
from swingrl.data.options.store import OptionsStore
from swingrl.monitoring.alerter import Alerter
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)
_ET = ZoneInfo("America/New_York")

FIXED_JOB_IDS = [
    "options_health_check",
    "options_data_audit",
    "options_offsite_backup",
]


def _snapshot_job_id(label: str) -> str:
    """Stable APScheduler job id for a snapshot label."""
    return f"options_{label}_snapshot"


def all_job_ids(config: SwingRLConfig) -> list[str]:
    """One snapshot job per configured snapshot + the fixed jobs (D4)."""
    return [_snapshot_job_id(s.label) for s in config.options_collector.snapshots] + FIXED_JOB_IDS


def _hhmm(time_et: str) -> tuple[int, int]:
    hour, minute = (int(x) for x in time_et.split(":"))
    return hour, minute


def build_app(config_path: str) -> dict[str, Any]:
    """Load config and wire logging, DB, alerter, and all collector components."""
    config = load_config(config_path)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    db = DatabaseManager(config)
    alerter = Alerter(
        webhook_url=config.alerting.alerts_webhook_url,
        alerts_webhook_url=config.alerting.alerts_webhook_url,
        daily_webhook_url=config.alerting.daily_webhook_url,
        cooldown_minutes=config.alerting.alert_cooldown_minutes,
        consecutive_failures_before_alert=config.alerting.consecutive_failures_before_alert,
        db=db,
    )
    client = CboeChainClient(config.options_collector)
    store = OptionsStore(config.options_collector, db=db)
    collector = OptionsCollector(config, client, store, alerter=alerter)
    return {
        "config": config, "db": db, "alerter": alerter,
        "client": client, "store": store, "collector": collector,
    }


def guarded_snapshot(
    collector: OptionsCollector,
    label: str,
    scheduled_pull_utc: datetime | None = None,
    now: datetime | None = None,
) -> None:
    """Run a snapshot only on NYSE trading days; thread the schedule through (D8)."""
    now = now or datetime.now(UTC)
    quote_date = now.astimezone(_ET).date()
    if not market_calendar.is_trading_day(quote_date):
        log.info("options_snapshot_skipped_non_trading_day", label=label,
                 date=quote_date.isoformat())
        return
    collector.run_snapshot(label, now=now, scheduled_pull_utc=scheduled_pull_utc)


def _scheduled_pull_utc(pull_time_et: str, now: datetime) -> datetime:
    hh, mm = _hhmm(pull_time_et)
    local = now.astimezone(_ET)
    return local.replace(hour=hh, minute=mm, second=0, microsecond=0).astimezone(UTC)


def run_health_check(
    config: SwingRLConfig, collector: OptionsCollector, store: OptionsStore,
    alerter: Alerter, now: datetime | None = None,
) -> None:
    """Verify snapshots over the last health_lookback_days sessions (D9 lookback)."""
    now = now or datetime.now(UTC)
    as_of = now.astimezone(_ET).date()
    sessions = market_calendar.recent_sessions(as_of, config.options_collector.health_lookback_days)
    symbols = collector.symbols()
    for session in sessions:
        for snap in config.options_collector.snapshots:
            present = [
                s for s in symbols
                if store.snapshot_exists_parquet(s, session, snap.label)
            ]
            if not present:
                alerter.send_alert(
                    "critical", f"Options {snap.label} MISSED",
                    f"No {snap.label} snapshot for any symbol on {session.isoformat()}.",
                )
            elif len(present) < len(symbols):
                missing = [s for s in symbols if s not in present]
                alerter.send_alert(
                    "warning", f"Options {snap.label} incomplete",
                    f"Missing {snap.label} for {missing} on {session.isoformat()}.",
                )


def boot_self_check(components: dict[str, Any]) -> None:
    """D9 boot trio: reconcile unsynced Parquet + lookback health check, every start."""
    components["store"].reconcile()
    run_health_check(
        components["config"], components["collector"],
        components["store"], components["alerter"],
    )
    log.info("options_boot_self_check_done")


def run_offsite_backup(config: SwingRLConfig, alerter: Alerter | None = None) -> None:
    """Sync captured data offsite via rclone (3-2-1 backup; spec §13)."""
    backup = config.options_collector.backup
    if not backup.enabled:
        return
    cmd = ["rclone", "sync", config.options_collector.output_dir, backup.rclone_remote]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info("options_offsite_backup_ok", remote=backup.rclone_remote)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.error("options_offsite_backup_failed", error=str(exc))
        if alerter is not None:
            alerter.send_alert("warning", "Options offsite backup failed", str(exc))


def register_jobs(scheduler: Any, components: dict[str, Any]) -> None:
    """Register per-snapshot + fixed cron jobs on the scheduler (D4, D8/D9)."""
    config: SwingRLConfig = components["config"]
    oc = config.options_collector
    collector = components["collector"]
    store = components["store"]
    alerter = components["alerter"]
    db = components["db"]

    for snap in oc.snapshots:
        sh, sm = _hhmm(snap.pull_time_et)

        def _job(label: str = snap.label, pull: str = snap.pull_time_et) -> None:
            now = datetime.now(UTC)
            guarded_snapshot(collector, label,
                             scheduled_pull_utc=_scheduled_pull_utc(pull, now), now=now)

        scheduler.add_job(
            _job, trigger="cron", day_of_week="mon-fri", hour=sh, minute=sm,
            timezone="America/New_York", id=_snapshot_job_id(snap.label),
            misfire_grace_time=snap.misfire_grace_s, replace_existing=True,
        )

    hh, hm = _hhmm(oc.health_check_time_et)
    scheduler.add_job(
        run_health_check, trigger="cron", day_of_week="mon-fri", hour=hh, minute=hm,
        timezone="America/New_York", args=[config, collector, store, alerter],
        id="options_health_check", replace_existing=True,
    )
    ah, am = _hhmm(oc.integrity.audit_time_et)
    scheduler.add_job(
        run_data_quality_audit, trigger="cron", day=oc.integrity.audit_day_of_month,
        hour=ah, minute=am, timezone="America/New_York",
        kwargs={"config": config, "db": db, "alerter": alerter},
        id="options_data_audit", replace_existing=True,
    )
    bh, bm = _hhmm(oc.backup.time_et)
    scheduler.add_job(
        run_offsite_backup, trigger="cron", hour=bh, minute=bm,
        timezone="America/New_York", args=[config, alerter],
        id="options_offsite_backup", replace_existing=True,
    )


def _make_signal_handler(scheduler: Any, stop_event: threading.Event):
    def handler(_signum, _frame) -> None:
        log.info("options_collector_shutting_down")
        scheduler.shutdown(wait=False)
        stop_event.set()

    return handler


def main() -> int:
    """Build, self-check, register jobs, start the scheduler, and block."""
    parser = argparse.ArgumentParser(description="SwingRL market-data collector")
    parser.add_argument("--config", default="config/swingrl.yaml")
    args = parser.parse_args()

    components = build_app(args.config)
    boot_self_check(components)  # D9: every restart is a self-audit

    scheduler = BackgroundScheduler(
        jobstores={
            "default": SQLAlchemyJobStore(
                url=f"sqlite:///{components['config'].options_collector.apscheduler_db_path}"
            )
        },
        executors={"default": ThreadPoolExecutor(max_workers=4)},
        job_defaults={"coalesce": True, "max_instances": 1},
    )
    register_jobs(scheduler, components)

    stop_event = threading.Event()
    handler = _make_signal_handler(scheduler, stop_event)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    scheduler.start()
    log.info("options_collector_started", jobs=all_job_ids(components["config"]))
    stop_event.wait()
    log.info("options_collector_exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_scheduler.py -v`
Expected: all 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/collector_main.py tests/test_options_scheduler.py
git commit -m "feat(options): collector_main — pull-time jobs, per-label grace, lookback health, boot self-check (spec §9, §17 C4)"
```

---

### Task 14: Docker service `swingrl-collector` (RESTRUCTURED 2026-07-14)

**Files:**
- Modify: `docker-compose.yml` (add the `swingrl-collector` service)

**Interfaces:**
- Consumes: the existing `production` Dockerfile target (ships `src/` + `scripts/` already — no Dockerfile change needed), `.env` (Discord webhooks + DB only — no provider secrets), `br0` network (pg16 access).
- Produces: an always-on `swingrl-collector` container running `scripts/collector_main.py` with its own lifecycle and its **own pinned image tag** (D9: Plan A/B image churn must never recreate it via a bare `up -d`). Rebuilding/restarting it never touches `swingrl` (A30) — and per the quiet-window rule it is itself only recreated **outside 15:30–16:45 ET** on trading days.

- [ ] **Step 1: Add the service to `docker-compose.yml`**

```yaml
  swingrl-collector:
    container_name: swingrl-collector
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    # Pinned tag (D9): bumped by hand, only outside the 15:30-16:45 ET quiet window.
    # A bare `docker compose up -d` after unrelated image builds must not recreate this.
    image: swingrl-collector:2026-07-15
    cpus: 2.0
    env_file: .env
    environment:
      - TZ=America/New_York
    command: ["python", "scripts/collector_main.py"]
    volumes:
      - ./data:/app/data
      - ./db:/app/db
      - ./config:/app/config
      - ./logs:/app/logs
    networks:
      - default
      - br0
    restart: unless-stopped
```

- [ ] **Step 2: Validate the compose file**

Run: `docker compose config >/dev/null && echo OK`
Expected: `OK` (compose parses; the new service resolves `br0`, `.env`, and the build target).

- [ ] **Step 3: Build the image (no-cache, per homelab convention)**

Run: `docker compose build --no-cache swingrl-collector`
Expected: build succeeds. **Service-scoped commands only** — never bare `docker compose build`/`up -d` on a host with any always-on swingrl service running.

- [ ] **Step 4: Smoke-test imports inside the image**

Run: `docker compose run --rm --no-deps swingrl-collector python -c "import scripts.collector_main as m; print(m.FIXED_JOB_IDS)"`
Expected: prints the 3 fixed job ids. (Confirms every module imports cleanly in the container; no scheduler is started.)

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml
git commit -m "feat(options): swingrl-collector always-on container — pinned tag, A30-isolated (spec §9.1, §17 C4)"
```

---

### Task 15: Documentation & runbooks (RESTRUCTURED 2026-07-14)

**Files:**
- Create: `docs/options/ops.md`
- Create: `docs/options/data-caveats.md`
- Create: `docs/options/week1-data-quality-audit.md`

(The Schwab-era OAuth docs — register-app, first-oauth, weekly-reauth — are NOT written;
they belong to the shelved fallback and their content lives in the spec + git history.)

- [ ] **Step 1: `ops.md`** — container lifecycle (`docker compose up -d swingrl-collector`, logs, restart; A30-isolated from the trader; **pinned-tag bump discipline + the 15:30–16:45 ET quiet window**; service-scoped compose commands only); the jobs and their times (pull 16:00 → 15:45 state, pull 16:35 → frozen close; health 17:15 with lookback; monthly audit; nightly backup); boot self-check behavior (every restart = reconcile + lookback health check — deliberate restarts are safe outside the quiet window); where data lands (Parquet dirs + Postgres tables); disk-growth monitoring + the `postgres_store_raw_json` flag (decision D5); how reconcile self-heals a pg16 outage; alert catalogue (spec §10.4, minus the retired auth alerts).

- [ ] **Step 2: `data-caveats.md`** — the "this data isn't quite what it looks like" doc:
  - **Delay convention (D8):** the feed is ~15-min delayed; `snapshot_time_utc` = market moment, `pulled_at_utc` = fetch time, `raw_header.payload_timestamp` = CBOE's own stamp, `raw_header.late_by_s` = job lateness. Measured offset recorded here from T6.
  - **`iv` is a decimal fraction** (0.1164 = 11.64%) — CBOE convention, opposite of Schwab's percent; illiquid contracts report `iv = 0.0` → stored as NaN.
  - **OI is T-1/once-daily** (decision D6) + the splice-time date-convention note.
  - **16:35 = close mark, not settlement** — official settlement values belong to the separate premium-project sourcing track.
  - **Greeks/IV are CBOE's vendor values** — recompute from quotes + FRED rate if cross-source consistency is ever needed; the trustworthy fields are quotes + contract identity.
  - **Endpoint has no SLA** — fallback ladder (Schwab #1 shelved w/ registered app; moomoo #2; spec §17 C2) and what breakage looks like (health-check CRITICAL, not silence).
  - **CBOE historical-candles endpoint** (`…/charts/historical/{symbol}.json`): `_SPX` daily OHLCV to 1975 (verified 2026-07-14) = the future premium env's underlying-history source + a cross-check for equity dailies. **Never a replacement for the trader's Alpaca/Binance sources** (source-seam discipline; Plan B `data_fingerprint` guards it).
  - **Real-time chains for the future live premium trader**: comes from the executing broker, decided in the premium-trader spec (candidates ranked: Schwab / moomoo / Tradier-IBKR); **source-seam note** — run an overlap capture (CBOE-delayed vs broker-real-time) at go-live to measure the offset.

- [ ] **Step 3: `week1-data-quality-audit.md`** — the manual checks (spec §10.6): reconstruct an SPX IV surface; confirm delta ∈ [-1,1] and monotone across strikes; OI populated on liquid names; `bid ≤ ask` with plausible spreads; decision→eod drift non-trivial and plausible; **delay-offset spot-check against the T6 finding**. Include the exact SQL to pull one snapshot from `options_chains`.

- [ ] **Step 4: Commit**

```bash
git add docs/options/
git commit -m "docs(options): ops, data-caveats (delay/iv/OI/seam/fallbacks), week-1 audit runbook (spec §13, §17)"
```

---

### Task 16 🛑: Homelab CI, live migration, deploy, first live run (RESTRUCTURED 2026-07-14)

**Human-in-the-loop gate.** Resolves the first-run unknowns empirically. Requires an
approved deploy per CLAUDE.md — **do not deploy without explicit approval.** All steps run
**outside the 15:30–16:45 ET quiet window**.

**Files:**
- Create: `tests/test_chain_parser_real_fixture.py` (pins the parser to the real capture)

- [ ] **Step 0: Verify the Wave-0 CI-cleanup fix is present** (prerequisite, C4/G-5):
  `grep -n "compose down" scripts/ci-homelab.sh` must show the cleanup scoped to the dev
  compose project (`$DEV_COMPOSE down`), never the production project. If not, STOP — the
  fix must merge first (every later CI run would kill the collector).

- [ ] **Step 1: Write the pin-to-real-fixture test**

```python
# tests/test_chain_parser_real_fixture.py
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from swingrl.data.options.chain_parser import parse_chain
from swingrl.data.options.collector import check_schema_drift

_FIXTURE = Path("tests/fixtures/cboe_chain_spx.json")


@pytest.mark.skipif(not _FIXTURE.exists(), reason="real fixture not yet captured (T6)")
def test_parse_real_spx_fixture_no_schema_drift() -> None:
    """OPT-PARSE-10: parser handles the REAL captured chain, no drift (spec §12, §17 C1)."""
    raw = json.loads(_FIXTURE.read_text())
    assert check_schema_drift(raw) == [], "real payload field names differ — update the mapping"
    parsed = parse_chain(
        raw, underlying_symbol="_SPX", snapshot_label="eod", quote_date=date(2026, 7, 14),
        snapshot_time_utc=datetime(2026, 7, 14, 20, 15, tzinfo=UTC),
        pulled_at_utc=datetime(2026, 7, 14, 20, 35, tzinfo=UTC),
        schema_version="v1", is_early_close=False,
    )
    assert len(parsed.contracts) > 0
    assert parsed.contracts["iv"].notna().any()
    assert parsed.contracts["strike"].gt(0).all()
    assert parsed.header["number_of_contracts"] == len(parsed.contracts)
```

Run it; if it fails on drift, the real field names differ from T7's mapping — fix the
mapping + `EXPECTED_CONTRACT_FIELDS`, re-run T7's tests, then this. Commit.

- [ ] **Step 2: Apply the additive migration to live pg16** (approval required)

```bash
cd ~/swingrl && git fetch origin && git checkout <branch> && git pull origin <branch>
DATABASE_URL=<pg16-url> uv run python scripts/migrations/add_options_capture_tables.py
```
Expected: `options_capture_migration_applied`. Verify `options_snapshots` + `options_chains` + current-month partition exist. Additive-only — the trader (if running) is unaffected (A30).

- [ ] **Step 3: Homelab CI** (per CLAUDE.md; outside the quiet window)

```bash
cd ~/swingrl && bash scripts/ci-homelab.sh --no-cache
```
Expected: PASS, including the DB-gated tests (T9/T10 against pg16), and the collector still up afterwards (Step 0's fix proven).

- [ ] **Step 4: Deploy the container** (approval required)

```bash
cd ~/swingrl && docker compose build --no-cache swingrl-collector \
  && docker compose up -d swingrl-collector && docker compose logs -f swingrl-collector
```
Confirm `options_boot_self_check_done` then `options_collector_started` with the 5 job ids.

- [ ] **Step 5: First live snapshots — RECORD the first-run findings**

Wait for the 16:00/16:35 crons (or trigger once manually inside the container via
`build_app` + `guarded_snapshot`). Record in `data/options_eod/cboe/metadata.json` + spec
§17 C1's table: **(a)** measured delay offset (confirms T6), **(b)** contract counts per
symbol (drift-guard baselines), **(c)** iv-zero sentinel behavior on illiquid names
(confirms the T7 rule), **(d)** wall-clock duration of a full 9-symbol run. Confirm a
Discord INFO digest arrived — **this is also the system's first end-to-end Discord proof**
(predates Plan A Task 16's trader-path proof; different container, same Alerter class).

- [ ] **Step 6: Stand up the offsite backup** (spec §13)

Configure the `rclone` remote (`b2:swingrl-options`), then verify:
```bash
docker compose exec swingrl-collector python -c \
  "from scripts.collector_main import build_app, run_offsite_backup; \
   c=build_app('config/swingrl.yaml'); run_offsite_backup(c['config'], c['alerter'])"
```
Confirm `options_offsite_backup_ok` and that objects appear at the remote.

- [ ] **Step 7: Week-1 watch** — run the week-1 data-quality audit runbook (T15 doc);
  verify the health check's lookback catches a synthetic hole (rename one Parquet for a
  past day in a scratch copy — or simply review its log output); confirm daily INFO
  digests. (No token watch — there is no token.)

- [ ] **Step 8: Commit findings/tuning**

```bash
git add config/swingrl.yaml docs/superpowers/specs/2026-07-14-schwab-options-collector-design.md docs/options/data-caveats.md
git commit -m "chore(options): record first-run delay/volume findings + tuning (spec §17 C1)"
```

---

## Self-Review — Spec Coverage

| Spec section (as amended §17) | Covered by |
|---|---|
| §2 goal, forward-capture, dual-use library | T2–T13 (library in `src/swingrl/data/options/`) |
| §5 symbols & config (nothing hardcoded) | T2 (`OptionsCollectorConfig`), T11 (`symbols()` = index + equity) |
| §6.1 two snapshots (config-driven) + early-close provenance | T2 (snapshots, pull vs market time), T11 (`is_early_close`, `run_snapshot`), T13 (per-snapshot job registration, D4) |
| §6.2 capture everything (typed + raw_json) | T7 (`CONTRACT_COLUMNS` + `raw_json`), T8 (Parquet), T10 (JSONB) |
| §6.3 flattened grain + column mapping | T7 (CBOE fields + OSI parsing) |
| §6.4 snapshot-level context / header | T7 (`header`), T8 (sidecar), T10 (`options_snapshots`) |
| §7 auth | **Retired for the primary path (C1/C2)** — T3/T4 tombstones; Schwab fallback design shelved in spec + git history |
| §8.1 Parquet layout + atomic + resume unit | T8 |
| §8.2 Postgres tables, partitions, idempotency, reconcile | T9 (schema/migration), T10 (sync/reconcile) |
| §8.3 metadata.json sidecar | T6 (delay findings), T15/T16 (provenance recorded) |
| §9 scheduling & runtime (own scheduler/jobstore) | T13, T14 |
| §10.1–10.2 idempotency, resumability, isolation | T8/T10 (skip + ON CONFLICT), T11 (per-symbol try/except) |
| §10.3 typed errors + retry | T5 (`swingrl_retry`, `DataError`), all modules |
| §10.4 Discord alert catalogue | T11/T13 (`send_alert` routing; auth alerts retired) |
| §10.5 silent-corruption guards (drift / contract-count / OI stability) | T11 (`check_schema_drift`, count-drop guard), T12 (`oi_stability_failures`, D6) |
| §10.6 data-quality audit | T12, T15 (runbook), T16 (week-1) |
| §11 entitlement check | **Superseded by C3** — delay-convention measurement (T6) + `late_by_s` provenance (T7/T11) |
| §12 testing (fixture-pinned) | every task's tests; T6 (real fixtures from day one, D2) + T16 (re-pin) |
| §13 security, offsite backup, storage growth | T13/T16 (rclone), T15 (data-caveats), T2/T10 (`postgres_store_raw_json`, D5); no secrets exist (C2) |
| §14 build sequence | T1–T16 as restructured (dependency diagram above) |
| §15 open questions (empirical) | T6 + T16 (delay offset, volumes, iv-zero convention — recorded, not guessed) |
| §16 success criteria | full plan; verified at T16 |
| §17 C1–C4 (CBOE, fallback ladder, timing, restart resilience) | T2 (config), T5 (client), T6 (delay 🛑), T7 (provenance), T11 (guards + late stamping), T13 (grace/lookback/boot), T14 (pinned tag), T16 Step 0 (CI fix check) |

**Placeholder scan:** no "TBD"/"implement later"/"add error handling" — every code step shows real code. **Type consistency:** `ParsedChain(header, contracts)`, `parse_chain(...)`, `parse_osi(...)`, `OptionsStore.write_snapshot/read_snapshot/sync_to_postgres/reconcile/last_snapshot_row_count`, `CboeChainClient.get_option_chain/chain_url`, `OptionsCollector.run_snapshot/symbols`, `SnapshotResult`, and `CONTRACT_COLUMNS`/`DB_CHAIN_COLUMNS` names are used identically across producing and consuming tasks. (T9/T10's DDL keeps its superset of nullable columns — CBOE-absent fields stay NULL; no DDL change.)

---

## Execution Handoff

Plan complete at `docs/superpowers/plans/2026-07-14-schwab-options-collector-plan.md`
(restructured 2026-07-14: CBOE primary).

**Master-sequence context (2026-07-14 reconciliation):** this plan is Track C, Wave 1 —
collector-first. Wave 0 prerequisites: cut `swingrl/2.R-C-options-collector` off the
integration branch; land the `ci-homelab.sh` dev-scoped-cleanup fix. Plan A Tasks 1–5 may
run in parallel (disjoint files). See `.planning/V1.1_EXECUTION_PLAN.md` § "Master
sequence".

Per the standing instruction for this work: **STOP here — do not begin implementation
without a separate go-ahead.** When you give the go-ahead, two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, two-stage review between tasks, fast iteration (`superpowers:subagent-driven-development`).
2. **Inline Execution** — tasks executed in this session with batch checkpoints (`superpowers:executing-plans`).

Gates that always require a human: **T6 Step 3** (trading-day delay-convention
measurement) and **T16** (live pg16 migration, homelab CI, deploy, offsite backup) — the
latter needs explicit deploy approval per CLAUDE.md. Fallback note: Schwab (registered
app, shelved design) and moomoo (researched 2026-07-14) are the ranked provider fallbacks
— spec §17 C2.
