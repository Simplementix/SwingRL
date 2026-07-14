# Schwab EOD Option-Chain Data Collector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reliably capture the full option chain for `$SPX` + the 8 equity ETFs twice per trading day (15:45 ET decision, 16:35 ET close) into durable, idempotent, resumable storage (Parquet → Postgres), running in a standalone always-on container that never touches the live trader.

**Architecture:** Three layers. **Layer 1** is a reusable Schwab client library (`src/swingrl/data/options/`): token manager, thin schwab-py wrapper, chain parser, and store. **Layer 2** is the EOD collector orchestration (`collector.py`) — per-symbol fetch→parse→store with isolation, silent-corruption guards, and alert routing. **Layer 3** is the container entrypoint (`scripts/options_collector_main.py`) — its own APScheduler, its own jobstore, six cron jobs. The library choice (schwab-py) is quarantined behind our wrapper so the future SPX premium-selling trader reuses Layer 1 unchanged.

**Tech Stack:** Python 3.11, `schwab-py` (OAuth + `/marketdata/v1/chains`), pandas + pyarrow (Parquet), psycopg 3 + `psycopg_pool` (Postgres 16, JSONB + monthly range partitions), APScheduler `BackgroundScheduler` (SQLAlchemy sqlite jobstore), `exchange_calendars` (XNYS), Pydantic v2 / pydantic-settings (config), structlog (logging), tenacity via `swingrl_retry` (retry), the existing `Alerter` (Discord), Docker Compose.

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
- **Secrets in `.env` only.** `SCHWAB_API_KEY` / `SCHWAB_APP_SECRET` read from env, never YAML or git. Token file `secrets/schwab_token.json`: gitignored, `chmod 600`, mounted into the container (never baked into the image).
- **TDD, tests-first.** Write the failing test, commit RED, then GREEN implementation. Test files `tests/test_<module>.py`, functions `test_<behavior>`, docstring `"""OPT-<id>: what is tested (spec §N)."""`, fixtures from `tests/conftest.py`. Run `uv run pytest tests/ -v`.
- **Never skip pre-commit.** Fix the hook failure; never pass `--no-verify`.
- **A30 compliance.** The collector writes only to its own tables (`options_snapshots`, `options_chains`) and `data/options_eod/`. It never writes to `models/active/` or any shared trader table. Schema changes are **additive only** (new tables). The container is a separate always-on service — rebuilding/restarting it never touches the trader.

---

## Glossary (no undefined shorthand)

Carried from spec §1, plus plan-specific terms. Read this before the tasks.

| Term | Plain meaning |
|---|---|
| **Option chain** | Full list of option contracts for one underlying — every strike × expiration × call/put — with prices and analytics. |
| **Underlying** | The thing the option is on (`SPY`, `$SPX`). |
| **`$SPX`** | Schwab's request symbol for S&P 500 **index** options (European, cash-settled). Exact form confirmed at first run (`$SPX` vs `$SPX.X`). |
| **Greeks** | Risk sensitivities: delta, gamma, theta, vega, rho. |
| **IV** | Implied volatility. Schwab returns it as a **percent** (12.34 = 12.34%). |
| **OI** | Open interest — contracts currently outstanding. |
| **DTE** | Days to expiration. |
| **OSI id** | The standardized option contract symbol (e.g. `SPXW  260718C05000000`) — the natural key part. |
| **EOD** | End of day. |
| **Decision snapshot** | Chain captured at **15:45 ET** — the moment the future premium agent trades. **Un-backfillable.** |
| **EOD snapshot** | Chain captured at **16:35 ET** — frozen post-close state (a few min past the 16:15 options close + the 15-min-delay boundary; see decision D3). |
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
| **schwab-py** | The Python library we use to talk to the Schwab API (handles OAuth). Quarantined behind our wrapper. |
| **Access token** | Short-lived (~30 min) API credential; auto-refreshed by schwab-py. |
| **Refresh token** | Longer-lived credential; Schwab's expires in **7 days** (design for hard expiry, verify in week 1). |
| **OAuth manual flow** | Headless login: print URL, log in in a browser, copy the redirected URL back. No local server needed. |
| **`isDelayed`** | Top-level response flag: real-time vs 15-min-delayed quotes. |
| **`isChainTruncated`** | Top-level flag meaning Schwab returned a **partial** chain. Must be checked or strikes are silently lost. |
| **Schema drift** | Schwab silently adding/renaming/removing response fields; can null out our typed columns with no error. |
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
| `OPT-CFG` | `OptionsCollectorConfig` schema | §5 |
| `OPT-AUTH` | `schwab_auth` token manager | §7 |
| `OPT-CLI` | `scripts/schwab_reauth.py` | §7.2 |
| `OPT-CLIENT` | `schwab_client.get_option_chain` | §4, §10.5 |
| `OPT-PARSE` | `chain_parser` | §6.3 |
| `OPT-STORE` | `store` Parquet + Postgres + reconcile | §8 |
| `OPT-SCHEMA` | Postgres tables + partitions + migration | §8.2 |
| `OPT-COLLECT` | `collector` orchestration + guards | §6, §10 |
| `OPT-AUDIT` | `audit` data-quality | §10.6 |
| `OPT-SCHED` | scheduler + calendar guards | §9 |

---

## File Structure

**New package** `src/swingrl/data/options/` (one responsibility per file):

| File | Responsibility |
|---|---|
| `__init__.py` | Package marker; re-exports the public surface. |
| `market_calendar.py` | `is_trading_day(date)`, `is_early_close(date)` over XNYS. |
| `schwab_auth.py` | `TokenManager`: load client, token-age math, reminder thresholds, `invalid_client`→CRITICAL. |
| `schwab_client.py` | `SchwabOptionsClient.get_option_chain(symbol)`: rate-limit, retry, truncation-aware re-fetch, raw dict out. |
| `chain_parser.py` | `parse_chain(...) -> ParsedChain`: raw dict → typed DataFrame + `raw_json` + header. |
| `schema.py` | `ensure_options_schema(conn)`, `ensure_monthly_partition(conn, quote_date)`; DDL constants. |
| `store.py` | `OptionsStore`: atomic Parquet write, skip-existing, Postgres JSONB sync, reconcile. |
| `collector.py` | `OptionsCollector.run_snapshot(label)`: per-symbol loop, isolation, guards, alert routing. |
| `audit.py` | `run_data_quality_audit(...)`: greeks/OI/spread sanity over a trailing window. |

**New scripts:**

| File | Responsibility |
|---|---|
| `scripts/schwab_reauth.py` | Manual-OAuth CLI: writes/refreshes `secrets/schwab_token.json`. |
| `scripts/capture_chain_fixture.py` | One-shot: pull one real chain, sanitize, save as the pinning fixture (T6). |
| `scripts/options_collector_main.py` | Container entrypoint: configure logging, build Alerter, own scheduler, register jobs, block. |
| `scripts/migrations/add_options_capture_tables.py` | Standalone additive migration (the "V011") for the initial live apply. |

**Modified:** `src/swingrl/config/schema.py` (add `OptionsCollectorConfig`), `config/swingrl.yaml` (add `options_collector:` block), `pyproject.toml` (add `schwab-py`), `.gitignore` (`secrets/`, `data/options_eod/` covered), `.env.example` (Schwab keys), `docker-compose.yml` (new `swingrl-options` service).

**Design decisions locked with the user (deviations from a literal reading of the spec):**
- **D1 — "V011" migration is self-contained.** There is no numbered migration ledger in the repo. We implement V011 as (a) a standalone additive script `scripts/migrations/add_options_capture_tables.py` (mirrors the existing `scripts/migrations/add_cps_columns.py`) for the one-time live apply, **and** (b) `schema.ensure_options_schema(conn)` called at collector startup (idempotent `CREATE TABLE IF NOT EXISTS`). The trader's `postgres_schema.py` is **not** touched — options tables live only in this subsystem. This keeps A30 isolation clean.
- **D2 — parser is TDD'd against a hand-authored fixture first, then pinned to a real capture.** `chain_parser` (T7) is built against a representative fixture derived from the documented Schwab schema so TDD isn't blocked on live OAuth. T6 (manual gate) captures one real sanitized chain; T16 re-runs the parser tests against it and reconciles any field-name differences.
- **D3 — EOD snapshot is 16:35, not 16:30.** The spec says 16:30, but 16:30 sits *exactly* on the 15-min-delay boundary (options freeze at 16:15; a delayed feed first shows that close at 16:30:00 sharp). Firing on the edge risks catching a pre-close quote if clock skew / fetch time / an occasional >15-min delay pushes it early. 16:35 clears the boundary with a 5-minute buffer at zero cost (values are frozen after ~16:15). Chosen from day one so the forward-captured history has no self-inflicted time-of-day seam. The 15:45 `decision` snapshot is unaffected (intraday, market open). We deliberately did **not** add a 16:00 "close" snapshot — the splice-alignment case for it largely washes out because the premium pipeline feeds z-scores/ratios/spreads, not raw levels, and its only independent benefit (simultaneous underlying+options for IV inversion) is small and helps only under real-time entitlement.
- **D4 — snapshot jobs are config-driven.** The scheduler registers **one job per entry in `options_collector.snapshots`** (T13), and the label validator accepts `{open, decision, close, eod}` — so revising the number/times of snapshots (e.g. adding 16:00 later) is a YAML edit, not a code change ("nothing hardcoded").
- **D5 — `postgres_store_raw_json` flag.** `raw_json` is always kept in Parquet (the durable archive). Its copy in Postgres JSONB — the fastest-growing, least-queried storage — is behind a config flag (default **on**, spec-compliant), so it can be flipped off at first-run once real GB/day is known. `options_chains.raw_json` is therefore nullable. If flipped off later, reconcile can re-load from Parquet.
- **D6 — OI is captured but understood as T-1/once-daily.** Open interest updates once overnight (OCC) and is stable intraday, so it is identical across same-day snapshots and one day in arrears — which is the correct no-lookahead value. The audit adds an invariant that flags any same-day OI *difference* (a bug signal), and a splice-time note to verify the OI date-convention matches the purchased history on the overlap.

---

## Task Dependency Order

```
T1 (setup) → T2 (config) → T3 (auth) → T4 (reauth CLI) → T5 (client)
   → T6 🛑 manual: first OAuth + real fixture + isDelayed
   → T7 (parser) → T8 (store: Parquet) → T9 (schema) → T10 (store: Postgres+reconcile)
   → T11 (collector) → T12 (audit) → T13 (scheduler) → T14 (docker) → T15 (docs)
   → T16 🛑 manual: homelab CI + first live run + pin fixture + offsite backup
```

T7 can begin in parallel with T6 (hand-authored fixture). Everything T8+ depends on T7. 🛑 = human-in-the-loop gate (spec first-run unknowns).

---

### Task 1: Project setup — dependency, gitignore, env template

**Files:**
- Modify: `pyproject.toml` (add `schwab-py` to `[project].dependencies`)
- Modify: `.gitignore` (add `secrets/` and confirm `data/options_eod/` coverage)
- Modify: `.env.example` (add Schwab keys)
- Create: `secrets/.gitkeep`
- Create: `data/options_eod/.gitkeep`
- Create: `src/swingrl/data/options/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the `swingrl.data.options` package namespace; `schwab-py` importable as `schwab`.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, under `[project].dependencies`, add (keep the list alphabetical if it already is):

```toml
    "schwab-py>=1.5.0",
```

- [ ] **Step 2: Lock and install**

Run: `uv lock && uv sync`
Expected: resolves and installs `schwab-py` (and its deps `authlib`, `httpx`). No errors.

- [ ] **Step 3: Verify the import**

Run: `uv run python -c "import schwab; from schwab.auth import client_from_manual_flow, client_from_token_file; print(schwab.__version__)"`
Expected: prints a version string (≥ 1.5.0). This pins the two auth entrypoints we depend on.

- [ ] **Step 4: gitignore + secrets/data scaffolding**

The `.gitignore` already ignores `data/*` (keeping `.gitkeep`) and `.env`. Add a `secrets/` rule. Append under the "Environment and secrets" group:

```gitignore
# Schwab options collector secrets (token file holds account-scoped access)
secrets/*
!secrets/.gitkeep
```

Create the placeholder dirs so the paths exist in a fresh clone:

```bash
mkdir -p secrets data/options_eod
touch secrets/.gitkeep data/options_eod/.gitkeep
chmod 700 secrets
```

- [ ] **Step 5: .env.example — Schwab keys**

Append to `.env.example` (double-quoted `KEY="value"` format, matching the file's convention):

```bash
# --- Charles Schwab (options-chain data collector; Market-Data-only app) ---
# Register a Market Data app at https://developer.schwab.com; callback https://127.0.0.1:8182
SCHWAB_API_KEY="your-schwab-app-key"          # pragma: allowlist secret
SCHWAB_APP_SECRET="your-schwab-app-secret"    # pragma: allowlist secret
```

- [ ] **Step 6: Confirm nothing secret is staged**

Run: `git status --short && git check-ignore secrets/schwab_token.json data/options_eod/SPX`
Expected: `git status` shows the new `.gitkeep`s, `pyproject.toml`, `uv.lock`, `.gitignore`, `.env.example`, `__init__.py` — **not** any token file. `git check-ignore` prints both paths (proving they're ignored).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .env.example secrets/.gitkeep data/options_eod/.gitkeep src/swingrl/data/options/__init__.py
git commit -m "chore(options): add schwab-py dep, secrets/ gitignore, env template, package scaffold"
```

---

### Task 2: `OptionsCollectorConfig` schema + YAML block

**Files:**
- Modify: `src/swingrl/config/schema.py` (add the config models + attach to `SwingRLConfig`)
- Modify: `config/swingrl.yaml` (add the `options_collector:` block)
- Test: `tests/test_options_config.py`

**Interfaces:**
- Consumes: `load_config()`, `SwingRLConfig`, `EquityConfig.symbols` (existing, `src/swingrl/config/schema.py`).
- Produces:
  - `OptionsCollectorConfig(BaseModel)` with nested `OptionsAuthConfig`, `OptionsChainConfig`, `OptionsSnapshotConfig`, `OptionsIntegrityConfig`, `OptionsBackupConfig`.
  - Attribute path `config.options_collector.*`.
  - Fields relied on downstream: `enabled: bool`, `provider: str`, `index_symbols: list[str]`, `include_equity_symbols: bool`, `output_dir: str`, `schema_version: str`, `snapshots: list[OptionsSnapshotConfig]` (each `label: str`, `time_et: str`), `chain.contract_type/strike_range/include_underlying_quote/from_date/to_date/strike_count`, `auth.token_path/api_key_env/app_secret_env/callback_url/max_token_age_days/reminder_days`, `rate_limit_per_sec: float`, `health_check_time_et: str`, `token_reminder_time_et: str`, `apscheduler_db_path: str`, `integrity.fail_on_truncated/refetch_dte_chunks/audit_day_of_month/audit_time_et`, `backup.enabled/rclone_remote/time_et`.

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
    """OPT-CFG-1: OptionsCollectorConfig has spec defaults (spec §5)."""
    cfg = OptionsCollectorConfig()
    assert cfg.enabled is True
    assert cfg.provider == "schwab"
    assert cfg.index_symbols == ["$SPX"]
    assert cfg.include_equity_symbols is True
    assert cfg.output_dir == "data/options_eod/schwab"
    assert cfg.schema_version == "v1"
    assert cfg.apscheduler_db_path == "db/apscheduler_options.sqlite"
    assert cfg.postgres_store_raw_json is True


def test_options_snapshots_are_decision_and_eod() -> None:
    """OPT-CFG-2: two snapshots, 15:45 decision + 16:35 eod (spec §6.1, decision D3)."""
    cfg = OptionsCollectorConfig()
    labels = [(s.label, s.time_et) for s in cfg.snapshots]
    assert labels == [("decision", "15:45"), ("eod", "16:35")]


def test_options_auth_defaults() -> None:
    """OPT-CFG-3: auth defaults match spec (spec §5, §7)."""
    cfg = OptionsCollectorConfig()
    assert cfg.auth.token_path == "secrets/schwab_token.json"
    assert cfg.auth.api_key_env == "SCHWAB_API_KEY"  # pragma: allowlist secret
    assert cfg.auth.app_secret_env == "SCHWAB_APP_SECRET"  # pragma: allowlist secret
    assert cfg.auth.callback_url == "https://127.0.0.1:8182"
    assert cfg.auth.max_token_age_days == pytest.approx(6.5)
    assert cfg.auth.reminder_days == [5, 6]


def test_options_config_attached_to_root() -> None:
    """OPT-CFG-4: load_config exposes options_collector (spec §5)."""
    cfg = load_config("config/swingrl.yaml")
    assert cfg.options_collector.enabled is True
    assert cfg.options_collector.integrity.fail_on_truncated is False


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
    """OPT-CFG-6: snapshot label validated against {decision, eod} (spec §6.1)."""
    from swingrl.utils.exceptions import ConfigError

    with pytest.raises(ConfigError):
        OptionsSnapshotConfig(label="lunchtime", time_et="12:00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'OptionsCollectorConfig'`.

- [ ] **Step 3: Add the config models**

In `src/swingrl/config/schema.py`, near the other sub-config models (e.g. after `SchedulerConfig`), add:

```python
class OptionsSnapshotConfig(BaseModel):
    """One scheduled snapshot: a label and an ET wall-clock time."""

    label: str = Field(default="decision")
    time_et: str = Field(default="15:45")

    @field_validator("label")
    @classmethod
    def label_known(cls, v: str) -> str:
        """Recognized snapshot labels (config-driven; add times via YAML, not code)."""
        allowed = {"open", "decision", "close", "eod"}
        if v not in allowed:
            raise ConfigError(f"options snapshot label must be one of {sorted(allowed)}, got {v!r}")
        return v


class OptionsAuthConfig(BaseModel):
    """Schwab OAuth token + credential-env configuration."""

    token_path: str = Field(default="secrets/schwab_token.json")
    api_key_env: str = Field(default="SCHWAB_API_KEY")  # pragma: allowlist secret
    app_secret_env: str = Field(default="SCHWAB_APP_SECRET")  # pragma: allowlist secret
    callback_url: str = Field(default="https://127.0.0.1:8182")
    max_token_age_days: float = Field(default=6.5, gt=0.0)
    reminder_days: list[int] = Field(default_factory=lambda: [5, 6])


class OptionsChainConfig(BaseModel):
    """Chain request bounds. Nulls = full chain (all strikes, all expirations)."""

    contract_type: str = Field(default="ALL")
    strike_range: str = Field(default="ALL")
    include_underlying_quote: bool = Field(default=True)
    from_date: str | None = Field(default=None)
    to_date: str | None = Field(default=None)
    strike_count: int | None = Field(default=None)


class OptionsIntegrityConfig(BaseModel):
    """Silent-corruption guards + audit schedule (spec §10.5, §10.6)."""

    fail_on_truncated: bool = Field(default=False)
    # DTE cut points for the bounded re-fetch on truncation; tuned at first live run (spec §15.7).
    refetch_dte_chunks: list[int] = Field(default_factory=lambda: [45, 180, 3650])
    audit_day_of_month: int = Field(default=1, ge=1, le=28)
    audit_time_et: str = Field(default="18:00")


class OptionsBackupConfig(BaseModel):
    """Offsite 3-2-1 backup of the un-backfillable capture (spec §13)."""

    enabled: bool = Field(default=True)
    rclone_remote: str = Field(default="b2:swingrl-options")
    time_et: str = Field(default="02:30")


class OptionsCollectorConfig(BaseModel):
    """EOD option-chain collector configuration (spec §5)."""

    enabled: bool = Field(default=True)
    provider: str = Field(default="schwab")
    index_symbols: list[str] = Field(default_factory=lambda: ["$SPX"])
    include_equity_symbols: bool = Field(default=True)
    output_dir: str = Field(default="data/options_eod/schwab")
    schema_version: str = Field(default="v1")
    snapshots: list[OptionsSnapshotConfig] = Field(
        default_factory=lambda: [
            OptionsSnapshotConfig(label="decision", time_et="15:45"),
            OptionsSnapshotConfig(label="eod", time_et="16:35"),
        ]
    )
    chain: OptionsChainConfig = Field(default_factory=OptionsChainConfig)
    auth: OptionsAuthConfig = Field(default_factory=OptionsAuthConfig)
    rate_limit_per_sec: float = Field(default=2.0, gt=0.0)
    health_check_time_et: str = Field(default="17:15")
    token_reminder_time_et: str = Field(default="09:00")
    apscheduler_db_path: str = Field(default="db/apscheduler_options.sqlite")
    # Keep raw_json in Postgres JSONB too (bulky). Default on; revisit at first-run (decision D5).
    postgres_store_raw_json: bool = Field(default=True)
    integrity: OptionsIntegrityConfig = Field(default_factory=OptionsIntegrityConfig)
    backup: OptionsBackupConfig = Field(default_factory=OptionsBackupConfig)
```

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
  provider: schwab
  index_symbols: ["$SPX"]           # confirm exact request symbol at first run
  include_equity_symbols: true      # also capture chains for config.equity.symbols
  output_dir: "data/options_eod/schwab"
  schema_version: "v1"
  apscheduler_db_path: db/apscheduler_options.sqlite   # SEPARATE from the trader's jobstore
  postgres_store_raw_json: true         # also store raw_json JSONB in Postgres; revisit at first-run
  rate_limit_per_sec: 2.0
  health_check_time_et: "17:15"
  token_reminder_time_et: "09:00"
  snapshots:
    - { label: decision, time_et: "15:45" }
    - { label: eod,      time_et: "16:35" }
  chain:
    contract_type: ALL
    strike_range: ALL
    include_underlying_quote: true
    from_date: null
    to_date: null
    strike_count: null
  auth:
    token_path: "secrets/schwab_token.json"
    api_key_env: SCHWAB_API_KEY          # pragma: allowlist secret
    app_secret_env: SCHWAB_APP_SECRET    # pragma: allowlist secret
    callback_url: "https://127.0.0.1:8182"
    max_token_age_days: 6.5
    reminder_days: [5, 6]
  integrity:
    fail_on_truncated: false
    refetch_dte_chunks: [45, 180, 3650]
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

### Task 3: `schwab_auth.py` — token manager

**Files:**
- Create: `src/swingrl/data/options/schwab_auth.py`
- Test: `tests/test_schwab_auth.py`

**Interfaces:**
- Consumes: `OptionsCollectorConfig` (T2), `Alerter.send_alert(level, title, message)` (`src/swingrl/monitoring/alerter.py`), `DataError` (`src/swingrl/utils/exceptions.py`), `schwab.auth.client_from_token_file`.
- Produces: `TokenManager` with:
  - `__init__(self, config: OptionsCollectorConfig, alerter: Alerter | None = None) -> None`
  - `token_issued_at(self) -> datetime | None` — reads `creation_timestamp` (epoch s) from the token JSON; falls back to file mtime; `None` if file missing.
  - `token_age_days(self, now: datetime | None = None) -> float | None`
  - `due_reminder_day(self, now: datetime | None = None) -> int | None` — the integer reminder day (5 or 6) the current age has reached, else `None`.
  - `check_token_age_and_alert(self, now: datetime | None = None) -> None` — WARNING at reminder days; CRITICAL if age ≥ 7.
  - `load_client(self) -> object` — schwab-py client from the token file; on missing file or `invalid_client` logs + CRITICAL alert + raises `DataError`.
  - Module constant `REFRESH_TOKEN_LIFETIME_DAYS = 7.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schwab_auth.py
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swingrl.config.schema import OptionsCollectorConfig
from swingrl.data.options.schwab_auth import REFRESH_TOKEN_LIFETIME_DAYS, TokenManager
from swingrl.utils.exceptions import DataError


def _write_token(path: Path, created: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "creation_timestamp": int(created.timestamp()),
        "token": {"refresh_token": "rt", "access_token": "at", "expires_in": 1800},
    }
    path.write_text(json.dumps(payload))


def _config(tmp_path: Path) -> OptionsCollectorConfig:
    cfg = OptionsCollectorConfig()
    cfg.auth.token_path = str(tmp_path / "schwab_token.json")
    return cfg


def test_token_age_days_from_creation_timestamp(tmp_path: Path) -> None:
    """OPT-AUTH-1: age computed from creation_timestamp (spec §7.2)."""
    cfg = _config(tmp_path)
    now = datetime(2026, 7, 14, tzinfo=UTC)
    _write_token(Path(cfg.auth.token_path), now - timedelta(days=3, hours=12))
    tm = TokenManager(cfg)
    assert tm.token_age_days(now) == pytest.approx(3.5, abs=0.01)


def test_token_age_none_when_missing(tmp_path: Path) -> None:
    """OPT-AUTH-2: missing token file -> age None (spec §7.2)."""
    tm = TokenManager(_config(tmp_path))
    assert tm.token_age_days() is None


def test_due_reminder_day_at_day_5(tmp_path: Path) -> None:
    """OPT-AUTH-3: reminder fires on day 5 (spec §7.2)."""
    cfg = _config(tmp_path)
    now = datetime(2026, 7, 14, tzinfo=UTC)
    _write_token(Path(cfg.auth.token_path), now - timedelta(days=5, hours=1))
    tm = TokenManager(cfg)
    assert tm.due_reminder_day(now) == 5


def test_no_reminder_before_day_5(tmp_path: Path) -> None:
    """OPT-AUTH-4: no reminder at day 4 (spec §7.2)."""
    cfg = _config(tmp_path)
    now = datetime(2026, 7, 14, tzinfo=UTC)
    _write_token(Path(cfg.auth.token_path), now - timedelta(days=4))
    tm = TokenManager(cfg)
    assert tm.due_reminder_day(now) is None


def test_check_age_sends_warning_on_reminder_day(tmp_path: Path) -> None:
    """OPT-AUTH-5: WARNING alert at reminder day (spec §10.4)."""
    cfg = _config(tmp_path)
    now = datetime(2026, 7, 14, tzinfo=UTC)
    _write_token(Path(cfg.auth.token_path), now - timedelta(days=6, hours=1))
    alerter = MagicMock()
    TokenManager(cfg, alerter=alerter).check_token_age_and_alert(now)
    alerter.send_alert.assert_called_once()
    assert alerter.send_alert.call_args.args[0] == "warning"


def test_check_age_sends_critical_past_lifetime(tmp_path: Path) -> None:
    """OPT-AUTH-6: CRITICAL when age >= 7 days (spec §10.4)."""
    cfg = _config(tmp_path)
    now = datetime(2026, 7, 14, tzinfo=UTC)
    _write_token(Path(cfg.auth.token_path), now - timedelta(days=8))
    alerter = MagicMock()
    TokenManager(cfg, alerter=alerter).check_token_age_and_alert(now)
    assert alerter.send_alert.call_args.args[0] == "critical"
    assert REFRESH_TOKEN_LIFETIME_DAYS == 7.0


def test_load_client_missing_file_raises_and_alerts(tmp_path: Path) -> None:
    """OPT-AUTH-7: missing token -> DataError + CRITICAL (spec §10.4)."""
    alerter = MagicMock()
    tm = TokenManager(_config(tmp_path), alerter=alerter)
    with pytest.raises(DataError):
        tm.load_client()
    assert alerter.send_alert.call_args.args[0] == "critical"


def test_load_client_invalid_client_raises_and_alerts(tmp_path: Path, monkeypatch) -> None:
    """OPT-AUTH-8: invalid_client on load -> DataError + CRITICAL (spec §10.4)."""
    cfg = _config(tmp_path)
    _write_token(Path(cfg.auth.token_path), datetime.now(UTC))

    def boom(*_a, **_k):
        raise Exception("invalid_client: refresh token invalid")

    monkeypatch.setattr(
        "swingrl.data.options.schwab_auth.client_from_token_file", boom
    )
    monkeypatch.setenv("SCHWAB_API_KEY", "k")  # pragma: allowlist secret
    monkeypatch.setenv("SCHWAB_APP_SECRET", "s")  # pragma: allowlist secret
    alerter = MagicMock()
    tm = TokenManager(cfg, alerter=alerter)
    with pytest.raises(DataError):
        tm.load_client()
    assert alerter.send_alert.call_args.args[0] == "critical"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schwab_auth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swingrl.data.options.schwab_auth'`.

- [ ] **Step 3: Write the implementation**

```python
# src/swingrl/data/options/schwab_auth.py
"""Schwab OAuth token manager: age tracking, reminders, client load (spec §7)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from schwab.auth import client_from_token_file

from swingrl.utils.exceptions import DataError

if TYPE_CHECKING:
    from swingrl.config.schema import OptionsCollectorConfig
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

REFRESH_TOKEN_LIFETIME_DAYS = 7.0
_SECONDS_PER_DAY = 86_400.0


class TokenManager:
    """Loads the Schwab client and tracks refresh-token age (spec §7)."""

    def __init__(self, config: OptionsCollectorConfig, alerter: Alerter | None = None) -> None:
        self._config = config
        self._alerter = alerter
        self._token_path = Path(config.auth.token_path)

    def token_issued_at(self) -> datetime | None:
        """Return refresh-token creation time (UTC), or None if the file is absent."""
        if not self._token_path.exists():
            return None
        try:
            payload = json.loads(self._token_path.read_text())
            created = payload.get("creation_timestamp")
            if created is not None:
                return datetime.fromtimestamp(float(created), tz=UTC)
        except (ValueError, OSError) as exc:
            log.warning("token_file_unreadable", path=str(self._token_path), error=str(exc))
        # Fallback: file mtime.
        return datetime.fromtimestamp(self._token_path.stat().st_mtime, tz=UTC)

    def token_age_days(self, now: datetime | None = None) -> float | None:
        """Return refresh-token age in days, or None if the token file is absent."""
        issued = self.token_issued_at()
        if issued is None:
            return None
        now = now or datetime.now(UTC)
        return (now - issued).total_seconds() / _SECONDS_PER_DAY

    def due_reminder_day(self, now: datetime | None = None) -> int | None:
        """Return the highest crossed reminder day (e.g. 5/6), else None."""
        age = self.token_age_days(now)
        if age is None:
            return None
        crossed = [d for d in sorted(self._config.auth.reminder_days) if age >= d]
        return crossed[-1] if crossed else None

    def check_token_age_and_alert(self, now: datetime | None = None) -> None:
        """Emit WARNING at reminder days and CRITICAL once past the 7-day lifetime."""
        age = self.token_age_days(now)
        if age is None:
            self._alert("critical", "Schwab token missing", "No token file — re-auth required.")
            return
        if age >= REFRESH_TOKEN_LIFETIME_DAYS:
            self._alert(
                "critical",
                "Schwab refresh token expired",
                f"Token age {age:.1f}d ≥ {REFRESH_TOKEN_LIFETIME_DAYS}d — run scripts/schwab_reauth.py.",
            )
            return
        day = self.due_reminder_day(now)
        if day is not None:
            self._alert(
                "warning",
                "Schwab re-auth due soon",
                f"Token age {age:.1f}d (day {day}); re-auth before day 7 to avoid a gap.",
            )

    def load_client(self) -> object:
        """Load the schwab-py client from the token file (spec §7.2).

        Raises DataError (and CRITICAL-alerts) on a missing file or invalid_client.
        """
        if not self._token_path.exists():
            self._alert("critical", "Schwab token missing", "No token file — re-auth required.")
            log.error("schwab_token_missing", path=str(self._token_path))
            raise DataError(f"Schwab token file not found: {self._token_path}")
        api_key = os.environ.get(self._config.auth.api_key_env, "")
        app_secret = os.environ.get(self._config.auth.app_secret_env, "")
        try:
            return client_from_token_file(
                token_path=str(self._token_path),
                api_key=api_key,
                app_secret=app_secret,
            )
        except Exception as exc:  # schwab-py raises generic errors on invalid_client
            log.error("schwab_client_load_failed", error=str(exc))
            self._alert(
                "critical",
                "Schwab auth failed",
                f"Client load failed ({exc}). Re-auth required.",
            )
            raise DataError(f"Schwab client load failed: {exc}") from exc

    def _alert(self, level: str, title: str, message: str) -> None:
        if self._alerter is not None:
            self._alerter.send_alert(level, title, message)  # type: ignore[arg-type]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schwab_auth.py -v`
Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/options/schwab_auth.py tests/test_schwab_auth.py
git commit -m "feat(options): TokenManager — age tracking, reminders, client load (spec §7)"
```

---

### Task 4: `scripts/schwab_reauth.py` — manual OAuth CLI

**Files:**
- Create: `scripts/schwab_reauth.py`
- Test: `tests/test_schwab_reauth.py`

**Interfaces:**
- Consumes: `load_config()`, `OptionsCollectorConfig.auth.*`, `schwab.auth.client_from_manual_flow`, `configure_logging`.
- Produces:
  - `run_reauth(config_path: str) -> int` — orchestrates the manual flow; returns exit code.
  - `chmod_600(path: Path) -> None` — sets `0o600` on the token file.
  - `main() -> int` (argparse `--config`), `sys.exit(main())`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schwab_reauth.py
from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import MagicMock

from scripts.schwab_reauth import chmod_600, run_reauth


def test_chmod_600_sets_owner_only(tmp_path: Path) -> None:
    """OPT-CLI-1: token file locked to 0o600 (spec §7.3)."""
    f = tmp_path / "tok.json"
    f.write_text("{}")
    chmod_600(f)
    assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_run_reauth_invokes_manual_flow(tmp_path: Path, monkeypatch) -> None:
    """OPT-CLI-2: run_reauth calls schwab-py manual flow with configured args (spec §7.2)."""
    token = tmp_path / "secrets" / "schwab_token.json"
    monkeypatch.setenv("SCHWAB_API_KEY", "key")  # pragma: allowlist secret
    monkeypatch.setenv("SCHWAB_APP_SECRET", "secret")  # pragma: allowlist secret

    fake_flow = MagicMock(return_value=MagicMock())

    def fake_load_config(_path):
        cfg = MagicMock()
        cfg.options_collector.auth.token_path = str(token)
        cfg.options_collector.auth.api_key_env = "SCHWAB_API_KEY"  # pragma: allowlist secret
        cfg.options_collector.auth.app_secret_env = "SCHWAB_APP_SECRET"  # pragma: allowlist secret
        cfg.options_collector.auth.callback_url = "https://127.0.0.1:8182"
        cfg.logging.json_logs = False
        cfg.logging.level = "INFO"
        return cfg

    monkeypatch.setattr("scripts.schwab_reauth.client_from_manual_flow", fake_flow)
    monkeypatch.setattr("scripts.schwab_reauth.load_config", fake_load_config)
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text("{}")  # simulate schwab-py having written it

    rc = run_reauth("config/swingrl.yaml")

    assert rc == 0
    kwargs = fake_flow.call_args.kwargs
    assert kwargs["api_key"] == "key"  # pragma: allowlist secret
    assert kwargs["app_secret"] == "secret"  # pragma: allowlist secret
    assert kwargs["callback_url"] == "https://127.0.0.1:8182"
    assert kwargs["token_path"] == str(token)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schwab_reauth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.schwab_reauth'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/schwab_reauth.py
"""Manual-OAuth CLI: register/refresh the Schwab token file (spec §7.2).

Usage:
    uv run python scripts/schwab_reauth.py --config config/swingrl.yaml

Prints the Schwab login URL, you log in + MFA in any browser, then copy the
redirected https://127.0.0.1:8182/?code=... URL back into the terminal.
That page failing to load is EXPECTED — nothing listens on 8182 (spec §7.4).
Always re-auth into THIS token file and nowhere else (spec §13).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import structlog
from schwab.auth import client_from_manual_flow

from swingrl.config.schema import load_config
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)


def chmod_600(path: Path) -> None:
    """Restrict the token file to owner read/write only (spec §7.3)."""
    path.chmod(0o600)


def run_reauth(config_path: str) -> int:
    """Run the schwab-py manual OAuth flow and lock the token file down."""
    config = load_config(config_path)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    auth = config.options_collector.auth
    api_key = os.environ.get(auth.api_key_env, "")
    app_secret = os.environ.get(auth.app_secret_env, "")
    if not api_key or not app_secret:
        log.error("schwab_credentials_missing", api_key_env=auth.api_key_env)
        print(f"ERROR: set {auth.api_key_env} and {auth.app_secret_env} in .env", file=sys.stderr)
        return 2
    token_path = Path(auth.token_path)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("schwab_reauth_starting", callback_url=auth.callback_url, token_path=str(token_path))
    client_from_manual_flow(
        api_key=api_key,
        app_secret=app_secret,
        callback_url=auth.callback_url,
        token_path=str(token_path),
    )
    if not token_path.exists():
        log.error("schwab_token_not_written", token_path=str(token_path))
        return 1
    chmod_600(token_path)
    log.info("schwab_reauth_complete", token_path=str(token_path))
    print(f"Token written and chmod 600: {token_path}")
    return 0


def main() -> int:
    """Parse args and run the re-auth flow."""
    parser = argparse.ArgumentParser(description="Schwab manual OAuth re-auth")
    parser.add_argument("--config", default="config/swingrl.yaml")
    args = parser.parse_args()
    return run_reauth(args.config)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schwab_reauth.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/schwab_reauth.py tests/test_schwab_reauth.py
git commit -m "feat(options): schwab_reauth.py manual-OAuth CLI (spec §7.2)"
```

---

### Task 5: `schwab_client.py` — chain fetch wrapper

**Files:**
- Create: `src/swingrl/data/options/schwab_client.py`
- Test: `tests/test_schwab_client.py`

**Interfaces:**
- Consumes: `TokenManager.load_client()` (T3), `OptionsCollectorConfig.chain.*` + `.rate_limit_per_sec` + `.integrity.*` (T2), `DataError`, `swingrl_retry` (`src/swingrl/utils/retry.py`).
- Produces: `SchwabOptionsClient` with:
  - `__init__(self, config: OptionsCollectorConfig, token_manager: TokenManager) -> None`
  - `get_option_chain(self, symbol: str) -> dict` — raw response dict; rate-limited; retried; raises `DataError` on HTTP error / empty. If `isChainTruncated` is true and `fail_on_truncated` is false, performs a bounded DTE-chunked re-fetch and merges; sets `_truncated_after_refetch` in the returned dict's `_provenance`.
  - Static `is_truncated(raw: dict) -> bool`.
- **schwab-py surface pinned here** (verify exact enum paths at T6 against the installed version): `client.get_option_chain(symbol, contract_type=..., strike_range=..., include_underlying_quote=..., from_date=..., to_date=..., strike_count=...)` returns an `httpx.Response`; `.json()` yields the dict; `.status_code` is the HTTP status.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schwab_client.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from swingrl.config.schema import OptionsCollectorConfig
from swingrl.data.options.schwab_client import SchwabOptionsClient
from swingrl.utils.exceptions import DataError


def _response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


def _client_with(raw_client: MagicMock) -> SchwabOptionsClient:
    cfg = OptionsCollectorConfig()
    cfg.rate_limit_per_sec = 1000.0  # no real sleeping in tests
    tm = MagicMock()
    tm.load_client.return_value = raw_client
    return SchwabOptionsClient(cfg, tm)


def test_get_option_chain_returns_dict() -> None:
    """OPT-CLIENT-1: happy path returns the parsed JSON dict (spec §4)."""
    ok = {"symbol": "SPY", "status": "SUCCESS", "isChainTruncated": False, "numberOfContracts": 2}
    raw = MagicMock()
    raw.get_option_chain.return_value = _response(ok)
    client = _client_with(raw)
    out = client.get_option_chain("SPY")
    assert out["symbol"] == "SPY"
    raw.get_option_chain.assert_called_once()


def test_get_option_chain_http_error_raises_dataerror() -> None:
    """OPT-CLIENT-2: non-200 -> DataError (spec §10.3)."""
    raw = MagicMock()
    raw.get_option_chain.return_value = _response({}, status=401)
    with pytest.raises(DataError):
        _client_with(raw).get_option_chain("SPY")


def test_get_option_chain_empty_raises_dataerror() -> None:
    """OPT-CLIENT-3: empty / non-SUCCESS status -> DataError (spec §10.3)."""
    raw = MagicMock()
    raw.get_option_chain.return_value = _response({"status": "FAILED"})
    with pytest.raises(DataError):
        _client_with(raw).get_option_chain("$SPX")


def test_is_truncated_reads_top_level_flag() -> None:
    """OPT-CLIENT-4: truncation flag read from top level (spec §10.5)."""
    assert SchwabOptionsClient.is_truncated({"isChainTruncated": True}) is True
    assert SchwabOptionsClient.is_truncated({"isChainTruncated": False}) is False
    assert SchwabOptionsClient.is_truncated({}) is False


def test_truncated_triggers_chunked_refetch_and_merges() -> None:
    """OPT-CLIENT-5: truncation -> DTE-chunked re-fetch, merged maps (spec §10.5)."""
    truncated = {
        "symbol": "$SPX",
        "status": "SUCCESS",
        "isChainTruncated": True,
        "callExpDateMap": {"2026-07-18:4": {"5000.0": [{"symbol": "A"}]}},
        "putExpDateMap": {},
    }
    chunk_a = {
        "status": "SUCCESS",
        "isChainTruncated": False,
        "callExpDateMap": {"2026-07-18:4": {"5000.0": [{"symbol": "A"}]}},
        "putExpDateMap": {},
    }
    chunk_b = {
        "status": "SUCCESS",
        "isChainTruncated": False,
        "callExpDateMap": {"2026-12-18:180": {"5200.0": [{"symbol": "B"}]}},
        "putExpDateMap": {},
    }
    chunk_c = {
        "status": "SUCCESS",
        "isChainTruncated": False,
        "callExpDateMap": {"2027-06-18:400": {"5500.0": [{"symbol": "C"}]}},
        "putExpDateMap": {},
    }
    raw = MagicMock()
    raw.get_option_chain.side_effect = [
        _response(truncated),
        _response(chunk_a),
        _response(chunk_b),
        _response(chunk_c),
    ]
    out = _client_with(raw).get_option_chain("$SPX")
    # 1 full attempt + 3 DTE chunks (defaults [45, 180, 3650]).
    assert raw.get_option_chain.call_count == 4
    calls = out["callExpDateMap"]
    assert set(calls) == {"2026-07-18:4", "2026-12-18:180", "2027-06-18:400"}
    assert out["_provenance"]["truncated"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schwab_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swingrl.data.options.schwab_client'`.

- [ ] **Step 3: Write the implementation**

```python
# src/swingrl/data/options/schwab_client.py
"""Thin, swappable wrapper over schwab-py's option-chain endpoint (spec §4)."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from swingrl.utils.exceptions import DataError
from swingrl.utils.retry import swingrl_retry

if TYPE_CHECKING:
    from swingrl.config.schema import OptionsCollectorConfig
    from swingrl.data.options.schwab_auth import TokenManager

log = structlog.get_logger(__name__)


class SchwabOptionsClient:
    """Fetch a full option chain as a raw dict, rate-limited and retried."""

    def __init__(self, config: OptionsCollectorConfig, token_manager: TokenManager) -> None:
        self._config = config
        self._token_manager = token_manager
        self._client: Any | None = None
        self._min_interval_s = 1.0 / config.rate_limit_per_sec
        self._last_call_ts = 0.0

    @staticmethod
    def is_truncated(raw: dict[str, Any]) -> bool:
        """Return the top-level isChainTruncated flag (defaults False)."""
        return bool(raw.get("isChainTruncated", False))

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._token_manager.load_client()
        return self._client

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)
        self._last_call_ts = time.monotonic()

    @swingrl_retry(max_attempts=4, retryable_exceptions=(ConnectionError, TimeoutError, OSError))
    def _raw_fetch(self, symbol: str, *, from_date: date | None, to_date: date | None) -> dict:
        self._throttle()
        client = self._ensure_client()
        chain = self._config.chain
        resp = client.get_option_chain(
            symbol,
            contract_type=chain.contract_type,
            strike_range=chain.strike_range,
            include_underlying_quote=chain.include_underlying_quote,
            strike_count=chain.strike_count,
            from_date=from_date,
            to_date=to_date,
        )
        if resp.status_code != 200:
            log.error("schwab_chain_http_error", symbol=symbol, status=resp.status_code)
            raise DataError(f"Schwab chain HTTP {resp.status_code} for {symbol}")
        payload: dict[str, Any] = resp.json()
        if payload.get("status") != "SUCCESS":
            log.error("schwab_chain_bad_status", symbol=symbol, status=payload.get("status"))
            raise DataError(f"Schwab chain status {payload.get('status')!r} for {symbol}")
        return payload

    def get_option_chain(self, symbol: str) -> dict[str, Any]:
        """Fetch the full chain; on truncation, bounded DTE-chunked re-fetch (spec §10.5)."""
        raw = self._raw_fetch(
            symbol,
            from_date=self._parse_date(self._config.chain.from_date),
            to_date=self._parse_date(self._config.chain.to_date),
        )
        truncated = self.is_truncated(raw)
        raw.setdefault("_provenance", {})["truncated"] = truncated
        if truncated and not self._config.integrity.fail_on_truncated:
            log.warning("schwab_chain_truncated_refetch", symbol=symbol)
            raw = self._refetch_chunked(symbol, raw)
        elif truncated and self._config.integrity.fail_on_truncated:
            raise DataError(f"Truncated chain for {symbol} and fail_on_truncated=true")
        return raw

    def _refetch_chunked(self, symbol: str, base: dict[str, Any]) -> dict[str, Any]:
        """Re-request in DTE windows and merge exp-date maps (spec §10.5)."""
        today = datetime.now(UTC).date()
        edges = [0, *sorted(self._config.integrity.refetch_dte_chunks)]
        merged_calls: dict[str, Any] = {}
        merged_puts: dict[str, Any] = {}
        for lo, hi in zip(edges[:-1], edges[1:], strict=False):
            chunk = self._raw_fetch(
                symbol,
                from_date=today + timedelta(days=lo),
                to_date=today + timedelta(days=hi),
            )
            merged_calls.update(chunk.get("callExpDateMap", {}))
            merged_puts.update(chunk.get("putExpDateMap", {}))
        base["callExpDateMap"] = merged_calls
        base["putExpDateMap"] = merged_puts
        base["_provenance"]["refetched_chunks"] = len(edges) - 1
        return base

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        return date.fromisoformat(value) if value else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schwab_client.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/options/schwab_client.py tests/test_schwab_client.py
git commit -m "feat(options): SchwabOptionsClient — fetch, throttle, truncation re-fetch (spec §4, §10.5)"
```

---

### Task 6 🛑: First OAuth + capture the pinning fixture (manual gate)

**This task requires a human and a live Schwab app — it is not TDD.** It produces the real token file, the entitlement finding, and the sanitized chain fixture that T16 pins the parser against.

**Files:**
- Create: `scripts/capture_chain_fixture.py` (one-shot helper)
- Create (output, gitignored): `secrets/schwab_token.json`
- Create (committed, sanitized): `tests/fixtures/schwab_chain_spx.json`, `tests/fixtures/schwab_chain_spy.json`

**Interfaces:**
- Consumes: `run_reauth` (T4), `SchwabOptionsClient` (T5), `load_config`.
- Produces: `sanitize_chain(raw: dict) -> dict` (drops account-scoped/PII-ish top-level keys, keeps structure + a bounded sample of strikes), `capture(symbol, out_path) -> dict`.

- [ ] **Step 1: Write `scripts/capture_chain_fixture.py`**

```python
# scripts/capture_chain_fixture.py
"""One-shot: pull one real chain, sanitize, and save a test fixture (spec §12).

Usage:
    uv run python scripts/capture_chain_fixture.py --symbol '$SPX' \
        --out tests/fixtures/schwab_chain_spx.json
Requires a valid secrets/schwab_token.json (run scripts/schwab_reauth.py first).
Keeps a bounded number of strikes per expiration so the fixture stays small.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

from swingrl.config.schema import load_config
from swingrl.data.options.schwab_auth import TokenManager
from swingrl.data.options.schwab_client import SchwabOptionsClient
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)
_DROP_KEYS = {"accountNumber", "accountId"}
_MAX_EXP = 2
_MAX_STRIKES = 4


def sanitize_chain(raw: dict, *, max_exp: int = _MAX_EXP, max_strikes: int = _MAX_STRIKES) -> dict:
    """Drop account-scoped keys and truncate to a small, representative sample."""
    out = {k: v for k, v in raw.items() if k not in _DROP_KEYS}
    for map_key in ("callExpDateMap", "putExpDateMap"):
        exp_map = out.get(map_key, {})
        trimmed: dict = {}
        for exp in list(exp_map)[:max_exp]:
            strikes = exp_map[exp]
            trimmed[exp] = {s: strikes[s] for s in list(strikes)[:max_strikes]}
        out[map_key] = trimmed
    return out


def capture(symbol: str, out_path: Path) -> dict:
    """Fetch, sanitize, and write one chain fixture; log the isDelayed finding."""
    config = load_config("config/swingrl.yaml")
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    tm = TokenManager(config.options_collector)
    client = SchwabOptionsClient(config.options_collector, tm)
    raw = client.get_option_chain(symbol)
    log.info(
        "chain_captured",
        symbol=symbol,
        is_delayed=raw.get("isDelayed"),
        is_truncated=raw.get("isChainTruncated"),
        number_of_contracts=raw.get("numberOfContracts"),
    )
    sanitized = sanitize_chain(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sanitized, indent=2))
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a sanitized Schwab chain fixture")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    capture(args.symbol, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Human — register/confirm the Schwab app and run first OAuth**

Confirm `SCHWAB_API_KEY` / `SCHWAB_APP_SECRET` are in `.env`, then:

```bash
uv run python scripts/schwab_reauth.py --config config/swingrl.yaml
```

Log in + MFA in a browser, copy the redirected `https://127.0.0.1:8182/?code=...` URL back into the terminal (the page failing to load is expected — spec §7.4). Confirm `secrets/schwab_token.json` exists and is `chmod 600`.

- [ ] **Step 3: Human — capture the two fixtures and RECORD findings**

```bash
uv run python scripts/capture_chain_fixture.py --symbol '$SPX' --out tests/fixtures/schwab_chain_spx.json
uv run python scripts/capture_chain_fixture.py --symbol 'SPY'  --out tests/fixtures/schwab_chain_spy.json
```

From the `chain_captured` log lines, **record in the plan tracker / spec §3**: the exact `$SPX` request symbol that worked (unknown #1), `isDelayed` (unknown #2 — gates decision-snapshot fidelity), whether `$SPX` tripped `isChainTruncated` (unknown #7), and `numberOfContracts` per symbol (unknown #4). These are un-guessable first-run findings.

- [ ] **Step 4: Sanity-eyeball the fixtures**

Open `tests/fixtures/schwab_chain_spx.json` and confirm the contract objects carry the fields the parser will map (`delta`, `gamma`, `theta`, `vega`, `rho`, `volatility`, `openInterest`, `bid`, `ask`, `strikePrice`, `expirationDate`, `daysToExpiration`, `putCall`, `optionRoot`, `settlementType`, `exerciseType`). If a field name differs, note it — T7's parser mapping and T11's schema-drift guard must use the real names.

- [ ] **Step 5: Commit the sanitized fixtures + helper (never the token)**

```bash
git status --short   # confirm secrets/schwab_token.json is NOT listed (gitignored)
git add scripts/capture_chain_fixture.py tests/fixtures/schwab_chain_spx.json tests/fixtures/schwab_chain_spy.json
git commit -m "chore(options): capture-fixture helper + sanitized SPX/SPY chain fixtures (spec §12)"
```

---

### Task 7: `chain_parser.py` — raw dict → typed rows + raw_json

Built against a **hand-authored fixture** (decision D2) so it isn't blocked on live OAuth. T16 adds a test pinning it to the real captured fixture.

**Files:**
- Create: `src/swingrl/data/options/chain_parser.py`
- Test: `tests/test_chain_parser.py`

**Interfaces:**
- Consumes: `DataError`.
- Produces:
  - `ParsedChain` — frozen dataclass: `header: dict[str, Any]`, `contracts: pd.DataFrame`.
  - `parse_chain(raw, *, underlying_symbol, snapshot_label, quote_date, snapshot_time_utc, pulled_at_utc, schema_version, is_early_close, source="schwab") -> ParsedChain`.
  - `epoch_ms_to_utc(ms: int | float | None) -> datetime | None`.
  - `clean_sentinel(value: float | int | None) -> float` (maps `-999`, `-999.0`, `NaN`, `None` → `float("nan")`).
  - Module constant `CONTRACT_COLUMNS: list[str]` — the exact ordered typed columns (the grain).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chain_parser.py
from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from swingrl.data.options.chain_parser import (
    CONTRACT_COLUMNS,
    clean_sentinel,
    epoch_ms_to_utc,
    parse_chain,
)


def _raw() -> dict:
    """Hand-authored representative chain (documented Schwab schema)."""
    contract = {
        "putCall": "CALL",
        "symbol": "SPXW  260718C05000000",
        "optionRoot": "SPXW",
        "bid": 12.3,
        "ask": 12.7,
        "last": 12.5,
        "mark": 12.5,
        "bidSize": 10,
        "askSize": 8,
        "lastSize": 1,
        "openPrice": 12.0,
        "highPrice": 13.0,
        "lowPrice": 11.8,
        "closePrice": 12.4,
        "totalVolume": 4200,
        "openInterest": 15000,
        "netChange": 0.1,
        "delta": 0.55,
        "gamma": 0.01,
        "theta": -0.9,
        "vega": 1.2,
        "rho": 0.3,
        "volatility": 12.34,
        "theoreticalOptionValue": 12.45,
        "timeValue": 2.5,
        "intrinsicValue": 10.0,
        "extrinsicValue": 2.5,
        "strikePrice": 5000.0,
        "daysToExpiration": 4,
        "expirationType": "W",
        "settlementType": "P",
        "exerciseType": "E",
        "multiplier": 100.0,
        "inTheMoney": True,
    }
    illiquid_put = {**contract, "putCall": "PUT", "symbol": "SPXW  260718P04000000",
                    "strikePrice": 4000.0, "delta": -999.0, "volatility": -999.0,
                    "gamma": float("nan"), "openInterest": 0}
    return {
        "symbol": "$SPX",
        "status": "SUCCESS",
        "isDelayed": False,
        "underlyingPrice": 5001.2,
        "interestRate": 5.0,
        "dividendYield": 1.3,
        "volatility": 13.0,
        "numberOfContracts": 2,
        "callExpDateMap": {"2026-07-18:4": {"5000.0": [contract]}},
        "putExpDateMap": {"2026-07-18:4": {"4000.0": [illiquid_put]}},
    }


def _parse():
    return parse_chain(
        _raw(),
        underlying_symbol="$SPX",
        snapshot_label="decision",
        quote_date=date(2026, 7, 14),
        snapshot_time_utc=datetime(2026, 7, 14, 19, 45, tzinfo=UTC),
        pulled_at_utc=datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        schema_version="v1",
        is_early_close=False,
    )


def test_epoch_ms_to_utc() -> None:
    """OPT-PARSE-1: epoch-ms -> tz-aware UTC datetime (spec §6.3)."""
    assert epoch_ms_to_utc(1_752_522_300_000) == datetime.fromtimestamp(1_752_522_300, tz=UTC)
    assert epoch_ms_to_utc(None) is None


def test_clean_sentinel_maps_to_nan() -> None:
    """OPT-PARSE-2: -999 / NaN / None -> NaN, real values pass (spec §6.3)."""
    assert math.isnan(clean_sentinel(-999.0))
    assert math.isnan(clean_sentinel(float("nan")))
    assert math.isnan(clean_sentinel(None))
    assert clean_sentinel(0.55) == 0.55


def test_contracts_flattened_one_row_per_contract() -> None:
    """OPT-PARSE-3: grain = one row per contract (spec §6.3)."""
    df = _parse().contracts
    assert len(df) == 2
    assert list(df.columns) == CONTRACT_COLUMNS


def test_call_put_split_uses_option_right() -> None:
    """OPT-PARSE-4: putCall -> option_right CALL/PUT (spec §6.3)."""
    df = _parse().contracts.set_index("contract_symbol")
    assert df.loc["SPXW  260718C05000000", "option_right"] == "CALL"
    assert df.loc["SPXW  260718P04000000", "option_right"] == "PUT"


def test_iv_percent_preserved() -> None:
    """OPT-PARSE-5: iv stored as percent, not fraction (spec §6.3)."""
    df = _parse().contracts.set_index("contract_symbol")
    assert df.loc["SPXW  260718C05000000", "iv"] == pytest.approx(12.34)


def test_sentinel_greeks_become_nan() -> None:
    """OPT-PARSE-6: illiquid -999 greeks/iv stored as NaN, never -999 (spec §6.3)."""
    df = _parse().contracts.set_index("contract_symbol")
    row = df.loc["SPXW  260718P04000000"]
    assert math.isnan(row["delta"])
    assert math.isnan(row["iv"])
    assert math.isnan(row["gamma"])


def test_dte_is_int_and_expiration_is_date() -> None:
    """OPT-PARSE-7: dte int, expiration from map key (spec §6.3)."""
    df = _parse().contracts.set_index("contract_symbol")
    row = df.loc["SPXW  260718C05000000"]
    assert int(row["dte"]) == 4
    assert row["expiration"] == date(2026, 7, 18)


def test_raw_json_populated_per_row() -> None:
    """OPT-PARSE-8: full original contract dict kept in raw_json (spec §6.2)."""
    df = _parse().contracts.set_index("contract_symbol")
    raw = df.loc["SPXW  260718C05000000", "raw_json"]
    assert isinstance(raw, dict)
    assert raw["strikePrice"] == 5000.0


def test_header_denormalized_context() -> None:
    """OPT-PARSE-9: snapshot-level context in header + on each row (spec §6.4)."""
    parsed = _parse()
    assert parsed.header["underlying_price"] == 5001.2
    assert parsed.header["is_delayed"] is False
    assert parsed.header["number_of_contracts"] == 2
    assert parsed.header["is_early_close"] is False
    assert "callExpDateMap" not in parsed.header["raw_header"]
    assert (parsed.contracts["underlying_price"] == 5001.2).all()


def test_empty_chain_raises_dataerror() -> None:
    """OPT-PARSE-10: no contracts -> DataError (spec §10.3)."""
    from swingrl.utils.exceptions import DataError

    empty = {"symbol": "VTI", "status": "SUCCESS", "callExpDateMap": {}, "putExpDateMap": {}}
    with pytest.raises(DataError):
        parse_chain(
            empty, underlying_symbol="VTI", snapshot_label="eod",
            quote_date=date(2026, 7, 14),
            snapshot_time_utc=datetime(2026, 7, 14, 20, 30, tzinfo=UTC),
            pulled_at_utc=datetime(2026, 7, 14, 20, 30, tzinfo=UTC),
            schema_version="v1", is_early_close=False,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chain_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swingrl.data.options.chain_parser'`.

- [ ] **Step 3: Write the implementation**

```python
# src/swingrl/data/options/chain_parser.py
"""Parse a raw Schwab chain dict into typed contract rows + raw_json (spec §6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import structlog

from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

_SENTINELS = {-999.0, -999}
_GREEK_COLUMNS = {"delta", "gamma", "theta", "vega", "rho", "iv"}

# Ordered typed columns — the grain (spec §6.3).
CONTRACT_COLUMNS: list[str] = [
    "underlying_symbol", "quote_date", "snapshot_label",
    "underlying_price", "is_delayed", "quote_time_utc", "trade_time_utc", "pulled_at_utc",
    "source", "schema_version",
    "contract_symbol", "option_root", "expiration", "dte", "strike", "option_right",
    "expiration_type", "settlement_type", "exercise_type", "multiplier", "in_the_money",
    "bid", "ask", "last", "mark", "bid_size", "ask_size", "last_size",
    "open", "high", "low", "close", "volume", "open_interest", "net_change",
    "delta", "gamma", "theta", "vega", "rho", "iv",
    "theoretical_value", "time_value", "intrinsic_value", "extrinsic_value",
    "raw_json",
]


@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain: snapshot-level header + one DataFrame row per contract."""

    header: dict[str, Any]
    contracts: pd.DataFrame


def epoch_ms_to_utc(ms: int | float | None) -> datetime | None:
    """Convert Schwab epoch-milliseconds to a tz-aware UTC datetime (None-safe)."""
    if ms is None or (isinstance(ms, float) and math.isnan(ms)):
        return None
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=UTC)


def clean_sentinel(value: float | int | None) -> float:
    """Map Schwab's -999 / NaN / None illiquid sentinels to real NaN (spec §6.3)."""
    if value is None:
        return float("nan")
    if isinstance(value, float) and math.isnan(value):
        return float("nan")
    if value in _SENTINELS:
        return float("nan")
    return float(value)


def _row(contract: dict, *, option_right: str, expiration: date, base: dict) -> dict:
    row = dict(base)
    row.update(
        contract_symbol=contract.get("symbol"),
        option_root=contract.get("optionRoot"),
        expiration=expiration,
        dte=int(contract.get("daysToExpiration", 0)),
        strike=_f(contract.get("strikePrice")),
        option_right=contract.get("putCall", option_right),
        expiration_type=contract.get("expirationType"),
        settlement_type=contract.get("settlementType"),
        exercise_type=contract.get("exerciseType"),
        multiplier=_f(contract.get("multiplier")),
        in_the_money=contract.get("inTheMoney"),
        bid=_f(contract.get("bid")), ask=_f(contract.get("ask")),
        last=_f(contract.get("last")), mark=_f(contract.get("mark")),
        bid_size=_i(contract.get("bidSize")), ask_size=_i(contract.get("askSize")),
        last_size=_i(contract.get("lastSize")),
        open=_f(contract.get("openPrice")), high=_f(contract.get("highPrice")),
        low=_f(contract.get("lowPrice")), close=_f(contract.get("closePrice")),
        volume=_i(contract.get("totalVolume")),
        open_interest=_i(contract.get("openInterest")),
        net_change=_f(contract.get("netChange")),
        delta=clean_sentinel(contract.get("delta")),
        gamma=clean_sentinel(contract.get("gamma")),
        theta=clean_sentinel(contract.get("theta")),
        vega=clean_sentinel(contract.get("vega")),
        rho=clean_sentinel(contract.get("rho")),
        iv=clean_sentinel(contract.get("volatility")),
        theoretical_value=_f(contract.get("theoreticalOptionValue")),
        time_value=_f(contract.get("timeValue")),
        intrinsic_value=_f(contract.get("intrinsicValue")),
        extrinsic_value=_f(contract.get("extrinsicValue")),
        quote_time_utc=epoch_ms_to_utc(contract.get("quoteTimeInLong")),
        trade_time_utc=epoch_ms_to_utc(contract.get("tradeTimeInLong")),
        raw_json=contract,
    )
    return row


def _f(value: Any) -> float | None:
    return None if value is None else float(value)


def _i(value: Any) -> int | None:
    return None if value is None else int(value)


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
    source: str = "schwab",
) -> ParsedChain:
    """Flatten a raw chain to typed rows + raw_json and build the header (spec §6)."""
    base = {
        "underlying_symbol": underlying_symbol,
        "quote_date": quote_date,
        "snapshot_label": snapshot_label,
        "underlying_price": _f(raw.get("underlyingPrice")),
        "is_delayed": raw.get("isDelayed"),
        "pulled_at_utc": pulled_at_utc,
        "source": source,
        "schema_version": schema_version,
    }
    rows: list[dict] = []
    for map_key, right in (("callExpDateMap", "CALL"), ("putExpDateMap", "PUT")):
        for exp_key, strike_map in raw.get(map_key, {}).items():
            expiration = date.fromisoformat(exp_key.split(":")[0])
            for _strike, contracts in strike_map.items():
                for contract in contracts:
                    rows.append(_row(contract, option_right=right, expiration=expiration, base=base))
    if not rows:
        log.error("options_empty_chain", underlying_symbol=underlying_symbol)
        raise DataError(f"Empty option chain for {underlying_symbol}")
    contracts_df = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)

    raw_header = {k: v for k, v in raw.items() if k not in ("callExpDateMap", "putExpDateMap")}
    header = {
        "underlying_symbol": underlying_symbol,
        "quote_date": quote_date,
        "snapshot_label": snapshot_label,
        "snapshot_time_utc": snapshot_time_utc,
        "pulled_at_utc": pulled_at_utc,
        "underlying_price": _f(raw.get("underlyingPrice")),
        "is_delayed": raw.get("isDelayed"),
        "is_early_close": is_early_close,
        "interest_rate": _f(raw.get("interestRate")),
        "dividend_yield": _f(raw.get("dividendYield")),
        "underlying_volatility": _f(raw.get("volatility")),
        "number_of_contracts": _i(raw.get("numberOfContracts")),
        "status": raw.get("status"),
        "source": source,
        "schema_version": schema_version,
        "raw_header": raw_header,
    }
    return ParsedChain(header=header, contracts=contracts_df)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chain_parser.py -v`
Expected: all 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/options/chain_parser.py tests/test_chain_parser.py
git commit -m "feat(options): chain_parser — raw dict -> typed rows + raw_json (spec §6)"
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
  - static `symbol_to_dir(symbol: str) -> str` — strips `$` (`$SPX` → `SPX`).
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
    cfg.output_dir = str(tmp_path / "options_eod" / "schwab")
    return OptionsStore(cfg)


def _parsed() -> ParsedChain:
    df = pd.DataFrame(
        [{"contract_symbol": "SPXW260718C05000000", "strike": 5000.0, "iv": 12.3,
          "raw_json": {"symbol": "SPXW260718C05000000", "strikePrice": 5000.0}}]
    )
    header = {
        "underlying_symbol": "$SPX", "quote_date": date(2026, 7, 14), "snapshot_label": "decision",
        "snapshot_time_utc": datetime(2026, 7, 14, 19, 45, tzinfo=UTC),
        "pulled_at_utc": datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        "number_of_contracts": 1, "is_early_close": False,
        "raw_header": {"symbol": "$SPX", "status": "SUCCESS"},
    }
    return ParsedChain(header=header, contracts=df)


def test_symbol_to_dir_strips_dollar() -> None:
    """OPT-STORE-1: $SPX -> dir SPX (spec §5)."""
    assert OptionsStore.symbol_to_dir("$SPX") == "SPX"
    assert OptionsStore.symbol_to_dir("SPY") == "SPY"


def test_parquet_path_layout(tmp_path: Path) -> None:
    """OPT-STORE-2: one file per (symbol,date,label) (spec §8.1)."""
    p = _store(tmp_path).parquet_path("$SPX", date(2026, 7, 14), "decision")
    assert p.name == "2026-07-14_decision.parquet"
    assert p.parent.name == "SPX"


def test_write_then_exists(tmp_path: Path) -> None:
    """OPT-STORE-3: write makes snapshot_exists_parquet true (spec §10.1)."""
    store = _store(tmp_path)
    assert store.snapshot_exists_parquet("$SPX", date(2026, 7, 14), "decision") is False
    store.write_snapshot(_parsed(), "$SPX", date(2026, 7, 14), "decision")
    assert store.snapshot_exists_parquet("$SPX", date(2026, 7, 14), "decision") is True


def test_write_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    """OPT-STORE-4: no .tmp file remains after write (spec §8.1)."""
    store = _store(tmp_path)
    path = store.write_snapshot(_parsed(), "$SPX", date(2026, 7, 14), "decision")
    assert not path.with_suffix(".parquet.tmp").exists()
    assert list(path.parent.glob("*.tmp")) == []


def test_roundtrip_restores_dicts_and_datetimes(tmp_path: Path) -> None:
    """OPT-STORE-5: read_snapshot restores raw_json dict + header datetimes (spec §8.1)."""
    store = _store(tmp_path)
    store.write_snapshot(_parsed(), "$SPX", date(2026, 7, 14), "decision")
    back = store.read_snapshot("$SPX", date(2026, 7, 14), "decision")
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
        """Filesystem-safe directory name for a symbol ($SPX -> SPX)."""
        return symbol.lstrip("$")

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
  source              text NOT NULL DEFAULT 'schwab',
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
  source text NOT NULL DEFAULT 'schwab', schema_version text NOT NULL,
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


def _parsed(symbol: str = "$SPX") -> ParsedChain:
    row = {c: None for c in CONTRACT_COLUMNS}
    row.update(
        underlying_symbol=symbol, quote_date=date(2026, 7, 14), snapshot_label="decision",
        contract_symbol="SPXW260718C05000000", strike=5000.0, dte=4, option_right="CALL",
        delta=0.55, iv=12.3, underlying_price=5001.2, is_delayed=False,
        pulled_at_utc=datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        expiration=date(2026, 7, 18), source="schwab", schema_version="v1",
        raw_json={"symbol": "SPXW260718C05000000", "strikePrice": 5000.0},
    )
    header = {
        "underlying_symbol": symbol, "quote_date": date(2026, 7, 14), "snapshot_label": "decision",
        "snapshot_time_utc": datetime(2026, 7, 14, 19, 45, tzinfo=UTC),
        "pulled_at_utc": datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        "underlying_price": 5001.2, "is_delayed": False, "is_early_close": False,
        "interest_rate": 5.0, "dividend_yield": 1.3, "underlying_volatility": 13.0,
        "number_of_contracts": 1, "status": "SUCCESS", "source": "schwab", "schema_version": "v1",
        "raw_header": {"symbol": symbol, "status": "SUCCESS"},
    }
    return ParsedChain(header=header, contracts=pd.DataFrame([row])[CONTRACT_COLUMNS])


def _store(tmp_path: Path, conn: psycopg.Connection) -> OptionsStore:
    cfg = OptionsCollectorConfig()
    cfg.output_dir = str(tmp_path / "options_eod" / "schwab")
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
        cfg.output_dir = str(tmp_path / "options_eod" / "schwab")
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
        store.write_snapshot(_parsed(), "$SPX", date(2026, 7, 14), "decision")  # Parquet only
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

### Task 11: `market_calendar.py` + `collector.py` — orchestration & guards

**Files:**
- Create: `src/swingrl/data/options/market_calendar.py`
- Create: `src/swingrl/data/options/collector.py`
- Test: `tests/test_options_market_calendar.py`, `tests/test_options_collector.py`

**Interfaces:**
- Consumes: `SchwabOptionsClient` (T5), `parse_chain` (T7), `OptionsStore` (T8/T10), `TokenManager` (T3), `Alerter.send_alert` (existing), `SwingRLConfig` (for both `options_collector` and `equity.symbols`), `DataError`.
- Produces:
  - `market_calendar.is_trading_day(quote_date: date) -> bool`, `market_calendar.is_early_close(quote_date: date) -> bool`.
  - `SnapshotResult` — dataclass `label, succeeded, failed, skipped, warnings, auth_failed`.
  - `OptionsCollector.__init__(self, config: SwingRLConfig, client, store, token_manager, alerter=None) -> None`
  - `OptionsCollector.symbols() -> list[str]`
  - `OptionsCollector.run_snapshot(self, snapshot_label: str, now: datetime | None = None) -> SnapshotResult`
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
```

Run: `uv run pytest tests/test_options_market_calendar.py -v` → 4 PASS. (If a hardcoded early-close date drifts in a future `exchange_calendars` release, adjust the fixture date — the logic is date-agnostic.)

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
    cfg.options_collector.index_symbols = ["$SPX"]
    cfg.options_collector.include_equity_symbols = True
    return cfg


def _good_parsed(symbol: str) -> ParsedChain:
    row = {c: None for c in CONTRACT_COLUMNS}
    row.update(underlying_symbol=symbol, contract_symbol=f"{symbol}_C", strike=1.0)
    header = {"underlying_symbol": symbol, "quote_date": date(2026, 7, 14),
              "snapshot_label": "decision", "number_of_contracts": 1}
    return ParsedChain(header=header, contracts=pd.DataFrame([row])[CONTRACT_COLUMNS])


def _collector(client, store, token_manager) -> OptionsCollector:
    return OptionsCollector(_cfg(), client, store, token_manager, alerter=MagicMock())


def test_symbols_combines_index_and_equity() -> None:
    """OPT-COLLECT-5: symbols = index + equity when enabled (spec §5)."""
    c = _collector(MagicMock(), MagicMock(), MagicMock())
    assert c.symbols() == ["$SPX", "SPY", "QQQ"]


def test_per_symbol_isolation_one_fails_others_succeed(monkeypatch) -> None:
    """OPT-COLLECT-6: one symbol failing does not abort the rest (spec §10.2)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: (
        (_ for _ in ()).throw(DataError("boom")) if s == "SPY" else {"symbol": s}
    )
    store = MagicMock()
    store.snapshot_exists_parquet.return_value = False
    monkeypatch.setattr(
        "swingrl.data.options.collector.parse_chain",
        lambda raw, **k: _good_parsed(raw["symbol"]),
    )
    result = _collector(client, store, MagicMock()).run_snapshot("decision", now=datetime(2026, 7, 14, 19, 45, tzinfo=UTC))
    assert "SPY" in result.failed
    assert set(result.succeeded) == {"$SPX", "QQQ"}


def test_skip_already_captured() -> None:
    """OPT-COLLECT-7: existing Parquet snapshot is skipped (spec §10.1)."""
    store = MagicMock()
    store.snapshot_exists_parquet.return_value = True
    client = MagicMock()
    result = _collector(client, store, MagicMock()).run_snapshot("decision", now=datetime(2026, 7, 14, 19, 45, tzinfo=UTC))
    client.get_option_chain.assert_not_called()
    assert set(result.skipped) == {"$SPX", "SPY", "QQQ"}


def test_auth_preflight_failure_aborts_all() -> None:
    """OPT-COLLECT-8: token load failure -> CRITICAL, no per-symbol fetch (spec §10.2)."""
    tm = MagicMock()
    tm.load_client.side_effect = DataError("invalid_client")
    alerter = MagicMock()
    c = OptionsCollector(_cfg(), MagicMock(), MagicMock(), tm, alerter=alerter)
    result = c.run_snapshot("decision", now=datetime(2026, 7, 14, 19, 45, tzinfo=UTC))
    assert result.auth_failed is True
    assert alerter.send_alert.call_args.args[0] == "critical"


def test_all_symbols_fail_is_critical(monkeypatch) -> None:
    """OPT-COLLECT-9: every symbol failing -> CRITICAL summary (spec §10.4)."""
    client = MagicMock()
    client.get_option_chain.side_effect = DataError("boom")
    store = MagicMock()
    store.snapshot_exists_parquet.return_value = False
    alerter = MagicMock()
    c = OptionsCollector(_cfg(), client, store, MagicMock(), alerter=alerter)
    c.run_snapshot("decision", now=datetime(2026, 7, 14, 19, 45, tzinfo=UTC))
    levels = [call.args[0] for call in alerter.send_alert.call_args_list]
    assert "critical" in levels


def test_schema_drift_detected() -> None:
    """OPT-COLLECT-10: missing expected greek field flagged (spec §10.5)."""
    raw = {"callExpDateMap": {"2026-07-18:4": {"5000.0": [{"symbol": "A", "bid": 1.0}]}},
           "putExpDateMap": {}}
    missing = check_schema_drift(raw)
    assert "delta" in missing and "openInterest" in missing
```

- [ ] **Step 4: Run → fail; then write `collector.py`**

```python
# src/swingrl/data/options/collector.py
"""EOD collector orchestration: per-symbol fetch->parse->store with guards (spec §6, §10)."""

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
    from swingrl.data.options.schwab_auth import TokenManager
    from swingrl.data.options.schwab_client import SchwabOptionsClient
    from swingrl.data.options.store import OptionsStore
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)
_ET = ZoneInfo("America/New_York")

EXPECTED_CONTRACT_FIELDS = {
    "delta", "gamma", "theta", "vega", "volatility", "openInterest",
    "bid", "ask", "strikePrice", "expirationDate",
}


@dataclass
class SnapshotResult:
    """Outcome of one snapshot run across all symbols."""

    label: str
    succeeded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    auth_failed: bool = False


def check_schema_drift(raw: dict[str, Any]) -> list[str]:
    """Return expected contract fields missing from the first contract (spec §10.5)."""
    for map_key in ("callExpDateMap", "putExpDateMap"):
        for _exp, strike_map in raw.get(map_key, {}).items():
            for _strike, contracts in strike_map.items():
                if contracts:
                    return sorted(EXPECTED_CONTRACT_FIELDS - set(contracts[0]))
    return []


class OptionsCollector:
    """Runs one snapshot across all configured symbols with per-symbol isolation."""

    def __init__(
        self,
        config: SwingRLConfig,
        client: SchwabOptionsClient,
        store: OptionsStore,
        token_manager: TokenManager,
        alerter: Alerter | None = None,
    ) -> None:
        self._config = config
        self._oc = config.options_collector
        self._client = client
        self._store = store
        self._token_manager = token_manager
        self._alerter = alerter

    def symbols(self) -> list[str]:
        """Index symbols first, then equity symbols if enabled (spec §5)."""
        symbols = list(self._oc.index_symbols)
        if self._oc.include_equity_symbols:
            symbols.extend(self._config.equity.symbols)
        return symbols

    def _snapshot_time_utc(self, snapshot_label: str, quote_date: date) -> datetime:
        time_et = next(s.time_et for s in self._oc.snapshots if s.label == snapshot_label)
        hh, mm = (int(x) for x in time_et.split(":"))
        return datetime(
            quote_date.year, quote_date.month, quote_date.day, hh, mm, tzinfo=_ET
        ).astimezone(UTC)

    def run_snapshot(self, snapshot_label: str, now: datetime | None = None) -> SnapshotResult:
        """Fetch+store every symbol's chain for one snapshot; alert on the summary."""
        now = now or datetime.now(UTC)
        quote_date = now.astimezone(_ET).date()
        result = SnapshotResult(label=snapshot_label)

        # Auth preflight — a token failure blocks every symbol (spec §10.2).
        try:
            self._token_manager.load_client()
        except DataError as exc:
            result.auth_failed = True
            self._alert("critical", "Schwab auth blocks options snapshot",
                        f"{snapshot_label} snapshot aborted: {exc}")
            return result

        early_close = market_calendar.is_early_close(quote_date)
        snapshot_time_utc = self._snapshot_time_utc(snapshot_label, quote_date)

        for symbol in self.symbols():
            if self._store.snapshot_exists_parquet(symbol, quote_date, snapshot_label):
                result.skipped.append(symbol)
                continue
            try:
                self._capture_one(
                    symbol, snapshot_label, quote_date, snapshot_time_utc, early_close, result
                )
                result.succeeded.append(symbol)
            except DataError as exc:
                log.error("options_symbol_failed", symbol=symbol, error=str(exc))
                result.failed.append(symbol)

        self._route_summary_alert(result)
        return result

    def _capture_one(
        self, symbol: str, snapshot_label: str, quote_date: date,
        snapshot_time_utc: datetime, early_close: bool, result: SnapshotResult,
    ) -> None:
        raw = self._client.get_option_chain(symbol)
        missing = check_schema_drift(raw)
        if missing:
            result.warnings.append(f"{symbol}: schema drift, missing {missing}")
        if raw.get("_provenance", {}).get("truncated"):
            result.warnings.append(f"{symbol}: chain truncated (partial data)")
        parsed = parse_chain(
            raw, underlying_symbol=symbol, snapshot_label=snapshot_label, quote_date=quote_date,
            snapshot_time_utc=snapshot_time_utc, pulled_at_utc=datetime.now(UTC),
            schema_version=self._oc.schema_version, is_early_close=early_close,
        )
        expected = parsed.header.get("number_of_contracts")
        if expected is not None and expected != len(parsed.contracts):
            result.warnings.append(
                f"{symbol}: row-count mismatch (header {expected} vs parsed {len(parsed.contracts)})"
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

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_market_calendar.py tests/test_options_collector.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/data/options/market_calendar.py src/swingrl/data/options/collector.py tests/test_options_market_calendar.py tests/test_options_collector.py
git commit -m "feat(options): collector orchestration + guards + XNYS calendar (spec §6, §10)"
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
    cfg.options_collector.index_symbols = ["$SPX"]
    assert audit_symbols(cfg) == ["$SPX", "SPY"]


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

### Task 13: `scripts/options_collector_main.py` — scheduler entrypoint

**Files:**
- Create: `scripts/options_collector_main.py`
- Test: `tests/test_options_scheduler.py`

**Interfaces:**
- Consumes: everything from Layers 1–2, `configure_logging`, `DatabaseManager`, `Alerter`, APScheduler (`BackgroundScheduler`, `SQLAlchemyJobStore`, `ThreadPoolExecutor`), `run_data_quality_audit` (T12), `market_calendar` (T11).
- Produces:
  - `build_app(config_path: str) -> dict` — wires all components (mirrors `scripts/main.py` construction).
  - `register_jobs(scheduler, components) -> None` — registers one job per configured snapshot + the fixed jobs.
  - `all_job_ids(config) -> list[str]` — snapshot job ids (per config) + `FIXED_JOB_IDS`.
  - `guarded_snapshot(collector, label, now=None) -> None` — trading-day guard around `run_snapshot`.
  - `run_health_check(config, collector, store, alerter, now=None) -> None` — missed-run safety net.
  - `run_offsite_backup(config, alerter=None) -> None` — rclone sync (subprocess).
  - `main() -> int`.
  - Module constants: `FIXED_JOB_IDS: list[str]` (the 4 non-snapshot jobs); `_SNAPSHOT_MISFIRE_GRACE_S = 3600`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options_scheduler.py
from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

from scripts.options_collector_main import (
    all_job_ids,
    guarded_snapshot,
    register_jobs,
    run_health_check,
)
from swingrl.config.schema import SwingRLConfig


def test_register_jobs_registers_snapshots_plus_fixed() -> None:
    """OPT-SCHED-1: one job per snapshot + fixed jobs, stable ids (spec §9.2, decision D4)."""
    cfg = SwingRLConfig()
    scheduler = MagicMock()
    components = {
        "config": cfg, "collector": MagicMock(), "store": MagicMock(),
        "token_manager": MagicMock(), "alerter": MagicMock(), "db": MagicMock(),
    }
    register_jobs(scheduler, components)
    registered = {call.kwargs["id"] for call in scheduler.add_job.call_args_list}
    assert registered == set(all_job_ids(cfg))
    assert {"options_decision_snapshot", "options_eod_snapshot"} <= registered
    assert len(registered) == 6  # 2 default snapshots + 4 fixed


def test_guarded_snapshot_skips_non_trading_day(monkeypatch) -> None:
    """OPT-SCHED-2: holiday/weekend -> run_snapshot NOT called (spec §9.2)."""
    monkeypatch.setattr(
        "scripts.options_collector_main.market_calendar.is_trading_day", lambda d: False
    )
    collector = MagicMock()
    guarded_snapshot(collector, "decision", now=datetime(2026, 12, 25, 20, 45, tzinfo=UTC))
    collector.run_snapshot.assert_not_called()


def test_guarded_snapshot_runs_on_trading_day(monkeypatch) -> None:
    """OPT-SCHED-3: trading day -> run_snapshot called (spec §9.2)."""
    monkeypatch.setattr(
        "scripts.options_collector_main.market_calendar.is_trading_day", lambda d: True
    )
    collector = MagicMock()
    guarded_snapshot(collector, "eod", now=datetime(2026, 7, 14, 20, 30, tzinfo=UTC))
    collector.run_snapshot.assert_called_once_with("eod")


def test_health_check_missed_run_is_critical(monkeypatch) -> None:
    """OPT-SCHED-4: a whole snapshot absent -> CRITICAL missed-run (spec §9.2, §10.4)."""
    monkeypatch.setattr(
        "scripts.options_collector_main.market_calendar.is_trading_day", lambda d: True
    )
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["$SPX"]
    collector = MagicMock()
    collector.symbols.return_value = ["$SPX", "SPY"]
    store = MagicMock()
    store.snapshot_exists_parquet.return_value = False  # nothing captured
    alerter = MagicMock()
    run_health_check(cfg, collector, store, alerter, now=datetime(2026, 7, 14, 21, 15, tzinfo=UTC))
    assert any(c.args[0] == "critical" for c in alerter.send_alert.call_args_list)


def test_health_check_partial_is_warning(monkeypatch) -> None:
    """OPT-SCHED-5: some symbols missing -> WARNING (spec §10.4)."""
    monkeypatch.setattr(
        "scripts.options_collector_main.market_calendar.is_trading_day", lambda d: True
    )
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["$SPX"]
    collector = MagicMock()
    collector.symbols.return_value = ["$SPX", "SPY"]
    store = MagicMock()
    # $SPX present for both snapshots; SPY missing.
    store.snapshot_exists_parquet.side_effect = lambda s, d, label: s == "$SPX"
    alerter = MagicMock()
    run_health_check(cfg, collector, store, alerter, now=datetime(2026, 7, 14, 21, 15, tzinfo=UTC))
    levels = [c.args[0] for c in alerter.send_alert.call_args_list]
    assert "warning" in levels and "critical" not in levels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.options_collector_main'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/options_collector_main.py
"""Standalone options-collector container entrypoint (spec §9).

Its OWN scheduler + jobstore + token file. Never touches the trader (A30).
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
from swingrl.data.options.collector import OptionsCollector
from swingrl.data.options.schwab_auth import TokenManager
from swingrl.data.options.schwab_client import SchwabOptionsClient
from swingrl.data.options.store import OptionsStore
from swingrl.monitoring.alerter import Alerter
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)
_ET = ZoneInfo("America/New_York")
_SNAPSHOT_MISFIRE_GRACE_S = 3600

FIXED_JOB_IDS = [
    "options_token_reminder",
    "options_health_check",
    "options_data_audit",
    "options_offsite_backup",
]


def _snapshot_job_id(label: str) -> str:
    """Stable APScheduler job id for a snapshot label."""
    return f"options_{label}_snapshot"


def all_job_ids(config: SwingRLConfig) -> list[str]:
    """One snapshot job per configured snapshot + the fixed jobs (spec §9.2, decision D4)."""
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
    token_manager = TokenManager(config.options_collector, alerter=alerter)
    client = SchwabOptionsClient(config.options_collector, token_manager)
    store = OptionsStore(config.options_collector, db=db)
    collector = OptionsCollector(config, client, store, token_manager, alerter=alerter)
    return {
        "config": config, "db": db, "alerter": alerter, "token_manager": token_manager,
        "client": client, "store": store, "collector": collector,
    }


def guarded_snapshot(collector: OptionsCollector, label: str, now: datetime | None = None) -> None:
    """Run a snapshot only on NYSE trading days (holiday guard; spec §9.2)."""
    now = now or datetime.now(UTC)
    quote_date = now.astimezone(_ET).date()
    if not market_calendar.is_trading_day(quote_date):
        log.info("options_snapshot_skipped_non_trading_day", label=label, date=quote_date.isoformat())
        return
    collector.run_snapshot(label)


def run_health_check(
    config: SwingRLConfig, collector: OptionsCollector, store: OptionsStore,
    alerter: Alerter, now: datetime | None = None,
) -> None:
    """Verify today's snapshots landed; CRITICAL on a missed run (spec §9.2)."""
    now = now or datetime.now(UTC)
    quote_date = now.astimezone(_ET).date()
    if not market_calendar.is_trading_day(quote_date):
        return
    symbols = collector.symbols()
    for snap in config.options_collector.snapshots:
        present = [s for s in symbols if store.snapshot_exists_parquet(s, quote_date, snap.label)]
        if not present:
            alerter.send_alert(
                "critical", f"Options {snap.label} MISSED",
                f"No {snap.label} snapshot for any symbol on {quote_date.isoformat()}.",
            )
        elif len(present) < len(symbols):
            missing = [s for s in symbols if s not in present]
            alerter.send_alert(
                "warning", f"Options {snap.label} incomplete",
                f"Missing {snap.label} for {missing} on {quote_date.isoformat()}.",
            )


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
    """Register per-snapshot + fixed cron jobs on the scheduler (spec §9.2, decision D4)."""
    config: SwingRLConfig = components["config"]
    oc = config.options_collector
    collector = components["collector"]
    store = components["store"]
    alerter = components["alerter"]
    db = components["db"]

    hh, hm = _hhmm(oc.health_check_time_et)
    th, tm = _hhmm(oc.token_reminder_time_et)
    ah, am = _hhmm(oc.integrity.audit_time_et)
    bh, bm = _hhmm(oc.backup.time_et)

    # One snapshot job per configured snapshot — number/times are pure config (decision D4).
    for snap in oc.snapshots:
        sh, sm = _hhmm(snap.time_et)
        scheduler.add_job(
            guarded_snapshot, trigger="cron", day_of_week="mon-fri", hour=sh, minute=sm,
            timezone="America/New_York", args=[collector, snap.label],
            id=_snapshot_job_id(snap.label), replace_existing=True,
        )
    scheduler.add_job(
        components["token_manager"].check_token_age_and_alert, trigger="cron", hour=th, minute=tm,
        timezone="America/New_York", id="options_token_reminder", replace_existing=True,
    )
    scheduler.add_job(
        run_health_check, trigger="cron", day_of_week="mon-fri", hour=hh, minute=hm,
        timezone="America/New_York", args=[config, collector, store, alerter],
        id="options_health_check", replace_existing=True,
    )
    scheduler.add_job(
        run_data_quality_audit, trigger="cron", day=oc.integrity.audit_day_of_month,
        hour=ah, minute=am, timezone="America/New_York",
        kwargs={"config": config, "db": db, "alerter": alerter},
        id="options_data_audit", replace_existing=True,
    )
    scheduler.add_job(
        run_offsite_backup, trigger="cron", hour=bh, minute=bm, timezone="America/New_York",
        args=[config, alerter], id="options_offsite_backup", replace_existing=True,
    )


def _make_signal_handler(scheduler: Any, stop_event: threading.Event):
    def handler(_signum, _frame) -> None:
        log.info("options_collector_shutting_down")
        scheduler.shutdown(wait=False)
        stop_event.set()

    return handler


def main() -> int:
    """Build, register jobs, reconcile, start the scheduler, and block."""
    parser = argparse.ArgumentParser(description="SwingRL options collector")
    parser.add_argument("--config", default="config/swingrl.yaml")
    args = parser.parse_args()

    components = build_app(args.config)
    components["store"].reconcile()  # self-heal any unsynced Parquet on boot (spec §8.2)

    scheduler = BackgroundScheduler(
        jobstores={
            "default": SQLAlchemyJobStore(
                url=f"sqlite:///{components['config'].options_collector.apscheduler_db_path}"
            )
        },
        executors={"default": ThreadPoolExecutor(max_workers=4)},
        job_defaults={"coalesce": True, "max_instances": 1,
                      "misfire_grace_time": _SNAPSHOT_MISFIRE_GRACE_S},
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
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/options_collector_main.py tests/test_options_scheduler.py
git commit -m "feat(options): scheduler entrypoint — config-driven jobs, trading-day guard, health check (spec §9)"
```

---

### Task 14: Docker service `swingrl-options`

**Files:**
- Modify: `docker-compose.yml` (add the `swingrl-options` service)

**Interfaces:**
- Consumes: the existing `production` Dockerfile target (ships `src/` + `scripts/` already — no Dockerfile change needed), `.env`, `br0` network (pg16 access).
- Produces: an always-on `swingrl-options` container running `scripts/options_collector_main.py` with its own lifecycle. Rebuilding/restarting it never touches `swingrl` (A30).

- [ ] **Step 1: Add the service to `docker-compose.yml`**

Insert a new service (mirrors the `swingrl` trader service, docker-compose.yml:63-95, but with its own command, the `secrets/` mount, no `depends_on` the trader, and lower `cpus`):

```yaml
  swingrl-options:
    container_name: swingrl-options
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    cpus: 2.0
    # Secrets (incl. secrets/schwab_token.json) loaded at runtime; never baked in.
    env_file: .env
    environment:
      - TZ=America/New_York
    command: ["python", "scripts/options_collector_main.py"]
    volumes:
      - ./data:/app/data
      - ./db:/app/db
      - ./config:/app/config
      - ./logs:/app/logs
      - ./secrets:/app/secrets
    networks:
      - default
      - br0
    restart: unless-stopped
```

- [ ] **Step 2: Validate the compose file**

Run: `docker compose config >/dev/null && echo OK`
Expected: `OK` (compose parses; the new service resolves `br0`, `.env`, and the build target).

- [ ] **Step 3: Build the image (no-cache, per homelab convention)**

Run: `docker compose build --no-cache swingrl-options`
Expected: build succeeds (same production image content as the trader).

- [ ] **Step 4: Smoke-test imports inside the image**

Run: `docker compose run --rm --no-deps swingrl-options python -c "import scripts.options_collector_main as m; print(m.FIXED_JOB_IDS)"`
Expected: prints the 4 fixed job ids. (Confirms every module imports cleanly in the container; no scheduler is started.)

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml
git commit -m "feat(options): swingrl-options always-on container (A30-isolated) (spec §9.1)"
```

---

### Task 15: Documentation & runbooks

**Files:**
- Create: `docs/options/register-schwab-app.md`
- Create: `docs/options/first-oauth.md`
- Create: `docs/options/weekly-reauth-runbook.md`
- Create: `docs/options/week1-data-quality-audit.md`
- Create: `docs/options/ops.md`
- Create: `docs/options/data-caveats.md`

**Interfaces:** none (documentation). Each doc opens with a one-line purpose and (where relevant) a glossary pointer to the spec.

- [ ] **Step 1: `register-schwab-app.md`** — how to create the Market-Data-only app at developer.schwab.com; set callback `https://127.0.0.1:8182`; where API key/secret go (`.env` as `SCHWAB_API_KEY`/`SCHWAB_APP_SECRET`); the "Approved – Pending" activation-lag caveat (spec §15.6); least-privilege note (spec §13).

- [ ] **Step 2: `first-oauth.md`** — run `uv run python scripts/schwab_reauth.py`; log in + MFA; copy the redirected `https://127.0.0.1:8182/?code=...` URL back; that the page failing to load is EXPECTED (spec §7.4); confirm `secrets/schwab_token.json` exists and is `chmod 600`; then run `scripts/capture_chain_fixture.py` and record `isDelayed` + the exact `$SPX` symbol (spec §11, §15).

- [ ] **Step 3: `weekly-reauth-runbook.md`** — the ~30-second weekly action; **single-source discipline** (a new login on ANY other machine silently kills the homelab token — always re-auth into the mounted token file and nowhere else; spec §13); what the WARNING (day 5/6) and CRITICAL (expired) alerts mean; the week-1 finding (does the token roll forward under daily use?, spec §7.2).

- [ ] **Step 4: `week1-data-quality-audit.md`** — the manual checks (spec §10.6): reconstruct an SPX IV surface; confirm delta ∈ [-1,1] and monotone across strikes; OI populated; `bid ≤ ask` with plausible spreads; decision→eod drift non-trivial and plausible. Include the exact SQL to pull one snapshot from `options_chains`.

- [ ] **Step 5: `ops.md`** — container lifecycle (`docker compose up -d swingrl-options`, logs, restart — and that it's A30-isolated from the trader); the jobs and their times (15:45 + 16:35 snapshots); where data lands (Parquet dirs + Postgres tables); disk-growth monitoring + the `postgres_store_raw_json` flag (spec §13, decision D5); how reconcile self-heals a pg16 outage; alert catalogue (spec §10.4).

- [ ] **Step 6: `data-caveats.md`** — the "this data isn't quite what it looks like" doc (decisions D3/D6): **OI** is T-1 / once-daily / stable intraday — identical across same-day snapshots, and the one-day-arrears value is the correct no-lookahead value; **splice note** — verify the OI date-convention matches the purchased historical dataset (DiscountOptionData/DataBento) on the overlap and shift by a day if it differs; **16:35 = close mark, not settlement** — official settlement (SET for AM-settled SPX / index close for PM-settled SPXW) is free/tiny from CBOE and belongs to the separate premium-project sourcing track, not this collector; **greeks/IV are Schwab's vendor black-box** — recompute from bid/ask + underlying + strike + dte (+ FRED rate, put-call-parity q); the trustworthy fields are the quotes and contract identity, not the vendor greeks.

- [ ] **Step 7: Commit**

```bash
git add docs/options/
git commit -m "docs(options): register-app, first-OAuth, weekly-reauth, week-1 audit, ops, data-caveats runbooks (spec §9,§13)"
```

---

### Task 16 🛑: Homelab CI, first live run, pin fixture, offsite backup

**Human-in-the-loop gate.** Resolves the spec's first-run unknowns (§15) empirically and pins the parser to real data (decision D2). Requires an approved deploy per CLAUDE.md — **do not deploy without explicit approval.**

**Files:**
- Create: `tests/test_chain_parser_real_fixture.py` (pins the parser to the real capture)

**Interfaces:**
- Consumes: the real `tests/fixtures/schwab_chain_spx.json` (from T6), `parse_chain` (T7), `EXPECTED_CONTRACT_FIELDS` (T11).

- [ ] **Step 1: Write the pin-to-real-fixture test**

```python
# tests/test_chain_parser_real_fixture.py
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from swingrl.data.options.chain_parser import parse_chain
from swingrl.data.options.collector import EXPECTED_CONTRACT_FIELDS, check_schema_drift

_FIXTURE = Path("tests/fixtures/schwab_chain_spx.json")


@pytest.mark.skipif(not _FIXTURE.exists(), reason="real fixture not yet captured (T6)")
def test_parse_real_spx_fixture_no_schema_drift() -> None:
    """OPT-PARSE-11: parser handles the REAL captured chain, no drift (spec §3, §12)."""
    raw = json.loads(_FIXTURE.read_text())
    assert check_schema_drift(raw) == [], "real payload field names differ — update the mapping"
    parsed = parse_chain(
        raw, underlying_symbol=raw["symbol"], snapshot_label="eod", quote_date=date(2026, 7, 14),
        snapshot_time_utc=datetime(2026, 7, 14, 20, 30, tzinfo=UTC),
        pulled_at_utc=datetime(2026, 7, 14, 20, 30, tzinfo=UTC),
        schema_version="v1", is_early_close=False,
    )
    assert len(parsed.contracts) > 0
    assert parsed.contracts["iv"].notna().any()
    assert parsed.header["number_of_contracts"] is not None
```

- [ ] **Step 2: Run it; reconcile any field-name differences**

Run: `uv run pytest tests/test_chain_parser_real_fixture.py -v`
If it fails on schema drift, the real Schwab field names differ from the hand-authored fixture — update `chain_parser._row(...)` mappings and `EXPECTED_CONTRACT_FIELDS` (T11) to the real names, re-run T7's tests, then re-run this. Commit the fix:

```bash
git add tests/test_chain_parser_real_fixture.py src/swingrl/data/options/chain_parser.py src/swingrl/data/options/collector.py
git commit -m "test(options): pin chain_parser to real captured SPX fixture (spec §12)"
```

- [ ] **Step 3: Apply the additive migration to live pg16** (approval required)

```bash
cd ~/swingrl && git fetch origin && git checkout <branch> && git pull origin <branch>
DATABASE_URL=<pg16-url> uv run python scripts/migrations/add_options_capture_tables.py
```
Expected: `options_capture_migration_applied`. Verify `options_snapshots` + `options_chains` + current-month partition exist. This is additive-only — the trader is unaffected (A30).

- [ ] **Step 4: Homelab CI** (per CLAUDE.md phase-closeout)

```bash
cd ~/swingrl && bash scripts/ci-homelab.sh --no-cache
```
Expected: PASS, including the DB-gated tests (T9/T10 run against pg16). If coverage or lint fails, fix before deploy.

- [ ] **Step 5: Deploy the container** (approval required)

```bash
cd ~/swingrl && docker compose up -d swingrl-options && docker compose logs -f swingrl-options
```
Confirm `options_collector_started` with the 6 job ids; confirm reconcile ran on boot.

- [ ] **Step 6: First live snapshot — RECORD the first-run findings**

Either wait for the 15:45/16:35 cron or trigger once manually inside the container:
```bash
docker compose exec swingrl-options python -c \
  "from scripts.options_collector_main import build_app, guarded_snapshot; \
   c=build_app('config/swingrl.yaml'); guarded_snapshot(c['collector'],'eod')"
```
Record in spec §3 / `data/options_eod/schwab/metadata.json`: **(a)** exact `$SPX` symbol, **(b)** `isDelayed` (gates decision-snapshot fidelity — spec §11), **(c)** whether `$SPX` tripped `isChainTruncated` (and if so tune `integrity.refetch_dte_chunks`), **(d)** `numberOfContracts` per symbol. Confirm a Discord INFO digest arrived.

- [ ] **Step 7: Stand up the offsite backup** (spec §13)

Configure the `rclone` remote (`b2:swingrl-options`), then verify the backup job:
```bash
docker compose exec swingrl-options python -c \
  "from scripts.options_collector_main import build_app, run_offsite_backup; \
   c=build_app('config/swingrl.yaml'); run_offsite_backup(c['config'], c['alerter'])"
```
Confirm `options_offsite_backup_ok` and that objects appear at the remote.

- [ ] **Step 8: Week-1 watch** — run the week-1 data-quality audit runbook (T15 doc); observe the true 7-day token behavior (does it roll forward under daily use?); confirm the day-5/6 WARNING fires. Record the token-behavior finding in `metadata.json` and, if it rolls forward, relax `reminder_days` per spec §7.2.

- [ ] **Step 9: Commit any findings/tuning**

```bash
git add config/swingrl.yaml docs/superpowers/specs/2026-07-14-schwab-options-collector-design.md
git commit -m "chore(options): record first-run entitlement/volume/token findings + tuning (spec §15)"
```

---

## Self-Review — Spec Coverage

| Spec section | Covered by |
|---|---|
| §2 goal, forward-capture, dual-use library | T2–T13 (library in `src/swingrl/data/options/`) |
| §5 symbols & config (nothing hardcoded) | T2 (`OptionsCollectorConfig`), T11 (`symbols()` = index + equity) |
| §6.1 two snapshots (15:45 + 16:35, config-driven) + early-close provenance | T2 (snapshots), T11 (`is_early_close`, `run_snapshot`), T13 (per-snapshot job registration, decision D4) |
| §6.2 capture everything (typed + raw_json) | T7 (`CONTRACT_COLUMNS` + `raw_json`), T8 (Parquet), T10 (JSONB) |
| §6.3 flattened grain + column mapping | T7 |
| §6.4 snapshot-level context / header | T7 (`header`), T8 (sidecar), T10 (`options_snapshots`) |
| §7 auth & 7-day token | T3 (`TokenManager`), T4 (re-auth CLI), T16 (week-1 behavior) |
| §8.1 Parquet layout + atomic + resume unit | T8 |
| §8.2 Postgres tables, partitions, idempotency, reconcile | T9 (schema/migration), T10 (sync/reconcile) |
| §8.3 metadata.json sidecar | T6 (fixture findings), T15/T16 (provenance recorded) |
| §9 scheduling & runtime (own scheduler/jobstore) | T13, T14 |
| §10.1–10.2 idempotency, resumability, isolation | T8/T10 (skip + ON CONFLICT), T11 (per-symbol try/except) |
| §10.3 typed errors + retry | T5 (`swingrl_retry`, `DataError`), all modules |
| §10.4 Discord alert catalogue | T3/T11/T13 (`send_alert` routing) |
| §10.5 silent-corruption guards (truncation/drift/row-count/OI stability) | T5 (re-fetch), T11 (`check_schema_drift`, row-count, truncation flag), T12 (`oi_stability_failures`, decision D6) |
| §10.6 data-quality audit | T12, T15 (runbook), T16 (week-1) |
| §11 first-run entitlement check | T6 (capture), T16 (record `isDelayed`) |
| §12 testing (fixture-pinned) | every task's tests; T6 + T16 (real fixture) |
| §13 security, secrets, single-source re-auth, offsite backup, storage growth | T1 (gitignore/secrets), T4 (chmod 600), T13/T16 (rclone), T15 (runbook + data-caveats), T2/T10 (`postgres_store_raw_json` flag, decision D5) |
| §14 build sequence (10 phases) | T1–T16 (mapped in the task table) |
| §15 open questions (empirical) | T6 + T16 (recorded, not guessed) |
| §16 success criteria | full plan; verified at T16 |

**Placeholder scan:** no "TBD"/"implement later"/"add error handling" — every code step shows real code. **Type consistency:** `ParsedChain(header, contracts)`, `parse_chain(...)`, `OptionsStore.write_snapshot/read_snapshot/sync_to_postgres/reconcile`, `SchwabOptionsClient.get_option_chain`, `TokenManager.load_client`, `OptionsCollector.run_snapshot/symbols`, `SnapshotResult`, and `CONTRACT_COLUMNS`/`DB_CHAIN_COLUMNS` names are used identically across producing and consuming tasks.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-14-schwab-options-collector-plan.md`.

Per the standing instruction for this work: **STOP here — do not begin implementation without a separate go-ahead.** When you give the go-ahead, two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, two-stage review between tasks, fast iteration (`superpowers:subagent-driven-development`).
2. **Inline Execution** — tasks executed in this session with batch checkpoints (`superpowers:executing-plans`).

Gates that always require a human: **T6** (first OAuth + fixture) and **T16** (live pg16 migration, homelab CI, deploy, offsite backup) — the latter needs explicit deploy approval per CLAUDE.md.
