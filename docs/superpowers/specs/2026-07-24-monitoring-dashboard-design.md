# Monitoring Dashboard — Design Spec

> **Status: DESIGN APPROVED (2026-07-24)** — all four design sections signed off by the
> user in the brainstorming session of the same date. Supersedes the Streamlit dashboard
> shipped in v1.0 (`dashboard/`), which is retired by this work.
>
> **Design source:** `docs/superpowers/design_handoff_swingrl_dashboard/` — a Claude Design
> handoff bundle (README, Streamlit-handoff, PWA-notes, `SwingRL Dashboard.dc.html`,
> `SwingRL Mobile Preview.dc.html`, `ios-frame.jsx`, `support.js`). This is the project's
> **first Claude Design → Claude Code handoff**.
>
> **Companion plans:** Plan A `2026-07-07-capture-foundation-plan.md` (paper-trading capture
> foundation — supplies the tables this dashboard reads). Plan B
> `2026-07-07-training-redesign-plan.md` (training redesign — supplies the metrics that
> eventually fill page 6).
>
> **Implementation plan:** `docs/superpowers/plans/2026-07-24-monitoring-dashboard-plan.md`

**Goal.** Give the operator a complete, live picture of the paper-trading system on both
desktop and phone, so that capital-preservation decisions can be made from evidence rather
than from Discord notification fragments. Discord tells you *that* something happened; the
dashboard tells you *where you stand*.

**Primary constraint (inherited from the product).** Capital preservation. The dashboard
must surface halts and data staleness loudly, and must **never present a stale number as
live** — in the UI, in the cache, or offline.

---

## Glossary — read this first

Every shorthand used in this spec, defined once (per the project's plain-English rule).

| Term | Meaning |
|---|---|
| **PWA** | Progressive Web App — a website the browser can install to the phone home screen so it launches full-screen like a native app |
| **Service worker** | A background script the browser runs for an installed PWA; it decides what is served from cache versus fetched from the network |
| **Manifest** | `manifest.json` — the file that tells the browser the app's name, icon, colours, and display mode, making it installable |
| **Secure context** | A page served over valid HTTPS (or localhost). Service workers and PWA install are refused anywhere else |
| **Jinja2** | Python's HTML templating engine — `{{ value }}` placeholders and `{% if %}` / `{% for %}` blocks, filled in on the server before the page is sent |
| **Server-rendered** | The HTML arrives from the server already complete, as opposed to being assembled in the browser by JavaScript |
| **SVG polyline** | A vector line drawn by listing its corner coordinates: `points="0,124 80,122 160,120"`. The chart format the design already uses |
| **viewBox** | An SVG's internal coordinate canvas (here `0 0 400 140`), which stretches to whatever pixel size the card gives it |
| **FOUC** | Flash Of Unstyled Content — the visible blink when a page renders one way and JavaScript immediately restyles it |
| **ITP** | Intelligent Tracking Prevention — WebKit's privacy system. Relevant here because it caps JavaScript-written cookies at 7 days on iOS |
| **Traefik** | The homelab reverse proxy that terminates HTTPS and routes hostnames to containers |
| **Vault CA** | The homelab's private certificate authority, which issues the `*.home.lab` certificates Traefik serves |
| **Forward-auth** | A proxy pattern where Traefik checks with an auth service before passing a request through. **Not used here** — see Deployment |
| **A30** | The project's deploy-isolation rule: while paper trading is live, no deploy may rebuild or restart the trader container |
| **Staleness window** | How old the newest bar may be before a feed is considered degraded: equity 26 h, crypto 5 h |
| **Circuit breaker** | The risk layer's automatic trading halt, tripped by drawdown or daily-loss thresholds |
| **Data gate** | A pause caused by missing/stale data rather than by a risk breach — breakers stay ARMED |
| **HWM** | High-water mark — the highest portfolio value reached, used as the drawdown denominator |
| **B&H** | Buy and hold — the passive benchmark the agent is measured against |
| **Vendoring** | Keeping your own copy of an outside file (here, the two font files) inside the app and serving it from your own server, instead of loading it from someone else's server every time the page opens |
| **Stub** | A page or panel that exists and is reachable but deliberately shows "coming soon" instead of real content, so the structure is in place before the data is |
| **TTL** | Time To Live — how long a cached answer is reused before the app fetches a fresh one |
| **HttpOnly** | A cookie flag meaning JavaScript cannot read the cookie; only the server sees it |
| **SameSite=Lax** | A cookie flag that stops the cookie being sent on requests originating from other websites |
| **Downsampling** | Drawing fewer points than the data contains, because a chart 400 units wide cannot show 5,000 distinct values |
| **CVE surface** | A dependency file that `pip-audit` scans for known vulnerabilities in CI |
| **pip-audit** | The CI stage that fails the build on known-vulnerable dependencies |

---

## Needs from you — decisions and actions, stated up front

Per the project's standing rule that user dependencies are surfaced early, never buried.

| # | Item | When | Status |
|---|---|---|---|
| U-1 | **Approve the read-only role SQL** (`CREATE ROLE swingrl_dashboard` + `GRANT SELECT`). Runs against the live database as a superuser; the migration runner cannot do it | Before the deploy task | ⏳ pending |
| U-2 | **Approve the deploy** of the new `swingrl-dashboard` service | Before the deploy task | ⏳ pending |
| U-3 | Confirm the `prefs_cookie_days` value if 180 is wrong | Any time — one-line YAML edit | ✅ defaulted to 180 |
| U-4 | Decide whether `alert_log` gains a `message` column | **Out of scope** — separate follow-up | 📋 logged |
| U-5 | Broker fill verification — on mismatch, alert only, or alert *and* correct the stored value? | **Out of scope** — trader bug fix, own workstream (see Adjacent work) | 📋 logged |

**Explicitly NOT needed:** root-CA trust on your devices. Every device on the LAN,
including the iPhone, already trusts the homelab Vault root CA, so `https://*.home.lab` is
a valid secure context and the PWA installs with no profile work. This is settled
infrastructure and is not a risk, gate, or prerequisite anywhere in this spec.

---

## Locked decisions (2026-07-24 session)

| # | Decision | Rationale |
|---|---|---|
| D-1 | Dashboard gets its **own spec + plan**, not a phase of Plan B | Five of six pages are paper-trading monitoring; Plan B is purely training-side. Plan B receives a cross-reference and one task for page 6 |
| D-2 | Stack is **FastAPI + Jinja2**, replacing Streamlit | The design file is already a template (`{{ }}`, `<sc-if>`, `<sc-for>`); it ports near-verbatim to Jinja, giving ~100% fidelity. Streamlit caps out near 60% and needs CSS that breaks on upgrades |
| D-3 | Host at **`swingrl.home.lab`**, LAN-only, Vault cert | Internal monitoring surface; no public exposure |
| D-4 | **No Authentik** / no application auth | LAN-only access is sufficient for the operator's threat model |
| D-5 | **Retire the Streamlit app**; page 6 ships as a "coming soon" stub | Its one valuable page reads tables Plan B replaces — building it now means building it twice |
| D-6 | **All five real pages ship in one deploy** | User preference over incremental slices |
| D-7 | Charts are **server-rendered SVG**, not a JS charting library | The design *is* SVG; generation is a pure Python function, so it is unit-testable and adds no JS dependency. Isolated in `charts.py` + one partial, so swapping later is contained |
| D-8 | UI preferences persist in a **server-set cookie**, 180 days, config-driven | Server-side rendering sets `data-theme`, so a cookie avoids a theme flash on first paint. Server-set (not JS-set) sidesteps WebKit's 7-day cap on script-written cookies |
| D-9 | Dashboard reads the database as a **SELECT-only role** | A monitoring surface must be structurally incapable of writing a trade, order, or snapshot |
| D-10 | Build this **before** resuming Stage 2.R training work | Paper trading is live and effectively unmonitored; the training redesign has no clock on it |
| D-11 | **Pending orders get their own panel**, on Dashboard and Trade Log | `pending_orders` is populated (8 rows) and the design has no home for it. "Did my order actually go through?" is a first-order operator question |
| D-12 | **Broker reconciliation is out of scope** — it is a trader bug fix, not dashboard work | See "Adjacent work discovered" below. The dashboard displays whatever the trader records; it does not verify it |
| D-13 | Refresh interval and cookie lifetime are **config values**, not constants | Equity trades once daily and crypto every 4 h, so a fixed 5-minute poll is mostly wasted phone battery. `dashboard.refresh_seconds` defaults to 300 to match the design |
| D-14 | The plan **opens with a panel-by-panel data audit**, before any code | A surface pass already found three mismatches between the design's assumptions and the real schema. Finding the rest costs far less before the templates exist |

---

## Verified facts this design rests on

Read directly from the repository, the running containers, and the live database on
2026-07-24. Anything not verified is marked UNKNOWN in Open Questions.

| Fact | How verified |
|---|---|
| A Streamlit dashboard exists: `dashboard/app.py` + 5 pages, 1,468 LOC incl. 429 lines of tests | file reads |
| It is **not deployed** — no `swingrl-dashboard` container is running | `docker ps -a` |
| `docker-compose.prod.yml`'s dashboard service is stale: SQLite-era `SWINGRL_DB_DIR=/app/db`, no `DATABASE_URL` | file read |
| `dashboard/requirements.txt` is one of three `pip-audit --strict` surfaces in `ci-homelab.sh` | capture-foundation plan Task 14 |
| Traefik terminates TLS; `web` → `websecure` redirect is global; `*.home.lab` uses `certresolver=vault` | `docker inspect traefik` |
| Traefik and `pg16` are both on the `br0` network; the trader is on `br0` + `swingrl_default` | `docker inspect` |
| Database host is `pg16:5432/swingrl` | trader environment |
| The design file uses `{{ }}` / `<sc-if>` / `<sc-for>` templating with inline styles and a CSS custom-property token block | file read |
| Chart curves are already data-bound (`{{ eqCurve }}`, `{{ eqArea }}`); the B&H baseline, gridlines, and axis labels are hardcoded mock values | file read |
| Live row counts (2026-07-24): `trades` 19 · `positions` 10 · `portfolio_snapshots` 38 · `fill_quality` 17 · `benchmark_baselines` 10 · `risk_decisions` 57 · `circuit_breaker_events` 3 · `alert_log` 109 · `data_ingestion_log` 1,542 · `inference_cycles` 40 · `inference_outcomes` 306 · `pending_orders` 8 | live DB query |
| Empty tables: `system_events` 0 · `api_errors` 0 · `emergency_flags` 0 · `shadow_trades` 0 | live DB query |
| `alert_log` columns are `alert_id, timestamp, level, title, message_hash, sent` — **no message body** | live DB query |
| `alert_log` level mix: info 71 · critical 32 · warning 6 | live DB query |
| No `config` table exists — thresholds live in `config/swingrl.yaml` | live DB query |
| `iteration_results` has 10 rows but is stale (latest 2026-04-02), consistent with training being paused | live DB query |

---

## Architecture

A single FastAPI process serves server-rendered Jinja pages. It reads PostgreSQL and
nothing else — no broker calls, no writes, no Docker socket.

```
                    LAN only
  iPhone / desktop ───────────► Traefik (443, websecure)
                                   │  Host(`swingrl.home.lab`)
                                   │  certresolver=vault
                                   ▼
                          swingrl-dashboard  (uvicorn, non-root, 512m)
                                   │  SELECT-only role
                                   ▼
                             pg16:5432/swingrl
```

**Boundaries.** Each unit has one purpose and is testable on its own:

| Unit | Responsibility | Depends on |
|---|---|---|
| `queries/` | Read rows. One module per domain. No formatting, no derivation | psycopg |
| `derive.py` | Turn rows into the numbers the UI shows | nothing (pure) |
| `charts.py` | Turn series into SVG coordinate strings | nothing (pure) |
| `templates/` | Present. No logic beyond loops and conditionals | the above |
| `app.py` | Route, assemble context, set cookies | all of the above |

`derive.py` and `charts.py` being pure is deliberate: they hold the logic most likely to be
wrong, and pure functions can be tested exhaustively without a database or a browser.

### Repo layout

Replaces the top-level `dashboard/` directory. Living under `src/swingrl/` means it
inherits the project's mypy (`disallow_untyped_defs`), ruff, and test conventions.

```
src/swingrl/dashboard/
  app.py               FastAPI factory, routes, cookie handling
  queries/
    portfolio.py       portfolio_snapshots, positions
    trades.py          trades, fill_quality
    risk.py            risk_decisions, circuit_breaker_events
    health.py          data_ingestion_log, inference_cycles, alert_log
    benchmark.py       benchmark_baselines
  derive.py            cash/P&L, env split, exposure, staleness, breaker state
  charts.py            pure SVG point-string generation
  templates/
    base.html          shell: header, nav, status banner, bottom tab bar
    partials/          kpi_tile, status_dot, chart_svg, table, empty_state
    pages/             dashboard, portfolio, trade_log, risk, health, training
  static/
    app.css            design tokens + component styles, lifted from the design file
    app.js             ~150 lines: hover crosshair, filters, toggles, prefs POST
    manifest.json
    service-worker.js
    icon-192.png  icon-512.png  icon-512-maskable.png
```

**Dependencies** move into `pyproject.toml` as an optional group so FastAPI never lands in
the trader image:

```toml
[project.optional-dependencies]
dashboard = ["fastapi", "uvicorn[standard]", "jinja2", "psycopg[binary]"]
```

---

## Data layer

**Connection.** A psycopg connection pool, opened at app startup, authenticating as the
SELECT-only role. Every query is read-only by construction, not by convention.

**Caching.** A small TTL cache per query function, its lifetime matching the configured
refresh interval. The cache stores rows, never rendered HTML.

**Configuration** (all in `config/swingrl.yaml`, no hardcoded values per CLAUDE.md):

| Key | Default | Purpose |
|---|---|---|
| `dashboard.refresh_seconds` | 300 | Auto-refresh interval, matching the design. Raise it — equity trades once a day and crypto every 4 h, so frequent polling mostly costs phone battery |
| `dashboard.prefs_cookie_days` | 180 | Preference cookie lifetime |
| `dashboard.max_chart_points` | 400 | Downsampling cap (see Charts) |
| staleness windows, breaker thresholds, symbols, capital | existing keys | Reused, not redefined |

**Time.** All timestamps are stored and compared in UTC; conversion to `America/New_York`
happens only at the display edge, in a Jinja filter. Every panel renders an **as-of**
timestamp alongside its numbers.

**Derivations** (all in `derive.py`, all pure):

| Value | Derivation |
|---|---|
| Cash / P&L | from `portfolio_snapshots` + the ledger; **not** stored per trade |
| Equity vs crypto split | from per-environment `portfolio_snapshots` rows |
| Exposure | `sum(abs(position_value)) / total_value`, display-capped at 100% |
| Feed staleness | `now() - max(bar_timestamp)` per environment, vs the configured window |
| Breaker state | latest `circuit_breaker_events` per environment, plus the global rule |
| vs B&H | agent value vs `benchmark_baselines` baseline price × capital |

**Thresholds** come from `config/swingrl.yaml` via `load_config()`, mounted read-only —
the handoff assumed a `config` table, which does not exist.

---

## Pages and panels

Ordering within every page follows the design's information hierarchy:
**status → money → risk → activity.**

### 1 · Dashboard (summary)

| Panel | Source |
|---|---|
| System status, grouped Infrastructure / Brokers / Data feeds | derived liveness (see System Health) |
| KPI tiles ×4: total value, open positions, current drawdown, breaker state | `portfolio_snapshots` latest, `positions` count, `circuit_breaker_events` |
| Equity curve (agent + dashed B&H baseline) | `portfolio_snapshots.total_value`, `benchmark_baselines` |
| Drawdown chart, with 5% / 8% threshold lines | `portfolio_snapshots.drawdown_pct` |
| Open positions table | `positions` |
| **Pending orders** — placed but not yet filled, with age | `pending_orders` |
| Recent trades table | `trades`, latest 5 |
| Alerts & notifications | `alert_log` |
| Circuit breakers, per env + global | `circuit_breaker_events` + config thresholds |

### 2 · Portfolio

Combined totals (value, return, vs B&H, cash) ×4 · per-environment Equity | Crypto panels
with metrics, caption (cash · HWM · symbols) and equity curve · daily P&L bar chart, green
up / red down · full holdings table (symbol, env, side, qty, cost, last, unrealised P&L,
stop, target).

### 3 · Trade Log

**Pending orders section**, pinned above the ledger and visually distinct: orders placed
but not yet filled, with symbol, side, qty, limit/type, age, and lifecycle state from
`pending_orders` (+ `V011__pending_order_lifecycle`). An order sitting unfilled for an
unusual length of time is an operational signal, so age is shown prominently rather than
buried in a column.

Then: summary stats ×4 recomputed from the **filtered** set · live filters (environment,
side, symbol, date range) · fill ledger (time ET, symbol, side, qty, fill price,
commission, slippage, env, broker, type, cycle) · row click drills into `fill_quality`
(decision vs expected vs actual price, slippage fraction, time to fill) · explicit empty
state when no rows match.

> **No per-trade P&L column.** It is not stored, and the handoff explicitly forbids
> synthesising one.

### 4 · Risk Metrics

Live tiles (current drawdown, HWM, daily loss, exposure) with threshold colouring — green
below 5%, amber below 8%, red at or above 8% · cooldown & ramp stepper, rendered **only
when halted** (Halt → Cooldown → 25/50/75/100%) · circuit-breaker status table with rule
tooltips · global & emergency halts list · drawdown history chart · risk decisions table
from `risk_decisions` + `circuit_breaker_events`.

Thresholds displayed from config: drawdown halt equity ≥10% / crypto ≥12%; daily-loss halt
equity ≥2% / crypto ≥3%; global combined drawdown ≥15% or combined daily ≥3%.

### 5 · System Health

| Panel | Source | Note |
|---|---|---|
| Services grid | **derived liveness** | see below |
| Data ingestion | `data_ingestion_log` | last bar + age; red at or beyond the window |
| Last trade cycle | `inference_cycles` | cycle id, completion time, orders placed, inference OK |
| **Event feed** | `alert_log` | see below |

**Two honest substitutions**, both approved:

1. **Services grid is derived, not probed.** There is no heartbeat table. Liveness is
   inferred: database = the query succeeded; trader = age of newest `inference_cycles`;
   collector = age of newest `data_ingestion_log`; brokers = last successful fill plus
   `api_errors` (currently empty). Rendered as **"last seen 4m ago"**, never as a green
   "healthy" light the system cannot actually verify. Claiming health we have not measured
   would be exactly the failure mode the capital-preservation rule exists to prevent.

2. **"Log stream" becomes an Event feed.** `alert_log` stores `level`, `title`, and a
   `message_hash` — there is **no message body**. The panel therefore shows time, level,
   and title (e.g. "🔴 SELL BTCUSDT", "Circuit Breaker Halted — equity"), colour-coded info
   / warning / critical. Container stdout is deliberately not used: reaching it would
   require a Docker socket mount, which is precisely the kind of write-capable access a
   read-only monitoring surface must not have.

### 6 · Training Dashboard — stub

Ships as the design's "coming soon" panel with the intended metric grid greyed out
(Sharpe / Sortino / Calmar, max drawdown, profit factor, win rate, max single loss,
iteration comparison). Built for real by Plan B once its evidence engine lands.

**Stub contract for Plan B.** The page exists at route `/training`, renders from
`templates/pages/training.html`, and expects a future `queries/training.py` returning
per-iteration rows keyed by `(iteration, environment, algo)`. Plan B's Phase 4 task fills
it from `season_results` / `fold_results` / `epoch_snapshots` — **not** from the legacy
`iteration_results` / `training_epochs` tables the retired Streamlit page used.

---

## States

The design's preview-state switcher is demo-only and is **not** built. The states it
previewed are real and are driven by live data.

| State | Trigger | Presentation |
|---|---|---|
| **Normal** | no open breaker, all feeds fresh | banner hidden |
| **Circuit halt** | open `circuit_breaker_events` | red banner; HALTED pill; cooldown & ramp stepper appears; flat-book empty state if liquidated |
| **Degraded feed** | newest bar age ≥ window (equity 26 h, crypto 5 h) | amber banner; that feed red; environment marked PAUSED as a **data gate** — breakers stay ARMED |

Both non-normal states can be active at once; the red halt banner takes precedence, with
the degraded feed still flagged on its own row.

**Empty states are rendered explicitly**, never as blank panels: no positions, no trades
matching filters, no data yet.

---

## Charts

Every chart is an inline SVG on a `viewBox="0 0 400 140"` canvas, generated server-side.

**Pure function contract** (`charts.py`):

```
series (list of (timestamp, value)) + viewbox dims
  → x = index scaled across 0..400
  → y = value scaled across 140..0   (inverted: SVG y grows downward)
  → "x1,y1 x2,y2 …"  polyline points string
  → closed polygon variant for the gradient area fill
  → y-axis tick labels + matching gridline positions from min/max
```

Because it is a pure function, tests assert the exact output string for known input — the
most error-prone part of the chart layer is also the most cheaply verified.

**Three things hardcoded in the design prototype become dynamic:** the buy & hold dashed
baseline (from `benchmark_baselines`), the y-axis labels, and the gridline positions.

**Downsampling is mandatory, not optional.** There are 38 snapshots today; after a year of
daily equity plus 4-hourly crypto there will be thousands, all crushed onto a 400-unit
canvas where they cannot be distinguished. `charts.py` therefore caps any series at **400
plotted points** — one per horizontal unit. Longer series are reduced by splitting them
into that many equal groups and keeping each group's most extreme value, **not** its
average. Averaging would smooth away exactly the drawdown spikes a risk chart exists to
show. The reduction happens inside the pure function, so tests assert it directly.

**Hover** follows the design's own technique — absolutely-positioned divs layered *over*
the SVG (crosshair line, snapped dot, tooltip), driven by a small JSON series embedded in
the page. Roughly 40 lines of JavaScript. Zoom and pan are not in the design and are not
built.

---

## Preferences

Theme (dark/light), accent (gold/blue/purple/green), and density (comfortable/compact)
persist across sessions, per browser and per installed app.

| Aspect | Choice |
|---|---|
| Transport | cookie `swingrl_prefs`, `SameSite=Lax`, `Path=/`, `HttpOnly` |
| Lifetime | `dashboard.prefs_cookie_days` in `swingrl.yaml`, **default 180** |
| Written by | the **server**, via `Set-Cookie` on `POST /prefs` |
| Applied by | FastAPI reads the cookie and sets `data-theme` / `data-accent` / `data-density` on the root element at render time |
| Toggle behaviour | JS flips the attribute instantly for feedback, then fires `POST /prefs` in the background. JS never needs to *read* the cookie — current state is read off the root element — so `HttpOnly` costs nothing |
| Override | `?theme=&accent=` URL params override for one request; the cookie remains the persisted default |

**Why the server sets it.** WebKit's ITP caps cookies written by `document.cookie` at
7 days on iOS regardless of the expiry requested; cookies delivered via the `Set-Cookie`
response header are exempt. Server-set also means the correct theme is present on first
paint, avoiding a FOUC flash.

---

## PWA

Because the application owns its HTML `<head>`, the manifest and service worker are
ordinary files — no iframe injection workaround.

**Manifest:** name "SwingRL Trading Dashboard", short name "SwingRL", `display: standalone`,
`orientation: portrait`, `theme_color` and `background_color` `#0d0f14`, icons at 192, 512,
and 512-maskable using the 📈 gold-square mark.

**Service worker caching policy** — the capital-preservation rule applied to cache:

| Asset class | Strategy |
|---|---|
| Shell: CSS, JS, icons, fonts | cache-first, versioned `swingrl-v1` (bumped whenever static assets change) |
| **Every page and data route** | **network-only — never cached** |
| Offline | an explicit "offline — no live data" screen |

An offline PWA showing yesterday's portfolio value would be worse than showing nothing.

**Mobile layout** comes from the design's own CSS: header extras collapse into a ⚙ menu
below 940 px; the top nav becomes a fixed bottom tab bar below 720 px.

### Browser compatibility — required fallbacks

The design uses two CSS features that are not universally available, and the primary target
device is an iPhone:

| Feature | Used | Requires | Fallback |
|---|---|---|---|
| `color-mix(in oklab, …)` | 6 places, incl. the sticky header background | Safari 16.2+ | A plain `background` declared **first**, so older engines take it and newer ones override |
| `backdrop-filter` (blur) | 2 places | broadly supported, prefixed on older WebKit | Include `-webkit-backdrop-filter`; the layout is unaffected if both are ignored |

Both fallbacks are single extra declarations. Skipping them means the sticky header renders
with no background on an older iPhone — text over content, effectively unreadable.

---

## Deployment and security

New `swingrl-dashboard` service in `docker-compose.yml`, and the stale service in
`docker-compose.prod.yml` corrected in the same change.

| Item | Value |
|---|---|
| Host rule | ``Host(`swingrl.home.lab`)`` |
| Entrypoint | `websecure`; the global `web` → `websecure` redirect already applies |
| Certificate | `certresolver=vault` |
| Networks | `br0` (reaches Traefik and `pg16`) + `swingrl_default` |
| Server | uvicorn, non-root user, `mem_limit: 512m`, `cpus: 0.5` |
| Healthcheck | `GET /healthz` — returns 200 only when the database is reachable |
| Restart | `unless-stopped` |
| Mounts | `config/swingrl.yaml` read-only. No models mount, no Docker socket |
| Auth | none — LAN-only access (D-4) |

**Database role.** `CREATE ROLE swingrl_dashboard LOGIN` + `GRANT SELECT` on exactly the
tables the pages read, and `GRANT USAGE` on the schema. Delivered as
`scripts/sql/dashboard_readonly_role.sql`, run once by the user as a superuser — the
in-container migration runner has no superuser rights, so this is deliberately not a
migration. It is additive and non-destructive.

**A30 deploy isolation holds trivially:** this is a brand-new service. Bringing it up is
`docker compose up -d swingrl-dashboard`, which never rebuilds, restarts, or otherwise
touches the running trader.

---

## Testing

TDD throughout, per CLAUDE.md: RED commit, then GREEN.

| Surface | Approach |
|---|---|
| `charts.py` | pure functions — assert exact SVG point strings, axis labels, gridline positions for known series; include awkward cases (empty series, single point, all-equal values) and **downsampling**: a 5,000-point series must yield ≤400 points and must still contain the series maximum and minimum |
| `derive.py` | pure functions — cash/P&L, exposure cap, env split, staleness at and either side of the window boundary, breaker precedence |
| `queries/` | against the test database using existing `tests/conftest.py` fixtures |
| Routes | FastAPI `TestClient` — all six routes return 200; `/healthz` returns non-200 when the database is unreachable |
| **States** | fixture data driving each of normal / halt / degraded, asserting the banner, pill, and stepper each render or stay absent |
| Preferences | `POST /prefs` sets the cookie; a request carrying the cookie renders the matching `data-theme` on first paint |
| Empty states | flat book, no matching trades, no data yet |

Tests live in `tests/dashboard/`, replacing the retired Streamlit tests. Pure-helper tests
whose logic survives the port are carried over rather than rewritten.

**CI change (required, not optional).** `dashboard/requirements.txt` is one of three
dependency surfaces scanned by `pip-audit --strict` in `ci-homelab.sh` Stage [5/6].
Deleting it without updating that stage silently removes a CVE surface. The optional
dependency group in `pyproject.toml` takes its place, and `ci-homelab.sh` is updated in the
same commit as the deletion.

---

## Retirement of the Streamlit dashboard

Removed: `dashboard/` (app, five pages, Dockerfile, requirements) and
`tests/dashboard/test_pages.py`. The old code remains recoverable in git history.

Nothing is lost operationally — the container was never deployed. The one page with real
value, `5_Iteration_History.py`, reads `iteration_results` / `training_epochs`, which Plan B
replaces with `season_results` / `fold_results` / `epoch_snapshots`; rebuilding it now would
mean rebuilding it twice inside the same milestone. Its analysis logic already lives in
`swingrl.reporting.iteration_report` and is untouched by this work.

---

## Documentation changes

| Document | Change |
|---|---|
| `plans/2026-07-24-monitoring-dashboard-plan.md` | **NEW** — the implementation plan |
| `plans/2026-07-07-training-redesign-plan.md` | Cross-reference block + one task: wire page 6 to the evidence engine, honouring the stub contract above |
| `.planning/V1.1_EXECUTION_PLAN.md` | New stage entry; refreshed ▶ RESUME HERE |
| `.planning/ROADMAP.md` | Phase 25 `dashboard-updates` scoped and pointed at this spec |
| `CLAUDE.md` | Key Paths: `dashboard/` → `src/swingrl/dashboard/` |
| `docs/execution/` | Deploy runbook: Traefik labels, read-only role SQL, PWA install verification |

---

## Success criteria

| # | Criterion | How it is verified |
|---|---|---|
| S-1 | All six routes render 200 with live production data | homelab smoke test after deploy |
| S-2 | The three states render correctly from fixture data | automated tests |
| S-3 | The PWA installs on the iPhone and launches standalone | manual, once |
| S-4 | No page displays a number without an as-of timestamp | review checklist |
| S-5 | Offline shows the offline screen, never stale values | manual, airplane mode |
| S-6 | Preferences survive a browser restart and an app relaunch | manual |
| S-7 | The dashboard role cannot write — `INSERT` is refused | automated test asserting the permission error |
| S-8 | Full CI passes, `pip-audit` still covers dashboard dependencies | `scripts/ci-homelab.sh` |
| S-9 | The trader container is untouched by the deploy | container id / uptime unchanged before and after |
| S-10 | Pending orders appear on Dashboard and Trade Log with their age | automated test + live check against the 8 existing rows |
| S-11 | A 5,000-point series renders ≤400 points and still shows the true max and min | automated test |
| S-12 | The sticky header has a readable background with `color-mix` unsupported | manual, or a test asserting the fallback declaration precedes it |

---

## Adjacent work discovered — not built here

Investigating the design surfaced a defect in the live trading path. It is recorded here
because this spec is where it was found, but it is **explicitly not dashboard work**.

### Broker fill verification (trader bug fix — separate workstream)

**The defect.** `PositionTracker.get_portfolio_value()` (`position_tracker.py:48-66`) reads
`total_value` from `portfolio_snapshots` — the table the position tracker itself wrote.
`pipeline.py:353` then feeds that number into position sizing. The system's notion of what
the portfolio is worth is therefore **self-referential**: the broker is never asked. A
missed fill, partial fill, unaccounted fee, or corporate action silently drifts the
recorded value from reality, and the drift then compounds into wrongly-sized orders with
nothing in the loop to catch it.

**The fix (user ruling, 2026-07-24).** The **trader** verifies its own entries against the
broker at the point of action: after recording a fill, confirm the broker shows that order
with the expected quantity and price. On disagreement, send a Discord notification.

| Aspect | Ruling |
|---|---|
| Owner | Trader — it already holds the authenticated broker client and is already mid-cycle |
| Scope | **Alpaca / equity only.** Binance.US has no paper trading, so crypto fills are simulated locally and there is no external state to compare against |
| Hook | The pending → filled transition, which `pending_orders` + `V011__pending_order_lifecycle` already model — not a separate polling loop |
| On mismatch | Discord alert. **Open decision:** whether to also correct the stored value to match the broker. Leaving a known-wrong number means position sizing keeps consuming it; auto-correction carries its own risk |
| Why the trader, not the collector | The check queries an external authority about a specific fact, so it is not self-checking in any meaningful sense — and catching it at the moment of action, where the operator can still act, beats discovering it hours later |

**Relationship to this dashboard:** none required. Optionally, if the fix records a
verification outcome per fill, the Trade Log can render a ✓/✗ per row at negligible cost —
noted, not assumed.

## Out of scope

- **Broker fill verification** — see above. A trader bug fix with its own workstream.
- **`alert_log` schema question** — it stores only `message_hash`, no message body, so the
  event feed cannot show detail text. Whether to add a `message` column is a separate
  follow-up (U-4), deliberately excluded here.
- **Page 6 for real** — Plan B owns it.
- **Push notifications from the dashboard.** Worth stating plainly: this dashboard is
  **pull, not push**. It gives the whole picture when opened; it cannot reach you. If a
  breaker trips at 09:15 and the app is opened at 19:00, the dashboard contributed nothing.
  Discord therefore remains the urgent channel and is *complemented*, not replaced. A PWA
  can do real web push, which would close that gap — deferred as its own decision.
- **Remote (off-LAN) access.** By decision D-3 the dashboard is unreachable outside the
  house. If monitoring while travelling matters, that needs a VPN decision — deferred.
- **A "what happened today" narrative panel.** The design shows *where you stand*, never
  *what happened and why*; the Discord digest does narrative. Whether state plus the event
  feed is enough is a product question — deferred.
- **Options monitoring.** `options_chains` / `options_positions` / `options_snapshots` are
  populated and the collector emits "Options decision captured" alerts, but the design has
  no options page. Deferred as its own scoped addition.
- **`shadow_trades`** — table exists, currently empty; no panel until it matters.
- **Writes of any kind** — no order entry, no manual halt, no config editing from the UI.
- **Historical backfill or data repair** — the dashboard reads what exists.
- **Zoom / pan on charts** — not in the design.
- **Public internet exposure** — LAN-only by decision D-3.

---

## Open questions and UNKNOWNs

Marked explicitly rather than guessed, per the project's honest-gap rule.

| # | Unknown | Impact | How to resolve |
|---|---|---|---|
| Q-1 | Exact query shape for current breaker state — `circuit_breaker_events` has only 3 rows and its resume semantics were not read | Risk page breaker matrix | Read `src/swingrl/risk/` during implementation; adjust `derive.py` |
| Q-2 | Whether `positions` rows are deleted or zeroed on close | Open-positions count and the flat-book empty state | Query the table during implementation |
| Q-3 | The design loads its two fonts (Space Grotesk, IBM Plex Mono) from Google's servers on every page open. Should we instead keep our own copies in `static/` and serve them from the homelab? | If we don't, the fonts fail whenever the phone has no internet — the installed app would render in a fallback typeface | Recommended: keep our own copies. It is a one-time download of a few files, removes an outside dependency, and makes the offline screen look correct |
| Q-4 | Per-environment `portfolio_snapshots` split reliability — `crypto_value` is known to be NULL and totals have drifted $46.96–$48.22 | Portfolio page env split | Known data quirk already tracked separately; display from per-env rows and show as-of timestamps |
| Q-5 | Whether the 5-minute refresh should be a full page reload or a partial fetch | Minor — affects perceived smoothness only | Decide at implementation; full reload is simpler and acceptable at this cadence |
