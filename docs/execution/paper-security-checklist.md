# Paper-trading security hardening checklist (Task 15)

**Date:** 2026-07-17 · **Branch:** `swingrl/2.R-A-capture-foundation` (HEAD at start `bba1adb`)
**Author:** Task 15 audit · **Status:** executed — 7 steps, 1 in-repo fix committed
(`4faf932`), 3 findings requiring operator action on homelab, 2 items that can only be
operator-attested (server-side key scope, key-rotation date).

This document is the **executed** run of the Task 15 checklist from
`.superpowers/sdd/planA/task-15-brief.md`. Each step below records what was checked, the exact
command/evidence used, and the result. It is the pre-deploy gate referenced by the tracker
("Plan A Task E before Plan A Task 16 AND before any Plan B homelab deploy").

---

## Glossary

| Term | Meaning |
|---|---|
| `~/swingrl` | The homelab **live** checkout — the actual running deployment (collector container today; trader/memory/dashboard not yet deployed). Read-only for this audit. |
| `swingrl-planA` (this repo) | The **branch checkout** used to do the work — `swingrl/2.R-A-capture-foundation`, not yet deployed. |
| Paper vs. live keys | Alpaca issues **separate API key pairs** per account (a "paper" trading account and a real-money "live" account each have their own key/secret). The `alpaca-py` SDK picks which Alpaca base URL to call via a `paper=True/False` flag — it does **not** infer this from the key itself. |
| `br0` | The homelab's shared Docker bridge network that `pg16` (the Postgres 16 container, a separate stack) lives on. Containers must explicitly join `br0` to reach `pg16` by its container name. |
| `env_file` | The Docker Compose directive that loads a `.env` file's variables into a container's environment, without baking secret values into the built image. |
| `structlog` | The structured-logging library SwingRL uses (`log.info("event", key="value")` style, see project `CLAUDE.md`). Distinct from Python's built-in `logging` module, though it can be layered on top of it. |
| `httpx` / `httpcore` | The HTTP client libraries SwingRL uses for outbound calls (Discord webhooks, Binance.US public API, LLM providers). Both ship their **own** internal Python `logging` loggers (named `"httpx"` / `"httpcore"`) that print one line per HTTP request, separate from anything the application code explicitly logs. |
| `exc_info=True` | A `structlog`/`logging` call argument that captures and renders the currently-handled exception's message/traceback into the log output. |
| `MockTransport` | An `httpx` test utility that lets a test simulate an HTTP request/response cycle with no real network call — used here to trigger the leak deterministically. |
| Webhook URL | A Discord "incoming webhook" URL. Discord's design embeds the entire secret credential **in the URL path itself** (`https://discord.com/api/webhooks/<id>/<token>`) — unlike most APIs, which put the secret in a header. This matters because anything that logs "the URL" logs the secret. |
| `pg16` | The homelab's shared PostgreSQL 16 container — a separate stack from this repo, referenced read-only here only to check its network exposure. |
| Operator-attested / attest-at-Task-16 | An item this audit could not verify from the repo or from read-only homelab inspection (e.g., a value stored only in a third-party provider's dashboard). Recorded here as a specific, closeable question for whoever runs Task 16 (go-live). |

---

## Checklist — results at a glance

| # | Step | Result | Evidence |
|---|------|--------|----------|
| 1 | Key scoping (Alpaca paper vs. live; Binance.US read-only) | **PARTIAL — verifiable parts pass; server-side scope is operator-attest** | See §1 |
| 2 | Key rotation (2026-03-24 leak → rotated) | **CANNOT VERIFY FROM REPO — operator-attest; tracker discrepancy found** | See §2 |
| 3 | Secrets never in image/repo/logs | **FAIL → FIXED** (1 real, live, active leak found and fixed) | See §3 |
| 4 | Network surface | **PASS with 1 additional finding** (dashboard has no auth, LAN-exposed) | See §4 |
| 5 | Container posture | **PASS — clean** | See §5 |
| 6 | Backup hygiene | **FAIL — homelab file permissions** (operator action required) | See §6 |
| 7 | Commit | Done — see §7 | — |

---

## §1 — Key scoping

**Alpaca.** `~/swingrl/.env` var names only (values never inspected):

```
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
```

Verified by reading `src/swingrl/execution/adapters/alpaca_adapter.py:63-76`: the adapter reads
these two generic var names and picks the Alpaca **paper** vs **live** endpoint via
`paper = config.trading_mode == "paper"`, passed to `TradingClient(..., paper=paper)`. Confirmed
`/home/varun/swingrl/config/swingrl.yaml:6` → `trading_mode: paper`, and confirmed no
`SWINGRL_TRADING_MODE` override exists in `~/swingrl/.env` (0 matches).

- **What this proves:** the *code path* will call Alpaca's paper endpoint.
- **What this does NOT prove:** whether the actual key *value* in `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`
  was generated from Alpaca's paper account or the live account. The var names are generic
  (no `_PAPER`/`_LIVE` suffix), so this is not verifiable from the repo or from the key names
  alone — Alpaca ties key pairs to the account they were generated under, and that binding is
  only checkable by looking at the key in the Alpaca dashboard (or by a live paper-mode order
  call succeeding/failing, which is exactly Task 16's job).
- **→ Operator-attest at Task 16:** confirm in the Alpaca dashboard that the key pair currently
  in `~/swingrl/.env` was generated under the **paper** account, not live.

**Binance.US.** Var names present (`BINANCE_API_KEY=`, `BINANCE_SECRET_KEY=`), but a full-repo
grep (`grep -rn "BINANCE_API_KEY\|BINANCE_SECRET_KEY" --include="*.py"`) found **zero** usages
in `src/` — these two vars are currently **unused by any running code**:
- `src/swingrl/data/binance.py` (market-data fetch) calls Binance.US's **public**,
  unauthenticated klines endpoint — no key attached.
- `src/swingrl/execution/adapters/binance_sim.py` (the crypto paper-fill simulator) hits
  Binance.US's **public** order-book endpoint via plain `requests`, no auth header, and never
  places a real order (confirmed by module docstring + code read: "No actual orders are
  placed").
- **What this proves:** even if the Binance.US key had trading scope, nothing in this
  codebase currently exercises it — the crypto env is fully simulated against public market
  data.
- **What this does NOT prove:** the actual permission scope configured for this key on
  Binance.US's side (withdrawals off, IP allowlist, reading-only) — server-side, unverifiable
  here.
- **→ Operator-attest at Task 16:** confirm in the Binance.US dashboard that withdrawals are
  OFF and the key is scoped to read-only (per `scripts/security_checklist.py`'s documented
  90-day rotation schedule, which already specifies this as the target scope).

## §2 — Key rotation (2026-03-24 leak)

Checked: git history (`git log --all -S"<pattern>"` and date-range log), and every memory file
mentioning rotation.

- `.planning/V1.1_EXECUTION_PLAN.md` — the project's **canonical tracker** — still lists this
  under "Open decisions & risks" (line ~406): *"API keys shared in chat 2026-03-24 must still
  be rotated (per project memory)."* No changelog entry recording completion exists in that
  file.
- The Claude memory `MEMORY.md` main file asserts, in a 2026-07-12 session-state bullet,
  "✅ Key rotation DONE" — but with no date, no evidence pointer, and no cross-reference to a
  commit or runbook execution log.
- `project_llm_providers.md` (memory) only records the original 2026-03-24 obligation, not a
  completion.
- **By design**, `.env` is gitignored, so there is **no git-trackable evidence possible** for
  when a key value changed — this isn't a gap in this audit's effort, it's structural (secrets
  correctly never touch git history).
- Circumstantial-only signal: `~/swingrl/.env`'s file mtime is 2026-07-16 (after the claimed
  2026-07-07 rotation date), but the file has been edited many times since for unrelated
  reasons (e.g., adding `SWINGRL_ALERTING__*` webhook vars during the 2026-07-15 Discord work)
  — this does not specifically confirm MISTRAL/GEMINI/OPENROUTER keys were touched, so it is
  **not** treated as evidence of rotation.
- **Finding (documentation discrepancy, not a security hole):** the canonical tracker and the
  memory file disagree on whether this is done. This should be reconciled by whoever can
  actually confirm the provider-dashboard rotation dates — not guessed here.
- **→ Operator-attest at Task 16:** confirm the current MISTRAL_API_KEY / GEMINI_API_KEY /
  OPENROUTER_API_KEY values were generated **after** 2026-03-24 (check each provider's
  dashboard key-creation timestamp), and correct the tracker's stale "Open decisions & risks"
  bullet once confirmed either way.

## §3 — Secrets never in image/repo/logs

**Gitignore / dockerignore** — both PASS, verified by direct read:
- `.gitignore:36` → `.env` excluded (`.env.example` explicitly kept, by design).
- `.dockerignore:26-27` → `.env` and `*.env` both excluded.

**git log -S spot-checks** (pattern search only, no content beyond commit refs/context
inspected) for `sk-`, `AKIA`, `discord.com/api/webhooks`, `xoxb-`, `ghp_`:
- `sk-` hits are all false positives — substring of the word "risk-" (e.g. "risk-adjusted") in
  docstrings/docs, confirmed by reading the actual diff context.
- `AKIA`, `xoxb-`, `ghp_` — zero hits.
- `discord.com/api/webhooks` hits are all test fixtures (`.../test/token`), format-validation
  regex documentation, or `.env.example`-style `...` placeholders — never a real token.
- **PASS — no leaked secret values found in git history.**

**structlog kwarg grep** in `src/swingrl/execution/` and `services/memory/` for any
key/secret/token/webhook-bearing kwarg:
- 4 hits total, all safe: `log.error("alpaca_credentials_missing")` (no kwargs at all),
  `log.warning("auth_rejected", has_key=bool(key))` (boolean, not the key),
  `log.debug("epoch_cloud_skipped_no_key", provider=cfg.get("provider"))` (provider name only),
  `log.warning("consolidation_missing_patterns_key")` (no kwargs).
- **PASS.**

**Alerter webhook URLs env-only** — confirmed by reading `scripts/main.py:255-262`:
`Alerter(webhook_url=config.alerting.alerts_webhook_url, ...)`, sourced from
`SwingRLConfig` (Pydantic schema, `alerts_webhook_url: str = Field(default="")` — no hardcoded
default). Config is populated via `load_config()` + env var overrides
(`SWINGRL_ALERTING__ALERTS_WEBHOOK_URL` / `__DAILY_WEBHOOK_URL`), never a CLI argument (which
would be visible in `ps aux`), never hardcoded. **PASS.**

### FINDING (critical, live, active) — `httpx`/`httpcore`'s own loggers leaked the webhook URL

This was **not** in the brief's literal bullet list, but was found while checking the
"structlog calls ... for any kwarg carrying a key/webhook" item, and is squarely in scope.

`httpx` prints `"HTTP Request: {method} {url} ...\"{status}\""` at **INFO** level on **every**
call — success or failure — via its own internal `logging.getLogger("httpx")`, entirely
separate from any `structlog` call our code makes. `httpcore` does the same at **DEBUG**.
Because a Discord webhook URL embeds its secret token directly in the path (see Glossary), this
line contains the full secret.

`configure_logging()` (`src/swingrl/utils/logging.py`) wires `root_logger.handlers = [...]` and
`root_logger.setLevel(log_level)` — this causes **every** stdlib logger that doesn't set its own
level, including `httpx`'s and `httpcore`'s, to propagate through to whichever renderer
(console or JSON) is configured.

**Confirmed empirically** (both isolated Python repro and the real render pipeline):
- `json_logs=False` (dev console): leaks the full URL, **plus full local variables** of the
  stack frame (via `rich`'s traceback rendering) — worse than just the URL.
- `json_logs=True` (the documented "production/Docker" mode): also leaks the full URL as plain
  text inside the JSON `"event"` field (a *different* code path — the `exc_info=True` variant
  of this bug does NOT leak in JSON mode, since nothing in the processor chain expands
  `exc_info` into text there; but `httpx`'s own per-request INFO line is a plain string message,
  independent of `exc_info`, and **does** leak in both modes).

**Confirmed live, not just theoretical:** `~/swingrl/config/swingrl.yaml:40` currently has
`json_logs: false` (comment says "true in production/Docker" but the value itself is `false`,
and no `SWINGRL_LOGGING__JSON_LOGS` override exists in `.env`). The **currently-running**
`swingrl-collector` container (the only swingrl container live on homelab right now — no trader
container is deployed yet) uses this same config and the same `Alerter`. A redacted,
count-only check of its live logs:

```
docker logs swingrl-collector 2>&1 | grep -c "discord.com/api/webhooks"
→ 3
```

confirmed 3 real historical occurrences (all `HTTP/1.1 204 No Content` — successful sends). The
matched lines were never printed in full to this session or the report — only a
token-redacted structure check was performed (path segment after `webhooks/` replaced with
`[REDACTED]` before being surfaced), confirming these are `httpx`'s own INFO request lines
(`logger: httpx`), not an application-level `structlog` call. **The container's own historical
logs contain the live secret token multiple times.**

Checked whether this also affects `services/memory/`: no. That service's `app.py` configures
`structlog.configure(..., logger_factory=structlog.PrintLoggerFactory())` — a structlog-native
setup that never wires Python's stdlib root logger, confirmed empirically (an isolated repro
with the exact same config produces zero output for an httpx request). Its LLM-provider calls
(Mistral/Gemini/OpenRouter/Cerebras/Groq) all use `Authorization: Bearer <key>` headers, never a
URL-embedded key (verified by reading `services/memory/memory_agents/query.py:1757` and
surrounding provider-config blocks) — so even if it were vulnerable to this exact mechanism, no
LLM key would be exposed by it. Only Discord webhook tokens are at risk, because only Discord's
webhook design embeds the secret in the URL.

**FIXED in-repo** (commit `4faf932`, TDD): added

```python
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
```

to `configure_logging()`. Regression test in `tests/utils/test_utils.py` uses
`httpx.MockTransport` to trigger the leak hermetically (no real network call), confirmed **RED**
before the fix (assertion failed, the fixture secret appeared verbatim in the rendered log
file) and **GREEN** after. Full `tests/utils/`, `tests/monitoring/test_alerter.py` suites
(43 passed, 2 skipped — DB-dependent, no `DATABASE_URL` in this sandbox) plus `ruff check`,
`ruff format --check`, and `mypy` all pass clean on the changed files.

**→ Operator action (separate from the code fix, already committed):** once this fix is
deployed, the *historical* 3 lines already inside `swingrl-collector`'s existing log storage
still contain the token. If Docker's log driver is `json-file` (the default) those lines
persist on disk under `/var/lib/docker/containers/.../…json.log` until that container is
recreated or its logs are pruned — worth a rotation of the Discord webhook token itself as a
belt-and-suspenders measure, since a secret that has already been written to a log file isn't
un-leaked by fixing the code that will stop leaking *future* ones. Recorded as a Task 16
decision point, not fixed here (out of scope — rotating a live webhook is an operator action).

## §4 — Network surface

- **`swingrl-memory` port 8889:** `docker-compose.prod.yml` publishes it as
  `"127.0.0.1:8889:8889"` — bound to the host's loopback interface only, **not** `0.0.0.0`.
  This is not the literal "docker-network-only, no host publish" wording in the brief, but is
  security-equivalent (arguably stronger): it is unreachable from the LAN/`br0` either way, and
  other containers on the same compose project can still reach it by container name over
  Docker's internal DNS regardless of any host port publish (that's how `swingrl-trader`'s soft
  dependency on it works). **PASS**, with this wording clarification.
- **Dashboard read-only mounts:** confirmed in `docker-compose.prod.yml`: `./db:/app/db:ro` and
  `./models:/app/models:ro`, both explicitly `:ro`. **PASS.** (The currently-deployed
  `docker-compose.yml` is even more restrictive — it doesn't mount `./db` into the dashboard at
  all.)
- **`pg16` exposure:** confirmed via read-only `docker inspect pg16 --format
  '{{json .NetworkSettings.Networks}}'` / `.Ports` — attached **only** to `br0`, and port 5432
  has a `null` host-publish mapping (exposed to the image, not published to any host port).
  Not reachable from outside `br0`. **PASS.**
  - **Observation (deploy-readiness, not a vulnerability):** `docker-compose.prod.yml` declares
    no `networks: br0` for any service (unlike the currently-deployed `docker-compose.yml`,
    which does join `br0` for exactly this reason — to reach `pg16`). If the trader/memory
    services' `DATABASE_URL` is meant to resolve `pg16` by container name, this will need a
    `networks: [default, br0]` addition before Task 16 deploy, or the containers won't be able
    to resolve the hostname. Not fixed here — didn't want to guess at the intended network
    topology without confirming what `DATABASE_URL` actually targets (a value, not inspected
    here).

### FINDING (additional, not in the brief's literal list) — dashboard has no authentication and is LAN-exposed

`swingrl-dashboard` publishes `"8501:8501"` in **both** `docker-compose.yml` and
`docker-compose.prod.yml` — bound to `0.0.0.0`, i.e., reachable from any device on the home LAN,
not just the host machine (unlike `swingrl-memory`'s deliberate loopback-only binding above). A
repo-wide search (`grep -rln "auth\|password\|login" dashboard/`) found **no** authentication
mechanism anywhere in the dashboard app. This may well be intentional (a human wants to check
the dashboard from a phone/laptop elsewhere on the LAN) — recorded as a finding for a decision,
not silently changed, since narrowing it to loopback-only would break that use case if it's
wanted. **→ User decision needed:** keep LAN-wide access as-is, or restrict to `127.0.0.1` /
add basic auth in front of it (e.g. a reverse proxy).

## §5 — Container posture

- **Non-root user:** `Dockerfile` has `USER trader` as the last `USER` directive in **both**
  build stages — `ci` (line 45) and `production` (line 89). Both `swingrl-trader` and
  `swingrl-trainer` compose services use `target: production`, so both run non-root (Task E
  note explicitly asked to check both — confirmed identical). Cross-checked with the repo's own
  `scripts/security_checklist.py` automated check (run read-only against `~/swingrl`): `[+]
  PASS: non-root-user`.
- **No `privileged`:** `grep -rn "privileged" docker-compose*.yml` — zero matches across all
  three compose files.
- **No capability/security overrides:** `grep -n "cap_add\|security_opt\|network_mode\|user:"
  docker-compose*.yml` — zero matches.
- **Restart policies:** `unless-stopped` on every always-on service (ollama, memory, trader,
  collector, dashboard). `swingrl-trainer` is deliberately `restart: "no"` — its default `CMD`
  is `sleep infinity` (only invoked via `docker compose run` for an actual training job), so
  auto-restarting it after a host reboot would just waste resources sitting idle; this is the
  documented, intentional A30 design, not an oversight.
- **PASS — clean across every check.**

## §6 — Backup hygiene

- **Retention:** confirmed in the live crontab (`crontab -l`):
  `find "$HOME/swingrl/backups" -name "swingrl-*.dump" -mtime +14 -delete` — matches the
  2026-07-16 user ruling (14-day retention). **PASS.**
- **`backup/offsite_sync.py` status:** exists (`src/swingrl/backup/offsite_sync.py`), and is
  **not** dead code — it's wired into the scheduler (`src/swingrl/scheduler/jobs.py`'s
  `monthly_offsite_job`). It performs `rsync -avz` over Tailscale to a NAS, gated entirely by
  `config.backup.offsite_host` being non-empty (skips with `log.info("offsite_rsync_skipped",
  ...)` and returns success if not configured). Confirmed `/home/varun/swingrl/config/
  swingrl.yaml`'s `backup:` section (lines 74-77) has no `offsite_host` key at all — this
  scheduled job is currently a **no-op every time it fires**, consistent with (not
  contradicting) the actual current mechanism being host-cron `pg_dump` + user-run Duplicati,
  not B2/rclone, not this function. **Not stale/broken — just presently inactive by config,**
  and its transport (Tailscale) would encrypt data in transit if ever enabled; at-rest
  encryption on the destination NAS is outside this function's control. No fix needed.

### FINDING — `~/swingrl/.env` is group- and world-readable (homelab, operator action required)

```
stat -c "%a %U:%G %n" /home/varun/swingrl/.env
→ 664 varun:varun /home/varun/swingrl/.env
```

Mode `664` = owner read/write, **group read/write**, **world read**. The repo's own
`scripts/security_checklist.py` (run read-only against `~/swingrl`) independently confirms this:
`[-] FAIL: env-permissions — .env permissions are 0o664, expected 0o600`. The file's group
(`varun`, gid 1000) has a **second member**: `hermes-ops` (uid 1001), an unrelated local service
account for a different agent running on this homelab host (`getent group varun` →
`varun:x:1000:hermes-ops`). At mode 664, that account currently has **read AND write** access to
every secret in this file (Alpaca, Binance.US, Discord webhook, all LLM provider keys), and
**any** other local user on the box can read it (world `r`).

**Not fixed here** — a homelab file-permission change, out of scope per the brief (do not touch
`~/swingrl`). **→ Operator action:**

```bash
chmod 600 ~/swingrl/.env
```

### FINDING — `~/swingrl/backups/*.dump` are world-readable (homelab, operator action required)

```
stat -c "%a %U:%G %n" /home/varun/swingrl/backups
→ 775 varun:varun /home/varun/swingrl/backups
ls -la /home/varun/swingrl/backups/
→ -rw-rw-r-- ... swingrl-2026-07-16.dump  (254 MB)
→ -rw-rw-r-- ... swingrl-2026-07-17.dump  (269 MB)
```

The backup directory is `775` (world-executable/listable) and the ~250MB `pg_dump` custom-format
dump files themselves are `664` (world-readable) — full production database contents (trade
records, positions, whatever the memory agent's tables hold) readable by any local account,
including `hermes-ops`. This happens because the cron job (`docker exec pg16 pg_dump ... >
"$HOME/swingrl/backups/swingrl-$(date +%F).dump"`) creates the file via shell redirection at
the process's default umask, with no explicit `chmod` afterward.

**Not fixed here** — homelab file/cron change, out of scope. **→ Operator action:**

```bash
chmod 700 ~/swingrl/backups
chmod 600 ~/swingrl/backups/*.dump
```

and consider tightening the cron entry itself (e.g. `umask 077` before the `pg_dump` line, or an
explicit `chmod 600` appended after it) so future dumps land safe by default instead of
depending on someone remembering to re-run this checklist.

**Duplicati** (whole-`~/swingrl` backup to a local drive) is **operator-attested only** — an
external application's state, not something this audit can introspect appropriately from a
read-only repo/homelab check.

## §7 — Commit

- `4faf932` — `fix(2.R-A): suppress httpx/httpcore INFO-DEBUG loggers to stop webhook URL leak`
  (the §3 fix + regression test).
- This document itself, committed separately per the brief's Step 7 instruction.

---

## Findings summary

| # | Finding | Severity | Disposition |
|---|---|---|---|
| 1 | `httpx`/`httpcore` internal loggers leak Discord webhook URL (incl. secret token) into application logs on every call, both console and JSON render modes; confirmed 3 real historical occurrences in the live `swingrl-collector` container | **Critical, live** | **Fixed in-repo**, commit `4faf932` |
| 2 | `~/swingrl/.env` mode `664` — group+world readable; a second, unrelated local service account (`hermes-ops`) has read+write access | High | Operator action: `chmod 600 ~/swingrl/.env` |
| 3 | `~/swingrl/backups/*.dump` mode `664`, dir mode `775` — full DB dumps world-readable | High | Operator action: `chmod 700`/`600` (commands above); consider cron umask fix |
| 4 | Dashboard (`8501`) published on `0.0.0.0`, no authentication anywhere in the app | Medium | User decision — confirm intended (LAN access wanted) or restrict |
| 5 | Historical leaked log lines (from Finding 1) may still exist in Docker's on-disk log storage for `swingrl-collector` even after the code fix | Medium | Operator decision at Task 16 — consider rotating the Discord webhook token as belt-and-suspenders |
| 6 | `docker-compose.prod.yml` has no `networks: br0` on any service, unlike the currently-deployed `docker-compose.yml` | Low (deploy-readiness, not security) | Verify at Task 16 whether the trader/memory `DATABASE_URL` needs `pg16` reachability; add `br0` if so |
| 7 | Tracker (`.planning/V1.1_EXECUTION_PLAN.md`) still lists the 2026-03-24 key rotation as an open risk; `MEMORY.md` asserts it's done — no dated evidence for either | Low (documentation) | Reconcile once rotation dates are confirmed against provider dashboards |

## Operator-attestation items for Task 16

1. Confirm `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` in `~/swingrl/.env` were generated under
   Alpaca's **paper** account, not live (server-side, unverifiable from the repo).
2. Confirm the Binance.US key's permission scope (withdrawals OFF, read-only / IP allowlist)
   in the Binance.US dashboard.
3. Confirm `MISTRAL_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` were actually rotated
   after 2026-03-24 (check each provider dashboard's key-creation date) — and reconcile the
   tracker/memory discrepancy either way.
4. Decide whether to rotate the Discord webhook token given it has already appeared in
   `swingrl-collector`'s historical container logs (Finding 5).
5. Duplicati's whole-`~/swingrl` backup — operator-attested, not independently checkable here.

## Honest gaps (could not verify)

- **Server-side key scope** (Alpaca paper/live binding, Binance.US permission flags) — these
  live entirely in the providers' own dashboards; nothing in this repo or on this host records
  them in a checkable form. This is a structural limitation, not a shortcut taken.
- **Exact key-rotation date** — `.env` is (correctly) gitignored, so there is no possible
  git-trackable evidence for when a secret value changed; the only real evidence would be each
  provider's own key-creation timestamp.
- **Duplicati state** — an external, whole-checkout backup tool; this audit's scope was the
  repo and read-only homelab inspection, not that tool's own configuration/logs.
