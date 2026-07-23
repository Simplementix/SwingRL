# Auction-Fill Status Normalization + Valid-Partial Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the enum-stringification bug that makes the 09:35 fill-confirmation job classify
every fully-filled auction order as "partially filled" (and leave its `pending_orders` row open
forever), and re-work the partial-fill alert so every VALID partial notifies (per-symbol, no
suppression) — per the user ruling of 2026-07-23.

**Architecture:** One-line status normalization in `_confirm_one_pending_order` (unwrap the
alpaca-py `OrderStatus` enum's `.value` before lowercase-comparing), plus an alert-policy change
in the still-live branch (symbol in the title, `bypass_suppression=True`, notional-aware
"requested" text). Regression tests use the REAL `alpaca.trading.enums.OrderStatus` enum —
the existing tests mock plain strings, which is exactly why this bug shipped (violation of the
"exact production payloads" house testing rule).

**Tech Stack:** Python 3.11, alpaca-py (`OrderStatus` enum), pytest 8 (fast lane `-n auto`,
full lane `-n 4` per docs/testing/best-practices.md).

## Evidence base (verified live 2026-07-23 ~10:15 ET)

- Alpaca (paper) shows all 8 of cycle 32's orders `OrderStatus.FILLED` at 09:30:00–01 ET,
  filled to the penny of requested notional; zero remainder working.
- `str(OrderStatus.FILLED)` == `'OrderStatus.FILLED'` → `.lower()` == `'orderstatus.filled'`
  (proven in the trader container) → `jobs.py:814` comparison `status == "filled"` never
  matches → all 8 fell into the still-live branch: false "PARTIALLY filled" alert, rows left
  open, and every future daily run would nag "unfilled" forever.
- `grep` confirms `src/swingrl/scheduler/jobs.py:814` is the ONLY raw-broker-object status
  comparison in the codebase (the 09:15 submit path uses the adapter's own normalized strings).
- The 8 currently-open rows self-heal after this fix deploys: first fixed run takes the
  `status == "filled"`, `delta_qty <= 0`, `cum_qty > 0` quiet-stamp path (books already match
  broker) and closes them silently. No data repair needed.

## Global Constraints

- **Branch:** `swingrl/2.R-G-status-fix` off `origin/swingrl/2.R-training-redesign`; PR
  targets the integration branch, never main.
- **This branch touches production `src/`** (`src/swingrl/scheduler/jobs.py`) — full suite
  (`-n 4`, ≤20 min) must be 0-failures before push; homelab CI literal `=== CI PASSED ===`
  before PR; **deploy is user-gated** (separate approval; not part of this plan).
- Python 3.11; `from __future__ import annotations`; type hints on all defs; absolute imports
  in src/; 100-char lines; structlog kwargs only; pre-commit always (never `--no-verify`).
- TDD: RED commit before GREEN. Regression tests MUST use the real
  `alpaca.trading.enums.OrderStatus` enum (production payload shape), not plain strings.
- Lockfile ruff is the CI authority: run `uv run ruff check src/ tests/` before push
  (pre-commit's pinned ruff is more lenient — 2026-07-23 CI-red lesson).
- **User ruling (2026-07-23, binding):** (a) partial-fill notifications fire ONLY for valid
  partials (broker genuinely reports a live/partially-filled order); (b) every valid partial
  notifies — per-symbol, never suppressed as a duplicate.
- Scope guard: do NOT change the "unfilled" (no-new-shares) branch's suppression behavior, the
  terminal branches, or any other alert — they are outside the ruling.
- CI/homelab: no container recreation or CI runs spanning 15:30–16:45 ET on trading days.

### Task boundary map

| File | Task 1 | Task 2 |
|---|---|---|
| `src/swingrl/scheduler/jobs.py:814` (status derivation) | fix | — |
| `src/swingrl/scheduler/jobs.py:988-1000` (partial alert) | — | fix |
| `tests/scheduler/test_jobs.py` | enum regression tests | alert-policy tests |

---

### Task 1: Normalize the broker status enum (the classification bug)

**Files:**
- Modify: `src/swingrl/scheduler/jobs.py:814` (one statement becomes two)
- Test: `tests/scheduler/test_jobs.py` (new tests in the existing fill-confirmation class,
  which uses the `_MockCtx` fixture and `mock_ctx.alpaca.order_status(...)` helper that stores
  a `SimpleNamespace` per order id — see existing `test_fill_confirmation_records_auction_fill`
  at ~line 1044 for the established arrange/act/assert shape)

**Interfaces:**
- Consumes: existing `_MockCtx` / `order_status()` test helpers; `alpaca.trading.enums.OrderStatus`.
- Produces: `status` local in `_confirm_one_pending_order` now normalized for BOTH plain
  strings and alpaca-py enums (Task 2's tests rely on enum payloads working).

- [ ] **Step 1: Write the failing regression tests**

Add to `tests/scheduler/test_jobs.py`, inside the same class as
`test_fill_confirmation_records_auction_fill`, directly after it (match the file's existing
import style — add `from alpaca.trading.enums import OrderStatus` to the module imports):

```python
    def test_fill_confirmation_enum_status_filled_closes_row(self, mock_ctx: _MockCtx) -> None:
        """Ruling 2026-07-23: a REAL alpaca-py OrderStatus.FILLED enum must close the row.

        Incident 2026-07-23: str(OrderStatus.FILLED).lower() == 'orderstatus.filled' never
        matched 'filled', so all 8 fully-filled auction orders were misclassified as
        still-live: false PARTIALLY-filled alerts, rows open forever, daily nag alerts.
        Production payloads carry the enum, not a string (proper-testing house rule).
        """
        _insert_pending(mock_ctx, "oenum1", symbol="SPY", side="buy")
        mock_ctx.alpaca.order_status(
            "oenum1",
            status=OrderStatus.FILLED,
            filled_avg_price=739.25,
            filled_qty=0.068434223,
        )
        jobs.equity_fill_confirmation_job()
        row = _pending_row(mock_ctx, "oenum1")
        assert row["resolved_at"] is not None, "enum FILLED must stamp resolved_at"
        assert row["disposition"] == "filled"
        titles = [c.kwargs.get("title", c.args[1] if len(c.args) > 1 else "")
                  for c in mock_ctx.alerter.send_alert.call_args_list]
        assert not any("PARTIALLY" in t for t in titles), titles

    def test_fill_confirmation_enum_status_partial_stays_open(self, mock_ctx: _MockCtx) -> None:
        """A REAL OrderStatus.PARTIALLY_FILLED enum takes the valid-partial path (row open)."""
        _insert_pending(mock_ctx, "oenum2", symbol="QQQ", side="buy")
        mock_ctx.alpaca.order_status(
            "oenum2",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=694.63,
            filled_qty=0.05,
            qty=0.0973036,
        )
        jobs.equity_fill_confirmation_job()
        row = _pending_row(mock_ctx, "oenum2")
        assert row["resolved_at"] is None, "genuine partial must stay open"

    def test_fill_confirmation_enum_status_canceled_closes_terminal(
        self, mock_ctx: _MockCtx
    ) -> None:
        """A REAL OrderStatus.CANCELED enum takes the terminal-dead path."""
        _insert_pending(mock_ctx, "oenum3", symbol="VTI", side="buy")
        mock_ctx.alpaca.order_status("oenum3", status=OrderStatus.CANCELED)
        jobs.equity_fill_confirmation_job()
        row = _pending_row(mock_ctx, "oenum3")
        assert row["resolved_at"] is not None
        assert row["disposition"] == "canceled"
```

NOTE for the implementer: `_insert_pending` / `_pending_row` above are placeholders for
whatever helper the existing tests in that class use to seed and re-read a `pending_orders`
row through `mock_ctx` (read `test_fill_confirmation_records_auction_fill` and
`test_fill_confirmation_closes_canceled_order` and copy their exact arrange/assert helpers —
do NOT invent a new seeding path). The three test BODIES' assertions and enum payloads are the
requirement; the seeding/read helpers must match the file's existing pattern. If the alerter
mock exposes titles differently (e.g. positional), adapt the title extraction to the file's
existing assertion style.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `env -u DATABASE_URL uv run pytest tests/scheduler/test_jobs.py -k "enum_status" -q`
Expected: 3 failures — FILLED: `resolved_at is None` (row wrongly left open) and a
"PARTIALLY" title present; PARTIALLY_FILLED: passes or fails depending on helper shape (if it
passes RED, that is acceptable — it is the pinned-behavior guard, note it in the report);
CANCELED: `resolved_at is None`.

- [ ] **Step 3: Commit RED**

```bash
git add tests/scheduler/test_jobs.py
git commit -m "test(scheduler): RED — broker OrderStatus enums must classify correctly"
```

- [ ] **Step 4: Implement the normalization**

In `src/swingrl/scheduler/jobs.py`, replace line 814:

```python
    status = str(getattr(order, "status", "") or "").lower()
```

with:

```python
    raw_status = getattr(order, "status", "")
    # alpaca-py returns an OrderStatus ENUM whose str() is 'OrderStatus.FILLED' — compare on
    # .value ('filled'). Plain strings (tests, other brokers) pass through unchanged.
    # Incident 2026-07-23: without this, every filled auction order was misclassified live.
    status = str(getattr(raw_status, "value", raw_status) or "").lower()
```

- [ ] **Step 5: Run the new tests + the whole fill-confirmation class to verify GREEN**

Run: `env -u DATABASE_URL uv run pytest tests/scheduler/test_jobs.py -q`
Expected: all pass, 0 failed (the existing string-status tests must still pass — strings have
no `.value` attribute and pass through `getattr(raw, "value", raw)` unchanged).

- [ ] **Step 6: Commit GREEN**

```bash
git add src/swingrl/scheduler/jobs.py
git commit -m "fix(scheduler): unwrap OrderStatus enum before status comparison — filled auction orders were misclassified as live"
```

---

### Task 2: Valid-partial alerts — per-symbol, never suppressed, notional-aware

**Files:**
- Modify: `src/swingrl/scheduler/jobs.py:988-1000` (the still-live `slice_recorded` alert)
- Test: `tests/scheduler/test_jobs.py` (extend the enum-partial test from Task 1 + one new test)

**Interfaces:**
- Consumes: Task 1's normalized `status`; `ctx.alerter.send_alert(..., bypass_suppression=...)`
  (existing signature, `src/swingrl/monitoring/alerter.py:143-150`).
- Produces: nothing consumed later; alert contract per the user ruling.

- [ ] **Step 1: Write the failing tests**

Add to the same test class:

```python
    def test_partial_fill_alert_is_per_symbol_and_unsuppressed(self, mock_ctx: _MockCtx) -> None:
        """Ruling 2026-07-23: every VALID partial notifies — symbol in title, no suppression.

        Two same-morning partials on different symbols are distinct events, not duplicates:
        the old shared title + consecutive-gate + cooldown delivered only 1 of 8.
        """
        _insert_pending(mock_ctx, "op1", symbol="SPY", side="buy")
        _insert_pending(mock_ctx, "op2", symbol="QQQ", side="buy")
        mock_ctx.alpaca.order_status(
            "op1", status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=739.25, filled_qty=0.03, qty=0.068434223,
        )
        mock_ctx.alpaca.order_status(
            "op2", status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=694.63, filled_qty=0.05, qty=0.0973036,
        )
        jobs.equity_fill_confirmation_job()
        partial_calls = [
            c for c in mock_ctx.alerter.send_alert.call_args_list
            if "PARTIALLY filled" in (c.kwargs.get("title") or "")
        ]
        assert len(partial_calls) == 2, partial_calls
        titles = sorted(c.kwargs["title"] for c in partial_calls)
        assert any("QQQ" in t for t in titles), titles
        assert any("SPY" in t for t in titles), titles
        assert all(c.kwargs.get("bypass_suppression") is True for c in partial_calls)

    def test_partial_fill_alert_notional_order_text(self, mock_ctx: _MockCtx) -> None:
        """Notional orders (qty=None) show '$X notional' instead of 'None requested'."""
        _insert_pending(mock_ctx, "op3", symbol="VTI", side="buy")
        mock_ctx.alpaca.order_status(
            "op3", status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=365.13, filled_qty=0.05, qty=None, notional=61.10,
        )
        jobs.equity_fill_confirmation_job()
        partial_calls = [
            c for c in mock_ctx.alerter.send_alert.call_args_list
            if "PARTIALLY filled" in (c.kwargs.get("title") or "")
        ]
        assert len(partial_calls) == 1
        msg = partial_calls[0].kwargs["message"]
        assert "None requested" not in msg, msg
        assert "$61.1 notional" in msg, msg
```

(Same NOTE as Task 1 about matching the file's existing seeding/assertion helpers. If
`send_alert` is asserted positionally in this file, adapt extraction accordingly — the
REQUIREMENTS are: exactly one alert per partial symbol, symbol in title,
`bypass_suppression=True`, and no literal `None requested` for notional orders.)

- [ ] **Step 2: Run to verify they fail**

Run: `env -u DATABASE_URL uv run pytest tests/scheduler/test_jobs.py -k "partial_fill_alert" -q`
Expected: 2 failures — old code sends a shared title without the symbol, without
`bypass_suppression`, and with `None requested` in the message.

- [ ] **Step 3: Commit RED**

```bash
git add tests/scheduler/test_jobs.py
git commit -m "test(scheduler): RED — valid partials alert per-symbol, unsuppressed, notional-aware"
```

- [ ] **Step 4: Implement the alert change**

In `src/swingrl/scheduler/jobs.py`, replace the still-live `slice_recorded` alert block
(previously lines 988-1000):

```python
    # Still live (partially_filled / new / accepted): warn per state, keep the row open.
    if slice_recorded:
        requested_qty = getattr(order, "qty", None)
        requested = (
            requested_qty
            if requested_qty is not None
            else f"${_safe_float(getattr(order, 'notional', None))} notional"
        )
        ctx.alerter.send_alert(
            level="warning",
            title=f"Equity auction order PARTIALLY filled — {row['symbol']} recorded",
            message=(
                f"{row['symbol']} {row['side']} (order {order_id}) partially filled: "
                f"{delta_qty} recorded now ({cum_qty} cumulative of "
                f"{requested} requested) at {cum_avg} avg — remainder "
                "still working, row stays open."
            ),
            environment="equity",
            # Ruling 2026-07-23: valid partials on different symbols are distinct events,
            # never duplicates — bypass the consecutive gate and the shared-title cooldown
            # (which delivered 1 of 8 on 2026-07-23). Title carries the symbol so two
            # same-morning partials cannot collide even at the Discord level.
            bypass_suppression=True,
        )
```

The `else:` "unfilled" branch below it stays byte-identical (out of ruling scope).

- [ ] **Step 5: Run the whole scheduler test file to verify GREEN**

Run: `env -u DATABASE_URL uv run pytest tests/scheduler/test_jobs.py -q`
Expected: all pass, 0 failed. If any pre-existing test asserted the OLD shared partial title,
update ONLY its expected title/kwargs to the new contract (that is the ruling changing the
contract, not a regression) and say so in the report.

- [ ] **Step 6: Commit GREEN**

```bash
git add src/swingrl/scheduler/jobs.py tests/scheduler/test_jobs.py
git commit -m "fix(scheduler): valid-partial alerts per-symbol + unsuppressed + notional-aware (ruling 2026-07-23)"
```

---

### Task 3: Branch verification, CI, PR

**Files:**
- No new code. Verification + PR only.

**Interfaces:**
- Consumes: everything above; the three-lane workflow (docs/testing/best-practices.md).
- Produces: evidence for the PR; PR to `swingrl/2.R-training-redesign`.

- [ ] **Step 1: Fast lane**

Run: `env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q`
Expected: green (<2 min).

- [ ] **Step 2: Lockfile ruff + mypy (CI authorities)**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/swingrl/scheduler/jobs.py tests/scheduler/test_jobs.py && uv run mypy src/`
Expected: clean on all three.

- [ ] **Step 3: Full suite (`-n 4`, per-worker clones; template already on pg16)**

Prepare a base scratch DB per docs/testing/best-practices.md (any `*_test` name; the per-worker
clones derive from it), then run in the background, harness-tracked:

```bash
DATABASE_URL=<base-scratch-url> uv run pytest tests/ -v -n 4
```

Expected: 0 failed (~18 min). After ANY failure: `--lf` first, never a blind full relaunch.

- [ ] **Step 4: Push, homelab CI, PR**

```bash
git push -u origin swingrl/2.R-G-status-fix
cd ~/swingrl && git fetch origin && git checkout swingrl/2.R-G-status-fix && \
  git pull origin swingrl/2.R-G-status-fix && bash scripts/ci-homelab.sh --no-cache
```

Literal `=== CI PASSED ===` is the only verdict. Then PR to `swingrl/2.R-training-redesign`
(never main) with: the bug evidence (enum stringification, 8 misclassified orders), the ruling
text, before/after alert behavior, and the self-heal note for the 8 open rows. **Stop after PR
— merge and deploy are the user's.**

- [ ] **Step 5 (post-deploy, USER-GATED — do not execute without explicit approval):**

The 8 open cycle-32 rows close silently on the first post-deploy 09:35 run. If the user wants
them closed immediately instead, run ONE manual confirmation pass in the deployed container
(`docker exec swingrl python -c "from swingrl.scheduler.jobs import equity_fill_confirmation_job; equity_fill_confirmation_job()"`)
— reads broker state, quiet-stamps the rows, records nothing new. This is a production action:
ask first.

---

## Self-Review (performed while writing — findings fixed inline)

- **Spec coverage:** enum bug → Task 1; "only valid instances" → Task 1 (filled orders never
  reach the partial branch); "all instances, never suppressed" → Task 2 (per-symbol +
  bypass_suppression); "None requested" cosmetic → Task 2; toggle/self-heal → evidence base +
  Task 3 Step 5; production-payload testing gap → real-enum tests in both tasks.
- **Placeholder scan:** the `_insert_pending`/`_pending_row` helper names are explicitly
  flagged as adapt-to-file placeholders with the source tests named — the assertions and
  payloads (the actual requirements) are complete. No TBDs.
- **Type consistency:** `raw_status`/`status` names consistent between Task 1 code and Task 2's
  reliance; `_safe_float` already exists in jobs.py (used at lines 815-816).
- **Scope check:** single subsystem (one function + its tests) — one plan, three tasks, no
  split needed. The "unfilled" branch and terminal branches explicitly untouched.
