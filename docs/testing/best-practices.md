# SwingRL Testing Best Practices

Source: test-infra reviews + implementation plan of 2026-07-22
(`docs/superpowers/plans/2026-07-22-test-suite-speed-robustness.md`). Each rule cites the
evidence that motivated it. Timing budgets were calibrated by that plan's measurement task.

## The three lanes

| Lane | Command | DB? | Budget | When |
|---|---|---|---|---|
| **fast** | `env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q` | none | < 2 min | every commit, every TDD RED/GREEN cycle |
| **db** | `DATABASE_URL=postgresql://swingrl:…@<pg16-ip>:5432/<name>_test uv run pytest tests/<pkg> -m "not slow" -q` | `*_test` scratch | < 5 min / package | before pushing a change that touches SQL / stores / migrations / that package |
| **full** | `bash scripts/ci-homelab.sh` (stage 3: `pytest tests/ -v -n 4`) | per-worker clones | ≤ 20 min | pre-push gate only |

pg16 has no host port — reach scratch DBs via the container IP
(`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' pg16`).

## The targeted-preflight rule (anti-2-hour rule)

**After ANY test failure, the next pytest invocation must be the failing subset, never the
full suite.** Escalation order:

1. `uv run pytest --lf -q` — the failure cache works out of the box (nothing disables the
   cacheprovider). Seconds to a minute.
2. Cache stale/gone (e.g. the failure was inside Docker CI — its cache died with the
   container): rerun the exact node IDs from the failure output
   (`ci-homelab.sh` runs `-v`, so node IDs are always in the log).
3. Only when the subset is green: relaunch the lane that failed.

A full-suite relaunch to verify a fix whose failing set is known is a rule violation
(2026-07-22 incident: 2 hours lost exactly this way).

## The checklist

1. **Always know which lane you are in** — fast / db / full, budgets above. *(Evidence: the
   only pre-2026-07-22 gate was one monolithic `pytest tests/ -v`, scripts/ci-homelab.sh:92 →
   60-minute minimum feedback.)*
2. **After any failure, rerun the failing subset first** (`--lf` / node IDs; rule above).
   *(Evidence: cacheprovider enabled but unused; 2026-07-22 incident cost 2 h.)*
3. **New DB-touching test ⇒ carries the `db` marker automatically** — derivation is
   collection-time (tests/db_marker.py: real-DB fixture requested, or module mentions
   DATABASE_URL). Do not hand-mark except to force with `@pytest.mark.db`. *(Evidence: DB
   gating was spelled 4 non-selectable ways across 51 files.)*
4. **One spelling for DB gating in new code**: module-level
   `pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), ...)` — no new inline
   mid-test `pytest.skip("DATABASE_URL...")`. *(Evidence: 43 inline skips were invisible to
   collection-time selection, e.g. tests/data/test_attribution_migration.py:78.)*
5. **Real DB only when SQL semantics are the subject** (migrations, upserts, FK/view order,
   store layers); mock (`make_mock_db`, tests/conftest.py:344) when the DB is a bystander.
   Rule of thumb: if changing the SQL string could break production while the test stays
   green, the test belongs on the real DB. *(Evidence: 61 mock files vs 16 real-DB files —
   split existed, unlabeled.)*
6. **Never name a real-DB fixture "mock"** — and vice versa. *(Evidence:
   tests/execution/conftest.py:73 `mock_db` is a real PostgreSQL DatabaseManager; root
   `make_mock_db` is a MagicMock — same word, opposite meanings, ~219 usages. Rename to
   `real_test_db` as one mechanical refactor when convenient.)*
7. **Session scope only for immutable values; freeze or don't promote.** DB fixtures stay
   function-scoped, non-negotiable (the autouse singleton reset at tests/conftest.py would
   hand wider-scoped fixtures a closed pool). *(Evidence: scope policy at
   tests/conftest.py:3-6; only `repo_root` is session-scoped.)*
8. **New autouse fixture ⇒ docstring must justify the per-test tax across ~1,900 tests** and
   state its per-test cost. *(Evidence: the unconditional wipe cost ~40 min/run before
   2026-07-22; contrast the well-justified structlog isolator, tests/conftest.py:66-93.)*
9. **Expensive schema fixtures (`db_with_legacy_schema`-class: de-migrate + re-migrate per
   test) live only in db/full lanes, never the commit loop.** *(Evidence:
   tests/data/conftest.py:64-112, ~79 test sites, full V001-V011 drop+reapply per test.)*
10. **NEVER `DROP TABLE` a production-named table from a test.** Use the canonical DDL
    (`src/swingrl/data/postgres_schema.py`) via `init_postgres_schema`/single-table constants,
    or a throwaway `*_probe`/`_mig_test_*` name. `CREATE TABLE IF NOT EXISTS` will never
    repair a wrong shape for you. *(Evidence: the 2026-07-22 hmm_state_history PK loss —
    3 suite-hours; the schema preflight now aborts the session if this rule is broken.)*
11. **Seeded factories over inline data**: pinned-RNG frames/arrays from conftest factories;
    package-local sizes in the package conftest. *(Evidence: good pattern at
    tests/conftest.py:166-321, tests/features/conftest.py — extend, don't fork.)*
12. **One config-YAML factory** — no fourth copy of the ~40-line YAML block. *(Evidence:
    three near-identical blocks: tests/conftest.py:105-143, 217-255,
    tests/execution/conftest.py:21-61.)*
13. **Docstring states the asserted behavior; cite ruling/REQ IDs when they exist.**
    *(Evidence: the literal `REQ-ID:` rule was followed by 23/1,905 tests — this matches live
    good practice instead, e.g. the rulings RED commits.)*
14. **Never weaken the DB-safety stack** — suite guard (tests/conftest.py::pytest_configure),
    `_test` classifier (tests/db_guard.py), pre-wipe re-check
    (tests/fixtures/db_cleanup.py:29-46), schema preflight
    (tests/fixtures/schema_preflight.py). Route around it, never through it.
15. **Long tests declare their own budget**: anything legitimately > 60 s gets
    `@pytest.mark.slow` + a tighter `@pytest.mark.timeout(n)`. *(Evidence: one global
    `timeout = 600` and zero per-test overrides — a hang burns 10 silent minutes.)*
16. **TDD stays RED-commit-then-GREEN; run the RED test via the fast lane or `--lf`, never
    the full suite.** *(Evidence: discipline alive in git history — keep the loop cheap so it
    survives.)*

## Isolated DBs and the template

- xdist workers automatically clone `swingrl_gwN_test` from `swingrl_test_template`
  (tests/db_worker.py). Serial sessions opt in with `SWINGRL_TEST_ISOLATED_DB=1`.
- Refresh the template after ANY new migration:
  `bash scripts/prepare-test-db-template.sh` (also reaps stale clones). A stale template
  fails loud (schema-version check at clone time).
- The schema preflight aborts any session whose target DB has a canonical table without its
  PRIMARY KEY — if you see `SCHEMA PREFLIGHT FAILED`, recreate the scratch DB; do not "fix"
  the guard.
