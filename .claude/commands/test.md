---
description: Run pytest FAST lane natively (no DB, parallel). Usage: /project:test [path] [-k filter] [-x]
---

Run the fast lane by default — no DB, `db`/`slow`/`integration` tests excluded, parallel:

```bash
env -u DATABASE_URL uv run pytest ${ARGUMENTS:-tests/} -m "not db and not slow and not integration" -n auto -q
```

Common patterns:
- Fast lane, whole suite (default): `/project:test`
- One package: `/project:test tests/execution`
- Single test by name: `/project:test -k test_valid_config_loads`
- Stop on first failure: `/project:test -x`
- Rerun last failures only: `uv run pytest --lf -q` (ALWAYS do this first after a failure)
- DB lane (targeted, real scratch DB — see docs/testing/best-practices.md for the URL recipe):
  `DATABASE_URL=postgresql://swingrl:…@<pg16-ip>:5432/<name>_test uv run pytest tests/<pkg> -m "not slow" -q`
- Full suite (pre-push only): `bash scripts/ci-homelab.sh` on homelab
