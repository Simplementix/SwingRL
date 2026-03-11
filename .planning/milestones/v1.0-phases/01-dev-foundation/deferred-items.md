# Deferred Items — Phase 01-dev-foundation

## pandas-ta Dependency Resolution

**Found during:** 01-01 Task 1 (dependency resolution)

**Issue:** `pandas-ta` (specified as `>=0.3.14b` in plan) is no longer installable on Python 3.11:
- PyPI only has versions 0.4.67b0 and 0.4.71b0, both requiring `Python>=3.12`
- The 0.3.14b0 version was removed from PyPI (404 on pythonhosted.org)
- The original GitHub repo (`twopirllc/pandas-ta`) has been deleted (404 on GitHub)
- All maintained forks found require `numpy>=2.0.0`, conflicting with the `numpy<2` constraint needed for Python 3.11 ecosystem compatibility
- The new official pandas-ta site (pandas-ta.dev) only serves Python>=3.12 packages

**Resolution chosen for Phase 1:** Removed `pandas-ta` from dependencies. Using `stockstats>=0.4.0` instead, which is FinRL's native technical indicator library and works on Python 3.11.

**Impact:**
- Smoke test updated to import `stockstats` instead of `pandas_ta`
- mypy override for `pandas_ta.*` kept in place for future use
- All FinRL-based indicator calculations will use `stockstats`

**Future resolution (Phase 6):**
- Evaluate `ta-lib` (C-based, robust, requires system binary) for custom indicators
- Evaluate `ta` library (pure Python, Python 3.11 compatible)
- Consider staying with `stockstats` if its indicator coverage is sufficient
- If `pandas_ta` functionality is essential, may need to upgrade to Python 3.12 (requires FinRL compatibility re-evaluation)

**Ticket:** Create issue before Phase 6 planning — "Resolve technical indicator library: stockstats vs ta vs pandas-ta"

---

## Docker/Linux pytorch-cpu Resolution (uv.lock multi-platform)

**Found during:** 01-01 Task 1 (uv.lock generation)

**Issue:** The `[tool.uv] environments = ["sys_platform == 'darwin'"]` constraint was added to avoid the Linux resolution split failing (due to the now-removed pandas-ta). This means the generated `uv.lock` only covers macOS, not Linux.

**Impact:** The Docker build (Phase 2, ENV-06) will fail with `uv sync --locked` because the Linux platform is not in the lockfile.

**Future resolution (Phase 2):**
- Remove the `environments` constraint from `[tool.uv]`
- Ensure all dependencies have Linux-compatible versions
- Run `uv lock` to regenerate a multi-platform lockfile
- Verify `uv sync --locked` succeeds in Docker before ENV-06 is marked complete
