# SwingRL — Streamlit build handoff

How each panel in the wireframe maps to Streamlit primitives and the PostgreSQL
tables/columns behind it. Built for `st` wide mode with a 5-minute auto-refresh
(`st_autorefresh(interval=300_000)`). All timestamps stored UTC, **displayed Eastern**.

Guiding principle carried from the product: **capital preservation first** — never show
stale numbers as if live; surface halts and staleness loudly (see banner + System Health).

---

## Source tables (recap)

| table | role | key columns |
|---|---|---|
| `trades` | append-only fill ledger (**no P&L**) | trade_id, timestamp, symbol, side, quantity, price, commission, slippage, environment, broker, order_type, trade_type, cycle_id |
| `positions` | current holdings | symbol, quantity, cost_basis, last_price, unrealized_pnl, stop_loss_price, take_profit_price, side |
| `portfolio_snapshots` | equity curve over time | total_value, cash_balance, high_water_mark, daily_pnl, drawdown_pct, timestamp (equity_value/crypto_value usually NULL — split comes from per-env rows) |
| `fill_quality` | execution quality | decision/expected/actual price, slippage_frac, time_to_fill_ms |
| `benchmark_baselines` | agent vs buy & hold | baseline_price, capital_usd |
| `risk_decisions` | every allow/veto | timestamp, environment, decision, reason |
| `circuit_breaker_events` | every trip/resume | timestamp, environment, breaching value, threshold, reason (e.g. `drawdown_0.1053_exceeds_0.1`) |
| `iteration_results` / `backtest_results` | research metrics (Training page) | sharpe, sortino, calmar, max_drawdown, profit_factor, win_rate, max_single_loss |
| `config` | thresholds (defaults) | per-env max_position_size, max_drawdown_pct, daily_loss_limit_pct, min_order |

Derived values (compute in a cached data layer, `@st.cache_data(ttl=300)`):
- **cash / P&L** — from the ledger + snapshots, not stored per trade.
- **equity vs crypto split** — from per-environment `portfolio_snapshots` rows.
- **exposure** — `sum(abs(position_value)) / total_value`, capped display at 100%.

---

## Shared shell

| element | Streamlit | notes |
|---|---|---|
| Page nav (Dashboard / Portfolio / Trade Log / Risk / System Health / Training) | `st.sidebar` radio or `st.navigation` / multipage `pages/` | one file per page |
| Range 7D / 30D / ALL | `st.segmented_control` (or `st.radio` horizontal) | scopes the time-series queries (`WHERE timestamp >= now() - interval`) |
| Auto-refresh chip | `st_autorefresh` + `st.caption("updated … ET · 5m")` | |
| Global status banner | `st.error` (halt) / `st.warning` (degraded) / hidden (normal) | driven by open `circuit_breaker_events` + feed staleness |
| Status dots (green/amber/red) | `st.markdown` with colored ● (unsafe_allow_html) | green healthy · amber warning · red down |

---

## 1 · Dashboard (summary)

| panel | Streamlit | data |
|---|---|---|
| System status (grouped: Infrastructure / Brokers / Data feeds) | `st.columns` of `st.markdown` dot + label | health probes; feed age vs staleness windows (equity 26h, crypto 5h) |
| KPI tiles: Total Portfolio Value, Open positions, Current drawdown, Circuit breakers | `st.metric` (value + delta) ×4 in `st.columns(4)` | `portfolio_snapshots` latest (total_value, daily_pnl, drawdown_pct); `positions` count; open `circuit_breaker_events` |
| Equity curve (agent vs buy & hold) | `st.plotly_chart` | `portfolio_snapshots.total_value` over range; baseline from `benchmark_baselines` |
| Drawdown | `st.plotly_chart` | `portfolio_snapshots.drawdown_pct`; threshold lines 5% / 8% |
| Circuit breakers (per env + global) | `st.markdown` rows w/ status pill | latest `circuit_breaker_events` / live risk state |
| Alerts & notifications | `st.container` list | recent `circuit_breaker_events` + risk_decisions + ingestion warnings (mirror Discord digest) |
| Open positions | `st.dataframe` | `positions` (symbol, quantity, last_price, unrealized_pnl) — empty state when flat |
| Recent trades | `st.dataframe` | `trades` latest 5 (time, symbol, side, qty, price) |

Ordering reflects capital-preservation priority: **status → money → risk → activity**.

## 2 · Portfolio

| panel | Streamlit | data |
|---|---|---|
| Combined totals (value, return, vs B&H, cash) | `st.metric` ×4 | `portfolio_snapshots` + `benchmark_baselines` |
| Per-environment panels (Equity \| Crypto) | `st.columns(2)`, each `st.metric` + `st.caption` + `st.plotly_chart` | per-env `portfolio_snapshots` rows; caption = cash · HWM · symbols |
| Daily P&L | `st.plotly_chart` (bar) | `portfolio_snapshots.daily_pnl` |
| Holdings | `st.dataframe` | `positions` full (env, side, qty, cost_basis, last_price, unrealized_pnl, stop, target) |

> Equity ≈ 8 ETFs (SPY QQQ VTI XLV XLI XLE XLF XLK) via Alpaca; Crypto = BTCUSDT/ETHUSDT via Binance.US.

## 3 · Trade Log

| panel | Streamlit | data |
|---|---|---|
| Summary stats (fills, buys/sells, symbols, commission) | `st.metric` ×4 | aggregates over the **filtered** `trades` set |
| Filters: environment, side, symbol, date range | `st.selectbox` ×3 + `st.date_input` | drive the `WHERE` clause; recompute stats + table live |
| Fill ledger | `st.dataframe` | `trades` (time ET, symbol, side, qty, fill price, commission, slippage, env, broker, trade_type, cycle_id) |
| (drill-in) Fill quality | `st.expander` / row select → `st.dataframe` | `fill_quality` (decision vs expected vs actual, slippage_frac, time_to_fill_ms) |

> **No per-trade P&L column** — it is not stored; do not synthesize one. `trade_type = 'signal'` marks a real agent trade; `cycle_id` links to the decision cycle.

## 4 · Risk Metrics

| panel | Streamlit | data |
|---|---|---|
| Live tiles: drawdown, HWM, daily loss, exposure | `st.metric` + colored dot | `portfolio_snapshots`; drawdown color green <5% · yellow <8% · red ≥8% |
| Circuit-breaker status matrix | `st.dataframe` / `st.markdown` table | live risk state vs `config` thresholds; tooltip = rule definition |
| Cooldown & ramp (only when halted) | `st.progress` / stepper via `st.columns` | `circuit_breaker_events`; equity 5 business days, crypto 3 calendar days → ramp 25/50/75/100% |
| Global & emergency halts | `st.markdown` rows | combined dd ≥15%, combined daily ≥3%; VIX>40 + 24h dd≥13%; ≥2 NaN inferences/24h; Binance HTTP 418/24h |
| Drawdown history | `st.plotly_chart` | `portfolio_snapshots.drawdown_pct` |
| Risk decisions | `st.dataframe` | `risk_decisions` + `circuit_breaker_events` (time, env, decision, reason) |

**Circuit-breaker thresholds (from `config`, ≥ comparisons, HWM denominator):**
- Drawdown halt: equity ≥10%, crypto ≥12%
- Daily-loss halt: equity ≥2%, crypto ≥3% (must be `<` the drawdown limit)
- Global: combined dd ≥15% **or** combined daily ≥3% → halts both
- Position-size veto (buys): >25% equity / >50% crypto
- Exposure veto: new total exposure >100%
- Turbulence: >97th percentile → hard halt + liquidation
- Emergency auto-stop: VIX>40 AND 24h dd ≥13%; or ≥2 NaN inferences/24h; or any Binance.US HTTP 418/24h

## 5 · System Health

| panel | Streamlit | data |
|---|---|---|
| Services (backend, DB, Alpaca, Binance.US, feeds, cron) | `st.columns` of dot + label | health probes / last-heartbeat checks |
| Data ingestion | `st.dataframe` | last bar time + age vs window; **red when age ≥ window** (equity 26h, crypto 5h) |
| Last trade cycle | `st.container` key/value | latest `cycle_id`, completion time, orders placed, inference OK (no NaN) |
| Log stream | `st.code` / styled `st.container` | app logs — info (blue) / warning (orange) / critical (red) |

## 6 · Training Dashboard *(placeholder — backend redesign in progress)*

Intended: Sharpe / Sortino / Calmar, max drawdown, profit factor, win rate, max single loss,
iteration comparison — from `iteration_results` / `backtest_results`
(`st.dataframe` + `st.plotly_chart`, incl. a Sharpe heatmap). Ship as a "coming soon" stub until ready.

---

## Colors (from Discord alerts — reuse for brand consistency)

digest gold `#F1C40F` · buy green `#00FF00` · sell red `#FF4444` · info blue `#3498DB` ·
cycle purple `#9B59B6` · warning orange `#FFA500` · critical red `#FF0000`.

*(The wireframe softens the neon buy/sell greens/reds for on-screen legibility — e.g.
`#2fd07a` / `#f75c5c` — while keeping the same semantics. Match to taste in Plotly.)*

## Notes for the build

- **Cache the data layer**, not the widgets: one `@st.cache_data(ttl=300)` query fn per table,
  re-scoped by the Range control. Auto-refresh invalidates on the 5-min tick.
- **Timezone**: store UTC, convert to `America/New_York` at the display edge only.
- **Empty/degraded states matter** — render the flat-book, halted, and stale-feed states
  explicitly (see the wireframe "Preview state" switcher) rather than blanking panels.
- **Mobile / PWA**: Streamlit reflows columns; for the installable PWA shell see `docs/PWA-notes.md`.
