# Handoff: SwingRL Trading Dashboard

## Overview
An operations dashboard for **SwingRL**, an RL-based swing-trading system currently paper-trading
equity (8 ETFs via Alpaca, daily) and crypto (BTC/ETH via Binance.US, 4-hour). It monitors the
live paper account across five pages plus a placeholder: **Dashboard** (summary), **Portfolio**,
**Trade Log**, **Risk Metrics**, **System Health**, and a **Training Dashboard** stub.
Guiding product principle: **capital preservation is the primary constraint** — surface halts and
data staleness loudly; never present stale numbers as live.

Target stack (per the owner): **Streamlit** frontend, **Python** backend, **PostgreSQL** database.

## About the Design Files
The files in this bundle are **design references created in HTML** (Design Components) — prototypes
showing intended look, layout, and behavior. They are **not production code to copy directly**.
The task is to **recreate these designs in the target environment (Streamlit + Python + PostgreSQL)**
using its established patterns. A complete component/table mapping already exists in
`Streamlit-handoff.md` (in this bundle) — treat that as the build spec and this README as the
design-detail reference.

## Fidelity
**Low-fidelity → medium.** These are structural wireframes (the owner asked for a wireframe), but
polished: real layout, a committed visual system, working interactions, and realistic mock data.
Use them as the authoritative guide for **layout, information hierarchy, states, and interactions**.
Styling is a coherent custom system (below) — reproduce the structure and behavior faithfully; match
the visual system as closely as Streamlit/Plotly reasonably allow rather than pixel-chasing.

## Screens / Views

### Shared shell (all pages)
- **Layout**: sticky top header (max-width 1220px, centered) + horizontal pill nav + content column.
  Content uses responsive `grid` with `repeat(auto-fit, minmax(...))` so cards reflow.
- **Header**: logo (📈 gold rounded square + "SwingRL" / "RL swing trading · paper"); right side:
  Range segmented control (7D/30D/ALL), auto-refresh chip ("14:32 ET · 5m", pulsing green dot),
  accent swatches, density toggle (▦/▤), theme toggle (☀/☾).
- **Preview-state switcher** (dashed, demo-only, below header): Normal / Circuit halt / Degraded feed.
- **Global status banner** (below switcher, all pages): hidden in Normal; red for halt; amber for degraded.
- **Nav items**: Dashboard, Portfolio, Trade Log, Risk Metrics, System Health, Training Dashboard.

### 1 · Dashboard (summary)
- **Purpose**: everything at a glance, ordered status → money → risk → activity.
- **Panels (top to bottom)**:
  1. **System status** card — grouped by *Infrastructure / Brokers / Data feeds*, each a status dot (green/amber/red) + name + mono detail.
  2. **KPI tiles** ×4: Total Portfolio Value ($451.72, +$3.19), Open positions (5), Current drawdown (-3.1%), Circuit breakers (ARMED).
  3. **Equity curve** + **Drawdown** charts side by side (agent line + dashed buy&hold baseline; drawdown with 5%/8% threshold lines). Y-axis labels + gridlines + date axis; **hover shows crosshair + value tooltip**.
  4. **Open positions** table + **Recent trades** table.
  5. **Alerts & notifications** + **Circuit breakers** (per env + global).

### 2 · Portfolio
- Combined totals (value, return, vs Buy&Hold, cash) as 4 metrics.
- Per-environment panels **Equity | Crypto** in 2 columns: total value + daily P&L metrics, caption (cash · HWM · symbols), equity curve (hover-enabled).
- **Daily P&L** bar chart (green up / red down days; hover shows day value).
- **Holdings** full table (symbol, env, side, qty, cost, last, unreal P&L, stop, target).

### 3 · Trade Log
- Summary stats ×4 (recompute from filtered set): Fills shown, Buys/sells, Symbols, Commission.
- **Filters** (live): environment, side, symbol (`select`), date range (`select`/date_input).
- **Fill ledger** table: time ET, symbol, side, qty, fill price, commission, slippage, env, broker, type, cycle. Rows hover-highlight + pointer (click → fill-quality drill-in). Empty state when no match.
- ⚠ **No per-trade P&L column** — not stored; do not synthesize.

### 4 · Risk Metrics
- Live tiles: Current drawdown, HWM, Daily loss, Exposure (colored dot: green <5% · amber <8% · red ≥8%).
- **Cooldown & ramp** stepper — shown only when halted (Halt → Cooldown → 25/50/75/100%).
- **Circuit-breaker status** table (rule tooltips on hover), **Global & emergency halts** list, **Drawdown history** chart (hover), **Risk decisions** table.

### 5 · System Health
- **Services** grid (backend, DB, Alpaca, Binance.US, equity/crypto feeds, cron) with status dots.
- **Data ingestion** table (last bar, age; red when age ≥ staleness window: equity 26h, crypto 5h).
- **Last trade cycle** key/values; **Log stream** (dark terminal card; info blue / warning orange / critical red).

### 6 · Training Dashboard (placeholder)
- "Coming soon" stub — backend RL-training redesign in progress. Grid of intended metrics
  (Sharpe/Sortino/Calmar, max drawdown, profit factor, win rate, max single loss, iteration comparison).

## Interactions & Behavior
- **Nav**: click switches page (single-page state; in Streamlit → sidebar/`st.navigation` multipage).
- **Range 7D/30D/ALL**: re-scopes chart curves + date axis.
- **Trade Log filters**: live-filter ledger + recompute stats; empty state.
- **Chart hover**: crosshair line + dot snapped to nearest point + tooltip (value · date) on every chart; bars show per-bar value.
- **Preview-state switcher**: Normal / Circuit halt (red banner, HALTED breaker, drawdown -10.5%, positions liquidated → empty state, cooldown stepper) / Degraded feed (amber banner, crypto feed STALE/red, crypto PAUSED as a *data gate* — breakers stay ARMED).
- **Theme toggle** (dark/light), **accent** (gold/blue/purple/green), **density** (comfortable/compact).
- **Responsive**: <940px header extras collapse into a ⚙ menu; <720px top nav → fixed **bottom tab bar** (mobile). Auto-refresh every 5 min.
- **URL params** on the dashboard: `?page=&theme=&accent=` (used by the mobile-preview frames).

## State Management
- `page`, `range`, `theme`, `accent`, `density`, `isMobile`/`isCompact`, `settingsOpen`,
  `scenario` (normal/halt/degraded), Trade Log `filterEnv/filterSide/filterSymbol/filterDate`,
  chart `hover {key,i}`.
- Data fetching: one cached query fn per table (`@st.cache_data(ttl=300)`), re-scoped by Range;
  auto-refresh invalidates on the 5-min tick. Derive cash/P&L, equity-crypto split, exposure.

## Design Tokens
Theming is driven by CSS custom properties (dark default / light override).

**Dark**: bg `#0d0f14`, card `#171a21`, inset `#1f232c`, line `#282d38`/`#333947`,
text `#e8eaef`, muted `#9aa1b0`, faint `#6b7280`.
**Light**: bg `#f3f4f7`, card `#ffffff`, inset `#f5f6f9`, line `#e7e9ef`/`#dcdfe7`,
text `#151821`, muted `#5c6472`, faint `#98a0ae`.

**Semantic (dark / light)**: positive `#2fd07a`/`#12a457`, negative `#f75c5c`/`#e03b3b`,
warning `#f6a63c`/`#d9820c`, info `#4aa3f0`/`#2f7fdb`, cycle purple `#b56ef0`/`#8b45cf`, gold `#F1C40F`.

**Brand palette (from Discord alerts — canonical)**: digest gold `#F1C40F`, buy green `#00FF00`,
sell red `#FF4444`, info blue `#3498DB`, cycle purple `#9B59B6`, warning orange `#FFA500`,
critical red `#FF0000`. *(The wireframe softens neon buy/sell to `#2fd07a`/`#f75c5c` for legibility; same semantics.)*

**Type**: `Space Grotesk` (UI/headings), `IBM Plex Mono` (numbers, timestamps, captions).
Sizes: page title 27px/700; card title 14px/600; KPI value 24–28px mono/600; body 12px; caption 10–11px.
**Radius**: cards 16px, controls 9–11px, pills 20px. **Gaps**: comfortable 14–18px, compact 10–12px.
Shadows: dark `0 1px 2px rgba(0,0,0,.5), 0 10px 30px rgba(0,0,0,.30)`; light softer.
Charts: 2.5px accent line, gradient area fill (accent 0.28→0), dashed 5% (gold) / 8% (red) thresholds.

## Assets
No image assets. Logo mark is the 📈 emoji on a gold rounded square. Status indicators are
CSS dots. Charts are inline SVG polylines/polygons (Plotly in the real build). Icon glyphs are
Unicode (☀ ☾ ▦ ▤ ⚙ ∅) and the brand emoji motif 📊/📈.

## Files (in this bundle)
- `SwingRL Dashboard.dc.html` — the full dashboard (all 6 pages, states, interactions). Primary reference.
- `SwingRL Mobile Preview.dc.html` — 3 iPhone frames showing the mobile layout (loads the dashboard in iframes).
- `ios-frame.jsx` — device-frame component used only by the mobile preview.
- `Streamlit-handoff.md` — **the build spec**: per-panel `st.*` mapping + tables/columns + SQL sketches + thresholds.
- `PWA-notes.md` — how to make the Streamlit app installable (manifest + service worker + verification).

> To view the HTML references: open the `.dc.html` files in a browser (they render standalone).
