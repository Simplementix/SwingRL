# SwingRL — Making the Streamlit app installable (PWA)

Streamlit does **not** emit a web-app manifest or service worker on its own, so a PWA
needs three pieces added around it: a `manifest.json`, an icon set, and a service worker,
all linked into the served HTML. Pick the option that matches how you deploy.

---

## What you need (any option)

**1. `manifest.json`**
```json
{
  "name": "SwingRL Trading Dashboard",
  "short_name": "SwingRL",
  "description": "RL-based swing trading system — portfolio, risk, and system health.",
  "start_url": "/",
  "display": "standalone",
  "orientation": "portrait",
  "background_color": "#0d0f14",
  "theme_color": "#0d0f14",
  "icons": [
    { "src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/static/icon-512-maskable.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ]
}
```

**2. Icons** — 192px and 512px PNGs (plus one "maskable" 512 with padding). Use the 📈
gold-square mark. `theme_color`/`background_color` above match the dashboard's dark theme
so the splash and status bar blend in.

**3. `service-worker.js`** — minimal offline shell (cache-first for static assets, network
for data so trades/prices stay live):
```js
const CACHE = "swingrl-v1";
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(["/", "/manifest.json"])));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  // Never cache Streamlit's websocket / data endpoints — keep the feed live.
  if (url.pathname.startsWith("/_stcore") || url.pathname.startsWith("/media")) return;
  e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
});
```

---

## Option A — Reverse proxy (recommended for a real deployment)

If Streamlit already sits behind nginx / Caddy / Traefik:

1. Serve `manifest.json`, the icons, and `service-worker.js` as static files at the domain root.
2. Inject two lines into the returned HTML `<head>` (nginx `sub_filter`, or a small edge worker):
   ```html
   <link rel="manifest" href="/manifest.json">
   <meta name="theme-color" content="#0d0f14">
   ```
3. Register the worker (also injectable, or via the component in Option B):
   ```html
   <script>
     if ("serviceWorker" in navigator)
       navigator.serviceWorker.register("/service-worker.js");
   </script>
   ```

`service-worker.js` **must** be served from the domain root (or as high as the pages it
should control) — a worker can only control paths at or below its own URL.

## Option B — In-app injection (no proxy, quickest)

From Python, drop the tags in with a components iframe near the top of the app:

```python
import streamlit.components.v1 as components

components.html(
    """
    <link rel="manifest" href="/app/static/manifest.json">
    <meta name="theme-color" content="#0d0f14">
    <script>
      if ('serviceWorker' in navigator)
        navigator.serviceWorker.register('/app/static/service-worker.js');
    </script>
    """,
    height=0,
)
```

Put the files in Streamlit's static dir and enable static serving in `.streamlit/config.toml`:
```toml
[server]
enableStaticServing = true
```
Files in `./static/` are then served at `/app/static/...`.

> Caveat: `components.html` renders inside an **iframe**, so the manifest link may not be
> picked up by every browser's install prompt as reliably as Option A. If the install
> banner doesn't appear, use the proxy approach.

## Option C — Community wrapper

Packages like `streamlit-pwa` automate the manifest + worker injection if you'd rather not
wire it by hand. Same three ingredients under the hood.

---

## Verifying

- Chrome/Edge DevTools → **Application** tab → Manifest (no errors) and Service Workers (activated).
- **Lighthouse → PWA** audit should pass "installable".
- On mobile: Chrome "Add to Home screen" / Safari "Add to Home Screen" gives a standalone,
  full-screen launch using the icon + theme color above.

## Notes specific to SwingRL

- `display: standalone` + `orientation: portrait` matches the mobile layout (bottom tab bar).
- Keep the service worker **network-first for data** (`/_stcore`, `/media` excluded above) so
  positions, P&L, and the 5-minute auto-refresh always show fresh values — capital-preservation
  monitoring should never render stale numbers from cache.
- Bump `CACHE` (`swingrl-v1` → `-v2`) whenever you ship new static assets to force an update.
