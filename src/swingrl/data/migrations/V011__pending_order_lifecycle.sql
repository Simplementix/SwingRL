-- Rulings 2026-07-22 #2/#3: pending-order lifecycle columns.
-- decision_price: the 09:15 sizing price (pipeline Step 9's get_current_price value),
--   persisted at submission so the 09:35 confirmation can compute auction slippage vs the
--   decision (fill_quality.decision_price_usd was NULL for every auction fill before this).
-- disposition: terminal state stamped together with resolved_at ('filled' | 'canceled' |
--   'expired'). A dead order is closed once — one final alert, then silence — instead of
--   re-warning daily forever. NULL while the row is still an open worklist item.
ALTER TABLE pending_orders ADD COLUMN decision_price DOUBLE PRECISION;
ALTER TABLE pending_orders ADD COLUMN disposition TEXT
    CHECK (disposition IN ('filled', 'canceled', 'expired'));
