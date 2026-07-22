-- Spec D11: pre-open opening-auction execution. The 09:15 equity cycle submits DAY market
-- orders before the 09:28 auction cutoff; those orders rest as "pending" (Alpaca fills them
-- at the official opening print) and are recorded by the ~09:35 fill-confirmation job.
--
-- pending_orders is that job's restart-safe worklist: it survives a process restart between
-- 09:15 submission and 09:35 confirmation (an in-memory queue would not). One row per
-- submitted-but-unconfirmed order; resolved_at is stamped once the fill is recorded (or the
-- order is confirmed dead), so the confirmation job only ever processes WHERE resolved_at IS
-- NULL. cycle_id links the fill back to its originating inference cycle (nullable — capture
-- is fail-open, so a cycle whose capture failed still records the fill with a NULL cycle_id,
-- matching trades.cycle_id's nullable convention).
CREATE TABLE pending_orders (
    order_id     TEXT PRIMARY KEY,                               -- Alpaca broker order id
    cycle_id     BIGINT REFERENCES inference_cycles (cycle_id),  -- originating cycle (nullable)
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    submitted_at TIMESTAMPTZ NOT NULL,                           -- 09:15 submission time
    resolved_at  TIMESTAMPTZ,                                    -- NULL until confirmed/recorded
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The confirmation job scans only unresolved orders in submission order; a partial index
-- keeps that scan cheap as resolved rows accumulate (append-only evidence, never deleted).
CREATE INDEX idx_pending_orders_unresolved
    ON pending_orders (submitted_at)
    WHERE resolved_at IS NULL;
