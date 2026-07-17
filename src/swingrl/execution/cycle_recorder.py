"""CycleRecorder: fail-open capture of each inference cycle.

Single writer for the V003 trade-time tables ``inference_cycles`` and
``cycle_algo_proposals``. Every capture is fail-open: any database failure is
logged (``cycle_capture_failed``) and alerted, and ``record_cycle`` returns
``None`` — capture must never raise into the money path (spec §4.7 / A27).

Usage:
    from swingrl.execution.cycle_recorder import CycleRecorder
    recorder = CycleRecorder(db=db, config=config, alerter=alerter)
    cycle_id = recorder.record_cycle(env_name="equity", mode="paper", ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from swingrl.utils.exceptions import DataError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

# Bumped whenever the JSONB payload shape changes so historical rows stay parseable.
_SCHEMA_VERSION = 1

_INFERENCE_INSERT_SQL = (
    "INSERT INTO inference_cycles "
    "(environment, mode, cycle_ts, deployed_iteration, hmm_p_bull, hmm_p_bear, "
    "vix, turbulence, active_event_ids, blended_actions) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
    "RETURNING cycle_id"
)

_PROPOSAL_INSERT_SQL = (
    "INSERT INTO cycle_algo_proposals "
    "(cycle_id, model_id, algorithm, proposed_actions, weight_in_blend_frac) "
    "VALUES (%s, %s, %s, %s::jsonb, %s)"
)

_ACTIVE_EVENTS_SQL = (
    "SELECT event_id FROM calendar_events "
    "WHERE window_start <= %s AND window_end >= %s "
    "ORDER BY event_id"
)

_ACTIVE_SPINE_SQL = (
    "SELECT tr.algorithm AS algorithm, m.model_id AS model_id, "
    "tr.iteration_number AS iteration_number "
    "FROM models m "
    "JOIN training_runs tr ON tr.run_pk = m.run_pk "
    "WHERE m.status = %s AND tr.environment = %s "
    "ORDER BY m.created_at DESC"
)


@dataclass(frozen=True)
class RegimeStamp:
    """Decision-time market regime captured for one inference cycle."""

    hmm_p_bull: float | None
    hmm_p_bear: float | None
    vix: float | None
    turbulence: float | None
    active_event_ids: list[int]


@dataclass
class AlgoProposal:
    """One algorithm's raw proposal and its snapshotted blend weight (D-T3.13)."""

    algorithm: str
    model_id: str
    raw_actions: dict[str, float]
    weight_in_blend_frac: float


class CycleRecorder:
    """Fail-open writer for per-cycle regime + per-algo proposal capture."""

    def __init__(
        self,
        db: DatabaseManager,
        config: SwingRLConfig,
        alerter: Alerter,
    ) -> None:
        """Initialize the recorder.

        Args:
            db: DatabaseManager for the capture writes.
            config: Validated SwingRLConfig (symbol ordering, trading mode).
            alerter: Discord alerter for the fail-open capture-failure warning.
        """
        self._db = db
        self._config = config
        self._alerter = alerter
        # Active models ⋈ training_runs spine, cached per env for the recorder's
        # lifetime. The trader is restarted on model promotions (A30 deploy
        # isolation runs them in market-safe windows), so a session-lifetime cache
        # never serves a model_id that changed mid-run.
        self._spine_cache: dict[str, list[dict[str, Any]]] = {}

    def record_cycle(
        self,
        *,
        env_name: str,
        mode: str,
        cycle_ts: datetime,
        regime: RegimeStamp,
        raw_actions: dict[str, list[float]],
        target_weights: dict[str, float],
        proposals: list[AlgoProposal],
        deployed_iteration: int | None,
        dry_run: bool = False,
        skip_reasons: dict[str, str] | None = None,
    ) -> int | None:
        """Write the inference_cycles row and one cycle_algo_proposals row per algo.

        Both tables are written in a single transaction so a partial capture rolls
        back. Fail-open: on any database error the failure is logged and alerted
        (a silent capture outage must not survive a day unnoticed) and ``None`` is
        returned — capture never raises into the trading path.

        Args:
            env_name: "equity" or "crypto".
            mode: Trading mode ("paper" or "live").
            cycle_ts: The single canonical cycle timestamp (UTC).
            regime: Decision-time RegimeStamp (HMM/VIX/turbulence/active events).
            raw_actions: Per-algo raw action vectors ``{algo: [floats]}``.
            target_weights: Blended per-symbol target weights ``{symbol: frac}``.
            proposals: One AlgoProposal per algorithm that produced actions.
            deployed_iteration: Max active-model iteration (display only, A20).
            dry_run: Whether this cycle was a dry-run — tagged in the payload so the
                ``run_cycle.py`` second-writer's simulations stay distinguishable (fold d).
            skip_reasons: Per-symbol M2 skip reasons ``{symbol: reason}`` (fold c);
                symbols the model held are absent.

        Returns:
            The new ``cycle_id``, or ``None`` if capture failed.
        """
        try:
            blended_payload = json.dumps(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "raw": raw_actions,
                    "target_weights_frac": target_weights,
                    "skip_reasons": skip_reasons or {},
                    "dry_run": dry_run,
                }
            )
            with self._db.connection() as conn:
                cycle_id = self._insert_cycle_row(
                    conn,
                    env_name=env_name,
                    mode=mode,
                    cycle_ts=cycle_ts,
                    deployed_iteration=deployed_iteration,
                    hmm_p_bull=regime.hmm_p_bull,
                    hmm_p_bear=regime.hmm_p_bear,
                    vix=regime.vix,
                    turbulence=regime.turbulence,
                    active_event_ids=regime.active_event_ids,
                    payload=blended_payload,
                )
                for proposal in proposals:
                    proposed_payload = json.dumps(
                        {"schema_version": _SCHEMA_VERSION, "raw": proposal.raw_actions}
                    )
                    conn.execute(
                        _PROPOSAL_INSERT_SQL,
                        (
                            cycle_id,
                            proposal.model_id,
                            proposal.algorithm,
                            proposed_payload,
                            proposal.weight_in_blend_frac,
                        ),
                    )

            log.info(
                "cycle_captured",
                env=env_name,
                cycle_id=cycle_id,
                proposals=len(proposals),
            )
            return cycle_id
        except Exception:
            self._alert_capture_failed(env_name)
            return None

    def record_halt(
        self,
        *,
        env_name: str,
        mode: str,
        cycle_ts: datetime,
        reason: str,
        turbulence: float | None = None,
        dry_run: bool = False,
    ) -> int | None:
        """Write a minimal inference_cycles row for an early-exit (halted) cycle.

        Regime fields are NULL (not computed before the exit) except ``turbulence``,
        which is passed when known (e.g. the turbulence-halt exit). No proposal rows
        are written; the payload carries the halt reason. Fail-open like
        ``record_cycle`` so a capture failure never changes the exit's behavior.

        Args:
            env_name: "equity" or "crypto".
            mode: Trading mode ("paper" or "live").
            cycle_ts: The canonical cycle timestamp (UTC).
            reason: Short halt-reason code (e.g. "circuit_breaker", "turbulence_halt").
            turbulence: Decision-time turbulence when available at the exit, else None.
            dry_run: Whether the halted cycle was a dry-run (tagged in the payload).

        Returns:
            The new ``cycle_id``, or ``None`` if capture failed.
        """
        try:
            payload = json.dumps(
                {"schema_version": _SCHEMA_VERSION, "halt_reason": reason, "dry_run": dry_run}
            )
            with self._db.connection() as conn:
                cycle_id = self._insert_cycle_row(
                    conn,
                    env_name=env_name,
                    mode=mode,
                    cycle_ts=cycle_ts,
                    deployed_iteration=None,
                    hmm_p_bull=None,
                    hmm_p_bear=None,
                    vix=None,
                    turbulence=turbulence,
                    active_event_ids=None,
                    payload=payload,
                )
            log.info("cycle_halt_captured", env=env_name, cycle_id=cycle_id, reason=reason)
            return cycle_id
        except Exception:
            self._alert_capture_failed(env_name)
            return None

    def _insert_cycle_row(
        self,
        conn: Any,
        *,
        env_name: str,
        mode: str,
        cycle_ts: datetime,
        deployed_iteration: int | None,
        hmm_p_bull: float | None,
        hmm_p_bear: float | None,
        vix: float | None,
        turbulence: float | None,
        active_event_ids: list[int] | None,
        payload: str,
    ) -> int:
        """Insert one inference_cycles row and return its cycle_id (shared writer)."""
        row = conn.execute(
            _INFERENCE_INSERT_SQL,
            (
                env_name,
                mode,
                cycle_ts,
                deployed_iteration,
                hmm_p_bull,
                hmm_p_bear,
                vix,
                turbulence,
                active_event_ids,
                payload,
            ),
        ).fetchone()
        if row is None:
            # RETURNING always yields a row on success; None means the write did not
            # land — treat as a capture failure (surfaced by the fail-open caller).
            raise DataError("inference_cycles insert returned no cycle_id")
        return int(row["cycle_id"])

    def _alert_capture_failed(self, env_name: str) -> None:
        """Log + alert a fail-open capture failure (a silent outage must not go unseen)."""
        log.warning("cycle_capture_failed", env=env_name, exc_info=True)
        if self._alerter is not None:
            self._alerter.send_alert(
                level="warning",
                title="Cycle capture failed",
                message=(
                    f"Inference-cycle capture failed for {env_name} — trading "
                    "continues, but this cycle was not recorded."
                ),
                environment=env_name,
            )

    def active_event_ids(self, cycle_ts: datetime) -> list[int]:
        """Return the ids of calendar events whose window brackets ``cycle_ts``.

        The ``window_start <= cycle_ts <= window_end`` filter is delegated to SQL.
        Fail-open: any query error yields an empty list (never raises).

        Args:
            cycle_ts: The canonical cycle timestamp (UTC).

        Returns:
            Active event ids, or ``[]`` on no match / query failure.
        """
        try:
            with self._db.connection() as conn:
                rows = conn.execute(_ACTIVE_EVENTS_SQL, (cycle_ts, cycle_ts)).fetchall()
            return [int(r["event_id"]) for r in rows]
        except Exception:
            log.warning("active_events_query_failed", exc_info=True)
            return []

    def active_model_ids(self, env_name: str) -> dict[str, str]:
        """Return ``{algorithm: model_id}`` for the environment's active models.

        Reads the cached ``models ⋈ training_runs`` spine (query issued once per
        env). When multiple active rows share an algorithm, the most recent (by
        ``created_at``) wins.

        Args:
            env_name: "equity" or "crypto".

        Returns:
            Mapping of algorithm to active model id (empty when none are active).
        """
        ids: dict[str, str] = {}
        for r in self._active_spine(env_name):
            algo = str(r["algorithm"])
            if algo not in ids:
                ids[algo] = str(r["model_id"])
        return ids

    def deployed_iteration(self, env_name: str) -> int | None:
        """Return the max iteration over the active-model spine (display only, A20).

        Args:
            env_name: "equity" or "crypto".

        Returns:
            The highest active-model iteration number, or ``None`` when none active.
        """
        iters = [
            int(r["iteration_number"])
            for r in self._active_spine(env_name)
            if r["iteration_number"] is not None
        ]
        return max(iters) if iters else None

    def _active_spine(self, env_name: str) -> list[dict[str, Any]]:
        """Query (and cache) the active ``models ⋈ training_runs`` rows for an env.

        A successful query — including one returning zero rows — is cached. A
        failure is not cached (retried next call) and yields an empty list.

        Args:
            env_name: "equity" or "crypto".

        Returns:
            List of ``{algorithm, model_id, iteration_number}`` rows.
        """
        if env_name in self._spine_cache:
            return self._spine_cache[env_name]
        try:
            with self._db.connection() as conn:
                rows = conn.execute(_ACTIVE_SPINE_SQL, ("active", env_name)).fetchall()
        except Exception:
            log.warning("active_spine_query_failed", env=env_name, exc_info=True)
            return []
        spine = [dict(r) for r in rows]
        self._spine_cache[env_name] = spine
        return spine
