"""Tests for CycleRecorder — per-cycle regime + per-algo proposal capture.

Uses the ``make_mock_db`` MagicMock fixture (no live PostgreSQL): the tests
verify the SQL issued, fail-open behaviour, and active-event window stamping.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from swingrl.execution.cycle_recorder import AlgoProposal, CycleRecorder, RegimeStamp

from tests.conftest import make_mock_db


def _recorder(db: Any, alerter: Any, config: Any) -> CycleRecorder:
    """Build a CycleRecorder from mocked collaborators."""
    return CycleRecorder(db=db, config=config, alerter=alerter)


def _regime() -> RegimeStamp:
    """A fully-populated RegimeStamp for insert tests."""
    return RegimeStamp(
        hmm_p_bull=0.7,
        hmm_p_bear=0.3,
        vix=18.5,
        turbulence=4.2,
        active_event_ids=[11, 22],
    )


class TestRecordCycle:
    """record_cycle issues the inference_cycles + cycle_algo_proposals writes."""

    def test_inserts_inference_cycle_then_one_proposal_per_algo(
        self, exec_config: Any, mock_alerter: Any
    ) -> None:
        """Task 9 (a): inference_cycles INSERT (RETURNING) then one proposal INSERT each."""
        db, conn = make_mock_db(fetchone_returns=[{"cycle_id": 42}])
        recorder = _recorder(db, mock_alerter, exec_config)

        proposals = [
            AlgoProposal(
                algorithm="ppo",
                model_id="equity-ppo-iter4",
                raw_actions={"SPY": 0.1, "QQQ": -0.2},
                weight_in_blend_frac=0.5,
            ),
            AlgoProposal(
                algorithm="a2c",
                model_id="equity-a2c-iter4",
                raw_actions={"SPY": 0.3, "QQQ": 0.0},
                weight_in_blend_frac=0.5,
            ),
        ]

        cycle_id = recorder.record_cycle(
            env_name="equity",
            mode="paper",
            cycle_ts=datetime(2026, 7, 16, 20, 0, tzinfo=UTC),
            regime=_regime(),
            raw_actions={"ppo": [0.1, -0.2], "a2c": [0.3, 0.0]},
            target_weights={"SPY": 0.2, "QQQ": 0.1},
            proposals=proposals,
            deployed_iteration=4,
        )

        assert cycle_id == 42

        sqls = [call.args[0] for call in conn.execute.call_args_list]
        # First write: the inference_cycles row with a RETURNING clause.
        assert "inference_cycles" in sqls[0]
        assert "RETURNING" in sqls[0].upper()
        # Then exactly one cycle_algo_proposals INSERT per proposal.
        proposal_sqls = [s for s in sqls if "cycle_algo_proposals" in s]
        assert len(proposal_sqls) == len(proposals)

        # The blended_actions payload carries schema_version=1 and both sub-maps.
        cycle_params = conn.execute.call_args_list[0].args[1]
        payload = next(
            json.loads(p) for p in cycle_params if isinstance(p, str) and "schema_version" in p
        )
        assert payload["schema_version"] == 1
        assert payload["raw"] == {"ppo": [0.1, -0.2], "a2c": [0.3, 0.0]}
        assert payload["target_weights_frac"] == {"SPY": 0.2, "QQQ": 0.1}

    def test_proposal_payload_carries_schema_version(
        self, exec_config: Any, mock_alerter: Any
    ) -> None:
        """Task 9: each proposed_actions JSONB payload is versioned."""
        db, conn = make_mock_db(fetchone_returns=[{"cycle_id": 7}])
        recorder = _recorder(db, mock_alerter, exec_config)

        recorder.record_cycle(
            env_name="equity",
            mode="paper",
            cycle_ts=datetime(2026, 7, 16, 20, 0, tzinfo=UTC),
            regime=_regime(),
            raw_actions={"ppo": [0.1]},
            target_weights={"SPY": 0.2},
            proposals=[
                AlgoProposal(
                    algorithm="ppo",
                    model_id="m1",
                    raw_actions={"SPY": 0.1},
                    weight_in_blend_frac=1.0,
                )
            ],
            deployed_iteration=None,
        )

        proposal_call = next(
            c for c in conn.execute.call_args_list if "cycle_algo_proposals" in c.args[0]
        )
        payload = next(
            json.loads(p)
            for p in proposal_call.args[1]
            if isinstance(p, str) and "schema_version" in p
        )
        assert payload == {"schema_version": 1, "raw": {"SPY": 0.1}}

    def test_returns_none_and_does_not_raise_on_db_error(
        self, exec_config: Any, mock_alerter: Any
    ) -> None:
        """Task 9 (b): a DB failure is swallowed (fail-open) and alerted, never raised."""
        db, conn = make_mock_db()
        conn.execute.side_effect = RuntimeError("connection reset")
        recorder = _recorder(db, mock_alerter, exec_config)

        result = recorder.record_cycle(
            env_name="equity",
            mode="paper",
            cycle_ts=datetime(2026, 7, 16, 20, 0, tzinfo=UTC),
            regime=_regime(),
            raw_actions={"ppo": [0.1]},
            target_weights={"SPY": 0.2},
            proposals=[],
            deployed_iteration=1,
        )

        assert result is None
        mock_alerter.send_alert.assert_called_once()
        kwargs = mock_alerter.send_alert.call_args.kwargs
        assert kwargs["level"] == "warning"
        assert kwargs["title"] == "Cycle capture failed"


class TestActiveEventStamping:
    """active_event_ids delegates the window filter to SQL and returns the ids."""

    def test_events_in_window_are_returned(self, exec_config: Any, mock_alerter: Any) -> None:
        """Task 9 (c): events with window_start <= cycle_ts <= window_end are stamped."""
        db, conn = make_mock_db(fetchall_returns=[[{"event_id": 11}, {"event_id": 22}]])
        recorder = _recorder(db, mock_alerter, exec_config)
        cycle_ts = datetime(2026, 7, 16, 20, 0, tzinfo=UTC)

        ids = recorder.active_event_ids(cycle_ts)

        assert ids == [11, 22]
        sql = conn.execute.call_args.args[0]
        assert "calendar_events" in sql
        assert "window_start <= %s" in sql
        assert "window_end >= %s" in sql
        # cycle_ts is bound as the window comparison value.
        assert cycle_ts in conn.execute.call_args.args[1]

    def test_returns_empty_on_db_error(self, exec_config: Any, mock_alerter: Any) -> None:
        """Fail-open: an events-query failure yields no ids, never raises."""
        db, conn = make_mock_db()
        conn.execute.side_effect = RuntimeError("no such table")
        recorder = _recorder(db, mock_alerter, exec_config)

        assert recorder.active_event_ids(datetime(2026, 7, 16, 20, 0, tzinfo=UTC)) == []


class TestActiveModelSpine:
    """active_model_ids / deployed_iteration read the active models ⋈ training_runs spine."""

    def test_active_model_ids_maps_algo_to_model_id(
        self, exec_config: Any, mock_alerter: Any
    ) -> None:
        """Task 9: algo -> model_id for the active spine rows."""
        db, conn = make_mock_db(
            fetchall_returns=[
                [
                    {"algorithm": "ppo", "model_id": "m-ppo", "iteration_number": 5},
                    {"algorithm": "a2c", "model_id": "m-a2c", "iteration_number": 4},
                ]
            ]
        )
        recorder = _recorder(db, mock_alerter, exec_config)

        assert recorder.active_model_ids("equity") == {"ppo": "m-ppo", "a2c": "m-a2c"}
        sql = conn.execute.call_args.args[0]
        assert "models" in sql
        assert "training_runs" in sql
        assert "active" in conn.execute.call_args.args[1]
        assert "equity" in conn.execute.call_args.args[1]

    def test_deployed_iteration_is_max_and_shares_one_query(
        self, exec_config: Any, mock_alerter: Any
    ) -> None:
        """Task 9: deployed_iteration = max iteration; spine is cached per call."""
        db, conn = make_mock_db(
            fetchall_returns=[
                [
                    {"algorithm": "ppo", "model_id": "m-ppo", "iteration_number": 5},
                    {"algorithm": "a2c", "model_id": "m-a2c", "iteration_number": 4},
                ]
            ]
        )
        recorder = _recorder(db, mock_alerter, exec_config)

        # First call primes the cache; the second reuses it (one query total).
        recorder.active_model_ids("equity")
        assert recorder.deployed_iteration("equity") == 5
        assert conn.execute.call_count == 1

    def test_deployed_iteration_none_when_no_active_models(
        self, exec_config: Any, mock_alerter: Any
    ) -> None:
        """Task 9: no active spine rows -> deployed_iteration None, empty id map."""
        db, _conn = make_mock_db(fetchall_returns=[[]])
        recorder = _recorder(db, mock_alerter, exec_config)

        assert recorder.deployed_iteration("equity") is None
        assert recorder.active_model_ids("equity") == {}
