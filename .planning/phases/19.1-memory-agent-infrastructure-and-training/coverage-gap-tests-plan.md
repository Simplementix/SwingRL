# Coverage Gap Test Plan
**Goal:** Reach 85% coverage (currently 82.88%, need ~135 lines)
**Branch:** gsd/phase-19.1-memory-agent-infrastructure-and-training

## Files to Create

### 1. `tests/memory/test_curriculum.py` (NEW)
Cover `src/swingrl/memory/training/curriculum.py` — currently 0%

#### TestCurriculumBuild
- `test_build_normalizes_weights` — non-zero weights normalize to sum=1.0 (prob array)
- `test_build_all_zero_weights_uses_uniform` — zero weights → uniform 1/n distribution
- `test_build_clamps_bar_ranges_to_dataset` — start/end bars clamped to [0, total_bars]
- `test_build_stores_labels` — _labels list matches window label order

#### TestCurriculumSampleDate
- `test_sample_date_within_window_range` — sampled bar falls within the selected window
- `test_sample_date_no_probs_falls_back_to_uniform` — _probs=None → _uniform_date()
- `test_last_sampled_label_tracks_window` — property returns most recently sampled label
- `test_uniform_date_in_valid_range` — _uniform_date() returns int in [0, total_bars)
- `test_uniform_date_zero_bars_returns_zero` — total_bars=0 → 0

#### TestCurriculumValidation
- `test_validate_raises_if_crisis_pct_exceeds_50pct` — ValueError when crisis_bars/total > 0.50
- `test_validate_warns_if_window_too_short` — warns but doesn't raise for < MIN_WINDOW_YEARS
- `test_validate_warns_crisis_below_min_pct` — warns if crisis exists but < 5%
- `test_validate_empty_windows_passes` — empty list is valid
- `test_validate_detects_crash_label_as_crisis` — "crash" in label counts as crisis

#### TestCurriculumStorePerformance
- `test_store_performance_calls_ingest_training` — calls client.ingest_training() with payload

#### TestCurriculumFromDateRanges
- `test_from_date_ranges_converts_dates_to_bar_indices` — start/end dates map to correct indices
- `test_from_date_ranges_handles_missing_date_gracefully` — unknown dates fall back to 0/len
- `test_from_date_ranges_equity_bars_per_year_252` — equity uses 252 bars/year
- `test_from_date_ranges_crypto_bars_per_year_2191` — crypto uses 2191 bars/year
- `test_from_date_ranges_can_call_build_immediately` — returned sampler is ready to build()

---

### 2. `tests/memory/test_meta_orchestrator.py` (NEW)
Cover `src/swingrl/memory/training/meta_orchestrator.py` — currently 0%

#### TestMetaOrchestratorColdStart
- `test_query_run_config_returns_empty_below_min_runs` — < 3 runs returns {}
- `test_query_run_config_calls_api_after_cold_start` — ≥ 3 runs makes HTTP POST
- `test_query_run_config_returns_empty_on_api_failure` — connection error → {}
- `test_query_run_config_returns_empty_on_timeout` — timeout → {}

#### TestMetaOrchestratorRunHistory
- `test_get_run_history_returns_completed_runs` — queries training_runs WHERE run_type='completed'
- `test_get_run_history_returns_empty_on_missing_table` — missing table → []
- `test_get_run_history_returns_empty_on_connection_error` — exception → []
- `test_get_run_history_filters_by_env_and_algo` — only returns matching env+algo rows

#### TestMetaOrchestratorRegimeVector
- `test_current_regime_vector_queries_hmm_state_history` — fetches from hmm_state_history
- `test_current_regime_vector_computes_sideways_as_remainder` — p_sideways = 1 - bull - bear - crisis
- `test_current_regime_vector_uses_defaults_on_no_row` — None row → {bull:0.33, bear:0.33, crisis:0.17, sideways:0.17}
- `test_current_regime_vector_handles_null_columns` — None columns treated as 0.33/0.17
- `test_current_regime_vector_uses_defaults_on_exception` — exception → defaults

#### TestMetaOrchestratorFinalMetrics
- `test_compute_final_metrics_returns_zero_placeholders` — all four values are 0.0
- `test_compute_final_metrics_returns_all_required_keys` — has final_sharpe, mdd, sortino, mean_reward

#### TestMetaOrchestratorBuildSummaryText
- `test_build_run_summary_text_contains_run_id` — run_id appears in output string
- `test_build_run_summary_text_contains_dominant_regime` — max(regime_vector) key appears
- `test_build_run_summary_text_uses_default_weights_if_none` — None weights → {profit:0.50, sharpe:0.25, ...}
- `test_build_run_summary_text_contains_timestamps` — ISO timestamps appear in output

#### TestMetaOrchestratorGenerateRunId
- `test_generate_run_id_format` — produces "{env}_{algo}_{YYYYMMDDTHHMMSSz}" format
- `test_generate_run_id_is_unique_per_call` — two calls produce different timestamps

---

### 3. `tests/memory/test_epoch_callback_extended.py` (NEW)
Cover uncovered lines in `epoch_callback.py` — currently 48%
(Lines: 115-134, 152-158, 169-171, 202-220, 244-267, 280-310, 343-354)

#### TestEpochCallbackShouldStore
- `test_should_store_every_5th_epoch` — EPOCH_STORE_CADENCE (5) returns True with None
- `test_should_store_not_on_non_cadence_epoch` — epoch 3 → False, None
- `test_should_store_kl_spike` — approx_kl > 0.02 → True, "kl_spike"
- `test_should_store_mdd_breach` — rolling_mdd < -0.08 → True, "mdd_breach"
- `test_should_store_kl_boundary` — approx_kl == 0.02 (not >) → False
- `test_should_store_mdd_boundary` — rolling_mdd == -0.08 (not <) → False

#### TestEpochCallbackCollectMetrics
- `test_collect_metrics_pulls_from_logger` — mean_reward, policy_loss, approx_kl from logger
- `test_collect_metrics_missing_keys_default_to_zero` — absent logger keys → 0.0
- `test_collect_metrics_calls_wrapper_rolling_methods` — rolling_sharpe(), rolling_mdd(), rolling_win_rate() called
- `test_collect_metrics_includes_reward_weights` — weights dict present in output

#### TestEpochCallbackIngestSnapshot
- `test_ingest_epoch_snapshot_calls_ingest_training` — client.ingest_training() called with text
- `test_ingest_epoch_snapshot_text_contains_run_id` — run_id in ingested text

#### TestEpochCallbackAdjustmentTrigger
- `test_ingest_adjustment_trigger_sets_pending_adjustment` — _pending_adjustment dict populated
- `test_ingest_adjustment_trigger_records_sharpe_at_trigger` — _sharpe_at_trigger set
- `test_ingest_adjustment_trigger_calls_ingest_training` — client.ingest_training() called

#### TestEpochCallbackResolvePendingAdjustment
- `test_resolve_pending_computes_sharpe_delta` — current_sharpe - sharpe_at_trigger
- `test_resolve_pending_computes_mdd_delta` — current_mdd - mdd_at_trigger
- `test_resolve_pending_effective_when_sharpe_improves` — sharpe_delta > 0 → effective=True
- `test_resolve_pending_effective_when_mdd_improves` — mdd_delta > 0 → effective=True
- `test_resolve_pending_clears_pending_adjustment` — _pending_adjustment = None after resolve
- `test_resolve_pending_calls_ingest_training` — outcome text ingested

---

## Implementation Notes

### Shared fixtures needed (add to tests/memory/conftest.py or inline)
```python
def _make_mock_memory_client():
    client = MagicMock()
    client._base_url = "http://localhost:8889"
    client.ingest_training.return_value = True
    return client

def _make_mock_wrapper(n_envs=1):
    mock = MagicMock()
    mock.num_envs = n_envs
    mock.observation_space = MagicMock()
    mock.action_space = MagicMock()
    mock.rolling_sharpe.return_value = 1.2
    mock.rolling_mdd.return_value = -0.05
    mock.rolling_win_rate.return_value = 0.55
    mock.weights = {"profit": 0.50, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.10}
    return mock
```

### DuckDB fixtures (for meta_orchestrator tests)
Use `duckdb.connect(":memory:")` to create in-memory DB with test data.
Create tables: `training_runs`, `hmm_state_history`.

### Curriculum window fixture
```python
WINDOWS = [
    {"label": "2022_bear", "start_bar": 0, "end_bar": 500, "weight": 2.0},
    {"label": "2020_crisis", "start_bar": 500, "end_bar": 700, "weight": 1.0},
    {"label": "2023_bull", "start_bar": 700, "end_bar": 1000, "weight": 1.5},
]
```

## Expected Coverage Gain
- test_curriculum.py: ~90 new lines (curriculum.py 0%→81%)
- test_meta_orchestrator.py: ~65 new lines (meta_orchestrator.py 0%→62%)
- test_epoch_callback_extended.py: ~40 new lines (epoch_callback.py 48%→88%)
- **Total: ~195 lines → 82.88% + (195/6419) ≈ 85.9%**

## Commit Message
```
test(19.1): add TRAIN-08/09/10 tests for curriculum, meta_orchestrator, epoch_callback

Covers 3 previously untested modules to reach ≥85% coverage threshold:
- TRAIN-08: MemoryCurriculumSampler (build, sampling, validation, from_date_ranges)
- TRAIN-09: MetaTrainingOrchestrator (cold-start, run history, regime vector, summary)
- TRAIN-10: MemoryEpochCallback extended (should_store, collect_metrics, two-pass tracking)
```
