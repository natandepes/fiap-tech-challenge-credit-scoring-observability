import json

import pandas as pd
import pytest

from credit_scoring import pipeline_log
from credit_scoring.observability import export
from credit_scoring.observability.metrics import (
    OBSERVED,
    WITHOUT_CONCEPT_DRIFT,
    MonitoringInputs,
    build_registry,
    read_events,
)
from credit_scoring.observability.step import tracked_step


def logged_events(stage: str) -> list[dict]:
    return [
        event for event in read_events(pipeline_log.PIPELINE_LOG_PATH) if event["stage"] == stage
    ]


def test_successful_step_logs_start_and_success_under_one_run_id():
    @tracked_step("test_success")
    def step():
        pipeline_log.get_pipeline_logger("test_success").info("work_done", extra={"rows": 3})
        return "result"

    assert step() == "result"
    events = logged_events("test_success")
    assert [event["event"] for event in events] == ["step_started", "work_done", "step_succeeded"]
    assert len({event["run_id"] for event in events}) == 1
    assert {"duration_seconds", "cpu_seconds", "peak_memory_bytes"} <= set(events[-1])


def test_failing_step_logs_failure_and_reraises():
    @tracked_step("test_failure")
    def step():
        raise ValueError("broken transformation")

    with pytest.raises(ValueError):
        step()
    failure = logged_events("test_failure")[-1]
    assert failure["event"] == "step_failed"
    assert failure["error_type"] == "ValueError"
    assert failure["error"] == "broken transformation"


def step_event(stage: str, event: str, run_id: str) -> dict:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "stage": stage,
        "event": event,
        "run_id": run_id,
        "duration_seconds": 1.5,
        "cpu_seconds": 1.0,
        "peak_memory_bytes": 1024,
    }


@pytest.fixture
def rejection_report(tmp_path):
    path = tmp_path / "bad_rejection.json"
    path.write_text(
        json.dumps(
            {
                "batch": "bad",
                "contract": "scoring_request_contract",
                "failures_by_check": [
                    {"column": "person_age", "check": "in_range(18, 100)", "failures": 6}
                ],
            }
        )
    )
    return path


@pytest.fixture
def monitoring_inputs(rejection_report) -> MonitoringInputs:
    events = [
        step_event("score", "step_succeeded", "first"),
        {
            "stage": "score",
            "event": "month_scored",
            "run_id": "second",
            "rows": 10,
            "month": "2026-01",
            "latency_ms": 50.0,
        },
        {
            "stage": "score",
            "event": "month_scored",
            "run_id": "second",
            "rows": 20,
            "month": "2026-02",
            "latency_ms": 70.0,
        },
        step_event("score", "step_failed", "second"),
        {"stage": "ingest", "event": "batch_accepted", "run_id": "third", "rows": 200},
        {
            "stage": "ingest",
            "event": "batch_rejected",
            "run_id": "fourth",
            "rows": 204,
            "report": str(rejection_report),
        },
    ]
    health_columns = {
        metric: [0.9, 0.8]
        for name in (
            "roc_auc",
            "recall",
            "precision",
            "brier_score",
            "observed_default_rate",
            "mean_default_probability",
            "predicted_default_rate",
        )
        for metric in (name, f"{name}_without_concept_drift")
    }
    model_health = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02"],
            "rows": [10, 20],
            "drifted_features": [0, 1],
            "prediction_psi": [0.01, 0.2],
            "target_psi": [0.0, 0.3],
            **health_columns,
        }
    )
    model_health["recall_without_concept_drift"] = [0.9, 0.85]
    feature_drift = pd.DataFrame(
        {
            "month": ["2026-02"],
            "feature": ["loan_int_rate"],
            "psi": [0.3],
            "test": ["kolmogorov_smirnov"],
            "p_value": [1e-6],
            "drifted": [True],
        }
    )
    alerts = [
        {
            "month": "2026-02",
            "type": "data_drift",
            "subject": "loan_int_rate",
            "value": 0.3,
            "threshold": 0.1,
            "severity": "critical",
        }
    ]
    return MonitoringInputs(events, feature_drift, model_health, alerts, {"roc_auc": 0.93})


def test_pipeline_metrics_reflect_the_latest_run_of_each_step(monitoring_inputs):
    registry = build_registry(monitoring_inputs)
    step = {"step": "score"}
    assert registry.get_sample_value("credit_pipeline_step_last_success", step) == 0
    assert registry.get_sample_value("credit_pipeline_step_last_rows_processed", step) == 30
    assert (
        registry.get_sample_value(
            "credit_pipeline_step_runs_total", {**step, "status": "succeeded"}
        )
        == 1
    )
    assert registry.get_sample_value(
        "credit_scoring_batch_latency_seconds", {"month": "2026-02"}
    ) == pytest.approx(0.07)


def test_ingestion_metrics_count_batches_and_contract_failures(monitoring_inputs):
    registry = build_registry(monitoring_inputs)
    assert registry.get_sample_value("credit_ingest_batches_total", {"outcome": "rejected"}) == 1
    assert registry.get_sample_value("credit_ingest_rows_total", {"outcome": "accepted"}) == 200
    failures = {
        "batch": "bad",
        "contract": "scoring_request_contract",
        "column": "person_age",
        "check": "in_range(18, 100)",
    }
    assert registry.get_sample_value("credit_contract_check_failures", failures) == 6


def test_drift_and_model_metrics_are_labelled_by_month(monitoring_inputs):
    registry = build_registry(monitoring_inputs)
    feature = {"month": "2026-02", "feature": "loan_int_rate"}
    assert registry.get_sample_value("credit_feature_psi", feature) == 0.3
    assert registry.get_sample_value("credit_feature_drifted", feature) == 1
    alert = {
        "month": "2026-02",
        "type": "data_drift",
        "subject": "loan_int_rate",
        "severity": "critical",
    }
    assert registry.get_sample_value("credit_drift_alert_value", alert) == 0.3
    assert (
        registry.get_sample_value("credit_monitoring_latest_month_info", {"month": "2026-02"}) == 1
    )
    recall = "credit_model_recall"
    assert registry.get_sample_value(recall, {"month": "2026-02", "ground_truth": OBSERVED}) == 0.8
    assert (
        registry.get_sample_value(
            recall, {"month": "2026-02", "ground_truth": WITHOUT_CONCEPT_DRIFT}
        )
        == 0.85
    )


def test_unreachable_pushgateway_is_reported_without_failing(monitoring_inputs, monkeypatch):
    monkeypatch.setattr(export, "PUSHGATEWAY_URL", "localhost:1")
    log = pipeline_log.get_pipeline_logger("test_monitor")
    assert export.push(build_registry(monitoring_inputs), log) is False
    assert logged_events("test_monitor")[-1]["event"] == "metrics_push_failed"
