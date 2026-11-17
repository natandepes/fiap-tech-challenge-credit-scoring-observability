import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from prometheus_client import CollectorRegistry, Counter, Gauge

from credit_scoring.config import (
    COUNTERFACTUAL_SUFFIX,
    DRIFT_ALERTS_PATH,
    DRIFT_SUMMARY_PATH,
    FEATURE_DRIFT_PATH,
    HEALTH_METRICS,
    MODEL_HEALTH_PATH,
    MONTH_COLUMN,
    PIPELINE_LOG_PATH,
)
from credit_scoring.observability.step import STEP_FAILED, STEP_SUCCEEDED

STEP_OUTCOMES = {STEP_SUCCEEDED: "succeeded", STEP_FAILED: "failed"}
INGEST_OUTCOMES = {"batch_accepted": "accepted", "batch_rejected": "rejected"}
CURATION_OUTCOMES = {
    "raw_rows": "raw",
    "exact_duplicates_removed": "duplicate",
    "rows_rejected_by_contract": "rejected",
    "curated_rows": "curated",
}
LAST_STEP_RUN_GAUGES = {
    "success": "1 if the latest run of the step succeeded, 0 if it failed",
    "duration_seconds": "Wall-clock duration of the latest run",
    "cpu_seconds": "CPU time used by the latest run",
    "peak_memory_bytes": "Peak resident memory of the process during the latest run",
    "rows_processed": "Rows reported by the latest run's events",
    "timestamp_seconds": "Unix time at which the latest run finished",
}
MONTHLY_HEALTH_GAUGES = {
    "rows": ("credit_production_scored_rows", "Applications scored in the month"),
    "drifted_features": ("credit_drifted_features", "Features with PSI above the drift threshold"),
    "prediction_psi": ("credit_prediction_psi", "PSI of default_probability vs. reference"),
    "target_psi": ("credit_target_psi", "PSI of the observed default label vs. reference"),
}
OBSERVED = "observed"
WITHOUT_CONCEPT_DRIFT = "simulated_without_concept_drift"


@dataclass(frozen=True)
class MonitoringInputs:
    events: list[dict]
    feature_drift: pd.DataFrame
    model_health: pd.DataFrame
    alerts: list[dict]
    baseline: dict

    @property
    def latest_month(self) -> str:
        return max(self.model_health[MONTH_COLUMN])


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as lines:
        return [json.loads(line) for line in lines if line.strip()]


def load_inputs(log_path: Path = PIPELINE_LOG_PATH) -> MonitoringInputs:
    return MonitoringInputs(
        events=read_events(log_path),
        feature_drift=pd.read_csv(FEATURE_DRIFT_PATH, dtype={MONTH_COLUMN: str}),
        model_health=pd.read_csv(MODEL_HEALTH_PATH, dtype={MONTH_COLUMN: str}),
        alerts=json.loads(DRIFT_ALERTS_PATH.read_text()),
        baseline=json.loads(DRIFT_SUMMARY_PATH.read_text())["reference_holdout_baseline"],
    )


def _events_named(events: list[dict], *names: str) -> list[dict]:
    return [event for event in events if event["event"] in names]


def _rows_in_run(events: list[dict], run_id: str) -> int:
    return sum(event.get("rows", 0) for event in events if event["run_id"] == run_id)


def _unix_time(event: dict) -> float:
    return datetime.fromisoformat(event["timestamp"]).timestamp()


def _last_run_values(step_event: dict, events: list[dict]) -> dict[str, float]:
    return {
        "success": float(step_event["event"] == STEP_SUCCEEDED),
        "duration_seconds": step_event["duration_seconds"],
        "cpu_seconds": step_event["cpu_seconds"],
        "peak_memory_bytes": step_event["peak_memory_bytes"],
        "rows_processed": _rows_in_run(events, step_event["run_id"]),
        "timestamp_seconds": _unix_time(step_event),
    }


def register_pipeline_metrics(registry: CollectorRegistry, events: list[dict]) -> None:
    step_events = _events_named(events, *STEP_OUTCOMES)
    runs = Counter(
        "credit_pipeline_step_runs",
        "Pipeline step executions by outcome",
        ["step", "status"],
        registry=registry,
    )
    for event in step_events:
        runs.labels(event["stage"], STEP_OUTCOMES[event["event"]]).inc()

    last_run = {
        name: Gauge(f"credit_pipeline_step_last_{name}", help_text, ["step"], registry=registry)
        for name, help_text in LAST_STEP_RUN_GAUGES.items()
    }
    latest_by_step = {event["stage"]: event for event in step_events}
    for step, step_event in latest_by_step.items():
        for name, value in _last_run_values(step_event, events).items():
            last_run[name].labels(step).set(value)


def _register_contract_failures(registry: CollectorRegistry, rejection: dict) -> None:
    report_path = Path(rejection["report"])
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    failures = Gauge(
        "credit_contract_check_failures",
        "Failure cases per data contract check in the latest rejected batch",
        ["batch", "contract", "column", "check"],
        registry=registry,
    )
    for failure in report["failures_by_check"]:
        failures.labels(
            report["batch"], report["contract"], failure["column"], failure["check"]
        ).set(failure["failures"])


def register_ingestion_metrics(registry: CollectorRegistry, events: list[dict]) -> None:
    batches = Counter(
        "credit_ingest_batches",
        "Batches gated by the data contract",
        ["outcome"],
        registry=registry,
    )
    rows = Counter(
        "credit_ingest_rows",
        "Rows in batches gated by the data contract",
        ["outcome"],
        registry=registry,
    )
    for outcome in INGEST_OUTCOMES.values():
        batches.labels(outcome)
        rows.labels(outcome)
    gated = _events_named(events, *INGEST_OUTCOMES)
    for event in gated:
        batches.labels(INGEST_OUTCOMES[event["event"]]).inc()
        rows.labels(INGEST_OUTCOMES[event["event"]]).inc(event["rows"])

    rejections = _events_named(events, "batch_rejected")
    if rejections:
        _register_contract_failures(registry, rejections[-1])

    curations = _events_named(events, "raw_dataset_curated")
    if curations:
        curation = Gauge(
            "credit_curation_rows",
            "Rows of the raw dataset by curation outcome",
            ["outcome"],
            registry=registry,
        )
        for field, outcome in CURATION_OUTCOMES.items():
            curation.labels(outcome).set(curations[-1][field])


def register_drift_metrics(registry: CollectorRegistry, inputs: MonitoringInputs) -> None:
    psi = Gauge(
        "credit_feature_psi",
        "PSI of a feature vs. reference",
        ["month", "feature"],
        registry=registry,
    )
    p_value = Gauge(
        "credit_feature_drift_p_value",
        "p-value of the KS / chi-square test vs. reference",
        ["month", "feature", "test"],
        registry=registry,
    )
    drifted = Gauge(
        "credit_feature_drifted",
        "1 when the feature's PSI crosses the drift threshold",
        ["month", "feature"],
        registry=registry,
    )
    for row in inputs.feature_drift.itertuples():
        psi.labels(row.month, row.feature).set(row.psi)
        p_value.labels(row.month, row.feature, row.test).set(row.p_value)
        drifted.labels(row.month, row.feature).set(int(row.drifted))

    alert_labels = ["month", "type", "subject", "severity"]
    alert_value = Gauge(
        "credit_drift_alert_value",
        "Observed value of a raised drift or degradation alert",
        alert_labels,
        registry=registry,
    )
    alert_threshold = Gauge(
        "credit_drift_alert_threshold",
        "Threshold crossed by a raised drift or degradation alert",
        alert_labels,
        registry=registry,
    )
    for alert in inputs.alerts:
        labels = [alert[name] for name in alert_labels]
        alert_value.labels(*labels).set(alert["value"])
        alert_threshold.labels(*labels).set(alert["threshold"])

    Gauge(
        "credit_monitoring_latest_month_info",
        "The most recent production month checked for drift",
        ["month"],
        registry=registry,
    ).labels(inputs.latest_month).set(1)


def register_model_metrics(registry: CollectorRegistry, inputs: MonitoringInputs) -> None:
    months = inputs.model_health.to_dict("records")
    for metric in HEALTH_METRICS:
        gauge = Gauge(
            f"credit_model_{metric}",
            f"Model {metric} per production month",
            ["month", "ground_truth"],
            registry=registry,
        )
        for month in months:
            gauge.labels(month[MONTH_COLUMN], OBSERVED).set(month[metric])
            gauge.labels(month[MONTH_COLUMN], WITHOUT_CONCEPT_DRIFT).set(
                month[f"{metric}{COUNTERFACTUAL_SUFFIX}"]
            )

    for column, (name, help_text) in MONTHLY_HEALTH_GAUGES.items():
        gauge = Gauge(name, help_text, ["month"], registry=registry)
        for month in months:
            gauge.labels(month[MONTH_COLUMN]).set(month[column])

    baseline = Gauge(
        "credit_model_baseline",
        "Model metric on the reference holdout",
        ["metric"],
        registry=registry,
    )
    for metric, value in inputs.baseline.items():
        baseline.labels(metric).set(value)


def register_scoring_metrics(registry: CollectorRegistry, events: list[dict]) -> None:
    score_runs = [
        event for event in _events_named(events, *STEP_OUTCOMES) if event["stage"] == "score"
    ]
    if not score_runs:
        return
    latency = Gauge(
        "credit_scoring_batch_latency_seconds",
        "Time to validate and score a monthly batch in the latest scoring run",
        ["month"],
        registry=registry,
    )
    run_id = score_runs[-1]["run_id"]
    for event in _events_named(events, "month_scored"):
        if event["run_id"] == run_id:
            latency.labels(event[MONTH_COLUMN]).set(event["latency_ms"] / 1000)


def build_registry(inputs: MonitoringInputs) -> CollectorRegistry:
    registry = CollectorRegistry()
    register_pipeline_metrics(registry, inputs.events)
    register_ingestion_metrics(registry, inputs.events)
    register_scoring_metrics(registry, inputs.events)
    register_drift_metrics(registry, inputs)
    register_model_metrics(registry, inputs)
    return registry
