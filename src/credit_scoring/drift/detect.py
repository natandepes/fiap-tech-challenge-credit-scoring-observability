import json
from dataclasses import dataclass

import joblib
import pandas as pd

from credit_scoring.config import (
    CONCEPT_DRIFT_TRUTH_PATH,
    COUNTERFACTUAL_SUFFIX,
    DRIFT_ALERTS_PATH,
    DRIFT_REPORTS_DIR,
    DRIFT_SUMMARY_PATH,
    FEATURE_DRIFT_PATH,
    HEALTH_METRICS,
    ID_COLUMN,
    MODEL_HEALTH_PATH,
    MODEL_PATH,
    MONTH_COLUMN,
    MONTHLY_PRODUCTION_DIR,
    PREDICTIONS_DIR,
    PROBABILITY_COLUMN,
    REFERENCE_PATH,
    SIMULATION_MANIFEST_PATH,
    TARGET_COLUMN,
)
from credit_scoring.drift.alerts import (
    data_drift_alerts,
    model_degradation_alerts,
    prediction_drift_alerts,
)
from credit_scoring.drift.evidently_reports import (
    data_drift_snapshot,
    model_health_snapshot,
    psi_by_column,
)
from credit_scoring.drift.scenario import PRODUCTION_MONTHS
from credit_scoring.drift.statistics import feature_drift_table
from credit_scoring.model.score import score
from credit_scoring.model.train import classification_metrics, split_train_holdout
from credit_scoring.observability.step import tracked_step
from credit_scoring.pipeline_log import get_pipeline_logger

EVIDENTLY_DIR = DRIFT_REPORTS_DIR / "evidently"
HTML_REPORT_MONTHS = (PRODUCTION_MONTHS[0], PRODUCTION_MONTHS[-1])
COUNTERFACTUAL_LABEL = "loan_status_without_concept_drift"


@dataclass(frozen=True)
class MonthCheck:
    month: str
    feature_drift: pd.DataFrame
    health: dict
    alerts: list[dict]


def reference_holdout_with_predictions(model, reference: pd.DataFrame) -> pd.DataFrame:
    _, holdout = split_train_holdout(reference)
    holdout = holdout.reset_index(drop=True)
    return holdout.merge(score(model, holdout), on=ID_COLUMN)


def scored_month(month: str) -> pd.DataFrame:
    batch = pd.read_parquet(MONTHLY_PRODUCTION_DIR / f"{month}.parquet")
    predictions = pd.read_parquet(PREDICTIONS_DIR / f"{month}.parquet")
    return batch.merge(predictions.drop(columns=MONTH_COLUMN), on=ID_COLUMN, validate="1:1")


def health_metrics(labels: pd.Series, default_probability: pd.Series, suffix: str = "") -> dict:
    metrics = classification_metrics(labels, default_probability)
    return {f"{name}{suffix}": metrics[name] for name in HEALTH_METRICS}


def counterfactual_labels(month: str, current: pd.DataFrame) -> pd.Series:
    truth = pd.read_parquet(CONCEPT_DRIFT_TRUTH_PATH, filters=[(MONTH_COLUMN, "==", month)])
    return current[[ID_COLUMN]].merge(truth, on=ID_COLUMN, validate="1:1")[COUNTERFACTUAL_LABEL]


def save_html(month: str, data_drift, model_health) -> None:
    if month not in HTML_REPORT_MONTHS:
        return
    EVIDENTLY_DIR.mkdir(parents=True, exist_ok=True)
    data_drift.save_html(str(EVIDENTLY_DIR / f"data_drift_{month}.html"))
    model_health.save_html(str(EVIDENTLY_DIR / f"model_health_{month}.html"))


def check_month(
    month: str, reference: pd.DataFrame, holdout: pd.DataFrame, baseline: dict
) -> MonthCheck:
    current = scored_month(month)
    data_drift = data_drift_snapshot(reference, current)
    model_health = model_health_snapshot(holdout, current, baseline)
    save_html(month, data_drift, model_health)

    feature_psi = psi_by_column(data_drift)
    prediction_psi = psi_by_column(model_health)[PROBABILITY_COLUMN]
    feature_drift = feature_drift_table(reference, current, feature_psi)
    probability = current[PROBABILITY_COLUMN]
    health = {
        MONTH_COLUMN: month,
        "rows": len(current),
        "drifted_features": int(feature_drift["drifted"].sum()),
        "prediction_psi": round(prediction_psi, 4),
        "target_psi": round(feature_psi[TARGET_COLUMN], 4),
        **health_metrics(current[TARGET_COLUMN], probability),
        **health_metrics(counterfactual_labels(month, current), probability, COUNTERFACTUAL_SUFFIX),
    }
    alerts = [
        *data_drift_alerts(month, feature_drift),
        *prediction_drift_alerts(month, prediction_psi),
        *model_degradation_alerts(month, health, baseline),
    ]
    return MonthCheck(month, feature_drift.assign(**{MONTH_COLUMN: month}), health, alerts)


def _flagged_features(feature_drift: pd.DataFrame, verdict: str) -> set[str]:
    return set(feature_drift.loc[feature_drift[verdict], "feature"])


def detection_against_injection(latest: MonthCheck) -> dict:
    injected = set(json.loads(SIMULATION_MANIFEST_PATH.read_text())["data_drifted_features"])
    comparison = {"month": latest.month, "injected": sorted(injected)}
    for verdict in ("drifted", "significant_change"):
        flagged = _flagged_features(latest.feature_drift, verdict)
        comparison[verdict] = {
            "flagged": sorted(flagged),
            "missed": sorted(injected - flagged),
            "false_positives": sorted(flagged - injected),
        }
    return comparison


def write_outputs(checks: list[MonthCheck], baseline: dict) -> dict:
    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_drift = pd.concat([check.feature_drift for check in checks], ignore_index=True)
    feature_drift.insert(0, MONTH_COLUMN, feature_drift.pop(MONTH_COLUMN))
    feature_drift.to_csv(FEATURE_DRIFT_PATH, index=False)
    pd.DataFrame([check.health for check in checks]).to_csv(MODEL_HEALTH_PATH, index=False)
    alerts = [alert for check in checks for alert in check.alerts]
    DRIFT_ALERTS_PATH.write_text(json.dumps(alerts, indent=2))

    latest = checks[-1]
    drifted = latest.feature_drift[latest.feature_drift["drifted"]]
    summary = {
        "reference_holdout_baseline": baseline,
        "latest_month": latest.month,
        "drifted_features_latest_month": drifted[["feature", "psi", "psi_band"]].to_dict("records"),
        "alerts_by_month": {check.month: len(check.alerts) for check in checks},
        "detection_against_injection": detection_against_injection(latest),
        "evidently_reports": sorted(str(path.name) for path in EVIDENTLY_DIR.glob("*.html")),
    }
    DRIFT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


@tracked_step("drift")
def detect_drift() -> None:
    log = get_pipeline_logger("drift")
    model = joblib.load(MODEL_PATH)
    reference = pd.read_parquet(REFERENCE_PATH)
    holdout = reference_holdout_with_predictions(model, reference)
    baseline = health_metrics(holdout[TARGET_COLUMN], holdout[PROBABILITY_COLUMN])
    checks = []
    for month in PRODUCTION_MONTHS:
        check = check_month(month, reference, holdout, baseline)
        checks.append(check)
        for alert in check.alerts:
            log.warning("drift_alert", extra=alert)
        log.info(
            "drift_check_completed",
            extra={**check.health, "alerts": len(check.alerts)},
        )
    summary = write_outputs(checks, baseline)
    log.info("drift_report_written", extra={"path": str(DRIFT_REPORTS_DIR), **summary})


if __name__ == "__main__":
    detect_drift()
