import pandas as pd

from credit_scoring.config import (
    CALIBRATION_GAP_ALERT,
    PSI_DRIFT_THRESHOLD,
    PSI_MAJOR_DRIFT_THRESHOLD,
    RECALL_DROP_ALERT,
    ROC_AUC_DROP_ALERT,
)

WARNING = "warning"
CRITICAL = "critical"


def _alert(month: str, kind: str, subject: str, value: float, threshold: float, severity: str):
    return {
        "month": month,
        "type": kind,
        "subject": subject,
        "value": round(float(value), 4),
        "threshold": threshold,
        "severity": severity,
    }


def _psi_severity(psi: float) -> str:
    return CRITICAL if psi >= PSI_MAJOR_DRIFT_THRESHOLD else WARNING


def data_drift_alerts(month: str, feature_drift: pd.DataFrame) -> list[dict]:
    drifted = feature_drift[feature_drift["drifted"]]
    return [
        _alert(
            month, "data_drift", row.feature, row.psi, PSI_DRIFT_THRESHOLD, _psi_severity(row.psi)
        )
        for row in drifted.itertuples()
    ]


def prediction_drift_alerts(month: str, prediction_psi: float) -> list[dict]:
    if prediction_psi < PSI_DRIFT_THRESHOLD:
        return []
    return [
        _alert(
            month,
            "prediction_drift",
            "default_probability",
            prediction_psi,
            PSI_DRIFT_THRESHOLD,
            _psi_severity(prediction_psi),
        )
    ]


def model_degradation_alerts(month: str, health: dict, baseline: dict) -> list[dict]:
    alerts = []
    roc_auc_drop = baseline["roc_auc"] - health["roc_auc"]
    if roc_auc_drop >= ROC_AUC_DROP_ALERT:
        alerts.append(
            _alert(month, "performance", "roc_auc_drop", roc_auc_drop, ROC_AUC_DROP_ALERT, CRITICAL)
        )
    recall_drop = baseline["recall"] - health["recall"]
    if recall_drop >= RECALL_DROP_ALERT:
        alerts.append(
            _alert(month, "performance", "recall_drop", recall_drop, RECALL_DROP_ALERT, CRITICAL)
        )
    calibration_gap = health["observed_default_rate"] - health["mean_default_probability"]
    if abs(calibration_gap) >= CALIBRATION_GAP_ALERT:
        alerts.append(
            _alert(
                month,
                "calibration",
                "observed_minus_predicted_default_rate",
                calibration_gap,
                CALIBRATION_GAP_ALERT,
                CRITICAL,
            )
        )
    return alerts
