import json

import joblib
import pandas as pd

from credit_scoring.config import (
    AGE_BANDS,
    DISPARATE_IMPACT_MIN_RATIO,
    FAIRNESS_AUDIT_PATH,
    FAIRNESS_BY_GROUP_PATH,
    FALSE_REJECTION_GAP_ALERT,
    MODEL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    PROXY_CORRELATION_ALERT,
    REFERENCE_PATH,
    TARGET_COLUMN,
)
from credit_scoring.drift.detect import reference_holdout_with_predictions, scored_month
from credit_scoring.drift.scenario import PRODUCTION_MONTHS
from credit_scoring.observability.step import tracked_step
from credit_scoring.pipeline_log import get_pipeline_logger

PROTECTED_ATTRIBUTE = "person_age"
GROUP_COLUMN = "age_group"
REFERENCE_HOLDOUT = "reference_holdout"


def age_group(age: pd.Series) -> pd.Series:
    groups = pd.Series(pd.NA, index=age.index, dtype="string")
    for name, (low, high) in AGE_BANDS.items():
        groups[age.between(low, high)] = name
    return groups


def _rate(flags: pd.Series) -> float:
    return round(float(flags.mean()), 4) if len(flags) else float("nan")


def group_outcomes(group: pd.DataFrame) -> dict:
    rejected = group[PREDICTION_COLUMN] == 1
    defaulted = group[TARGET_COLUMN] == 1
    return {
        "rows": len(group),
        "observed_default_rate": _rate(defaulted),
        "mean_default_probability": _rate(group[PROBABILITY_COLUMN]),
        "approval_rate": _rate(~rejected),
        "false_rejection_rate": _rate(rejected[~defaulted]),
        "missed_default_rate": _rate(~rejected[defaulted]),
    }


def outcomes_by_age_group(scored: pd.DataFrame, dataset: str) -> pd.DataFrame:
    groups = age_group(scored[PROTECTED_ATTRIBUTE])
    rows = [
        {"dataset": dataset, GROUP_COLUMN: name, **group_outcomes(scored[groups == name])}
        for name in AGE_BANDS
    ]
    return pd.DataFrame(rows)


def disparity_summary(by_group: pd.DataFrame) -> dict:
    approval = by_group["approval_rate"]
    false_rejection = by_group["false_rejection_rate"]
    disparate_impact = round(float(approval.min() / approval.max()), 4)
    false_rejection_gap = round(float(false_rejection.max() - false_rejection.min()), 4)
    return {
        "least_approved_group": by_group.loc[approval.idxmin(), GROUP_COLUMN],
        "most_approved_group": by_group.loc[approval.idxmax(), GROUP_COLUMN],
        "disparate_impact_ratio": disparate_impact,
        "passes_four_fifths_rule": disparate_impact >= DISPARATE_IMPACT_MIN_RATIO,
        "false_rejection_gap": false_rejection_gap,
        "false_rejection_gap_alert": false_rejection_gap >= FALSE_REJECTION_GAP_ALERT,
        "missed_default_gap": round(
            float(by_group["missed_default_rate"].max() - by_group["missed_default_rate"].min()), 4
        ),
    }


def proxy_correlations(reference: pd.DataFrame) -> pd.DataFrame:
    correlation = (
        reference[[PROTECTED_ATTRIBUTE, *NUMERIC_FEATURES]]
        .corr(method="spearman")[PROTECTED_ATTRIBUTE]
        .drop(PROTECTED_ATTRIBUTE)
        .round(4)
    )
    return (
        pd.DataFrame({"feature": correlation.index, "spearman_with_age": correlation.to_numpy()})
        .assign(
            strong_proxy=lambda frame: frame["spearman_with_age"].abs() >= PROXY_CORRELATION_ALERT
        )
        .sort_values("spearman_with_age", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def scored_datasets(model, reference: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        REFERENCE_HOLDOUT: reference_holdout_with_predictions(model, reference),
        **{month: scored_month(month) for month in PRODUCTION_MONTHS},
    }


def write_audit(by_group: pd.DataFrame, summaries: dict, proxies: pd.DataFrame) -> None:
    FAIRNESS_BY_GROUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(FAIRNESS_BY_GROUP_PATH, index=False)
    audit = {
        "protected_attribute": PROTECTED_ATTRIBUTE,
        "used_as_model_feature": PROTECTED_ATTRIBUTE in MODEL_FEATURES,
        "age_groups": {name: list(bounds) for name, bounds in AGE_BANDS.items()},
        "thresholds": {
            "disparate_impact_min_ratio": DISPARATE_IMPACT_MIN_RATIO,
            "false_rejection_gap_alert": FALSE_REJECTION_GAP_ALERT,
            "proxy_correlation_alert": PROXY_CORRELATION_ALERT,
        },
        "disparity_by_dataset": summaries,
        "proxy_correlations": proxies.to_dict("records"),
    }
    FAIRNESS_AUDIT_PATH.write_text(json.dumps(audit, indent=2, default=str))


@tracked_step("fairness")
def audit_fairness() -> None:
    log = get_pipeline_logger("fairness")
    model = joblib.load(MODEL_PATH)
    reference = pd.read_parquet(REFERENCE_PATH)
    tables = [
        outcomes_by_age_group(scored, dataset)
        for dataset, scored in scored_datasets(model, reference).items()
    ]
    by_group = pd.concat(tables, ignore_index=True)
    summaries = {table["dataset"].iloc[0]: disparity_summary(table) for table in tables}
    proxies = proxy_correlations(reference)
    write_audit(by_group, summaries, proxies)

    for dataset, summary in summaries.items():
        log.info("fairness_checked", extra={"dataset": dataset, **summary})
        if not summary["passes_four_fifths_rule"] or summary["false_rejection_gap_alert"]:
            log.warning("fairness_alert", extra={"dataset": dataset, **summary})
    log.info(
        "proxy_correlations_checked",
        extra={"strong_proxies": proxies.loc[proxies["strong_proxy"], "feature"].tolist()},
    )


if __name__ == "__main__":
    audit_fairness()
