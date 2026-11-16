import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.legacy.calculations.stattests.registry import StatTest, register_stattest
from evidently.legacy.core import ColumnType
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset
from evidently.tests import gte

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    DATASET_DRIFT_FEATURE_SHARE,
    DECISION_THRESHOLD,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    PSI_DRIFT_THRESHOLD,
    RECALL_DROP_ALERT,
    ROC_AUC_DROP_ALERT,
    TARGET_COLUMN,
)
from credit_scoring.drift.statistics import population_stability_index

PSI = "quantile_psi"
VALUE_DRIFT_TYPE = "evidently:metric_v2:ValueDrift"

FEATURE_DEFINITION = DataDefinition(
    numerical_columns=NUMERIC_FEATURES,
    categorical_columns=[*CATEGORICAL_FEATURES, TARGET_COLUMN],
)
MODEL_OUTPUT_DEFINITION = DataDefinition(
    numerical_columns=[*NUMERIC_FEATURES, PROBABILITY_COLUMN],
    categorical_columns=[*CATEGORICAL_FEATURES, TARGET_COLUMN, PREDICTION_COLUMN],
    classification=[
        BinaryClassification(
            target=TARGET_COLUMN,
            prediction_labels=PREDICTION_COLUMN,
            prediction_probas=PROBABILITY_COLUMN,
            pos_label=1,
        )
    ],
)


def _quantile_psi(
    reference: pd.Series, current: pd.Series, feature_type: ColumnType, threshold: float
) -> tuple[float, bool]:
    psi = population_stability_index(
        reference, current, categorical=feature_type == ColumnType.Categorical
    )
    return psi, psi >= threshold


register_stattest(
    StatTest(
        name=PSI,
        display_name="PSI (reference decile bins)",
        allowed_feature_types=[ColumnType.Categorical, ColumnType.Numerical],
        default_threshold=PSI_DRIFT_THRESHOLD,
    ),
    _quantile_psi,
)


def _dataset(frame: pd.DataFrame, definition: DataDefinition) -> Dataset:
    return Dataset.from_pandas(frame, data_definition=definition)


def _psi_drift(column: str) -> ValueDrift:
    return ValueDrift(column=column, method=PSI, threshold=PSI_DRIFT_THRESHOLD)


def data_drift_snapshot(reference: pd.DataFrame, current: pd.DataFrame) -> Snapshot:
    report = Report(
        [
            DataDriftPreset(
                columns=MODEL_FEATURES,
                method=PSI,
                threshold=PSI_DRIFT_THRESHOLD,
                drift_share=DATASET_DRIFT_FEATURE_SHARE,
            ),
            _psi_drift(TARGET_COLUMN),
        ],
        include_tests=True,
    )
    feature_columns = [*MODEL_FEATURES, TARGET_COLUMN]
    return report.run(
        _dataset(current[feature_columns], FEATURE_DEFINITION),
        _dataset(reference[feature_columns], FEATURE_DEFINITION),
    )


def _minimum(baseline: dict, metric: str, allowed_drop: float) -> list:
    return [gte(round(baseline[metric] - allowed_drop, 4))]


def model_health_snapshot(
    reference: pd.DataFrame, current: pd.DataFrame, baseline: dict
) -> Snapshot:
    report = Report(
        [
            _psi_drift(PROBABILITY_COLUMN),
            ClassificationPreset(
                probas_threshold=DECISION_THRESHOLD,
                rocauc_tests=_minimum(baseline, "roc_auc", ROC_AUC_DROP_ALERT),
                recall_tests=_minimum(baseline, "recall", RECALL_DROP_ALERT),
                tpr_tests=_minimum(baseline, "recall", RECALL_DROP_ALERT),
            ),
        ],
        include_tests=True,
    )
    return report.run(
        _dataset(current, MODEL_OUTPUT_DEFINITION), _dataset(reference, MODEL_OUTPUT_DEFINITION)
    )


def psi_by_column(snapshot: Snapshot) -> dict[str, float]:
    return {
        metric["config"]["column"]: float(metric["value"])
        for metric in snapshot.dict()["metrics"]
        if metric["config"]["type"] == VALUE_DRIFT_TYPE
    }
