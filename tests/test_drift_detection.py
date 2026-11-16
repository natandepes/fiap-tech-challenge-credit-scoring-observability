import numpy as np
import pandas as pd
import pytest

from credit_scoring.config import ID_COLUMN, REFERENCE_SAMPLE_PATH
from credit_scoring.drift.alerts import (
    data_drift_alerts,
    model_degradation_alerts,
    prediction_drift_alerts,
)
from credit_scoring.drift.evidently_reports import data_drift_snapshot, psi_by_column
from credit_scoring.drift.scenario import DEFAULT_SHOCK
from credit_scoring.drift.simulate import apply_data_drift
from credit_scoring.drift.statistics import (
    feature_drift_table,
    population_stability_index,
    psi_band,
)

BASELINE = {"roc_auc": 0.93, "recall": 0.71}


@pytest.fixture
def reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_SAMPLE_PATH, dtype={ID_COLUMN: "string"})


def test_psi_is_zero_for_identical_samples(reference):
    assert population_stability_index(
        reference["person_income"], reference["person_income"], categorical=False
    ) == pytest.approx(0)


def test_psi_grows_with_the_size_of_the_shift():
    reference = pd.Series(np.random.default_rng(0).normal(size=5_000))
    small, large = (
        population_stability_index(reference, reference + shift, categorical=False)
        for shift in (0.1, 1.0)
    )
    assert small < 0.1 < large


def test_psi_handles_unseen_categories():
    reference = pd.Series(["RENT"] * 50 + ["OWN"] * 50)
    current = pd.Series(["RENT"] * 50 + ["OTHER"] * 50)
    assert population_stability_index(reference, current, categorical=True) > 0.25


@pytest.mark.parametrize(("psi", "band"), [(0.05, "stable"), (0.1, "moderate"), (0.3, "major")])
def test_psi_bands(psi, band):
    assert psi_band(psi) == band


def test_evidently_flags_the_shocked_features_and_spares_the_others(reference):
    shocked = apply_data_drift(reference, 1.0, DEFAULT_SHOCK, np.random.default_rng(0))
    psi = psi_by_column(data_drift_snapshot(reference, shocked))
    table = feature_drift_table(reference, shocked, psi)
    drifted = set(table.loc[table["drifted"], "feature"])
    assert {"loan_int_rate", "loan_amnt"} <= drifted
    assert drifted.isdisjoint({"loan_grade", "loan_intent", "cb_person_cred_hist_length"})


def test_no_data_drift_alert_without_drift(reference):
    psi = psi_by_column(data_drift_snapshot(reference, reference))
    assert data_drift_alerts("m", feature_drift_table(reference, reference, psi)) == []


def test_prediction_drift_alert_fires_above_threshold():
    assert prediction_drift_alerts("m", 0.05) == []
    assert prediction_drift_alerts("m", 0.3)[0]["severity"] == "critical"


def test_degradation_alerts_name_what_broke():
    degraded = {
        "roc_auc": 0.85,
        "recall": 0.55,
        "observed_default_rate": 0.55,
        "mean_default_probability": 0.42,
    }
    alerts = model_degradation_alerts("m", degraded, BASELINE)
    assert {alert["subject"] for alert in alerts} == {
        "roc_auc_drop",
        "recall_drop",
        "observed_minus_predicted_default_rate",
    }


def test_healthy_model_raises_no_degradation_alert():
    healthy = {
        "roc_auc": 0.93,
        "recall": 0.70,
        "observed_default_rate": 0.30,
        "mean_default_probability": 0.29,
    }
    assert model_degradation_alerts("m", healthy, BASELINE) == []
