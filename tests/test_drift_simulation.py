import numpy as np
import pandas as pd
import pytest

from credit_scoring.config import ID_COLUMN, REFERENCE_SAMPLE_PATH, TARGET_COLUMN
from credit_scoring.contracts.credit_contract import LABELED_CREDIT_CONTRACT
from credit_scoring.contracts.enforcement import enforce_contract
from credit_scoring.drift.scenario import DEFAULT_SHOCK
from credit_scoring.drift.simulate import (
    DATA_DRIFTED_FEATURES,
    apply_data_drift,
    apply_nominal_inflation,
    resample_labels_under_reference_mechanism,
    simulate_month,
    split_into_months,
)


def burden_risk(frame: pd.DataFrame) -> np.ndarray:
    return np.clip(frame["loan_percent_income"].to_numpy() * 2, 0.01, 0.99)


@pytest.fixture
def labeled_batch() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_SAMPLE_PATH, dtype={ID_COLUMN: "string"})


def simulate(batch: pd.DataFrame, severity: float, seed: int = 0):
    return simulate_month(batch, severity, burden_risk, np.random.default_rng(seed))


def test_months_partition_the_production_base_without_overlap(labeled_batch):
    months = split_into_months(labeled_batch, np.random.default_rng(0))
    ids = pd.concat([month[ID_COLUMN] for month in months.values()])
    assert len(months) == 6
    assert ids.is_unique
    assert set(ids) == set(labeled_batch[ID_COLUMN])


def test_control_month_is_left_untouched(labeled_batch):
    simulated = simulate(labeled_batch, severity=0.0)
    pd.testing.assert_frame_equal(simulated.production, labeled_batch)
    assert not simulated.concept_drift_defaults.any()


def test_shock_moves_only_the_declared_features(labeled_batch):
    drifted = apply_data_drift(labeled_batch, 1.0, DEFAULT_SHOCK, np.random.default_rng(0))
    changed = [
        column for column in labeled_batch if not drifted[column].equals(labeled_batch[column])
    ]
    assert sorted(changed) == sorted(DATA_DRIFTED_FEATURES)
    rate_hike = drifted["loan_int_rate"] - labeled_batch["loan_int_rate"]
    assert rate_hike.round(2).eq(DEFAULT_SHOCK.interest_rate_hike_pp).all()


def test_nominal_inflation_raises_money_values_but_not_the_debt_burden(labeled_batch):
    inflated = apply_nominal_inflation(labeled_batch, 1.0, DEFAULT_SHOCK)
    price_level = 1 + DEFAULT_SHOCK.nominal_inflation
    for column in ("person_income", "loan_amnt"):
        ratio = inflated[column] / labeled_batch[column]
        assert ratio.round(2).eq(round(price_level, 2)).all()
    burden_change = inflated["loan_percent_income"] - labeled_batch["loan_percent_income"]
    assert burden_change.abs().le(0.11).all()


def test_drifted_production_still_honours_the_data_contract(labeled_batch):
    simulated = simulate(labeled_batch, severity=1.0)
    enforce_contract(simulated.production, LABELED_CREDIT_CONTRACT)


def test_concept_drift_only_adds_defaults(labeled_batch):
    simulated = simulate(labeled_batch, severity=1.0)
    flipped = simulated.concept_drift_defaults
    assert flipped.any()
    assert simulated.production.loc[flipped, TARGET_COLUMN].eq(1).all()
    assert simulated.labels_without_concept_drift[flipped].eq(0).all()


def test_simulation_is_deterministic(labeled_batch):
    first, second = simulate(labeled_batch, 0.6, seed=7), simulate(labeled_batch, 0.6, seed=7)
    pd.testing.assert_frame_equal(first.production, second.production)


@pytest.mark.parametrize(("risk_before", "risk_after"), [(0.2, 0.5), (0.6, 0.3)])
def test_label_resampling_matches_the_new_risk(risk_before, risk_after):
    rng = np.random.default_rng(0)
    size = 200_000
    labels = pd.Series((rng.random(size) < risk_before).astype(int))
    resampled = resample_labels_under_reference_mechanism(
        labels, np.full(size, risk_before), np.full(size, risk_after), rng.random(size)
    )
    assert resampled.mean() == pytest.approx(risk_after, abs=0.005)
