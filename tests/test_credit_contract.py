import numpy as np
import pandas as pd
import pytest

from credit_scoring.config import ID_COLUMN, REFERENCE_SAMPLE_PATH, TARGET_COLUMN
from credit_scoring.contracts.credit_contract import (
    LABELED_CREDIT_CONTRACT,
    SCORING_REQUEST_CONTRACT,
)
from credit_scoring.contracts.enforcement import DataContractViolation, enforce_contract
from credit_scoring.data.batches import inject_anomalies


@pytest.fixture
def labeled_batch() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_SAMPLE_PATH, dtype={ID_COLUMN: "string"})


@pytest.fixture
def scoring_batch(labeled_batch) -> pd.DataFrame:
    return labeled_batch.drop(columns=TARGET_COLUMN)


def failed_checks_of(batch: pd.DataFrame, contract) -> set[str]:
    with pytest.raises(DataContractViolation) as violation:
        enforce_contract(batch, contract)
    return violation.value.failed_checks


def with_value(batch: pd.DataFrame, column: str, value) -> pd.DataFrame:
    corrupted = batch.astype({column: object})
    corrupted.loc[0, column] = value
    return corrupted


def test_clean_labeled_batch_passes(labeled_batch):
    assert len(enforce_contract(labeled_batch, LABELED_CREDIT_CONTRACT)) == len(labeled_batch)


def test_clean_scoring_batch_passes(scoring_batch):
    assert len(enforce_contract(scoring_batch, SCORING_REQUEST_CONTRACT)) == len(scoring_batch)


@pytest.mark.parametrize(
    ("column", "value", "expected_check"),
    [
        ("person_income", np.nan, "not_nullable"),
        ("person_income", 0.0, "greater_than(0)"),
        ("person_income", 20_000_000.0, "less_than_or_equal_to(10000000)"),
        ("person_age", 17, "in_range(18, 100)"),
        ("person_age", 144, "in_range(18, 100)"),
        ("person_emp_length", -1.0, "in_range(0, 86)"),
        ("loan_amnt", 0.0, "greater_than(0)"),
        ("loan_amnt", "unknown", "coerce_dtype('float64')"),
        ("loan_int_rate", np.nan, "not_nullable"),
        ("loan_int_rate", 250.0, "less_than_or_equal_to(100)"),
        ("loan_percent_income", -0.1, "greater_than_or_equal_to(0)"),
        ("person_home_ownership", "UNKNOWN", "isin(['RENT', 'MORTGAGE', 'OWN', 'OTHER'])"),
        ("loan_grade", "Z", "isin(['A', 'B', 'C', 'D', 'E', 'F', 'G'])"),
        ("cb_person_default_on_file", "maybe", "isin(['N', 'Y'])"),
        (ID_COLUMN, pd.NA, "not_nullable"),
    ],
)
def test_cell_anomaly_is_rejected(scoring_batch, column, value, expected_check):
    corrupted = with_value(scoring_batch, column, value)
    assert expected_check in failed_checks_of(corrupted, SCORING_REQUEST_CONTRACT)


def test_employment_longer_than_working_life_is_rejected(scoring_batch):
    corrupted = scoring_batch.assign(person_age=25, person_emp_length=20.0)
    assert "employment_within_working_life" in failed_checks_of(corrupted, SCORING_REQUEST_CONTRACT)


def test_credit_history_longer_than_working_life_is_rejected(scoring_batch):
    corrupted = with_value(scoring_batch, "cb_person_cred_hist_length", 80)
    assert "credit_history_within_working_life" in failed_checks_of(
        corrupted, SCORING_REQUEST_CONTRACT
    )


def test_inconsistent_loan_to_income_is_rejected(scoring_batch):
    corrupted = with_value(scoring_batch, "loan_percent_income", 3.0)
    assert "loan_to_income_consistent" in failed_checks_of(corrupted, SCORING_REQUEST_CONTRACT)


def test_invalid_target_is_rejected(labeled_batch):
    corrupted = with_value(labeled_batch, TARGET_COLUMN, 2)
    assert "isin([0, 1])" in failed_checks_of(corrupted, LABELED_CREDIT_CONTRACT)


def test_reused_application_id_is_rejected(scoring_batch):
    corrupted = with_value(scoring_batch, ID_COLUMN, scoring_batch[ID_COLUMN].iloc[1])
    assert "field_uniqueness" in failed_checks_of(corrupted, SCORING_REQUEST_CONTRACT)


def test_replayed_record_under_new_id_is_rejected(scoring_batch):
    replayed = scoring_batch.head(1).assign(**{ID_COLUMN: "replayed-id"})
    corrupted = pd.concat([scoring_batch, replayed], ignore_index=True)
    assert "no_duplicate_records" in failed_checks_of(corrupted, SCORING_REQUEST_CONTRACT)


def test_unexpected_column_is_rejected(scoring_batch):
    corrupted = scoring_batch.assign(national_id="12345678901")
    assert "column_in_schema" in failed_checks_of(corrupted, SCORING_REQUEST_CONTRACT)


def test_missing_column_is_rejected(scoring_batch):
    corrupted = scoring_batch.drop(columns="person_age")
    assert "column_in_dataframe" in failed_checks_of(corrupted, SCORING_REQUEST_CONTRACT)


def test_demo_bad_batch_trips_every_rule_family(scoring_batch):
    bad_batch = inject_anomalies(scoring_batch, np.random.default_rng(0))
    failed = failed_checks_of(bad_batch, SCORING_REQUEST_CONTRACT)
    assert {
        "not_nullable",
        "greater_than(0)",
        "in_range(18, 100)",
        "isin(['RENT', 'MORTGAGE', 'OWN', 'OTHER'])",
        "isin(['A', 'B', 'C', 'D', 'E', 'F', 'G'])",
        "less_than_or_equal_to(100)",
        "coerce_dtype('float64')",
        "employment_within_working_life",
        "loan_to_income_consistent",
        "field_uniqueness",
        "no_duplicate_records",
        "column_in_schema",
    } <= failed


def test_failures_are_traceable_to_application_ids(scoring_batch):
    corrupted = with_value(scoring_batch, "person_age", 144)
    with pytest.raises(DataContractViolation) as violation:
        enforce_contract(corrupted, SCORING_REQUEST_CONTRACT)
    broken_rules = violation.value.broken_rules_by_row()
    assert broken_rules.loc[0] == "in_range(18, 100)"
    assert scoring_batch[ID_COLUMN].iloc[0] in set(violation.value.failure_cases[ID_COLUMN])
