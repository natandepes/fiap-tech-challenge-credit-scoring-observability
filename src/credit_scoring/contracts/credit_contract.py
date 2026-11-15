import pandas as pd
import pandera.pandas as pa

from credit_scoring.config import CATEGORY_DOMAINS, ID_COLUMN, TARGET_COLUMN

MIN_AGE = 18
MAX_AGE = 100
MIN_WORKING_AGE = 14
MAX_INCOME = 10_000_000
MAX_LOAN_AMOUNT = 1_000_000
MAX_INTEREST_RATE = 100
MAX_YEARS_OF_HISTORY = MAX_AGE - MIN_WORKING_AGE
LOAN_TO_INCOME_TOLERANCE = 0.1
TARGET_CODES = [0, 1]


def _required(dtype, *checks: pa.Check, unique: bool = False) -> pa.Column:
    return pa.Column(dtype, checks=list(checks), nullable=False, unique=unique, coerce=True)


def _category_columns() -> dict[str, pa.Column]:
    return {
        name: _required(str, pa.Check.isin(values)) for name, values in CATEGORY_DOMAINS.items()
    }


def _feature_columns() -> dict[str, pa.Column]:
    return {
        ID_COLUMN: _required(str, pa.Check.str_length(min_value=1), unique=True),
        "person_age": _required(int, pa.Check.in_range(MIN_AGE, MAX_AGE)),
        "person_income": _required(float, pa.Check.gt(0), pa.Check.le(MAX_INCOME)),
        "person_emp_length": _required(float, pa.Check.in_range(0, MAX_YEARS_OF_HISTORY)),
        "loan_amnt": _required(float, pa.Check.gt(0), pa.Check.le(MAX_LOAN_AMOUNT)),
        "loan_int_rate": _required(float, pa.Check.gt(0), pa.Check.le(MAX_INTEREST_RATE)),
        "loan_percent_income": _required(float, pa.Check.ge(0)),
        "cb_person_cred_hist_length": _required(int, pa.Check.in_range(0, MAX_YEARS_OF_HISTORY)),
        **_category_columns(),
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _within_working_life(column: str):
    def check(frame: pd.DataFrame) -> pd.Series:
        working_life = _numeric(frame, "person_age") - MIN_WORKING_AGE
        years = _numeric(frame, column)
        return (years <= working_life) | years.isna() | working_life.isna()

    return check


def _loan_to_income_is_consistent(frame: pd.DataFrame) -> pd.Series:
    implied_ratio = _numeric(frame, "loan_amnt") / _numeric(frame, "person_income")
    gap = (_numeric(frame, "loan_percent_income") - implied_ratio).abs()
    return (gap <= LOAN_TO_INCOME_TOLERANCE) | gap.isna()


def _no_duplicate_records(record_columns: list[str]):
    def is_unique_record(frame: pd.DataFrame) -> pd.Series:
        present = [column for column in record_columns if column in frame.columns]
        return ~frame[present].duplicated(keep=False)

    return is_unique_record


def _record_checks(record_columns: list[str]) -> list[pa.Check]:
    return [
        pa.Check(_no_duplicate_records(record_columns), name="no_duplicate_records"),
        pa.Check(_within_working_life("person_emp_length"), name="employment_within_working_life"),
        pa.Check(
            _within_working_life("cb_person_cred_hist_length"),
            name="credit_history_within_working_life",
        ),
        pa.Check(_loan_to_income_is_consistent, name="loan_to_income_consistent"),
    ]


def _build_schema(name: str, columns: dict[str, pa.Column]) -> pa.DataFrameSchema:
    record_columns = [column for column in columns if column != ID_COLUMN]
    return pa.DataFrameSchema(
        columns, checks=_record_checks(record_columns), strict=True, name=name
    )


SCORING_REQUEST_CONTRACT = _build_schema("scoring_request_contract", _feature_columns())

LABELED_CREDIT_CONTRACT = _build_schema(
    "labeled_credit_contract",
    {**_feature_columns(), TARGET_COLUMN: _required(int, pa.Check.isin(TARGET_CODES))},
)
