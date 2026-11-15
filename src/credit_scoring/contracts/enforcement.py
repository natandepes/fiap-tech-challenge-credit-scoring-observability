import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from credit_scoring.config import ID_COLUMN

FAILURE_CASE_COLUMNS = ["schema_context", "column", "check", "failure_case", "index"]
REPORT_COLUMNS = ["index", ID_COLUMN, "column", "check", "failure_case"]
RECORD_LEVEL = "(record)"


class DataContractViolation(Exception):
    def __init__(self, contract_name: str, failure_cases: pd.DataFrame):
        self.contract_name = contract_name
        self.failure_cases = failure_cases
        super().__init__(
            f"{contract_name} rejected the batch: {len(failure_cases)} failure case(s) "
            f"across checks {sorted(self.failed_checks)}"
        )

    @property
    def failing_rows(self) -> set:
        return set(self.failure_cases["index"].dropna())

    def broken_rules_by_row(self) -> pd.Series:
        located = self.failure_cases.dropna(subset=["index"])
        return located.groupby("index")["check"].agg(lambda checks: "; ".join(sorted(set(checks))))

    @property
    def failed_checks(self) -> set[str]:
        return set(self.failure_cases["check"].astype(str))

    def summary(self) -> pd.DataFrame:
        return (
            self.failure_cases.groupby(["column", "check"], dropna=False)
            .size()
            .rename("failures")
            .reset_index()
            .sort_values("failures", ascending=False, ignore_index=True)
        )


def enforce_contract(frame: pd.DataFrame, contract: pa.DataFrameSchema) -> pd.DataFrame:
    try:
        return contract.validate(frame, lazy=True)
    except SchemaErrors as errors:
        failure_cases = _one_row_per_record_failure(errors.failure_cases[FAILURE_CASE_COLUMNS])
        failure_cases = _with_record_ids(failure_cases, frame)
        raise DataContractViolation(contract.name, failure_cases) from errors


def _one_row_per_record_failure(failure_cases: pd.DataFrame) -> pd.DataFrame:
    is_record_check = (failure_cases["schema_context"] == "DataFrameSchema") & failure_cases[
        "index"
    ].notna()
    keep = ~is_record_check | (failure_cases["column"] == ID_COLUMN)
    collapsed = failure_cases[keep].reset_index(drop=True)
    record_rows = is_record_check[keep].to_numpy()
    collapsed.loc[record_rows, "column"] = RECORD_LEVEL
    collapsed.loc[record_rows, "failure_case"] = pd.NA
    return collapsed


def _with_record_ids(failure_cases: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    located = failure_cases.assign(index=failure_cases["index"].astype("Int64"))
    ids = frame[ID_COLUMN] if ID_COLUMN in frame.columns else pd.Series(dtype="string")
    located[ID_COLUMN] = located["index"].map(ids)
    return located[REPORT_COLUMNS]
