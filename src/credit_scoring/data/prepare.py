import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring.config import (
    ID_COLUMN,
    PRODUCTION_BASE_PATH,
    PRODUCTION_FRACTION,
    RANDOM_SEED,
    RAW_CURATION_FAILURE_CASES_PATH,
    RAW_CURATION_REPORT_PATH,
    RAW_PATH,
    RAW_REJECTED_ROWS_PATH,
    REFERENCE_PATH,
    REFERENCE_SAMPLE_PATH,
    SAMPLE_SIZE,
    TARGET_COLUMN,
)
from credit_scoring.contracts.credit_contract import LABELED_CREDIT_CONTRACT
from credit_scoring.contracts.enforcement import DataContractViolation, enforce_contract
from credit_scoring.pipeline_log import get_pipeline_logger


def surrogate_ids(count: int) -> list[str]:
    return [
        hashlib.sha256(f"{RANDOM_SEED}:{row}".encode()).hexdigest()[:16] for row in range(count)
    ]


def with_application_ids(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(**{ID_COLUMN: surrogate_ids(len(frame))})[[ID_COLUMN, *frame.columns]]


def curate(frame: pd.DataFrame) -> tuple[pd.DataFrame, DataContractViolation | None]:
    try:
        return enforce_contract(frame, LABELED_CREDIT_CONTRACT), None
    except DataContractViolation as violation:
        valid = frame.drop(index=list(violation.failing_rows)).reset_index(drop=True)
        return enforce_contract(valid, LABELED_CREDIT_CONTRACT), violation


def write_curation_report(counts: dict[str, int], violation: DataContractViolation | None) -> None:
    RAW_CURATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "contract": LABELED_CREDIT_CONTRACT.name,
        **counts,
        "failures_by_check": [] if violation is None else violation.summary().to_dict("records"),
    }
    RAW_CURATION_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))


def write_rejected_rows(frame: pd.DataFrame, violation: DataContractViolation) -> None:
    violation.failure_cases.to_csv(RAW_CURATION_FAILURE_CASES_PATH, index=False)
    broken_rules = violation.broken_rules_by_row()
    rejected = frame.loc[broken_rules.index].assign(broken_rules=broken_rules)
    RAW_REJECTED_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rejected.to_csv(RAW_REJECTED_ROWS_PATH, index=False)


def split_reference_and_production(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference, production = train_test_split(
        frame,
        test_size=PRODUCTION_FRACTION,
        stratify=frame[TARGET_COLUMN],
        random_state=RANDOM_SEED,
    )
    return reference.reset_index(drop=True), production.reset_index(drop=True)


def write_parquet(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def prepare_datasets() -> None:
    log = get_pipeline_logger("prepare")
    raw = pd.read_csv(RAW_PATH)
    unique = with_application_ids(raw.drop_duplicates().reset_index(drop=True))
    curated, violation = curate(unique)
    counts = {
        "raw_rows": len(raw),
        "exact_duplicates_removed": len(raw) - len(unique),
        "rows_rejected_by_contract": len(unique) - len(curated),
        "curated_rows": len(curated),
    }
    write_curation_report(counts, violation)
    if violation is not None:
        write_rejected_rows(unique, violation)
    log.info(
        "raw_dataset_curated",
        extra={
            **counts,
            "report": str(RAW_CURATION_REPORT_PATH),
            "failure_cases": str(RAW_CURATION_FAILURE_CASES_PATH),
            "rejected_rows": str(RAW_REJECTED_ROWS_PATH),
        },
    )

    reference, production = split_reference_and_production(curated)
    for name, frame, path in (
        ("reference", reference, REFERENCE_PATH),
        ("production_base", production, PRODUCTION_BASE_PATH),
    ):
        write_parquet(frame, path)
        log.info(
            "dataset_written",
            extra={
                "dataset": name,
                "rows": len(frame),
                "default_rate": round(frame[TARGET_COLUMN].mean(), 4),
                "path": str(path),
            },
        )

    REFERENCE_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference.sample(SAMPLE_SIZE, random_state=RANDOM_SEED).to_csv(
        REFERENCE_SAMPLE_PATH, index=False
    )


if __name__ == "__main__":
    prepare_datasets()
