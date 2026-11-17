import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from credit_scoring.config import ID_COLUMN, QUARANTINE_DIR, VALIDATED_DIR, VALIDATION_REPORTS_DIR
from credit_scoring.contracts.credit_contract import (
    LABELED_CREDIT_CONTRACT,
    SCORING_REQUEST_CONTRACT,
)
from credit_scoring.contracts.enforcement import DataContractViolation, enforce_contract
from credit_scoring.observability.step import tracked_step
from credit_scoring.pipeline_log import get_pipeline_logger

CONTRACTS = {"scoring": SCORING_REQUEST_CONTRACT, "labeled": LABELED_CREDIT_CONTRACT}
EXIT_REJECTED = 1


def read_batch(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={ID_COLUMN: "string"})


def write_rejection_report(batch_name: str, violation: DataContractViolation) -> Path:
    VALIDATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    violation.failure_cases.to_csv(
        VALIDATION_REPORTS_DIR / f"{batch_name}_failure_cases.csv", index=False
    )
    report_path = VALIDATION_REPORTS_DIR / f"{batch_name}_rejection.json"
    report = {
        "batch": batch_name,
        "contract": violation.contract_name,
        "status": "rejected",
        "failure_cases": len(violation.failure_cases),
        "failures_by_check": violation.summary().to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    return report_path


def quarantine(path: Path) -> Path:
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    destination = QUARANTINE_DIR / path.name
    shutil.copy2(path, destination)
    return destination


def accept(batch: pd.DataFrame, batch_name: str) -> Path:
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    destination = VALIDATED_DIR / f"{batch_name}.parquet"
    batch.to_parquet(destination, index=False)
    return destination


@tracked_step("ingest")
def ingest(path: Path, contract_key: str) -> int:
    log = get_pipeline_logger("ingest")
    contract = CONTRACTS[contract_key]
    batch = read_batch(path)
    batch_context = {"batch": path.name, "rows": len(batch), "contract": contract.name}
    try:
        validated = enforce_contract(batch, contract)
    except DataContractViolation as violation:
        report_path = write_rejection_report(path.stem, violation)
        log.error(
            "batch_rejected",
            extra={
                **batch_context,
                "failure_cases": len(violation.failure_cases),
                "failed_checks": sorted(violation.failed_checks),
                "quarantined_to": str(quarantine(path)),
                "report": str(report_path),
            },
        )
        print(violation.summary().to_string(index=False))
        return EXIT_REJECTED
    log.info(
        "batch_accepted", extra={**batch_context, "stored_to": str(accept(validated, path.stem))}
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a data batch through its data contract.")
    parser.add_argument("batch", type=Path)
    parser.add_argument("--contract", choices=CONTRACTS, default="scoring")
    arguments = parser.parse_args()
    raise SystemExit(ingest(arguments.batch, arguments.contract))


if __name__ == "__main__":
    main()
