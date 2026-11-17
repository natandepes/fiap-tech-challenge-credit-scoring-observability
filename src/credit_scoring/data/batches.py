import numpy as np
import pandas as pd

from credit_scoring.config import (
    BATCHES_DIR,
    ID_COLUMN,
    PRODUCTION_BASE_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from credit_scoring.observability.step import tracked_step
from credit_scoring.pipeline_log import get_pipeline_logger

BATCH_SIZE = 200
GOOD_BATCH_PATH = BATCHES_DIR / "good_batch.csv"
BAD_BATCH_PATH = BATCHES_DIR / "bad_batch.csv"

CELL_ANOMALIES = [
    ("person_income", np.nan, 5),
    ("person_income", -42_000.0, 3),
    ("person_age", 16, 4),
    ("person_age", 144, 2),
    ("person_home_ownership", "UNKNOWN", 3),
    ("loan_grade", "Z", 2),
    ("loan_int_rate", 250.0, 2),
    ("loan_amnt", "unknown", 2),
    (ID_COLUMN, np.nan, 1),
]
INCONSISTENT_EMPLOYMENT_RECORDS = 3
INCONSISTENT_LOAN_TO_INCOME_RECORDS = 3
REPLAYED_RECORDS = 4
REUSED_IDS = 2
UNEXPECTED_PII_COLUMN = "national_id"


def sample_scoring_requests(rng: np.random.Generator) -> pd.DataFrame:
    production = pd.read_parquet(PRODUCTION_BASE_PATH).drop(columns=TARGET_COLUMN)
    rows = rng.choice(len(production), size=BATCH_SIZE, replace=False)
    return production.iloc[rows].reset_index(drop=True)


def corrupt_cells(batch: pd.DataFrame, target_rows) -> pd.DataFrame:
    corrupted = batch.astype(object)
    for column, value, count in CELL_ANOMALIES:
        rows = [next(target_rows) for _ in range(count)]
        corrupted.loc[rows, column] = value
    return corrupted


def break_record_consistency(batch: pd.DataFrame, target_rows) -> pd.DataFrame:
    inconsistent = batch.copy()
    for _ in range(INCONSISTENT_EMPLOYMENT_RECORDS):
        row = next(target_rows)
        inconsistent.loc[row, "person_emp_length"] = float(inconsistent.loc[row, "person_age"] - 10)
    for _ in range(INCONSISTENT_LOAN_TO_INCOME_RECORDS):
        row = next(target_rows)
        inconsistent.loc[row, "loan_percent_income"] = (
            inconsistent.loc[row, "loan_percent_income"] + 0.5
        )
    return inconsistent


def replay_records(batch: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    replayed = batch.dropna().sample(REPLAYED_RECORDS, random_state=RANDOM_SEED)
    replayed[ID_COLUMN] = [f"replay{rng.integers(10**9):010d}" for _ in range(REPLAYED_RECORDS)]
    return pd.concat([batch, replayed], ignore_index=True)


def reuse_application_ids(batch: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    donors, receivers = np.split(rng.choice(batch.index, size=2 * REUSED_IDS, replace=False), 2)
    reused = batch.copy()
    reused.loc[receivers, ID_COLUMN] = batch.loc[donors, ID_COLUMN].to_numpy()
    return reused


def leak_pii_column(batch: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    national_ids = [f"{rng.integers(10**10, 10**11)}" for _ in range(len(batch))]
    return batch.assign(**{UNEXPECTED_PII_COLUMN: national_ids})


def inject_anomalies(batch: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    target_rows = iter(rng.permutation(batch.index))
    corrupted = break_record_consistency(batch, target_rows)
    corrupted = corrupt_cells(corrupted, target_rows)
    corrupted = reuse_application_ids(replay_records(corrupted, rng), rng)
    return leak_pii_column(corrupted, rng)


@tracked_step("batches")
def build_demo_batches() -> None:
    log = get_pipeline_logger("batches")
    rng = np.random.default_rng(RANDOM_SEED)
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    good_batch = sample_scoring_requests(rng)
    bad_batch = inject_anomalies(sample_scoring_requests(rng), rng)
    good_batch.to_csv(GOOD_BATCH_PATH, index=False)
    bad_batch.to_csv(BAD_BATCH_PATH, index=False)
    for path, frame in ((GOOD_BATCH_PATH, good_batch), (BAD_BATCH_PATH, bad_batch)):
        log.info("demo_batch_written", extra={"path": str(path), "rows": len(frame)})


if __name__ == "__main__":
    build_demo_batches()
