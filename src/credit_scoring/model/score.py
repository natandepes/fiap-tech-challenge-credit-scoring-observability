import time

import joblib
import pandas as pd

from credit_scoring.config import (
    DECISION_THRESHOLD,
    ID_COLUMN,
    MODEL_FEATURES,
    MODEL_PATH,
    MONTH_COLUMN,
    MONTHLY_PRODUCTION_DIR,
    PREDICTION_COLUMN,
    PREDICTIONS_DIR,
    PROBABILITY_COLUMN,
    TARGET_COLUMN,
)
from credit_scoring.contracts.credit_contract import SCORING_REQUEST_CONTRACT
from credit_scoring.contracts.enforcement import enforce_contract
from credit_scoring.pipeline_log import get_pipeline_logger


def score(model, requests: pd.DataFrame) -> pd.DataFrame:
    default_probability = model.predict_proba(requests[MODEL_FEATURES])[:, 1]
    return pd.DataFrame(
        {
            ID_COLUMN: requests[ID_COLUMN].to_numpy(),
            PROBABILITY_COLUMN: default_probability,
            PREDICTION_COLUMN: (default_probability >= DECISION_THRESHOLD).astype(int),
        }
    )


def score_month(model, month: str) -> pd.DataFrame:
    batch = pd.read_parquet(MONTHLY_PRODUCTION_DIR / f"{month}.parquet")
    requests = enforce_contract(batch.drop(columns=TARGET_COLUMN), SCORING_REQUEST_CONTRACT)
    return score(model, requests).assign(**{MONTH_COLUMN: month})


def score_production() -> None:
    log = get_pipeline_logger("score")
    model = joblib.load(MODEL_PATH)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(MONTHLY_PRODUCTION_DIR.glob("*.parquet")):
        started = time.perf_counter()
        predictions = score_month(model, path.stem)
        predictions.to_parquet(PREDICTIONS_DIR / path.name, index=False)
        log.info(
            "month_scored",
            extra={
                MONTH_COLUMN: path.stem,
                "rows": len(predictions),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "mean_default_probability": round(predictions[PROBABILITY_COLUMN].mean(), 4),
                "predicted_default_rate": round(predictions[PREDICTION_COLUMN].mean(), 4),
            },
        )


if __name__ == "__main__":
    score_production()
