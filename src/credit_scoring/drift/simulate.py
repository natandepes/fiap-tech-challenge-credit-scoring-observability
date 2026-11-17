import json
from collections.abc import Callable
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from credit_scoring.config import (
    CONCEPT_DRIFT_TRUTH_PATH,
    ID_COLUMN,
    MODEL_FEATURES,
    MODEL_PATH,
    MONTH_COLUMN,
    MONTHLY_PRODUCTION_DIR,
    PRODUCTION_BASE_PATH,
    RANDOM_SEED,
    SIMULATION_MANIFEST_PATH,
    TARGET_COLUMN,
)
from credit_scoring.contracts.credit_contract import LABELED_CREDIT_CONTRACT
from credit_scoring.contracts.enforcement import enforce_contract
from credit_scoring.drift.scenario import (
    DEFAULT_SHOCK,
    PRODUCTION_MONTHS,
    SHOCK_SEVERITY_BY_MONTH,
    EconomicShock,
)
from credit_scoring.observability.step import tracked_step
from credit_scoring.pipeline_log import get_pipeline_logger

RiskFunction = Callable[[pd.DataFrame], np.ndarray]
PROBABILITY_FLOOR = 1e-6
DATA_DRIFTED_FEATURES = [
    "person_income",
    "person_home_ownership",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
]


@dataclass(frozen=True)
class SimulatedMonth:
    production: pd.DataFrame
    labels_without_concept_drift: pd.Series
    is_new_profile: pd.Series

    @property
    def concept_drift_defaults(self) -> pd.Series:
        return self.production[TARGET_COLUMN] != self.labels_without_concept_drift


def split_into_months(production_base: pd.DataFrame, rng: np.random.Generator) -> dict:
    shuffled = rng.permutation(len(production_base))
    chunks = np.array_split(shuffled, len(PRODUCTION_MONTHS))
    return {
        month: production_base.iloc[rows].reset_index(drop=True)
        for month, rows in zip(PRODUCTION_MONTHS, chunks, strict=True)
    }


def _with_consistent_loan_to_income(drifted: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    terms_changed = (drifted["loan_amnt"] != frame["loan_amnt"]) | (
        drifted["person_income"] != frame["person_income"]
    )
    drifted["loan_percent_income"] = frame["loan_percent_income"].mask(
        terms_changed, (drifted["loan_amnt"] / drifted["person_income"]).round(2)
    )
    return drifted


def apply_real_shift(
    frame: pd.DataFrame, severity: float, shock: EconomicShock, rng: np.random.Generator
) -> pd.DataFrame:
    is_new_profile = rng.random(len(frame)) < shock.new_profile_share * severity
    income_factor = np.where(
        is_new_profile, rng.uniform(*shock.new_profile_income_factor, len(frame)), 1.0
    )
    shifted = frame.copy()
    shifted["person_income"] = (frame["person_income"] * income_factor).round()
    shifted["loan_int_rate"] = (
        frame["loan_int_rate"] + shock.interest_rate_hike_pp * severity
    ).round(2)
    shifted["person_home_ownership"] = frame["person_home_ownership"].where(
        ~is_new_profile, shock.new_profile_home_ownership
    )
    return _with_consistent_loan_to_income(shifted, frame)


def apply_nominal_inflation(
    frame: pd.DataFrame, severity: float, shock: EconomicShock
) -> pd.DataFrame:
    price_level = 1.0 + shock.nominal_inflation * severity
    inflated = frame.copy()
    for column in ("person_income", "loan_amnt"):
        inflated[column] = (frame[column] * price_level).round()
    return _with_consistent_loan_to_income(inflated, frame)


def apply_data_drift(
    frame: pd.DataFrame, severity: float, shock: EconomicShock, rng: np.random.Generator
) -> pd.DataFrame:
    return apply_nominal_inflation(apply_real_shift(frame, severity, shock, rng), severity, shock)


def resample_labels_under_reference_mechanism(
    labels: pd.Series, risk_before: np.ndarray, risk_after: np.ndarray, uniform: np.ndarray
) -> pd.Series:
    before = np.clip(risk_before, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    after = np.clip(risk_after, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    became_default = (labels == 0) & (after > before) & (uniform < (after - before) / (1 - before))
    became_repaid = (labels == 1) & (after < before) & (uniform < 1 - after / before)
    return labels.mask(became_default, 1).mask(became_repaid, 0)


def financial_stress(frame: pd.DataFrame, shock: EconomicShock) -> pd.Series:
    burden_span = shock.stress_burden_ceiling - shock.stress_burden_floor
    burden = ((frame["loan_percent_income"] - shock.stress_burden_floor) / burden_span).clip(0, 1)
    exposure = frame["person_home_ownership"].map(dict(shock.stress_exposure_by_home_ownership))
    return burden * exposure


def apply_concept_drift(
    frame: pd.DataFrame, severity: float, shock: EconomicShock, uniform: np.ndarray
) -> pd.Series:
    stress_default = shock.stress_default_probability * severity * financial_stress(frame, shock)
    return frame[TARGET_COLUMN].mask(uniform < stress_default, 1)


def simulate_month(
    frame: pd.DataFrame,
    severity: float,
    risk: RiskFunction,
    rng: np.random.Generator,
    shock: EconomicShock = DEFAULT_SHOCK,
) -> SimulatedMonth:
    shifted = apply_real_shift(frame, severity, shock, rng)
    shifted[TARGET_COLUMN] = resample_labels_under_reference_mechanism(
        frame[TARGET_COLUMN], risk(frame), risk(shifted), rng.random(len(frame))
    )
    drifted = apply_nominal_inflation(shifted, severity, shock)
    labels_without_concept_drift = drifted[TARGET_COLUMN].copy()
    drifted[TARGET_COLUMN] = apply_concept_drift(drifted, severity, shock, rng.random(len(frame)))
    is_new_profile = shifted["person_income"] < frame["person_income"]
    return SimulatedMonth(drifted, labels_without_concept_drift, is_new_profile)


def model_risk_function(model) -> RiskFunction:
    def risk(frame: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(frame[MODEL_FEATURES])[:, 1]

    return risk


def month_generator(month_index: int) -> np.random.Generator:
    return np.random.default_rng([RANDOM_SEED, month_index])


def month_summary(month: str, simulated: SimulatedMonth) -> dict:
    production = simulated.production
    return {
        "month": month,
        "shock_severity": SHOCK_SEVERITY_BY_MONTH[month],
        "rows": len(production),
        "default_rate": round(float(production[TARGET_COLUMN].mean()), 4),
        "default_rate_without_concept_drift": round(
            float(simulated.labels_without_concept_drift.mean()), 4
        ),
        "concept_drift_defaults": int(simulated.concept_drift_defaults.sum()),
        "new_profile_share": round(float(simulated.is_new_profile.mean()), 4),
    }


def write_manifest(summaries: list[dict], shock: EconomicShock) -> None:
    SIMULATION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "random_seed": RANDOM_SEED,
        "source": str(PRODUCTION_BASE_PATH.relative_to(PRODUCTION_BASE_PATH.parents[2])),
        "data_drifted_features": DATA_DRIFTED_FEATURES,
        "concept_drift": "stress defaults: P(y=1|X) rises with debt burden and housing exposure",
        "shock_at_full_severity": shock.as_dict(),
        "months": summaries,
    }
    SIMULATION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def concept_drift_truth(month: str, simulated: SimulatedMonth) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COLUMN: simulated.production[ID_COLUMN],
            MONTH_COLUMN: month,
            "loan_status_without_concept_drift": simulated.labels_without_concept_drift,
            "concept_drift_default": simulated.concept_drift_defaults,
        }
    )


@tracked_step("simulate")
def simulate_production(shock: EconomicShock = DEFAULT_SHOCK) -> None:
    log = get_pipeline_logger("simulate")
    risk = model_risk_function(joblib.load(MODEL_PATH))
    months = split_into_months(
        pd.read_parquet(PRODUCTION_BASE_PATH), np.random.default_rng(RANDOM_SEED)
    )
    MONTHLY_PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    summaries, truth = [], []
    for index, (month, frame) in enumerate(months.items()):
        simulated = simulate_month(
            frame, SHOCK_SEVERITY_BY_MONTH[month], risk, month_generator(index), shock
        )
        production = enforce_contract(simulated.production, LABELED_CREDIT_CONTRACT)
        production.to_parquet(MONTHLY_PRODUCTION_DIR / f"{month}.parquet", index=False)
        summaries.append(month_summary(month, simulated))
        truth.append(concept_drift_truth(month, simulated))
        log.info("production_month_simulated", extra=summaries[-1])
    CONCEPT_DRIFT_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(truth, ignore_index=True).to_parquet(CONCEPT_DRIFT_TRUTH_PATH, index=False)
    write_manifest(summaries, shock)


if __name__ == "__main__":
    simulate_production()
