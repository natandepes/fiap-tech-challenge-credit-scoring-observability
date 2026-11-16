import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    CATEGORY_DOMAINS,
    DECISION_THRESHOLD,
    MODEL_FEATURES,
    MODEL_PATH,
    MODEL_REPORTS_DIR,
    MODEL_TEST_FRACTION,
    NUMERIC_FEATURES,
    PROTECTED_ATTRIBUTES,
    RANDOM_SEED,
    REFERENCE_PATH,
    TARGET_COLUMN,
)
from credit_scoring.contracts.credit_contract import LABELED_CREDIT_CONTRACT
from credit_scoring.contracts.enforcement import enforce_contract
from credit_scoring.pipeline_log import get_pipeline_logger

PERMUTATION_REPEATS = 5


def build_model() -> Pipeline:
    category_encoder = OrdinalEncoder(
        categories=[CATEGORY_DOMAINS[name] for name in CATEGORICAL_FEATURES],
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", category_encoder, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    classifier = HistGradientBoostingClassifier(
        categorical_features=CATEGORICAL_FEATURES, random_state=RANDOM_SEED
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def classification_metrics(y_true: pd.Series, default_probability: pd.Series) -> dict[str, float]:
    predicted_default = (default_probability >= DECISION_THRESHOLD).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, default_probability),
        "average_precision": average_precision_score(y_true, default_probability),
        "brier_score": brier_score_loss(y_true, default_probability),
        "accuracy": accuracy_score(y_true, predicted_default),
        "precision": precision_score(y_true, predicted_default, zero_division=0),
        "recall": recall_score(y_true, predicted_default),
        "f1": f1_score(y_true, predicted_default),
        "observed_default_rate": y_true.mean(),
        "predicted_default_rate": predicted_default.mean(),
        "mean_default_probability": np.mean(default_probability),
    }
    return {name: round(float(value), 4) for name, value in metrics.items()}


def feature_importance(model, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    result = permutation_importance(
        model,
        features,
        target,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_SEED,
    )
    return (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": result.importances_mean.round(5),
                "importance_std": result.importances_std.round(5),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def write_reports(metrics: dict, importance: pd.DataFrame, training_rows: int) -> None:
    MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_card = {
        "model": "HistGradientBoostingClassifier, scikit-learn defaults",
        "target": f"{TARGET_COLUMN} (1 = loan in default)",
        "decision_threshold": DECISION_THRESHOLD,
        "features": MODEL_FEATURES,
        "excluded_protected_attributes": PROTECTED_ATTRIBUTES,
        "random_seed": RANDOM_SEED,
        "training_rows": training_rows,
        "holdout_metrics": metrics,
    }
    (MODEL_REPORTS_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2))
    importance.to_csv(MODEL_REPORTS_DIR / "feature_importance.csv", index=False)


def split_train_holdout(reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        reference,
        test_size=MODEL_TEST_FRACTION,
        stratify=reference[TARGET_COLUMN],
        random_state=RANDOM_SEED,
    )


def train_baseline() -> None:
    log = get_pipeline_logger("train")
    reference = enforce_contract(pd.read_parquet(REFERENCE_PATH), LABELED_CREDIT_CONTRACT)
    log.info("reference_contract_passed", extra={"rows": len(reference)})

    train, holdout = split_train_holdout(reference)
    model = build_model().fit(train[MODEL_FEATURES], train[TARGET_COLUMN])
    default_probability = model.predict_proba(holdout[MODEL_FEATURES])[:, 1]
    metrics = classification_metrics(holdout[TARGET_COLUMN], default_probability)
    importance = feature_importance(model, holdout[MODEL_FEATURES], holdout[TARGET_COLUMN])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    write_reports(metrics, importance, len(train))
    log.info(
        "baseline_model_trained",
        extra={
            "model_path": str(MODEL_PATH),
            "training_rows": len(train),
            "holdout_rows": len(holdout),
            **metrics,
            "top_features": importance["feature"].head(5).tolist(),
        },
    )


if __name__ == "__main__":
    train_baseline()
