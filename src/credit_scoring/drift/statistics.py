import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    PSI_DRIFT_THRESHOLD,
    PSI_MAJOR_DRIFT_THRESHOLD,
    STATISTICAL_TEST_ALPHA,
)

PSI_BINS = 10
BONFERRONI_ALPHA = STATISTICAL_TEST_ALPHA / len(MODEL_FEATURES)
PSI_EMPTY_BIN_SHARE = 1e-4


def _category_shares(reference: pd.Series, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    categories = sorted(set(reference) | set(current), key=str)
    return tuple(
        sample.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
        for sample in (reference, current)
    )


def _quantile_bin_shares(reference: pd.Series, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, PSI_BINS + 1)))[1:-1]
    return tuple(
        np.bincount(np.searchsorted(edges, sample, side="right"), minlength=len(edges) + 1)
        / len(sample)
        for sample in (reference, current)
    )


def population_stability_index(
    reference: pd.Series, current: pd.Series, categorical: bool
) -> float:
    shares = _category_shares if categorical else _quantile_bin_shares
    expected, actual = (
        np.clip(share, PSI_EMPTY_BIN_SHARE, None) for share in shares(reference, current)
    )
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def kolmogorov_smirnov(reference: pd.Series, current: pd.Series) -> tuple[str, float, float]:
    result = ks_2samp(reference, current)
    return "kolmogorov_smirnov", float(result.statistic), float(result.pvalue)


def chi_square(reference: pd.Series, current: pd.Series) -> tuple[str, float, float]:
    frequencies = pd.concat(
        {"reference": reference.value_counts(), "current": current.value_counts()}, axis=1
    ).fillna(0)
    result = chi2_contingency(frequencies.T)
    return "chi_square", float(result.statistic), float(result.pvalue)


def distribution_test(feature: str, reference: pd.Series, current: pd.Series):
    test = chi_square if feature in CATEGORICAL_FEATURES else kolmogorov_smirnov
    return test(reference, current)


def psi_band(psi: float) -> str:
    if psi >= PSI_MAJOR_DRIFT_THRESHOLD:
        return "major"
    if psi >= PSI_DRIFT_THRESHOLD:
        return "moderate"
    return "stable"


def feature_drift_table(
    reference: pd.DataFrame, current: pd.DataFrame, psi_by_feature: dict[str, float]
) -> pd.DataFrame:
    rows = []
    for feature in MODEL_FEATURES:
        test, statistic, p_value = distribution_test(feature, reference[feature], current[feature])
        psi = psi_by_feature[feature]
        rows.append(
            {
                "feature": feature,
                "psi": round(psi, 4),
                "psi_band": psi_band(psi),
                "test": test,
                "test_statistic": round(statistic, 4),
                "p_value": p_value,
                "significant_change": p_value < BONFERRONI_ALPHA,
                "drifted": psi >= PSI_DRIFT_THRESHOLD,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
