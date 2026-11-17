import pandas as pd
import pytest

from credit_scoring.config import (
    ID_COLUMN,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    REFERENCE_SAMPLE_PATH,
    TARGET_COLUMN,
)
from credit_scoring.governance.fairness import (
    age_group,
    disparity_summary,
    outcomes_by_age_group,
    proxy_correlations,
)


def scored_applicants(approval_by_age: dict[int, float], rows_per_age: int = 100) -> pd.DataFrame:
    frames = []
    for age, approval_rate in approval_by_age.items():
        approved = round(rows_per_age * approval_rate)
        frames.append(
            pd.DataFrame(
                {
                    "person_age": age,
                    TARGET_COLUMN: [0, 1] * (rows_per_age // 2),
                    PREDICTION_COLUMN: [0] * approved + [1] * (rows_per_age - approved),
                    PROBABILITY_COLUMN: 0.3,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_age_groups_cover_the_contract_age_range():
    groups = age_group(pd.Series([18, 25, 26, 35, 36, 100]))
    assert groups.tolist() == ["18-25", "18-25", "26-35", "26-35", "36+", "36+"]


def test_equal_approval_rates_pass_the_four_fifths_rule():
    by_group = outcomes_by_age_group(scored_applicants({20: 0.8, 30: 0.8, 40: 0.8}), "test")
    summary = disparity_summary(by_group)
    assert summary["disparate_impact_ratio"] == pytest.approx(1.0)
    assert summary["passes_four_fifths_rule"]


def test_a_group_approved_far_less_often_fails_the_four_fifths_rule():
    by_group = outcomes_by_age_group(scored_applicants({20: 0.5, 30: 0.9, 40: 0.9}), "test")
    summary = disparity_summary(by_group)
    assert summary["least_approved_group"] == "18-25"
    assert summary["disparate_impact_ratio"] == pytest.approx(0.5 / 0.9, abs=1e-3)
    assert not summary["passes_four_fifths_rule"]


def test_credit_history_length_is_flagged_as_an_age_proxy():
    reference = pd.read_csv(REFERENCE_SAMPLE_PATH, dtype={ID_COLUMN: "string"})
    proxies = proxy_correlations(reference).set_index("feature")
    assert proxies.loc["cb_person_cred_hist_length", "strong_proxy"]
    assert not proxies.loc["loan_int_rate", "strong_proxy"]
