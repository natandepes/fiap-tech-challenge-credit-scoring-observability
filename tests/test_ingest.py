import numpy as np
import pandas as pd
import pytest

from credit_scoring import ingest
from credit_scoring.config import ID_COLUMN, REFERENCE_SAMPLE_PATH, TARGET_COLUMN
from credit_scoring.data.batches import inject_anomalies


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    for name in ("QUARANTINE_DIR", "VALIDATED_DIR", "VALIDATION_REPORTS_DIR"):
        monkeypatch.setattr(ingest, name, tmp_path / name.lower())
    return tmp_path


@pytest.fixture
def scoring_batch() -> pd.DataFrame:
    sample = pd.read_csv(REFERENCE_SAMPLE_PATH, dtype={ID_COLUMN: "string"})
    return sample.drop(columns=TARGET_COLUMN)


def write_csv(frame: pd.DataFrame, path):
    frame.to_csv(path, index=False)
    return path


def test_clean_batch_is_accepted(scoring_batch, tmp_path):
    batch_path = write_csv(scoring_batch, tmp_path / "clean.csv")
    assert ingest.ingest(batch_path, "scoring") == 0
    assert (ingest.VALIDATED_DIR / "clean.parquet").exists()
    assert not ingest.QUARANTINE_DIR.exists()


def test_bad_batch_is_blocked_and_quarantined(scoring_batch, tmp_path):
    bad_batch = inject_anomalies(scoring_batch, np.random.default_rng(0))
    batch_path = write_csv(bad_batch, tmp_path / "bad.csv")
    assert ingest.ingest(batch_path, "scoring") == ingest.EXIT_REJECTED
    assert (ingest.QUARANTINE_DIR / "bad.csv").exists()
    assert (ingest.VALIDATION_REPORTS_DIR / "bad_rejection.json").exists()
    assert not ingest.VALIDATED_DIR.exists()
