from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PRODUCTION_DIR = DATA_DIR / "production"
BATCHES_DIR = DATA_DIR / "batches"
VALIDATED_DIR = DATA_DIR / "validated"
QUARANTINE_DIR = DATA_DIR / "quarantine"
SAMPLE_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VALIDATION_REPORTS_DIR = REPORTS_DIR / "validation"
MODEL_REPORTS_DIR = REPORTS_DIR / "model"
LOGS_DIR = PROJECT_ROOT / "logs"

KAGGLE_DATASET = "laotse/credit-risk-dataset"
RAW_FILE_NAME = "credit_risk_dataset.csv"
RAW_FILE_SHA256 = "ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8"
RAW_PATH = RAW_DIR / RAW_FILE_NAME

REFERENCE_PATH = REFERENCE_DIR / "reference.parquet"
PRODUCTION_BASE_PATH = PRODUCTION_DIR / "production_base.parquet"
REFERENCE_SAMPLE_PATH = SAMPLE_DIR / "reference_sample.csv"
RAW_CURATION_REPORT_PATH = VALIDATION_REPORTS_DIR / "raw_curation_report.json"
RAW_CURATION_FAILURE_CASES_PATH = VALIDATION_REPORTS_DIR / "raw_curation_failure_cases.csv"
RAW_REJECTED_ROWS_PATH = QUARANTINE_DIR / "raw_rejected_rows.csv"
MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline.jsonl"

RANDOM_SEED = 42
PRODUCTION_FRACTION = 0.4
MODEL_TEST_FRACTION = 0.25
SAMPLE_SIZE = 500

ID_COLUMN = "application_id"
TARGET_COLUMN = "loan_status"

CATEGORY_DOMAINS = {
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT",
    ],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}

PROTECTED_ATTRIBUTES = ["person_age"]

NUMERIC_FEATURES = [
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_FEATURES = list(CATEGORY_DOMAINS)
MODEL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
