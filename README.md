# Credit Scoring Observability

Sustainment and reliability layer for a credit scoring model in production: data contracts that
block bad data, drift simulation and detection, pipeline observability and LGPD governance.

FIAP MLET Tech Challenge, Phase 4.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. No Kaggle account is needed.

```bash
make install   # create the virtual environment
make stage1    # download, curate, split, train, then run the good and bad demo batches
make test      # contract and ingestion tests
make clean     # delete everything generated (data, model, logs, reports) to start from scratch
```

`make stage1` ends with `Error 1` from `validate-bad`. That is the contract rejecting the bad batch,
which is the expected result.

## Dataset

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle): 32,581
loan applications with a binary target `loan_status` (1 = loan in default, 21.8% positive rate).

| Column | Meaning |
| --- | --- |
| `person_age`, `person_income`, `person_emp_length`, `person_home_ownership` | Applicant profile |
| `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income` | Loan terms |
| `cb_person_default_on_file`, `cb_person_cred_hist_length` | Credit bureau history |

The source of this dataset is not well documented, and it is widely believed to be at least partly
simulated. It was chosen because its columns map directly onto the economic scenario: income,
interest rates, loan-to-income and employment are what inflation and unemployment move. It also
contains real defects for the contract to catch.

`make data` downloads the file anonymously with `kagglehub` and verifies its SHA-256 checksum. Raw
data is not committed. Only a 500-row sample of the curated reference data lives in `data/sample/`
for the tests.

Preparation (`credit_scoring.data.prepare`):

1. drops exact duplicate rows
2. assigns each application an opaque surrogate key (`application_id`)
3. runs the raw data through the labeled data contract and removes every row that breaks it
4. re-validates the curated data, then splits it with a fixed seed, stratified by target, into two
   physically separate files:

| Dataset | Path | Rows | Purpose |
| --- | --- | --- | --- |
| Reference | `data/reference/reference.parquet` | 17,097 | Model training and the drift baseline |
| Production base | `data/production/production_base.parquet` | 11,398 | Never used for training. Source of incoming batches and of the drifted production data |

## Stage 1: data contracts

Contracts are written with [Pandera](https://pandera.readthedocs.io/) in
`src/credit_scoring/contracts/credit_contract.py`:

- `scoring_request_contract`: an incoming scoring batch (features plus `application_id`)
- `labeled_credit_contract`: the same rules plus the target, used for raw curation and training

| # | Rule | Columns |
| --- | --- | --- |
| 1 | Exact schema: no unexpected columns (e.g. a leaked `national_id`) and no missing ones | all |
| 2 | No nulls | all |
| 3 | Values must have the declared type. Text in a numeric column is rejected | all |
| 4 | Unique, non-empty application identifier | `application_id` |
| 5 | No duplicate records (the same application replayed under a new id) | all except `application_id` |
| 6 | Age between 18 and 100 | `person_age` |
| 7 | Income > 0 and ≤ 10,000,000 | `person_income` |
| 8 | Loan amount > 0 and ≤ 1,000,000. Interest rate > 0 and ≤ 100% | `loan_amnt`, `loan_int_rate` |
| 9 | Employment and credit-history years in [0, 86] | `person_emp_length`, `cb_person_cred_hist_length` |
| 10 | Employment and credit history cannot be longer than the applicant's working life (age − 14) | cross-field |
| 11 | `loan_percent_income` within ±0.10 of `loan_amnt / person_income` | cross-field |
| 12 | Category values from a closed list | `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file` |
| 13 | Target is 0 (repaid) or 1 (default) | `loan_status` (labeled contract only) |

The bounds are sanity limits, not the ranges seen in the reference data. The contract blocks data
that cannot be valid. It does not block legitimate shifts in the population, which are drift and are
handled in Stage 2.

### The contract on real data

On the raw dataset (`reports/validation/raw_curation_report.json`), 165 exact duplicates are removed
first. The contract then rejects 3,921 applications:

| Defect | Rows |
| --- | --- |
| Missing interest rate | 3,095 |
| Missing employment length | 887 |
| Impossible ages (123, 144) | 5 |
| Employment length of 123 years | 2 |
| Employment longer than working life | 2 |

Some rows have more than one defect. 28,495 clean applications remain.

Every removed row is traceable:

| File | Content | Committed |
| --- | --- | --- |
| `reports/validation/raw_curation_report.json` | Row counts and failures per rule | yes |
| `reports/validation/raw_curation_failure_cases.csv` | One line per failure: `application_id`, column, rule and the offending value | yes |
| `data/quarantine/raw_rejected_rows.csv` | The removed rows as they were, plus a `broken_rules` column | no (derived from raw data, rebuilt by `make data`) |

### Ingestion gate

```bash
uv run python -m credit_scoring.ingest <batch.csv|batch.parquet> [--contract scoring|labeled]
```

The gate validates the whole batch lazily, so every violation is collected in one run:

- **Accepted:** the batch is written to `data/validated/`. Exit code 0.
- **Rejected:** nothing reaches `data/validated/`. The batch is copied to `data/quarantine/`, a
  report is written to `reports/validation/` (`<batch>_rejection.json` with failures per rule, and
  `<batch>_failure_cases.csv` with one line per failure, keyed by `application_id`), a
  `batch_rejected` event is logged and the exit code is **1**.

`make batches` builds two seeded demo batches from the production base: a clean 200-row
`good_batch.csv` and a `bad_batch.csv` with these injected anomalies:

| Anomaly | Count | Caught by |
| --- | --- | --- |
| Null income | 5 | `not_nullable` |
| Negative income | 3 | `greater_than(0)` |
| Under-age (16) and impossible (144) ages | 4 + 2 | `in_range(18, 100)` |
| Unknown home ownership / loan grade | 3 + 2 | `isin([...])` |
| Interest rate of 250% | 2 | `less_than_or_equal_to(100)` |
| Text in a numeric column (`"unknown"` loan amount) | 2 | `coerce_dtype('float64')` |
| Employment longer than working life | 3 | `employment_within_working_life` |
| Loan-to-income ratio inconsistent with amount and income | 3 | `loan_to_income_consistent` |
| Null application id | 1 | `not_nullable` |
| Reused application id | 2 | `field_uniqueness` |
| Application replayed under a new id | 4 | `no_duplicate_records` |
| Unexpected PII column `national_id` | 1 | `column_in_schema` |

A single bad value can break more than one rule. For example, a 16-year-old applicant also fails
the working-life checks. So the rejection report shows more failures than anomalies injected.

```bash
make validate-good   # exit 0, batch accepted
make validate-bad    # exit 1, batch rejected and quarantined
```

### Baseline model

`credit_scoring.model.train` re-validates the reference data against the labeled contract, then
trains a scikit-learn pipeline on a seeded, stratified 75/25 split of the reference data. The
pipeline is an ordinal encoder with a fixed category list, feeding a `HistGradientBoostingClassifier`
with default settings. The model is a means, not the deliverable, so it is not tuned.

`person_age` is a protected attribute and is excluded from the features (see the governance
section). It is still validated, because the contract enforces the legal minimum age.

Holdout metrics (`reports/model/model_card.json`): ROC AUC 0.936, average precision 0.882,
accuracy 0.930, recall 0.710 at a 0.5 threshold.

Permutation importance (`reports/model/feature_importance.csv`):

| Feature | Importance (ROC AUC drop) |
| --- | --- |
| `loan_percent_income` | 0.099 |
| `person_home_ownership` | 0.081 |
| `person_income` | 0.070 |
| `loan_grade` | 0.055 |
| `loan_intent` | 0.042 |
| `loan_int_rate` | 0.021 |

Income, loan-to-income and interest rate are the features an inflationary shock moves, which makes
them the drift targets for Stage 2.

### Logs

Every step writes structured JSON lines with `stage`, `run_id` and an `event` name to
`logs/pipeline.jsonl` and to stderr. These logs are the input for Stage 3 observability.

## Project layout

```
src/credit_scoring/
  config.py                 paths, seed, feature lists, category domains
  pipeline_log.py           structured JSON-lines logging
  ingest.py                 contract-enforcing ingestion gate (CLI)
  contracts/                Pandera contracts and enforcement
  data/                     download, curation and split, demo batches
  model/                    baseline training
tests/                      contract and ingestion tests
reports/                    generated evidence (validation, model)
```
