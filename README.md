# Credit Scoring Observability

Sustainment and reliability layer for a credit scoring model in production: data contracts that
block bad data, drift simulation and detection, pipeline observability and LGPD governance.

FIAP MLET Tech Challenge, Phase 4.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. No Kaggle account is needed.

```bash
make install   # create the virtual environment
make stage1    # download, curate, split, train, then run the good and bad demo batches
make stage2    # simulate six production months, score them, detect drift (needs stage1)
make all       # stage1 + stage2
make test      # contract, ingestion, simulation and drift detection tests
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

## Stage 2: drift simulation and detection

```bash
make stage2   # simulate -> score -> drift
```

### Scenario

The production base is split, seeded and without overlap, into six monthly batches of about 1,900
applications (2026-01 to 2026-06). An economic crisis builds up over these months. Its strength
(severity) goes 0, 0.2, 0.4, 0.6, 0.8, 1.0. **January has severity 0 and is the control month.** It
is the untouched production base, so any drift reported for it would be a false alarm.

Each month gets the same three changes, in this order, scaled by its severity. The values below are
at full severity (June). The settings are in `src/credit_scoring/drift/scenario.py`, and the
simulation is `credit_scoring.drift.simulate`.

**1. Real changes in the population (data drift the model understands).**

| Change | Feature(s) |
| --- | --- |
| Interest rate + 4.0 pp (central bank rate hike) | `loan_int_rate` |
| 40% of applicants are a new profile: income × U(0.5, 0.8), home ownership `RENT` | `person_income`, `person_home_ownership`, `loan_percent_income` |

These applicants really are riskier. Their outcomes are resampled from the reference risk of the
changed row, using the baseline model's probability as the stand-in for that risk. The model was
trained on this kind of relationship, so it reads these applicants correctly.

**2. Nominal inflation (data drift the model misreads).** Prices and wages rise 40%: income and loan
amount × 1.4. People are not actually richer, so their outcomes do not change. The model learned
risk from pre-inflation money values. It sees higher incomes, reads them as safer and underestimates
risk. Loan-to-income does not change, because both sides inflate.

**3. Concept drift (P(y | X) changes).** Inflation eats into real disposable income, which none of
the features measure. A repaid loan becomes a default with probability `0.9 × severity × stress`.
`stress` rises linearly with `loan_percent_income` from 0.05 to 0.35 and is weighted by housing
exposure: renters 1.0, mortgage 0.5, owners 0.25. Applicants with the same features now default more
than they did in the reference data.

Changes 1 and 2 move the input distribution (data drift). Inflation also changes what a given income
means, which is why it can mislead the model. Change 3 leaves the inputs alone and changes only the
outcomes.

Every monthly batch passes the labeled data contract (`make stage2` fails otherwise). The crisis is a
legitimate change in the population, not bad data. During development the contract did catch a
simulator bug: a `loan_percent_income` that no longer matched amount ÷ income.

To measure each kind of drift separately, the simulator also stores each application's outcome
**without** concept drift (after changes 1 and 2 only) in
`data/simulation/concept_drift_ground_truth.parquet`. A real lender would never have this.
`reports/drift/simulation_manifest.json` records what was injected:

| Month | Severity | Rows | Default rate | Without concept drift | Concept-drift defaults | New profiles |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-01 | 0.0 | 1,900 | 23.9% | 23.9% | 0 | 0.0% |
| 2026-02 | 0.2 | 1,900 | 26.8% | 23.7% | 60 | 7.7% |
| 2026-03 | 0.4 | 1,900 | 35.7% | 29.0% | 128 | 18.6% |
| 2026-04 | 0.6 | 1,900 | 39.8% | 31.1% | 167 | 23.5% |
| 2026-05 | 0.8 | 1,899 | 46.7% | 36.1% | 200 | 33.1% |
| 2026-06 | 1.0 | 1,899 | 53.9% | 37.7% | 308 | 40.8% |

The reference default rate is 21.8%.

**Time is compressed.** In real lending, a loan's outcome is known only months after it is granted.
Here every month is written with its outcomes already known, so the detection step looks back after
those outcomes have arrived. Checks that need outcomes (performance, calibration) would lag that much
in a real deployment. Checks on inputs and predictions would not.

### Predictions

`credit_scoring.model.score` sends each month's applications (without the label) through the
scoring contract. It writes `application_id`, `default_probability` and `predicted_default`
(threshold 0.5) to `data/production/predictions/<month>.parquet`, and logs volume, latency and the
predicted default rate.

### Detection

`credit_scoring.drift.detect` checks each month against the reference data with **Evidently**
(`DataDriftPreset`, `ValueDrift`, `ClassificationPreset`) and with scipy tests:

| Check | Needs outcomes? | Compared | Method | Alert when |
| --- | --- | --- | --- | --- |
| Feature drift | no | Reference dataset (17,097 rows) vs. month | PSI on reference decile bins (categories for categorical features) | PSI ≥ 0.10 (warning), ≥ 0.25 (critical) |
| Statistical significance | no | same | Kolmogorov–Smirnov (numeric), chi-square (categorical), Bonferroni-corrected α = 0.05 / 10 | Reported, no alert |
| Prediction drift | no | Reference holdout scores vs. month scores | PSI of `default_probability` | PSI ≥ 0.10 |
| Performance | yes | Reference holdout vs. month | ROC AUC and recall at 0.5 | AUC drop ≥ 0.05, recall drop ≥ 0.10 |
| Calibration | yes | Month | Observed default rate − mean predicted probability | gap ≥ 5 pp |

Why these choices:

- **PSI uses quantile bins.** Evidently's built-in `psi` uses equal-width bins. Income is heavily
  right-skewed, so almost every applicant lands in one bin, and the built-in test missed a large
  income shift. Credit scoring normally bins PSI on reference deciles. That version is in
  `drift/statistics.py` and is registered as an Evidently stat test (`quantile_psi`), so the HTML
  reports and the CSV tables show the same statistic.
- **Two verdicts: material and significant.** PSI measures the size of a shift and drives the
  alerts. KS and chi-square say whether a shift is statistically real. With about 1,900 rows a
  month they also flag shifts too small to matter, so they are reported but raise no alert.
- **Calibration catches what the inputs cannot show.** Concept drift does not change the inputs,
  so only outcome-based checks see it. A model whose predicted default rate falls below the observed
  rate no longer reflects reality.

### Results

**Which features drifted.** PSI by month (`reports/drift/feature_drift_by_month.csv`). Bold marks
drift (PSI ≥ 0.10):

| Month | `loan_int_rate` | `loan_amnt` | `person_home_ownership` | `person_income` | `loan_percent_income` | Highest of the 5 untouched |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-01 | 0.007 | 0.014 | 0.002 | 0.001 | 0.008 | 0.004 |
| 2026-02 | **0.168** | 0.012 | 0.004 | 0.020 | 0.004 | 0.006 |
| 2026-03 | **0.875** | 0.063 | 0.037 | 0.047 | 0.027 | 0.007 |
| 2026-04 | **1.750** | **0.165** | 0.057 | 0.065 | 0.033 | 0.009 |
| 2026-05 | **1.966** | **0.163** | **0.107** | **0.111** | 0.081 | 0.007 |
| 2026-06 | **2.698** | **0.219** | **0.187** | 0.085 | 0.081 | 0.010 |

- **No false alarms.** No feature passes PSI 0.10 in January. The five untouched features stay
  below 0.011 in every month and are never significant (lowest p = 0.03, above the corrected 0.005).
- **The rate hike is caught first:** PSI 0.17 in February, at 20% severity.
- **The statistical tests find every injected feature.** From March on they flag exactly the five
  injected features (p < 1e-5) and nothing else.
- **PSI flags three of the five in June, and four in May.** `person_income` and
  `loan_percent_income` move less than expected, because the changes partly cancel out: inflation
  raises incomes while the new profiles lower them, and inflation leaves loan-to-income unchanged.
  Their shift is statistically clear but small. This is why the statistical tests are reported next
  to PSI.

**How much each kind of drift hurts the model.** `reports/drift/model_health_by_month.csv`. The
reference holdout scored ROC AUC 0.936 and recall 0.710:

| Month | AUC, data drift only | Recall, data drift only | AUC, with concept drift | Recall, with concept drift | Observed default rate | Mean predicted | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01 | 0.948 | 0.738 | 0.948 | 0.738 | 23.9% | 23.5% | +0.4 pp |
| 2026-02 | 0.905 | 0.696 | 0.869 | 0.624 | 26.8% | 23.5% | +3.3 pp |
| 2026-03 | 0.915 | 0.693 | 0.838 | 0.564 | 35.7% | 27.9% | +7.9 pp |
| 2026-04 | 0.910 | 0.680 | 0.840 | 0.531 | 39.8% | 30.0% | +9.9 pp |
| 2026-05 | 0.908 | 0.682 | 0.842 | 0.536 | 46.7% | 33.2% | +13.5 pp |
| 2026-06 | 0.910 | 0.679 | 0.829 | 0.486 | 53.9% | 35.6% | +18.3 pp |

- **Data drift alone does moderate damage.** Inflation makes applicants look richer than they are,
  so AUC drops from 0.95 to about 0.91 and recall from 0.74 to 0.68. The damage appears as soon as
  inflation starts and then levels off. It stays limited because the model's most important
  feature, loan-to-income, is not affected by inflation.
- **Concept drift does severe damage on top.** In June recall is 0.49: half of the loans that
  default were approved. The model predicts a 35.6% default rate against 53.9% in reality, an
  18 pp underestimate.
- **Input checks warn early. Outcome checks confirm the damage.** Feature and prediction drift
  checks run as applications arrive. They fired first on the rate hike (February) and then on loan
  amount, prediction scores, home ownership and income. Only the performance and calibration checks,
  which need outcomes, show how badly the model is failing, and only they can see the concept drift.

### Alerts

`reports/drift/alerts.json` lists each alert with month, type, subject, value, threshold and
severity. Each alert is also logged as a `drift_alert` event in `logs/pipeline.jsonl` for Stage 3.

| Month | Alerts | What fired |
| --- | --- | --- |
| 2026-01 | 0 | none (control) |
| 2026-02 | 2 | `loan_int_rate` drift, AUC drop |
| 2026-03 | 4 | rate drift, AUC drop, recall drop, calibration (+7.9 pp) |
| 2026-04 | 6 | + `loan_amnt` drift, prediction drift |
| 2026-05 | 8 | + `person_home_ownership` and `person_income` drift |
| 2026-06 | 7 | rate, loan amount and home ownership drift, prediction drift, AUC drop, recall drop, calibration (+18.3 pp) |

### Outputs

| File | Content |
| --- | --- |
| `reports/drift/evidently/data_drift_<month>.html` | Evidently feature and target drift report, for the control month (2026-01) and the latest month (2026-06) |
| `reports/drift/evidently/model_health_<month>.html` | Evidently prediction drift and classification performance against the reference holdout |
| `reports/drift/feature_drift_by_month.csv` | PSI, band, KS / chi-square statistic and p-value, and both verdicts, per feature per month |
| `reports/drift/model_health_by_month.csv` | Performance, calibration and prediction drift per month, with and without concept drift |
| `reports/drift/alerts.json` | Every alert raised |
| `reports/drift/drift_summary.json` | Baseline, features drifted in the latest month, alerts per month, injected vs. detected |
| `reports/drift/simulation_manifest.json` | What the simulator injected, month by month |

The CSV and JSON outputs are byte-for-byte reproducible. The Evidently HTML files embed random
element ids, so a rerun changes the files but not what they show.

Evidently 0.7 does not work with pandas 3, so pandas is pinned to `<3`.

## Project layout

```
src/credit_scoring/
  config.py                 paths, seed, feature lists, category domains
  pipeline_log.py           structured JSON-lines logging
  ingest.py                 contract-enforcing ingestion gate (CLI)
  contracts/                Pandera contracts and enforcement
  data/                     download, curation and split, demo batches
  model/                    baseline training and production scoring
  drift/                    shock scenario, simulation, PSI and tests, Evidently reports, alerts
tests/                      contract, ingestion, simulation and drift detection tests
reports/                    generated evidence (validation, model, drift)
```
