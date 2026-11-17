.PHONY: install data batches validate-good validate-bad train simulate score drift metrics fairness monitor-up monitor-down test lint stage1 stage2 stage3 stage4 pipeline all clean

install:
	uv sync

data:
	uv run python -m credit_scoring.data.download
	uv run python -m credit_scoring.data.prepare

batches:
	uv run python -m credit_scoring.data.batches

validate-good: batches
	uv run python -m credit_scoring.ingest data/batches/good_batch.csv

validate-bad: batches
	uv run python -m credit_scoring.ingest data/batches/bad_batch.csv

train:
	uv run python -m credit_scoring.model.train

simulate:
	uv run python -m credit_scoring.drift.simulate

score:
	uv run python -m credit_scoring.model.score

drift:
	uv run python -m credit_scoring.drift.detect

metrics:
	uv run python -m credit_scoring.observability.export

fairness:
	uv run python -m credit_scoring.governance.fairness

monitor-up:
	docker compose -f monitoring/compose.yaml up -d

monitor-down:
	docker compose -f monitoring/compose.yaml down

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

stage1: data train validate-good
	-$(MAKE) validate-bad

stage2: simulate score drift

stage3: metrics

stage4: fairness

pipeline: stage1 stage2 stage4 stage3

all: monitor-up pipeline

clean:
	rm -rf data/raw data/reference data/production data/batches data/validated data/quarantine data/simulation
	rm -rf models logs reports
