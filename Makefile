.PHONY: install data batches validate-good validate-bad train test lint stage1 clean

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

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

stage1: data train validate-good
	-$(MAKE) validate-bad

clean:
	rm -rf data/raw data/reference data/production data/batches data/validated data/quarantine
	rm -rf models logs reports
