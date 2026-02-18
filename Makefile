.PHONY: install lint test fetch run

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest

fetch:
	python scripts/fetch_pubmed.py --protocol config/protocol.yaml --output-dir data/raw

run:
	python scripts/run_pipeline.py --protocol config/protocol.yaml --raw-glob "data/raw/pubmed_records_*.jsonl"
