.PHONY: install lint test fetch run api web

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest

fetch:
	python scripts/fetch_pubmed.py --protocol config/protocol.yaml --output-dir data/raw

run:
	python scripts/run_pipeline.py --protocol config/protocol.yaml --raw-glob "data/raw/pubmed_records_*.jsonl" --period-mode dynamic --with-funding

api:
	python scripts/run_api.py --host 127.0.0.1 --port 8000 --reload

web:
	cd apps/web && npm run dev
