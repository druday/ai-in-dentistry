# Reproducibility Checklist

Use this checklist before creating or updating a release.

## Environment
- [ ] Create clean virtual environment (`python -m venv .venv`).
- [ ] Install dependencies (`pip install -r requirements-dev.txt`).
- [ ] Set `NCBI_EMAIL` (required) and optional `NCBI_API_KEY`.

## Data and Pipeline
- [ ] Verify `config/protocol.yaml` has:
  - [ ] `study.manuscript_period_end: 2025`
  - [ ] `study.hard_publication_end_date: "2025-12-31"`
  - [ ] expected fixed bins under `temporal_segmentation.fixed_periods`
- [ ] Run:
  - [ ] `python scripts/fetch_pubmed.py --protocol config/protocol.yaml --output-dir data/raw --clean-output`
  - [ ] `python scripts/run_pipeline.py --protocol config/protocol.yaml --raw-glob "data/raw/pubmed_records_ai_dentistry_*.jsonl" --period-mode fixed --clean-output`

## Visualizations
- [ ] Generate role Sankey outputs:
  - [ ] `python scripts/generate_interactive_sankey.py --project-root . --mode role --output-html outputs/figures/flow_options/role_flow_sankey.html --write-field-small-multiples --field-small-multiples-html outputs/figures/flow_options/role_flow_sankey_fields.html --save-data-csv`
- [ ] Confirm files open correctly in browser.

## Quality Checks
- [ ] Run tests: `.venv/bin/pytest -q`
- [ ] Run lint: `PYTHONPATH=src .venv/bin/ruff check .`

## Repository Hygiene
- [ ] Ensure secrets are not tracked (`.env`, API keys).
- [ ] Ensure raw/processed data and generated outputs are ignored (except `.gitkeep` where intended).
- [ ] Confirm manuscript/notebook source artifacts are not tracked.

## Release
- [ ] Update `CHANGELOG.md`.
- [ ] Commit release prep.
- [ ] Create annotated tag (example: `v1.0.0`).
- [ ] Push branch and tag.
- [ ] Create GitHub Release using `docs/RELEASE_NOTES_v1.0.0.md`.
