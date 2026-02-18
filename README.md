# AI in Dentistry: Temporal Institutional Collaboration Analysis

This repository contains the reproducible analysis pipeline for:

**From Silos to Synergy: A Temporal Network Analysis of AI-Dentistry Collaborations (1946-2025)**.

The pipeline operationalizes the manuscript Methods using PubMed records, institutional affiliation parsing, period-specific collaboration networks, and longitudinal network metrics.

PubMed search is explicitly hard-capped to **2025-12-31** to exclude 2026 publications.

## Repository layout

```text
config/
  protocol.yaml                 # frozen protocol: queries, periods, keywords, outputs
data/
  raw/                          # PubMed snapshots (JSONL; generated, gitignored)
  interim/                      # intermediate artifacts (generated, gitignored)
  processed/                    # publication tables (generated, gitignored)
docs/
  METHODS_TRACEABILITY.md       # manuscript Methods -> code mapping
  SM1_PROTOCOL.md               # supplementary query and keyword specification
notebooks/
  (optional cleaned analysis notebooks)
outputs/
  tables/                       # generated result tables
  networks/                     # period graph files (GraphML)
  networks_html/                # interactive period network visualizations (HTML)
scripts/
  fetch_pubmed.py               # data acquisition from Entrez
  run_pipeline.py               # end-to-end preprocessing + analysis
src/ai_dentistry/
  package modules implementing the pipeline
tests/
  unit tests for method-critical logic
```

## Reproducibility guarantees in this repo

1. Fixed temporal bins with final period **2025-2025**.
2. Protocol-driven query definitions in `config/protocol.yaml`.
3. Environment-variable based Entrez credentials (`.env.example`).
4. Deterministic institution extraction from raw PubMed affiliation strings (exact mode by default) with within-publication deduplication.
5. Explicit institutional domain classification rules (Dental/Medical/Technical/Other).
6. Re-runnable metrics pipeline that writes standardized CSV/GraphML outputs.

## Quick start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2) Configure Entrez credentials

```bash
cp .env.example .env
# Edit .env and set NCBI_EMAIL (and optional NCBI_API_KEY)
set -a; source .env; set +a
```

### 3) Fetch raw PubMed snapshots

```bash
python scripts/fetch_pubmed.py \
  --protocol config/protocol.yaml \
  --output-dir data/raw \
  --clean-output
```

### 4) Run the full pipeline

```bash
python scripts/run_pipeline.py \
  --protocol config/protocol.yaml \
  --raw-glob "data/raw/pubmed_records_ai_dentistry_*.jsonl" \
  --clean-output
```

### 5) Choose cohort strategy (optional)

Default mode is fixed bins from `config/protocol.yaml`.

```bash
# Use fixed hardcoded bins
python scripts/run_pipeline.py --period-mode fixed --clean-output

# Use dynamically balanced bins (same number of bins; data-driven boundaries)
python scripts/run_pipeline.py --period-mode balanced --clean-output
```

### 5b) Institution extraction mode (optional)

Default is exact raw affiliation labels:

```yaml
preprocessing:
  institution_extraction_mode: "exact_affiliation"
```

Legacy collapsed-label behavior is still available for comparison:

```yaml
preprocessing:
  institution_extraction_mode: "keyword_hint_segment"
```

### 6) Run quality checks

```bash
ruff check .
pytest
```

## Outputs

Primary generated outputs:

- `data/processed/publications.parquet`
- `outputs/tables/publication_counts_by_period.csv`
- `outputs/tables/global_network_metrics.csv`
- `outputs/tables/core_newcomer_metrics.csv`
- `outputs/tables/cluster_interdisciplinarity_metrics.csv`
- `outputs/tables/who_region_summary.csv`
- `outputs/tables/period_definitions.csv`
- `outputs/networks/institutions_<period>.graphml`
- `outputs/networks_html/institutions_<period>.html`

Flow visualization prototypes:

- `outputs/figures/flow_options/option1_all_institutions_role_flow.(png|pdf)`
- `outputs/figures/flow_options/option2_field_small_multiples_role_flow.(png|pdf)`
- `outputs/figures/flow_options/option3_top_institution_trajectories.(png|pdf)`
- `outputs/figures/flow_options/flow_transition_counts.csv`
- `outputs/figures/flow_options/role_flow_sankey.html`
- `outputs/figures/flow_options/role_flow_sankey_fields.html`

Generate them with:

```bash
MPLBACKEND=Agg python scripts/generate_flow_prototypes.py \
  --project-root . \
  --output-dir outputs/figures/flow_options
```

Interactive role-flow Sankey (HTML, recommended):

```bash
python scripts/generate_interactive_sankey.py \
  --project-root . \
  --mode role \
  --output-html outputs/figures/flow_options/role_flow_sankey.html \
  --write-field-small-multiples \
  --field-small-multiples-html outputs/figures/flow_options/role_flow_sankey_fields.html \
  --save-data-csv
```

Legacy institution-level Sankey (can be visually dense):

```bash
python scripts/generate_interactive_sankey.py \
  --project-root . \
  --mode institution \
  --output-html outputs/figures/flow_options/institution_flow_sankey.html \
  --top-n-institutions 120 \
  --min-periods-present 2 \
  --save-data-csv
```

## Existing source artifacts

Original notebook and manuscript currently in project root:

- `AI in Dentistry - Final Analysis.ipynb`
- `From Silos to Synergy Manuscript v1_in_progress.docx`

They are preserved as source artifacts; the scripted pipeline is the reproducible execution path.
