# AI in Dentistry v1.0.0

Initial public reproducible release for:
**From Silos to Synergy: A Temporal Network Analysis of AI-Dentistry Collaborations (1946-2025)**.

## Highlights
- Protocol-controlled PubMed retrieval and preprocessing.
- Hard publication end-date cap at `2025-12-31`.
- Default six-bin fixed temporal cohorts aligned to manuscript.
- Per-period collaboration networks exported as GraphML and HTML.
- Reproducible network metrics and output tables.
- Interactive Sankey suite:
  - default aggregated role-flow Sankey (`role_flow_sankey.html`),
  - field-specific role-flow small multiples (`role_flow_sankey_fields.html`),
  - optional institution-level Sankey for exploratory analysis.
- Tests and CI included for core analytical logic.

## Reproducibility
- All core parameters are versioned in `config/protocol.yaml`.
- Data fetch and analysis run via:
  - `scripts/fetch_pubmed.py`
  - `scripts/run_pipeline.py`
- Environment setup documented in `README.md`.

## Known Constraints
- Institution labels use exact raw affiliations by default; this preserves granularity but does not perform cross-language disambiguation.
- PubMed access requires valid `NCBI_EMAIL` (and optional `NCBI_API_KEY`) in environment.

## Outputs of Interest
- `outputs/tables/*.csv`
- `outputs/networks/*.graphml`
- `outputs/networks_html/*.html`
- `outputs/figures/flow_options/role_flow_sankey.html`
- `outputs/figures/flow_options/role_flow_sankey_fields.html`
