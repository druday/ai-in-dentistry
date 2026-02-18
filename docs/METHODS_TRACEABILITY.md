# Methods Traceability

This document maps the manuscript Methods section to implementation files in this repository.

## Data Source and Search Strategy

- `scripts/fetch_pubmed.py`
- `src/ai_dentistry/pubmed.py`
- `config/protocol.yaml` (`queries`, `entrez`, `study.retrieval_date`)
- Hard publication-date ceiling (`2025-12-31`): `config/protocol.yaml::study.hard_publication_end_date`

## Unit of Analysis and Network Scope

- Unit of analysis = institution (not author): `src/ai_dentistry/affiliations.py`
- Undirected weighted inter-institution edges: `src/ai_dentistry/network_metrics.py::build_collaboration_graph`

## Temporal Segmentation

- Fixed bins (default): `config/protocol.yaml::temporal_segmentation.fixed_periods`
- Optional balanced bins (data-driven): `config/protocol.yaml::temporal_segmentation.dynamic`
- Resolver logic: `src/ai_dentistry/config.py::resolve_periods`
- Period assignment: `src/ai_dentistry/config.py::assign_period_label`

## Affiliation Parsing and Institution Resolution

- Extraction modes (default exact raw affiliation string): `src/ai_dentistry/affiliations.py`
- Protocol control: `config/protocol.yaml::preprocessing.institution_extraction_mode`
- Within-publication deduplication: `dedupe_preserve_order` and `extract_affiliation_entries`

## Institutional Field Classification

- Keyword sets: `config/protocol.yaml::classification`
- Regex compiler and classifier: `src/ai_dentistry/classification.py`

## Network Construction

- Graph building per period: `src/ai_dentistry/pipeline.py`
- Graph export format: GraphML in `outputs/networks/`
- Interactive visualization export (HTML): `src/ai_dentistry/visualization.py`

## Network Metrics

- Global metrics: `src/ai_dentistry/network_metrics.py::global_metrics`
- Core/newcomer metrics: `core_newcomer_metrics`
- Centrality metrics: `top_centrality_table` (path-sensitive metrics on largest connected component)
- Cluster interdisciplinarity: `cluster_interdisciplinarity_metrics`

## Geographic Distribution

- Country extraction from affiliations: `src/ai_dentistry/geography.py::extract_country_from_affiliation`
- WHO region mapping: `src/ai_dentistry/geography.py::WHO_REGION_MAP`
- Regional summary table: `src/ai_dentistry/pipeline.py::_save_region_summary`

## Software and Reproducibility

- Environment pinning: `requirements.txt`, `pyproject.toml`
- CI checks: `.github/workflows/ci.yml`
- Protocol-controlled reproducibility: `config/protocol.yaml`
