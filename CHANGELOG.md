# Changelog

All notable changes to this project will be documented in this file.

## [v1.0.0] - 2026-02-18

### Added
- Reproducible end-to-end PubMed pipeline (`fetch_pubmed.py` -> `run_pipeline.py`) with protocol-driven configuration.
- Fixed and balanced period segmentation support, with manuscript-aligned fixed bins as default.
- Study date ceiling enforcement through `2025-12-31` in protocol and pipeline.
- Period-specific collaboration network exports as GraphML and interactive HTML.
- Interactive Sankey visualizations:
  - aggregated role-flow Sankey (default, publication-friendly),
  - field-specific role-flow Sankey small multiples,
  - optional legacy institution-level Sankey.
- Unit test suite and CI workflow for core pipeline logic.
- Documentation set for methods traceability and protocol details.

### Changed
- Institution extraction default set to exact affiliation labels (`exact_affiliation`) for disaggregated clustering.
- Added optional legacy extraction mode (`keyword_hint_segment`) for comparison.

### Notes
- Local manuscript/notebook source artifacts are intentionally excluded from repository tracking.
