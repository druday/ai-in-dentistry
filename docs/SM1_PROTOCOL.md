# Supplementary Material SM1: Search and Classification Protocol

This file records the key protocol elements referenced in the manuscript.

## Retrieval date

- `2026-02-17`
- Publication-date upper bound in query: **`2025-12-31`**

## PubMed query blocks

The full executable query strings are stored in `config/protocol.yaml` under `queries`.

- `ai_dentistry_1930_2019`
- `ai_dentistry_2020_2025`

## Temporal bins

Default hardcoded bins:

- 1946-2008
- 2009-2016
- 2017-2020
- 2021-2022
- 2023-2024
- 2025-2025

Optional mode:

- Dynamically balanced six-bin segmentation via `temporal_segmentation.mode: balanced`

## Institutional domain keyword sets

Configured in `config/protocol.yaml` under:

- `classification.dental`
- `classification.medical`
- `classification.technical`

Institutions not matching these keyword sets are assigned to `Other`.

## Notes

- Default implementation uses exact raw affiliation strings as institution labels, with within-record deduplication.
- Legacy keyword-segment extraction remains available via `preprocessing.institution_extraction_mode: keyword_hint_segment`.
- Comprehensive cross-language institutional disambiguation is intentionally out of scope, as described in the manuscript.
