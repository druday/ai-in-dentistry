import json
from pathlib import Path

import pandas as pd

from ai_dentistry.pipeline import _load_all_raw_records
from ai_dentistry.pipeline import run_pipeline


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_load_all_raw_records_supports_absolute_glob(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    file_path = raw_dir / "pubmed_records_test.jsonl"
    _write_jsonl(file_path, [{"PMID": "1"}, {"PMID": "2"}])

    absolute_pattern = str(raw_dir / "pubmed_records_*.jsonl")
    rows = _load_all_raw_records(absolute_pattern, project_root=tmp_path)

    assert len(rows) == 2


def test_run_pipeline_enforces_study_year_ceiling(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "outputs" / "tables"
    networks_dir = tmp_path / "outputs" / "networks"
    html_dir = tmp_path / "outputs" / "networks_html"
    processed_dir = tmp_path / "data" / "processed"
    raw_file = raw_dir / "pubmed_records_test.jsonl"

    records = [
        {
            "PMID": "100",
            "TI": "In range",
            "DP": "2025",
            "AD": ["School of Dentistry, University A, United States."],
        },
        {
            "PMID": "200",
            "TI": "Out of range",
            "DP": "2026",
            "AD": ["Department of Medicine, University B, United States."],
        },
    ]
    _write_jsonl(raw_file, records)

    protocol = {
        "study": {
            "manuscript_period_start": 1946,
            "manuscript_period_end": 2025,
        },
        "queries": [{"id": "q1", "query": "dummy"}],
        "temporal_segmentation": {
            "mode": "fixed",
            "fixed_periods": [
                {"label": "2024-2025", "start_year": 2024, "end_year": 2025},
            ],
        },
        "preprocessing": {
            "remove_punctuation": True,
            "institution_keyword_hints": ["university", "school", "department"],
        },
        "classification": {
            "dental": ["dental"],
            "medical": ["medical", "medicine", "hospital"],
            "technical": ["engineering", "computer science"],
        },
        "outputs": {
            "publications_table": str(processed_dir / "publications.parquet"),
            "publication_summary": str(output_dir / "publication_counts.csv"),
            "period_definitions": str(output_dir / "period_definitions.csv"),
            "global_metrics": str(output_dir / "global.csv"),
            "core_newcomer_metrics": str(output_dir / "core.csv"),
            "cluster_metrics": str(output_dir / "cluster.csv"),
            "region_summary": str(output_dir / "region.csv"),
            "centrality_dir": str(output_dir / "centrality"),
            "graph_dir": str(networks_dir),
            "graph_html_dir": str(html_dir),
        },
        "visualization": {"export_html": False},
    }
    protocol_file = tmp_path / "config.yaml"
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")

    summary = run_pipeline(
        protocol_path=protocol_file,
        raw_glob=str(raw_file),
        project_root=tmp_path,
        clean_output_dirs=True,
    )

    assert summary["raw_records"] == 2
    assert summary["publications_after_preprocessing"] == 1

    publications_df = pd.read_parquet(processed_dir / "publications.parquet")
    assert len(publications_df) == 1
    institutions_cell = str(publications_df.iloc[0]["institutions"])
    assert "School of Dentistry, University A, United States." in institutions_cell
