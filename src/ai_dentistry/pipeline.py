from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from ai_dentistry.affiliations import (
    deduplicate_publications,
    read_jsonl_records,
    transform_records_to_publications,
)
from ai_dentistry.classification import compile_category_patterns, make_classifier
from ai_dentistry.config import (
    assign_period_label,
    load_protocol,
    periods_to_rows,
    resolve_periods,
)
from ai_dentistry.geography import map_country_to_who_region
from ai_dentistry.network_metrics import (
    annotate_node_types,
    build_collaboration_graph,
    cluster_interdisciplinarity_metrics,
    core_newcomer_metrics,
    global_metrics,
    top_centrality_table,
)
from ai_dentistry.visualization import export_pyvis_html


def _resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def _load_all_raw_records(raw_glob: str, project_root: Path) -> list[dict[str, Any]]:
    expanded = str(Path(raw_glob).expanduser())
    if Path(expanded).is_absolute():
        files = sorted(Path(p) for p in glob(expanded))
    else:
        files = sorted(project_root.glob(expanded))

    if not files:
        raise FileNotFoundError(
            f"No raw files matched '{raw_glob}'. "
            "Run scripts/fetch_pubmed.py first, or provide existing snapshots in data/raw/."
        )

    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(read_jsonl_records(file))
    return rows


def _serialize_list_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        output[column] = output[column].apply(lambda x: json.dumps(x, ensure_ascii=False))
    return output


def _save_publications_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _serialize_list_columns(
        df,
        ["institutions", "countries", "institution_country_pairs", "affiliation_raw"],
    )
    serializable.to_parquet(out_path, index=False)


def _save_region_summary(publications_df: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for _, row in publications_df.iterrows():
        for pair in row["institution_country_pairs"]:
            country = pair.get("country", "")
            if not country:
                continue
            rows.append(
                {
                    "period": row["period"],
                    "pmid": row["pmid"],
                    "institution": pair.get("institution", ""),
                    "country": country,
                    "who_region": map_country_to_who_region(country),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame(
            columns=[
                "period",
                "who_region",
                "publication_count",
                "country_count",
                "institution_count",
            ]
        ).to_csv(output_path, index=False)
        return

    region_df = pd.DataFrame(rows).drop_duplicates(
        subset=["period", "pmid", "institution", "country", "who_region"]
    )
    summary = (
        region_df.groupby(["period", "who_region"], as_index=False)
        .agg(
            publication_count=("pmid", "nunique"),
            country_count=("country", "nunique"),
            institution_count=("institution", "nunique"),
        )
        .sort_values(["period", "publication_count", "who_region"], ascending=[True, False, True])
    )
    summary.to_csv(output_path, index=False)


def _with_period_order(df: pd.DataFrame, ordered_labels: list[str]) -> pd.DataFrame:
    output = df.copy()
    output["period"] = pd.Categorical(output["period"], categories=ordered_labels, ordered=True)
    return output


def _clean_directory_files(directory: Path, suffixes: tuple[str, ...]) -> None:
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            path.unlink()


def run_pipeline(
    protocol_path: str | Path = "config/protocol.yaml",
    raw_glob: str = "data/raw/pubmed_records_*.jsonl",
    project_root: str | Path = ".",
    period_mode_override: str | None = None,
    clean_output_dirs: bool = False,
) -> dict[str, int]:
    root = Path(project_root).resolve()
    protocol = load_protocol(_resolve_path(str(protocol_path), root))
    preprocessing_cfg = protocol.get("preprocessing", {})

    records = _load_all_raw_records(raw_glob=raw_glob, project_root=root)
    transformed = transform_records_to_publications(
        records=records,
        keyword_hints=preprocessing_cfg.get("institution_keyword_hints", []),
        remove_punctuation=bool(preprocessing_cfg.get("remove_punctuation", True)),
        extraction_mode=str(
            preprocessing_cfg.get("institution_extraction_mode", "exact_affiliation")
        ),
    )
    publications = deduplicate_publications(transformed)

    df = pd.DataFrame(publications)
    if df.empty:
        raise ValueError("No publications remain after preprocessing.")

    study_cfg = protocol.get("study", {})
    min_year = study_cfg.get("manuscript_period_start")
    max_year = study_cfg.get("manuscript_period_end")
    if min_year is not None:
        df = df[df["publication_year"] >= int(min_year)]
    if max_year is not None:
        df = df[df["publication_year"] <= int(max_year)]
    df = df.reset_index(drop=True)

    periods = resolve_periods(
        protocol=protocol,
        publication_years=df["publication_year"].tolist(),
        mode_override=period_mode_override,
    )
    period_labels = [period.label for period in periods]

    df["period"] = df["publication_year"].apply(lambda y: assign_period_label(y, periods))
    df = df.dropna(subset=["period"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No publications fall within configured period boundaries.")
    df = _with_period_order(df, period_labels).sort_values("period").reset_index(drop=True)

    publications_out = _resolve_path(protocol["outputs"]["publications_table"], root)
    _save_publications_table(df, publications_out)

    summary_out = _resolve_path(protocol["outputs"]["publication_summary"], root)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    publication_summary = (
        df.groupby("period", as_index=False, observed=False)
        .agg(
            publication_count=("pmid", "count"),
            institution_mentions=("institutions", lambda col: int(sum(len(v) for v in col))),
        )
        .sort_values("period", ascending=True)
    )
    publication_summary["period"] = publication_summary["period"].astype(str)
    publication_summary.to_csv(summary_out, index=False)

    patterns = compile_category_patterns(protocol["classification"])
    classifier = make_classifier(patterns)

    graph_dir = _resolve_path(protocol["outputs"]["graph_dir"], root)
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_html_dir = _resolve_path(
        protocol["outputs"].get("graph_html_dir", "outputs/networks_html"),
        root,
    )
    graph_html_dir.mkdir(parents=True, exist_ok=True)
    centrality_dir = _resolve_path(protocol["outputs"]["centrality_dir"], root)
    centrality_dir.mkdir(parents=True, exist_ok=True)
    period_def_out = _resolve_path(
        protocol["outputs"].get("period_definitions", "outputs/tables/period_definitions.csv"),
        root,
    )
    period_def_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(periods_to_rows(periods)).to_csv(period_def_out, index=False)

    vis_cfg = protocol.get("visualization", {})
    export_html = bool(vis_cfg.get("export_html", True))
    node_colors = vis_cfg.get("node_colors", {})

    if clean_output_dirs:
        _clean_directory_files(graph_dir, suffixes=(".graphml",))
        _clean_directory_files(graph_html_dir, suffixes=(".html",))
        _clean_directory_files(centrality_dir, suffixes=(".csv",))

    period_graphs: dict[str, nx.Graph] = {}
    global_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []

    for period in periods:
        period_df = df[df["period"] == period.label]
        graph = build_collaboration_graph(period_df["institutions"].tolist())
        annotate_node_types(graph, classifier)
        period_graphs[period.label] = graph

        graph_file = graph_dir / f"institutions_{period.label.replace('-', '_')}.graphml"
        nx.write_graphml(graph, graph_file)
        if export_html:
            html_file = graph_html_dir / f"institutions_{period.label.replace('-', '_')}.html"
            export_pyvis_html(
                graph=graph,
                out_path=html_file,
                period_label=period.label,
                node_colors=node_colors,
            )

        global_rows.append({"period": period.label, **global_metrics(graph)})
        cluster_rows.append({"period": period.label, **cluster_interdisciplinarity_metrics(graph)})

        centrality = top_centrality_table(graph, top_n=10, use_lcc_for_path_metrics=True)
        centrality.to_csv(centrality_dir / f"centrality_top10_{period.label.replace('-', '_')}.csv", index=False)

    global_out = _resolve_path(protocol["outputs"]["global_metrics"], root)
    global_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(global_rows).to_csv(global_out, index=False)

    cluster_out = _resolve_path(protocol["outputs"]["cluster_metrics"], root)
    cluster_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cluster_rows).to_csv(cluster_out, index=False)

    core_rows: list[dict[str, Any]] = []
    for left, right in zip(periods, periods[1:]):
        metrics = core_newcomer_metrics(period_graphs[left.label], period_graphs[right.label])
        core_rows.append({"from": left.label, "to": right.label, **metrics})
    core_out = _resolve_path(protocol["outputs"]["core_newcomer_metrics"], root)
    core_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(core_rows).to_csv(core_out, index=False)

    region_out = _resolve_path(protocol["outputs"]["region_summary"], root)
    _save_region_summary(df.assign(period=df["period"].astype(str)), region_out)

    return {
        "raw_records": len(records),
        "publications_after_preprocessing": len(df),
        "periods": len(periods),
    }
