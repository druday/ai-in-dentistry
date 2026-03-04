from __future__ import annotations

from datetime import datetime, timezone
from glob import glob
import hashlib
import json
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
    Period,
    assign_period_label,
    load_protocol,
    periods_to_rows,
    resolve_periods,
)
from ai_dentistry.funding import (
    FUNDING_OUTPUT_ORDER,
    default_funding_config,
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
from ai_dentistry.role_flow import (
    FIELD_ORDER,
    build_role_transition_counts,
    write_role_sankey_html,
    write_role_sankey_small_multiples_html,
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


def _serialize_json_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column not in output.columns:
            continue
        output[column] = output[column].apply(lambda x: json.dumps(x, ensure_ascii=False))
    return output


def _save_publications_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _serialize_json_columns(
        df,
        [
            "institutions",
            "countries",
            "institution_country_pairs",
            "affiliation_raw",
            "grant_entries",
            "funding_flags",
        ],
    )
    serializable.to_parquet(out_path, index=False)


def _build_region_summary_df(publications_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in publications_df.iterrows():
        pairs = row.get("institution_country_pairs", [])
        if isinstance(pairs, str):
            try:
                pairs = json.loads(pairs)
            except Exception:
                pairs = []
        if not isinstance(pairs, list):
            continue
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            country = str(pair.get("country", "")).strip()
            if not country:
                continue
            rows.append(
                {
                    "period": str(row["period"]),
                    "pmid": str(row["pmid"]),
                    "institution": str(pair.get("institution", "")).strip(),
                    "country": country,
                    "who_region": map_country_to_who_region(country),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "period",
                "who_region",
                "publication_count",
                "country_count",
                "institution_count",
            ]
        )

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
    return summary


def _save_region_summary(publications_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _build_region_summary_df(publications_df).to_csv(output_path, index=False)


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


def _period_publication_summary(df: pd.DataFrame, period_labels: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            {
                "period": period_labels,
                "publication_count": [0] * len(period_labels),
                "institution_mentions": [0] * len(period_labels),
            }
        )

    summary = (
        df.groupby("period", as_index=False, observed=False)
        .agg(
            publication_count=("pmid", "count"),
            institution_mentions=("institutions", lambda col: int(sum(len(v) for v in col))),
        )
        .assign(period=lambda x: x["period"].astype(str))
    )
    all_periods = pd.DataFrame({"period": period_labels})
    merged = (
        all_periods.merge(summary, on="period", how="left")
        .fillna({"publication_count": 0, "institution_mentions": 0})
        .astype({"publication_count": int, "institution_mentions": int})
    )
    return merged


def _write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _funding_slug(category: str) -> str:
    return str(category).strip().lower().replace("-", "_").replace(" ", "_")


def _analyze_publication_subset(
    publications_df: pd.DataFrame,
    periods: list[Period],
    classifier,
    graph_dir: Path,
    graph_html_dir: Path,
    export_html: bool,
    node_colors: dict[str, str],
    centrality_dir: Path | None = None,
    clean_output_dirs: bool = False,
) -> dict[str, Any]:
    period_labels = [period.label for period in periods]
    if clean_output_dirs:
        _clean_directory_files(graph_dir, suffixes=(".graphml",))
        if export_html:
            _clean_directory_files(graph_html_dir, suffixes=(".html",))
        if centrality_dir is not None:
            _clean_directory_files(centrality_dir, suffixes=(".csv",))

    graph_dir.mkdir(parents=True, exist_ok=True)
    if export_html:
        graph_html_dir.mkdir(parents=True, exist_ok=True)
    if centrality_dir is not None:
        centrality_dir.mkdir(parents=True, exist_ok=True)

    period_graphs: dict[str, nx.Graph] = {}
    global_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []

    for period in periods:
        period_df = publications_df[publications_df["period"] == period.label]
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

        if centrality_dir is not None:
            centrality = top_centrality_table(graph, top_n=10, use_lcc_for_path_metrics=True)
            centrality.to_csv(
                centrality_dir / f"centrality_top10_{period.label.replace('-', '_')}.csv",
                index=False,
            )

    core_rows: list[dict[str, Any]] = []
    for left, right in zip(periods, periods[1:], strict=False):
        metrics = core_newcomer_metrics(period_graphs[left.label], period_graphs[right.label])
        core_rows.append({"from": left.label, "to": right.label, **metrics})

    return {
        "period_labels": period_labels,
        "period_graphs": period_graphs,
        "publication_summary": _period_publication_summary(publications_df, period_labels),
        "global_metrics": pd.DataFrame(global_rows),
        "cluster_metrics": pd.DataFrame(cluster_rows),
        "core_newcomer_metrics": pd.DataFrame(core_rows),
        "region_summary": _build_region_summary_df(publications_df.assign(period=publications_df["period"].astype(str))),
    }


def _write_run_manifest(
    *,
    root: Path,
    protocol: dict[str, Any],
    protocol_path: str | Path,
    raw_glob: str,
    period_mode_override: str | None,
    with_funding: bool,
    funding_policy_override: str | None,
    outputs: dict[str, str],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "outputs" / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    protocol_bytes = json.dumps(protocol, sort_keys=True).encode("utf-8")
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    manifest = {
        "run_timestamp_utc": timestamp,
        "protocol_path": str(Path(protocol_path)),
        "protocol_sha256": protocol_hash,
        "raw_glob": raw_glob,
        "period_mode_override": period_mode_override,
        "with_funding": with_funding,
        "funding_policy_override": funding_policy_override,
        "outputs": outputs,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def run_pipeline(
    protocol_path: str | Path = "config/protocol.yaml",
    raw_glob: str = "data/raw/pubmed_records_*.jsonl",
    project_root: str | Path = ".",
    period_mode_override: str | None = None,
    clean_output_dirs: bool = False,
    with_funding: bool = True,
    funding_policy_override: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    protocol = load_protocol(_resolve_path(str(protocol_path), root))
    preprocessing_cfg = protocol.get("preprocessing", {})
    funding_cfg = {**default_funding_config(), **protocol.get("funding", {})}
    if funding_policy_override:
        funding_cfg["classification_policy"] = funding_policy_override

    records = _load_all_raw_records(raw_glob=raw_glob, project_root=root)
    transformed = transform_records_to_publications(
        records=records,
        keyword_hints=preprocessing_cfg.get("institution_keyword_hints", []),
        remove_punctuation=bool(preprocessing_cfg.get("remove_punctuation", True)),
        extraction_mode=str(
            preprocessing_cfg.get("institution_extraction_mode", "exact_affiliation")
        ),
        funding_cfg=funding_cfg,
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

    period_def_out = _resolve_path(
        protocol["outputs"].get("period_definitions", "outputs/tables/period_definitions.csv"),
        root,
    )
    _write_csv(pd.DataFrame(periods_to_rows(periods)), period_def_out)

    patterns = compile_category_patterns(protocol["classification"])
    classifier = make_classifier(patterns)

    vis_cfg = protocol.get("visualization", {})
    export_html = bool(vis_cfg.get("export_html", True))
    node_colors = vis_cfg.get("node_colors", {})

    graph_dir = _resolve_path(protocol["outputs"]["graph_dir"], root)
    graph_html_dir = _resolve_path(
        protocol["outputs"].get("graph_html_dir", "outputs/networks_html"),
        root,
    )
    centrality_dir = _resolve_path(protocol["outputs"]["centrality_dir"], root)

    base = _analyze_publication_subset(
        publications_df=df,
        periods=periods,
        classifier=classifier,
        graph_dir=graph_dir,
        graph_html_dir=graph_html_dir,
        export_html=export_html,
        node_colors=node_colors,
        centrality_dir=centrality_dir,
        clean_output_dirs=clean_output_dirs,
    )

    summary_out = _resolve_path(protocol["outputs"]["publication_summary"], root)
    _write_csv(base["publication_summary"], summary_out)

    global_out = _resolve_path(protocol["outputs"]["global_metrics"], root)
    _write_csv(base["global_metrics"], global_out)

    cluster_out = _resolve_path(protocol["outputs"]["cluster_metrics"], root)
    _write_csv(base["cluster_metrics"], cluster_out)

    core_out = _resolve_path(protocol["outputs"]["core_newcomer_metrics"], root)
    _write_csv(base["core_newcomer_metrics"], core_out)

    region_out = _resolve_path(protocol["outputs"]["region_summary"], root)
    _write_csv(base["region_summary"], region_out)

    output_registry: dict[str, str] = {
        "publications_table": str(publications_out),
        "period_definitions": str(period_def_out),
        "publication_summary": str(summary_out),
        "global_metrics": str(global_out),
        "cluster_metrics": str(cluster_out),
        "core_newcomer_metrics": str(core_out),
        "region_summary": str(region_out),
    }

    funding_enabled = bool(with_funding and funding_cfg.get("enabled", True))
    if funding_enabled:
        funding_categories = [
            str(value).strip()
            for value in funding_cfg.get("output_categories", FUNDING_OUTPUT_ORDER)
            if str(value).strip()
        ]
        if not funding_categories:
            funding_categories = list(FUNDING_OUTPUT_ORDER)

        funding_pub_rows: list[pd.DataFrame] = []
        funding_global_rows: list[pd.DataFrame] = []
        funding_cluster_rows: list[pd.DataFrame] = []
        funding_core_rows: list[pd.DataFrame] = []
        funding_region_rows: list[pd.DataFrame] = []
        funding_transition_rows: list[pd.DataFrame] = []
        comparison_rows: list[pd.DataFrame] = []

        funding_figure_root = root / "outputs" / "figures" / "flow_options" / "funding"
        funding_figure_root.mkdir(parents=True, exist_ok=True)
        if clean_output_dirs:
            _clean_directory_files(funding_figure_root, suffixes=(".html", ".csv"))

        for category in funding_categories:
            subset = df[df["funding_category"] == category].copy()
            subset["period"] = subset["period"].astype(str)

            slug = _funding_slug(category)
            funding_graph_dir = root / "outputs" / "networks_funding" / slug
            funding_graph_html_dir = root / "outputs" / "networks_html_funding" / slug
            if clean_output_dirs:
                _clean_directory_files(funding_graph_dir, suffixes=(".graphml",))
                _clean_directory_files(funding_graph_html_dir, suffixes=(".html",))

            analyzed = _analyze_publication_subset(
                publications_df=subset,
                periods=periods,
                classifier=classifier,
                graph_dir=funding_graph_dir,
                graph_html_dir=funding_graph_html_dir,
                export_html=export_html,
                node_colors=node_colors,
                centrality_dir=None,
                clean_output_dirs=False,
            )

            pub_df = analyzed["publication_summary"].copy()
            pub_df["funding_category"] = category
            funding_pub_rows.append(pub_df)

            global_df = analyzed["global_metrics"].copy()
            global_df["funding_category"] = category
            funding_global_rows.append(global_df)

            cluster_df = analyzed["cluster_metrics"].copy()
            cluster_df["funding_category"] = category
            funding_cluster_rows.append(cluster_df)

            core_df = analyzed["core_newcomer_metrics"].copy()
            core_df["funding_category"] = category
            funding_core_rows.append(core_df)

            region_df = analyzed["region_summary"].copy()
            region_df["funding_category"] = category
            funding_region_rows.append(region_df)

            transitions = build_role_transition_counts(
                period_graphs=analyzed["period_graphs"],
                period_labels=period_labels,
                include_absent=True,
            )
            transitions["funding_category"] = category
            funding_transition_rows.append(transitions)

            sankey_path = funding_figure_root / f"{slug}_role_flow_sankey.html"
            write_role_sankey_html(
                transitions=transitions,
                period_labels=period_labels,
                out_html=sankey_path,
                title=f"Institution Role Flow Across Periods ({category})",
                subtitle="Funding-stratified transitions with Absent entry/exit states.",
                include_absent=True,
                include_period_annotations=True,
            )
            output_registry[f"funding_sankey_{slug}"] = str(sankey_path)

            if bool(funding_cfg.get("write_field_small_multiples", False)):
                field_transitions: dict[str, pd.DataFrame] = {}
                for field in FIELD_ORDER:
                    field_t = build_role_transition_counts(
                        period_graphs=analyzed["period_graphs"],
                        period_labels=period_labels,
                        include_absent=True,
                        field_filter=field,
                    )
                    if not field_t.empty:
                        field_transitions[field] = field_t
                if field_transitions:
                    field_sankey_path = funding_figure_root / f"{slug}_role_flow_sankey_fields.html"
                    write_role_sankey_small_multiples_html(
                        field_transitions=field_transitions,
                        period_labels=period_labels,
                        out_html=field_sankey_path,
                        title=f"Field-Specific Role Flows ({category})",
                        include_absent=True,
                    )
                    output_registry[f"funding_sankey_fields_{slug}"] = str(field_sankey_path)

            merged = pub_df.merge(global_df, on=["period", "funding_category"], how="left")
            if not core_df.empty:
                core_copy = core_df.rename(columns={"to": "period"}).drop(columns=["from"])
                merged = merged.merge(core_copy, on=["period", "funding_category"], how="left")
            merged["lcc_share_pct"] = merged.apply(
                lambda r: (100.0 * float(r["largest_connected_component_size"]) / float(r["nodes"]))
                if float(r["nodes"]) > 0
                else 0.0,
                axis=1,
            )
            comparison_rows.append(merged)

        funding_publication_summary = pd.concat(funding_pub_rows, ignore_index=True)
        funding_global_metrics = pd.concat(funding_global_rows, ignore_index=True)
        funding_cluster_metrics = pd.concat(funding_cluster_rows, ignore_index=True)
        funding_core_metrics = pd.concat(funding_core_rows, ignore_index=True)
        funding_region_summary = (
            pd.concat(funding_region_rows, ignore_index=True)
            if funding_region_rows
            else pd.DataFrame(
                columns=[
                    "period",
                    "who_region",
                    "publication_count",
                    "country_count",
                    "institution_count",
                    "funding_category",
                ]
            )
        )
        funding_transitions = pd.concat(funding_transition_rows, ignore_index=True)
        structural_comparison = pd.concat(comparison_rows, ignore_index=True)

        funding_publication_summary_out = _resolve_path(
            protocol["outputs"].get(
                "funding_publication_summary",
                "outputs/tables/funding_publication_counts_by_period.csv",
            ),
            root,
        )
        funding_global_out = _resolve_path(
            protocol["outputs"].get(
                "funding_global_metrics",
                "outputs/tables/funding_global_network_metrics.csv",
            ),
            root,
        )
        funding_cluster_out = _resolve_path(
            protocol["outputs"].get(
                "funding_cluster_metrics",
                "outputs/tables/funding_cluster_interdisciplinarity_metrics.csv",
            ),
            root,
        )
        funding_core_out = _resolve_path(
            protocol["outputs"].get(
                "funding_core_newcomer_metrics",
                "outputs/tables/funding_core_newcomer_metrics.csv",
            ),
            root,
        )
        funding_region_out = _resolve_path(
            protocol["outputs"].get(
                "funding_region_summary",
                "outputs/tables/funding_who_region_summary.csv",
            ),
            root,
        )
        funding_transition_out = _resolve_path(
            protocol["outputs"].get(
                "funding_role_transition_counts",
                "outputs/tables/funding_role_transition_counts.csv",
            ),
            root,
        )
        funding_comparison_out = _resolve_path(
            protocol["outputs"].get(
                "funding_structural_comparison_summary",
                "outputs/tables/funding_structural_comparison_summary.csv",
            ),
            root,
        )

        _write_csv(funding_publication_summary, funding_publication_summary_out)
        _write_csv(funding_global_metrics, funding_global_out)
        _write_csv(funding_cluster_metrics, funding_cluster_out)
        _write_csv(funding_core_metrics, funding_core_out)
        _write_csv(funding_region_summary, funding_region_out)
        _write_csv(funding_transitions, funding_transition_out)
        _write_csv(structural_comparison, funding_comparison_out)

        output_registry.update(
            {
                "funding_publication_summary": str(funding_publication_summary_out),
                "funding_global_metrics": str(funding_global_out),
                "funding_cluster_metrics": str(funding_cluster_out),
                "funding_core_newcomer_metrics": str(funding_core_out),
                "funding_region_summary": str(funding_region_out),
                "funding_role_transition_counts": str(funding_transition_out),
                "funding_structural_comparison_summary": str(funding_comparison_out),
            }
        )

    manifest_path = _write_run_manifest(
        root=root,
        protocol=protocol,
        protocol_path=protocol_path,
        raw_glob=raw_glob,
        period_mode_override=period_mode_override,
        with_funding=with_funding,
        funding_policy_override=funding_policy_override,
        outputs=output_registry,
    )
    output_registry["run_manifest"] = str(manifest_path)

    funding_counts = (
        {str(key): int(value) for key, value in df["funding_category"].value_counts().to_dict().items()}
        if "funding_category" in df.columns
        else {}
    )
    return {
        "raw_records": int(len(records)),
        "publications_after_preprocessing": int(len(df)),
        "periods": int(len(periods)),
        "funding_enabled": bool(funding_enabled),
        "funding_counts": funding_counts,
        "run_manifest": str(manifest_path),
    }
