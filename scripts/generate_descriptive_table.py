#!/usr/bin/env python3
# ruff: noqa: E402,E501
from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_dentistry.classification import compile_category_patterns, make_classifier
from ai_dentistry.config import load_protocol
from ai_dentistry.network_metrics import (
    annotate_node_types,
    build_collaboration_graph,
    cluster_interdisciplinarity_metrics,
    global_metrics,
    largest_connected_component_subgraph,
)

ROLE_ORDER = ["Core", "Semi-Peripheral", "Peripheral", "Isolate"]
WHO_REGION_ORDER = ["AFRO", "PAHO", "SEARO", "EMRO", "EURO", "WPRO", "Unknown"]


def role_labels_from_graph(graph: nx.Graph) -> dict[str, str]:
    if graph.number_of_nodes() == 0:
        return {}

    degree = nx.degree_centrality(graph)
    non_zero = [float(v) for v in degree.values() if float(v) > 0]
    q50 = float(pd.Series(non_zero).quantile(0.50)) if non_zero else 0.0
    q90 = float(pd.Series(non_zero).quantile(0.90)) if non_zero else 0.0

    labels: dict[str, str] = {}
    for node in graph.nodes():
        score = float(degree.get(node, 0.0))
        if score <= 0:
            labels[str(node)] = "Isolate"
        elif score >= q90:
            labels[str(node)] = "Core"
        elif score >= q50:
            labels[str(node)] = "Semi-Peripheral"
        else:
            labels[str(node)] = "Peripheral"
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-ready descriptive table across periods."
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Repository root path.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: outputs/tables/table_descriptive_by_period.csv).",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Output Markdown path (default: outputs/tables/table_descriptive_by_period.md).",
    )
    return parser.parse_args()


def parse_listlike(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def parse_institution_country_pairs(value: Any) -> list[dict[str, str]]:
    items = parse_listlike(value)
    output: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        institution = str(item.get("institution", "")).strip()
        country = str(item.get("country", "")).strip()
        output.append({"institution": institution, "country": country})
    return output


def load_period_order(project_root: Path) -> tuple[list[str], int, int]:
    period_path = project_root / "outputs" / "tables" / "period_definitions.csv"
    if period_path.exists():
        period_df = pd.read_csv(period_path)
        labels = period_df["label"].astype(str).tolist()
        start_year = int(period_df["start_year"].min())
        end_year = int(period_df["end_year"].max())
        return labels, start_year, end_year

    protocol = load_protocol(project_root / "config" / "protocol.yaml")
    fixed = (
        protocol.get("temporal_segmentation", {}).get("fixed_periods")
        or protocol.get("periods", [])
    )
    if not fixed:
        raise ValueError("No period definitions found in outputs/tables or config.")
    labels = [str(row["label"]) for row in fixed]
    start_year = int(min(int(row["start_year"]) for row in fixed))
    end_year = int(max(int(row["end_year"]) for row in fixed))
    return labels, start_year, end_year


def load_publications(project_root: Path, period_labels: list[str]) -> pd.DataFrame:
    pub_path = project_root / "data" / "processed" / "publications.parquet"
    if not pub_path.exists():
        raise FileNotFoundError(f"Missing processed publication table: {pub_path}")
    df = pd.read_parquet(pub_path)
    df = df.copy()
    df["period"] = df["period"].astype(str)
    df = df[df["period"].isin(period_labels)].copy()
    df["institutions_parsed"] = df["institutions"].apply(parse_listlike)
    df["institution_country_pairs_parsed"] = df["institution_country_pairs"].apply(
        parse_institution_country_pairs
    )
    return df


def load_graphs(project_root: Path, period_labels: list[str]) -> dict[str, nx.Graph]:
    graph_dir = project_root / "outputs" / "networks"
    graphs: dict[str, nx.Graph] = {}
    for period in period_labels:
        graph_path = graph_dir / f"institutions_{period.replace('-', '_')}.graphml"
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Missing period graph '{graph_path}'. Run pipeline before generating the table."
            )
        graphs[period] = nx.read_graphml(graph_path)
    return graphs


def role_counts_from_graph(graph: nx.Graph) -> dict[str, int]:
    counts = {role: 0 for role in ROLE_ORDER}
    for role in role_labels_from_graph(graph).values():
        counts[role] += 1
    return counts


def role_transition_metrics(
    role_by_period: dict[str, dict[str, str]],
    period_labels: list[str],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    by_target_period: dict[str, dict[str, int]] = {}
    totals = {
        "active_both": 0,
        "role_change_active": 0,
        "p_to_s": 0,
        "s_to_c": 0,
        "core_down": 0,
        "core_keep": 0,
        "core_exit": 0,
        "entry_from_absent": 0,
        "exit_to_absent": 0,
    }

    for left, right in zip(period_labels[:-1], period_labels[1:], strict=False):
        left_map = role_by_period.get(left, {})
        right_map = role_by_period.get(right, {})
        institutions = set(left_map) | set(right_map)

        row = {
            "active_both": 0,
            "role_change_active": 0,
            "p_to_s": 0,
            "s_to_c": 0,
            "core_down": 0,
            "core_keep": 0,
            "core_exit": 0,
            "entry_from_absent": 0,
            "exit_to_absent": 0,
        }

        for inst in institutions:
            source = left_map.get(inst, "Absent")
            target = right_map.get(inst, "Absent")

            if source == "Absent" and target != "Absent":
                row["entry_from_absent"] += 1
            if source != "Absent" and target == "Absent":
                row["exit_to_absent"] += 1

            if source != "Absent" and target != "Absent":
                row["active_both"] += 1
                if source != target:
                    row["role_change_active"] += 1
                if source == "Peripheral" and target == "Semi-Peripheral":
                    row["p_to_s"] += 1
                if source == "Semi-Peripheral" and target == "Core":
                    row["s_to_c"] += 1
                if source == "Core" and target in {"Semi-Peripheral", "Peripheral", "Isolate"}:
                    row["core_down"] += 1
                if source == "Core" and target == "Core":
                    row["core_keep"] += 1

            if source == "Core" and target == "Absent":
                row["core_exit"] += 1

        by_target_period[right] = row
        for key in totals:
            totals[key] += row[key]

    return by_target_period, totals


def centrality_summary(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {
            "degree_max": 0.0,
            "degree_top10_median": 0.0,
            "betweenness_max": 0.0,
            "betweenness_top10_median": 0.0,
        }

    degree_scores = nx.degree_centrality(graph)
    degree_vals = sorted((float(v) for v in degree_scores.values()), reverse=True)
    degree_top10 = degree_vals[:10]

    lcc_graph = largest_connected_component_subgraph(graph)
    betweenness_scores = (
        nx.betweenness_centrality(lcc_graph, weight="weight", normalized=True)
        if lcc_graph.number_of_nodes()
        else {}
    )
    between_vals = sorted((float(v) for v in betweenness_scores.values()), reverse=True)
    between_top10 = between_vals[:10]

    return {
        "degree_max": degree_vals[0] if degree_vals else 0.0,
        "degree_top10_median": statistics.median(degree_top10) if degree_top10 else 0.0,
        "betweenness_max": between_vals[0] if between_vals else 0.0,
        "betweenness_top10_median": statistics.median(between_top10) if between_top10 else 0.0,
    }


def aggregate_graph_metrics(graph: nx.Graph) -> dict[str, Any]:
    g = global_metrics(graph)
    c = cluster_interdisciplinarity_metrics(graph)
    roles = role_counts_from_graph(graph)
    cent = centrality_summary(graph)
    lcc_share = (
        (100.0 * int(g["largest_connected_component_size"]) / int(g["nodes"]))
        if int(g["nodes"]) > 0
        else 0.0
    )
    return {
        **g,
        **c,
        **roles,
        **cent,
        "lcc_share_pct": lcc_share,
    }


def load_region_summary(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "tables" / "who_region_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing WHO region summary: {path}")
    df = pd.read_csv(path)
    df["period"] = df["period"].astype(str)
    df["who_region"] = df["who_region"].astype(str)
    return df


def load_core_newcomer(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "tables" / "core_newcomer_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing core-newcomer summary: {path}")
    df = pd.read_csv(path)
    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)
    return df


def fmt_n(value: Any) -> str:
    return f"{int(round(float(value))):,}"


def fmt_dec(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}%"


def fmt_n_pct(n: Any, denom: Any, pct_digits: int = 1) -> str:
    n_val = int(round(float(n)))
    d_val = float(denom)
    pct = (100.0 * n_val / d_val) if d_val > 0 else 0.0
    return f"{n_val:,} ({pct:.{pct_digits}f}%)"


def empty_row(metric: str, columns: list[str]) -> dict[str, str]:
    row = {"Metric": metric}
    for col in columns:
        row[col] = ""
    return row


def markdown_table(df: pd.DataFrame) -> str:
    headers = df.columns.tolist()
    rows = [headers] + df.values.tolist()
    widths = [0] * len(headers)
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def fmt_row(values: list[Any]) -> str:
        return "| " + " | ".join(str(values[i]).ljust(widths[i]) for i in range(len(values))) + " |"

    header = fmt_row(headers)
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = "\n".join(fmt_row(list(row)) for row in df.values.tolist())
    return f"{header}\n{sep}\n{body}\n"


def generate_table(project_root: Path) -> tuple[pd.DataFrame, str]:
    period_labels, start_year, end_year = load_period_order(project_root)
    complete_col = f"Complete ({start_year}-{end_year})"
    value_cols = period_labels + [complete_col]

    protocol = load_protocol(project_root / "config" / "protocol.yaml")
    patterns = compile_category_patterns(protocol["classification"])
    classifier = make_classifier(patterns)

    publications = load_publications(project_root, period_labels)
    period_graphs = load_graphs(project_root, period_labels)
    region_df = load_region_summary(project_root)
    core_df = load_core_newcomer(project_root)

    # Publication-level summaries.
    pub_summary = (
        publications.groupby("period", as_index=False)
        .agg(
            publication_count=("pmid", "count"),
            institution_mentions=("institutions_parsed", lambda x: int(sum(len(v) for v in x))),
        )
        .set_index("period")
    )

    # Per-period graph metrics and complete-period recomputation.
    graph_metrics: dict[str, dict[str, Any]] = {}
    for period in period_labels:
        graph = period_graphs[period]
        annotate_node_types(graph, classifier)
        graph_metrics[period] = aggregate_graph_metrics(graph)

    full_graph = build_collaboration_graph(publications["institutions_parsed"].tolist())
    annotate_node_types(full_graph, classifier)
    graph_metrics[complete_col] = aggregate_graph_metrics(full_graph)

    pub_complete = {
        "publication_count": int(len(publications)),
        "institution_mentions": int(sum(len(v) for v in publications["institutions_parsed"])),
    }

    # Regional summaries.
    region_pub_by_period = (
        region_df.groupby(["period", "who_region"], as_index=False)["publication_count"].sum()
    )
    region_pub_complete = (
        region_pub_by_period.groupby("who_region", as_index=False)["publication_count"].sum()
    )
    region_pub_complete_total = float(region_pub_complete["publication_count"].sum())

    # Unique geographic entities per period and overall (complete column).
    unique_geo_by_period: dict[str, dict[str, int]] = {}
    all_countries: set[str] = set()
    all_institutions: set[str] = set()
    for period in period_labels:
        sub = publications[publications["period"] == period]
        countries: set[str] = set()
        institutions: set[str] = set()
        for entries in sub["institution_country_pairs_parsed"]:
            for pair in entries:
                country = str(pair.get("country", "")).strip()
                institution = str(pair.get("institution", "")).strip()
                if not country:
                    continue
                countries.add(country)
                all_countries.add(country)
                if institution:
                    institutions.add(institution)
                    all_institutions.add(institution)
        unique_geo_by_period[period] = {
            "country_count": len(countries),
            "institution_count": len(institutions),
        }

    country_complete_total = len(all_countries)
    institution_complete_total = len(all_institutions)

    region_pub_lookup = {
        (str(r["period"]), str(r["who_region"])): int(r["publication_count"])
        for _, r in region_pub_by_period.iterrows()
    }
    region_pub_complete_lookup = {
        str(r["who_region"]): int(r["publication_count"]) for _, r in region_pub_complete.iterrows()
    }

    role_by_period = {period: role_labels_from_graph(period_graphs[period]) for period in period_labels}
    flow_by_period, flow_totals = role_transition_metrics(role_by_period, period_labels)

    # Transition summaries (mapped to target period).
    core_lookup = {str(r["to"]): r for _, r in core_df.iterrows()}
    core_means = {
        "density_existing": float(core_df["density_existing"].mean()) if len(core_df) else 0.0,
        "density_new": float(core_df["density_new"].mean()) if len(core_df) else 0.0,
        "edge_growth_per_new": float(core_df["edge_growth_per_new"].mean()) if len(core_df) else 0.0,
        "new_nodes": int(core_df["new_nodes"].sum()) if len(core_df) else 0,
        "returning_nodes": int(core_df["returning_nodes"].sum()) if len(core_df) else 0,
    }

    rows: list[dict[str, str]] = []

    # Panel A
    rows.append(empty_row("Panel A. Corpus volume", value_cols))
    row = {"Metric": "  A1 Publications, n"}
    for col in value_cols:
        value = pub_complete["publication_count"] if col == complete_col else int(pub_summary.loc[col, "publication_count"])
        row[col] = fmt_n(value)
    rows.append(row)

    row = {"Metric": "  A2 Institution mentions, n"}
    for col in value_cols:
        value = pub_complete["institution_mentions"] if col == complete_col else int(pub_summary.loc[col, "institution_mentions"])
        row[col] = fmt_n(value)
    rows.append(row)

    row = {"Metric": "  A3 Unique institutions (nodes), n"}
    for col in value_cols:
        row[col] = fmt_n(graph_metrics[col]["nodes"])
    rows.append(row)

    row = {"Metric": "  A4 Unique collaborations (edges), n"}
    for col in value_cols:
        row[col] = fmt_n(graph_metrics[col]["edges"])
    rows.append(row)

    # Panel B
    rows.append(empty_row("Panel B. Institutional role composition, n (%)", value_cols))
    for role in ROLE_ORDER:
        row = {"Metric": f"  B{ROLE_ORDER.index(role)+1} {role}, n (%)"}
        for col in value_cols:
            row[col] = fmt_n_pct(graph_metrics[col][role], graph_metrics[col]["nodes"])
        rows.append(row)

    # Panel C
    rows.append(empty_row("Panel C. Network topology", value_cols))
    row = {"Metric": "  C1 Density"}
    for col in value_cols:
        row[col] = fmt_dec(graph_metrics[col]["density"])
    rows.append(row)

    row = {"Metric": "  C2 Average degree"}
    for col in value_cols:
        row[col] = fmt_dec(graph_metrics[col]["avg_degree"])
    rows.append(row)

    row = {"Metric": "  C3 Weighted clustering coefficient"}
    for col in value_cols:
        row[col] = fmt_dec(graph_metrics[col]["avg_clustering_weighted"])
    rows.append(row)

    row = {"Metric": "  C4 Largest connected component size, n"}
    for col in value_cols:
        row[col] = fmt_n(graph_metrics[col]["largest_connected_component_size"])
    rows.append(row)

    row = {"Metric": "  C5 Largest connected component share, %"}
    for col in value_cols:
        row[col] = fmt_pct(graph_metrics[col]["lcc_share_pct"])
    rows.append(row)

    # Panel D
    rows.append(empty_row("Panel D. Centrality summary", value_cols))
    row = {"Metric": "  D1 Max degree centrality"}
    for col in value_cols:
        row[col] = fmt_dec(graph_metrics[col]["degree_max"])
    rows.append(row)

    row = {"Metric": "  D2 Median top-10 degree centrality"}
    for col in value_cols:
        row[col] = fmt_dec(graph_metrics[col]["degree_top10_median"])
    rows.append(row)

    row = {"Metric": "  D3 Max betweenness centrality"}
    for col in value_cols:
        row[col] = fmt_dec(graph_metrics[col]["betweenness_max"])
    rows.append(row)

    row = {"Metric": "  D4 Median top-10 betweenness centrality"}
    for col in value_cols:
        row[col] = fmt_dec(graph_metrics[col]["betweenness_top10_median"])
    rows.append(row)

    # Panel E
    rows.append(empty_row("Panel E. Component interdisciplinarity, n (%)", value_cols))
    row = {"Metric": "  E1 Number of components, n"}
    for col in value_cols:
        row[col] = fmt_n(graph_metrics[col]["num_components"])
    rows.append(row)

    for idx, (label, key) in enumerate(
        [
            ("Single-field components", "single_field_components"),
            ("Two-field components", "two_field_components"),
            ("Three-field components", "three_field_components"),
        ],
        start=2,
    ):
        row = {"Metric": f"  E{idx} {label}, n (%)"}
        for col in value_cols:
            row[col] = fmt_n_pct(graph_metrics[col][key], graph_metrics[col]["num_components"])
        rows.append(row)

    # Panel F
    rows.append(empty_row("Panel F. Geographic distribution (WHO region), publications n (%)", value_cols))
    row = {"Metric": "  F1 Countries represented, n"}
    for col in value_cols:
        if col == complete_col:
            row[col] = fmt_n(country_complete_total)
        else:
            row[col] = fmt_n(unique_geo_by_period.get(col, {}).get("country_count", 0))
    rows.append(row)

    row = {"Metric": "  F2 Institutions geo-coded, n"}
    for col in value_cols:
        if col == complete_col:
            row[col] = fmt_n(institution_complete_total)
        else:
            row[col] = fmt_n(unique_geo_by_period.get(col, {}).get("institution_count", 0))
    rows.append(row)

    f_idx = 3
    for region in WHO_REGION_ORDER:
        row = {"Metric": f"  F{f_idx} {region}, n (%)"}
        for col in value_cols:
            if col == complete_col:
                n = region_pub_complete_lookup.get(region, 0)
                denom = region_pub_complete_total
            else:
                n = region_pub_lookup.get((col, region), 0)
                denom = float(
                    sum(region_pub_lookup.get((col, r), 0) for r in WHO_REGION_ORDER)
                )
            row[col] = fmt_n_pct(n, denom)
        rows.append(row)
        f_idx += 1

    # Panel G
    rows.append(empty_row("Panel G. Newcomer-returning dynamics (period t vs t-1)", value_cols))

    def transition_value(period: str, key: str) -> str:
        if period == period_labels[0]:
            return "—"
        if period not in core_lookup:
            return "—"
        return str(core_lookup[period][key])

    for idx, (label, key, style) in enumerate(
        [
            ("New institutions, n", "new_nodes", "n"),
            ("Returning institutions, n", "returning_nodes", "n"),
            ("Density among returning institutions", "density_existing", "dec"),
            ("Density among new institutions", "density_new", "dec"),
            ("Edge growth per new institution", "edge_growth_per_new", "dec"),
        ],
        start=1,
    ):
        row = {"Metric": f"  G{idx} {label}"}
        for col in value_cols:
            if col == complete_col:
                base = core_means[key]
            else:
                raw = transition_value(col, key)
                if raw == "—":
                    row[col] = "—"
                    continue
                base = raw
            if style == "n":
                row[col] = fmt_n(base)
            else:
                row[col] = fmt_dec(base)
        rows.append(row)

    # Panel H
    rows.append(empty_row("Panel H. Role-flow transition summary (period t vs t-1)", value_cols))

    def flow_value(period: str, key: str) -> str:
        if period == period_labels[0]:
            return "—"
        if period not in flow_by_period:
            return "—"
        return str(flow_by_period[period][key])

    for idx, (label, key) in enumerate(
        [
            ("Active institutions in both periods, n", "active_both"),
            ("Role changes among active institutions, n", "role_change_active"),
            ("Upward Peripheral -> Semi-Peripheral, n", "p_to_s"),
            ("Upward Semi-Peripheral -> Core, n", "s_to_c"),
            ("Downward Core -> lower tiers, n", "core_down"),
            ("Core retained (Core -> Core), n", "core_keep"),
            ("Core exits (Core -> Absent), n", "core_exit"),
            ("Entries (Absent -> active tiers), n", "entry_from_absent"),
            ("Exits (active tiers -> Absent), n", "exit_to_absent"),
        ],
        start=1,
    ):
        row = {"Metric": f"  H{idx} {label}"}
        for col in value_cols:
            if col == complete_col:
                base = flow_totals[key]
            else:
                raw = flow_value(col, key)
                if raw == "—":
                    row[col] = "—"
                    continue
                base = raw
            row[col] = fmt_n(base)
        rows.append(row)

    table_df = pd.DataFrame(rows, columns=["Metric", *value_cols])

    notes = [
        "Notes:",
        f"1) Complete-period network metrics are recomputed on pooled data ({start_year}-{end_year}), not summed from period-specific estimates.",
        "2) Betweenness centrality is computed with NetworkX weight='weight', where raw edge weights are used as path costs.",
        "3) density_existing and density_new are induced-subgraph densities with subgroup-specific possible ties as denominator.",
        "4) Interdisciplinarity uses key domains {Dental, Medical, Technical}; Other-only components are included in the total component count but excluded from key-domain numerators.",
        "5) Panel G complete-period values are transition aggregates: counts are sums across adjacent transitions; density and edge-growth terms are transition means.",
        "6) WHO-region publication percentages are calculated relative to total region-linked publication counts within each column.",
        "7) Panel H values are institution-level role transitions between adjacent periods; complete-period values are sums across all adjacent transitions.",
    ]
    markdown = markdown_table(table_df) + "\n" + "\n".join(notes) + "\n"
    return table_df, markdown


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else project_root / "outputs" / "tables" / "table_descriptive_by_period.csv"
    )
    out_md = (
        Path(args.output_md).resolve()
        if args.output_md
        else project_root / "outputs" / "tables" / "table_descriptive_by_period.md"
    )

    table_df, table_md = generate_table(project_root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(out_csv, index=False)
    out_md.write_text(table_md, encoding="utf-8")

    print(
        json.dumps(
            {
                "csv": str(out_csv),
                "markdown": str(out_md),
                "rows": int(len(table_df)),
                "columns": int(len(table_df.columns)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
