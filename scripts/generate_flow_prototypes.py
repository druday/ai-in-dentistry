#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_dentistry.classification import compile_category_patterns, make_classifier
from ai_dentistry.config import load_protocol
from ai_dentistry.network_metrics import annotate_node_types, build_collaboration_graph


ROLE_ORDER = ["Core", "Semi-Peripheral", "Peripheral", "Isolate", "Absent"]
ROLE_COLORS = {
    "Core": "#1f77b4",
    "Semi-Peripheral": "#17becf",
    "Peripheral": "#ff7f0e",
    "Isolate": "#bcbd22",
    "Absent": "#c7c7c7",
}
FIELD_ORDER = ["Dental", "Medical", "Technical", "Other"]
FIELD_COLORS = {
    "Dental": "#e31a1c",
    "Medical": "#1f78b4",
    "Technical": "#33a02c",
    "Other": "#6b7280",
}


@dataclass
class NodeSpan:
    y0: float
    y1: float


def parse_listlike(x: Any) -> list[str]:
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        text = x.strip()
        if not text:
            return []
        try:
            val = json.loads(text)
            if isinstance(val, list):
                return [str(v).strip() for v in val if str(v).strip()]
        except Exception:
            pass
        try:
            val = ast.literal_eval(text)
            if isinstance(val, list):
                return [str(v).strip() for v in val if str(v).strip()]
        except Exception:
            pass
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [v.strip().strip("'\"") for v in text.split(",") if v.strip()]
    return []


def load_period_order(project_root: Path) -> list[str]:
    period_file = project_root / "outputs" / "tables" / "period_definitions.csv"
    if period_file.exists():
        period_df = pd.read_csv(period_file)
        if "label" in period_df.columns:
            return period_df["label"].astype(str).tolist()
    protocol = load_protocol(project_root / "config" / "protocol.yaml")
    fixed = (
        protocol.get("temporal_segmentation", {}).get("fixed_periods")
        or protocol.get("periods", [])
    )
    if not fixed:
        raise ValueError("No period definitions available in outputs or config.")
    return [str(row["label"]) for row in fixed]


def load_publications(project_root: Path, periods: list[str]) -> pd.DataFrame:
    pub_path = project_root / "data" / "processed" / "publications.parquet"
    if not pub_path.exists():
        raise FileNotFoundError(f"Missing processed table: {pub_path}")
    df = pd.read_parquet(pub_path)
    if "institutions" not in df.columns or "period" not in df.columns:
        raise ValueError("publications.parquet is missing required columns.")
    df = df.copy()
    df["period"] = df["period"].astype(str)
    df = df[df["period"].isin(periods)].copy()
    df["institutions_parsed"] = df["institutions"].apply(parse_listlike)
    return df


def build_role_table_from_graphs(
    project_root: Path,
    period_order: list[str],
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    graph_dir = project_root / "outputs" / "networks"
    patterns = compile_category_patterns(protocol["classification"])
    classifier = make_classifier(patterns)
    rows: list[dict[str, Any]] = []
    used_periods: list[str] = []

    for period in period_order:
        graph_path = graph_dir / f"institutions_{period.replace('-', '_')}.graphml"
        if not graph_path.exists():
            continue

        graph = nx.read_graphml(graph_path)
        annotate_node_types(graph, classifier)
        if graph.number_of_nodes() == 0:
            continue
        used_periods.append(period)

        degree = nx.degree_centrality(graph)
        non_zero = np.array([v for v in degree.values() if v > 0], dtype=float)
        q50 = float(np.quantile(non_zero, 0.50)) if len(non_zero) else 0.0
        q90 = float(np.quantile(non_zero, 0.90)) if len(non_zero) else 0.0

        for inst in graph.nodes():
            deg = float(degree.get(inst, 0.0))
            if deg <= 0:
                role = "Isolate"
            elif deg >= q90:
                role = "Core"
            elif deg >= q50:
                role = "Semi-Peripheral"
            else:
                role = "Peripheral"
            rows.append(
                {
                    "institution": inst,
                    "period": period,
                    "role": role,
                    "degree_centrality": deg,
                    "field": str(graph.nodes[inst].get("type", "Other")),
                }
            )

    role_df = pd.DataFrame(rows)
    if role_df.empty:
        return role_df, []

    order_map = {p: i for i, p in enumerate(used_periods)}
    first_seen = (
        role_df.assign(period_idx=role_df["period"].map(order_map))
        .groupby("institution", as_index=False)["period_idx"]
        .min()
        .rename(columns={"period_idx": "first_idx"})
    )
    role_df = role_df.merge(first_seen, on="institution", how="left")
    role_df["period_idx"] = role_df["period"].map(order_map)
    role_df["status"] = np.where(role_df["period_idx"] == role_df["first_idx"], "Newcomer", "Consistent")
    return role_df, used_periods


def build_role_table(
    publications: pd.DataFrame,
    period_order: list[str],
    protocol: dict[str, Any],
) -> pd.DataFrame:
    patterns = compile_category_patterns(protocol["classification"])
    classifier = make_classifier(patterns)
    rows: list[dict[str, Any]] = []

    for period in period_order:
        subset = publications[publications["period"] == period]
        graph = build_collaboration_graph(subset["institutions_parsed"].tolist())
        annotate_node_types(graph, classifier)

        if graph.number_of_nodes() == 0:
            continue

        degree = nx.degree_centrality(graph)
        non_zero = np.array([v for v in degree.values() if v > 0], dtype=float)
        q50 = float(np.quantile(non_zero, 0.50)) if len(non_zero) else 0.0
        q90 = float(np.quantile(non_zero, 0.90)) if len(non_zero) else 0.0

        for inst in graph.nodes():
            deg = float(degree.get(inst, 0.0))
            if deg <= 0:
                role = "Isolate"
            elif deg >= q90:
                role = "Core"
            elif deg >= q50:
                role = "Semi-Peripheral"
            else:
                role = "Peripheral"
            rows.append(
                {
                    "institution": inst,
                    "period": period,
                    "role": role,
                    "degree_centrality": deg,
                    "field": str(graph.nodes[inst].get("type", "Other")),
                }
            )

    role_df = pd.DataFrame(rows)
    if role_df.empty:
        raise ValueError("Role table is empty; verify processed publications exist.")

    order_map = {p: i for i, p in enumerate(period_order)}
    first_seen = (
        role_df.assign(period_idx=role_df["period"].map(order_map))
        .groupby("institution", as_index=False)["period_idx"]
        .min()
        .rename(columns={"period_idx": "first_idx"})
    )
    role_df = role_df.merge(first_seen, on="institution", how="left")
    role_df["period_idx"] = role_df["period"].map(order_map)
    role_df["status"] = np.where(role_df["period_idx"] == role_df["first_idx"], "Newcomer", "Consistent")
    return role_df


def build_transition_table(
    role_df: pd.DataFrame,
    period_order: list[str],
    field_filter: str | None = None,
) -> pd.DataFrame:
    order_map = {p: i for i, p in enumerate(period_order)}
    filtered = role_df.copy()
    if field_filter is not None:
        filtered = filtered[filtered["field"] == field_filter].copy()

    by_period: dict[str, pd.DataFrame] = {
        p: grp for p, grp in filtered.groupby("period")
    }
    transitions: list[dict[str, Any]] = []

    for left, right in zip(period_order[:-1], period_order[1:]):
        ldf = by_period.get(left, pd.DataFrame(columns=filtered.columns))
        rdf = by_period.get(right, pd.DataFrame(columns=filtered.columns))

        lmap = dict(zip(ldf["institution"], ldf["role"]))
        rmap = dict(zip(rdf["institution"], rdf["role"]))
        institutions = sorted(set(lmap) | set(rmap))

        for inst in institutions:
            src = lmap.get(inst, "Absent")
            dst = rmap.get(inst, "Absent")
            transitions.append(
                {
                    "period_from": left,
                    "period_to": right,
                    "state_from": src,
                    "state_to": dst,
                    "institution": inst,
                    "value": 1,
                    "pair_index": order_map[left],
                }
            )

    if not transitions:
        return pd.DataFrame(
            columns=["period_from", "period_to", "state_from", "state_to", "institution", "value"]
        )
    df = pd.DataFrame(transitions)
    return (
        df.groupby(["period_from", "period_to", "state_from", "state_to", "pair_index"], as_index=False)[
            "value"
        ]
        .sum()
    )


def draw_alluvial(
    ax: plt.Axes,
    transition_df: pd.DataFrame,
    stage_labels: list[str],
    states: list[str],
    title: str,
    state_colors: dict[str, str],
) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim(-0.2, len(stage_labels) - 0.8)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if transition_df.empty:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center", transform=ax.transAxes)
        return

    node_totals: dict[tuple[int, str], float] = {}
    for i, stage in enumerate(stage_labels):
        if i == len(stage_labels) - 1:
            incoming = transition_df[transition_df["period_to"] == stage].groupby("state_to")["value"].sum()
            for state in states:
                node_totals[(i, state)] = float(incoming.get(state, 0))
        else:
            outgoing = transition_df[transition_df["period_from"] == stage].groupby("state_from")["value"].sum()
            for state in states:
                node_totals[(i, state)] = float(outgoing.get(state, 0))

    stage_totals = [sum(node_totals[(i, s)] for s in states) for i in range(len(stage_labels))]
    max_total = max(stage_totals) if stage_totals else 1.0
    if max_total <= 0:
        max_total = 1.0

    gap = 0.015
    unit = (0.96 - gap * (len(states) - 1)) / max_total
    node_spans: dict[tuple[int, str], NodeSpan] = {}

    for i, stage_total in enumerate(stage_totals):
        used = stage_total * unit + gap * (len(states) - 1)
        y_top = 0.98 - (0.96 - used) / 2.0
        for state in states:
            h = node_totals[(i, state)] * unit
            y_bot = y_top - h
            node_spans[(i, state)] = NodeSpan(y0=y_bot, y1=y_top)
            y_top = y_bot - gap

    x_positions = {stage: i for i, stage in enumerate(stage_labels)}
    out_cursor = {(i, s): node_spans[(i, s)].y1 for i in range(len(stage_labels)) for s in states}
    in_cursor = {(i, s): node_spans[(i, s)].y1 for i in range(len(stage_labels)) for s in states}

    draw_df = transition_df.copy().sort_values(
        ["pair_index", "state_from", "state_to"], ascending=[True, True, True]
    )

    for _, row in draw_df.iterrows():
        sf = row["state_from"]
        st = row["state_to"]
        period_from = row["period_from"]
        period_to = row["period_to"]
        i0 = x_positions[period_from]
        i1 = x_positions[period_to]
        h = row["value"] * unit
        if h <= 0:
            continue

        y0_top = out_cursor[(i0, sf)]
        y0_bot = y0_top - h
        out_cursor[(i0, sf)] = y0_bot

        y1_top = in_cursor[(i1, st)]
        y1_bot = y1_top - h
        in_cursor[(i1, st)] = y1_bot

        t = np.linspace(0, 1, 40)
        smooth = t * t * (3 - 2 * t)
        x_curve = i0 + (i1 - i0) * t
        y_top = (1 - smooth) * y0_top + smooth * y1_top
        y_bot = (1 - smooth) * y0_bot + smooth * y1_bot

        poly_xy = np.column_stack(
            [np.concatenate([x_curve, x_curve[::-1]]), np.concatenate([y_top, y_bot[::-1]])]
        )
        face = state_colors.get(sf, "#9ca3af")
        patch = Polygon(poly_xy, closed=True, facecolor=face, edgecolor="none", alpha=0.28)
        ax.add_patch(patch)

    for i, stage in enumerate(stage_labels):
        for state in states:
            span = node_spans[(i, state)]
            height = span.y1 - span.y0
            if height <= 0:
                continue
            rect = Rectangle(
                (i - 0.045, span.y0),
                width=0.09,
                height=height,
                facecolor=state_colors.get(state, "#9ca3af"),
                edgecolor="white",
                linewidth=0.5,
                alpha=0.95,
            )
            ax.add_patch(rect)
        ax.text(i, 1.01, stage, ha="center", va="bottom", fontsize=9, rotation=20)

    legend_items = [s for s in states if s != "Absent"]
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=state_colors[s], edgecolor="none", alpha=0.9)
        for s in legend_items
    ]
    ax.legend(handles, legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)


def plot_top_institution_trajectories(
    role_df: pd.DataFrame,
    period_order: list[str],
    out_path: Path,
    top_n: int = 40,
) -> None:
    role_score = {"Isolate": 0, "Peripheral": 1, "Semi-Peripheral": 2, "Core": 3}
    order_map = {p: i for i, p in enumerate(period_order)}

    trajectory_stats = (
        role_df.assign(score=role_df["role"].map(role_score))
        .groupby("institution", as_index=False)
        .agg(
            periods_present=("period", "nunique"),
            role_variants=("score", "nunique"),
            mean_degree=("degree_centrality", "mean"),
            field=("field", lambda s: s.mode().iloc[0] if not s.mode().empty else "Other"),
        )
    )
    candidates = trajectory_stats[
        (trajectory_stats["periods_present"] >= 3) & (trajectory_stats["role_variants"] >= 2)
    ].copy()
    if candidates.empty:
        candidates = trajectory_stats[trajectory_stats["periods_present"] >= 3].copy()
    if candidates.empty:
        candidates = trajectory_stats.copy()

    summary = (
        candidates.sort_values(
            ["periods_present", "role_variants", "mean_degree"],
            ascending=[False, False, False],
        )
        .head(top_n)
    )
    keep = set(summary["institution"])
    sub = role_df[role_df["institution"].isin(keep)].copy()
    sub["period_idx"] = sub["period"].map(order_map)
    sub["role_score"] = sub["role"].map(role_score)

    fig, ax = plt.subplots(figsize=(14, 8))
    for inst, grp in sub.groupby("institution"):
        grp = grp.sort_values("period_idx")
        field = str(grp["field"].iloc[0])
        color = FIELD_COLORS.get(field, FIELD_COLORS["Other"])
        jitter = ((abs(hash(inst)) % 100) / 100.0 - 0.5) * 0.10
        y_vals = grp["role_score"] + jitter
        ax.plot(grp["period_idx"], y_vals, marker="o", linewidth=1.4, color=color, alpha=0.45)

    ax.set_xticks(range(len(period_order)))
    ax.set_xticklabels(period_order, rotation=25, ha="right")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Isolate", "Peripheral", "Semi-Peripheral", "Core"])
    ax.set_ylabel("Role Tier")
    ax.set_title("Option 3: Role Trajectories for Institutions with Role Changes")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color=FIELD_COLORS[f], lw=3, label=f)
        for f in FIELD_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run(
    project_root: Path,
    output_dir: Path,
) -> dict[str, str]:
    protocol = load_protocol(project_root / "config" / "protocol.yaml")
    period_order = load_period_order(project_root)

    role_df, used_periods = build_role_table_from_graphs(project_root, period_order, protocol)
    if len(used_periods) >= 2 and not role_df.empty:
        period_order = used_periods
    else:
        publications = load_publications(project_root, period_order)
        role_df = build_role_table(publications, period_order, protocol)

    output_dir.mkdir(parents=True, exist_ok=True)

    option1_transitions = build_transition_table(role_df, period_order)
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    draw_alluvial(
        ax=ax1,
        transition_df=option1_transitions,
        stage_labels=period_order,
        states=ROLE_ORDER,
        title="Option 1: Institution Role Flow Across Periods (All Fields Combined)",
        state_colors=ROLE_COLORS,
    )
    option1_path = output_dir / "option1_all_institutions_role_flow.png"
    option1_pdf = output_dir / "option1_all_institutions_role_flow.pdf"
    fig1.tight_layout()
    fig1.savefig(option1_path, dpi=220)
    fig1.savefig(option1_pdf)
    plt.close(fig1)

    fig2, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    for ax, field in zip(axes.flatten(), FIELD_ORDER):
        tdf = build_transition_table(role_df, period_order, field_filter=field)
        draw_alluvial(
            ax=ax,
            transition_df=tdf,
            stage_labels=period_order,
            states=ROLE_ORDER,
            title=f"Option 2: {field} Institutions",
            state_colors=ROLE_COLORS,
        )
    fig2.suptitle("Field-Specific Role Flows (Small Multiples)", fontsize=14, y=1.01)
    fig2.tight_layout()
    option2_path = output_dir / "option2_field_small_multiples_role_flow.png"
    option2_pdf = output_dir / "option2_field_small_multiples_role_flow.pdf"
    fig2.savefig(option2_path, dpi=220, bbox_inches="tight")
    fig2.savefig(option2_pdf, bbox_inches="tight")
    plt.close(fig2)

    option3_path = output_dir / "option3_top_institution_trajectories.png"
    option3_pdf = output_dir / "option3_top_institution_trajectories.pdf"
    plot_top_institution_trajectories(
        role_df=role_df,
        period_order=period_order,
        out_path=option3_path,
        top_n=25,
    )
    # Save vector copy for publication workflows.
    # Re-rendering keeps text crisp in PDF.
    plot_top_institution_trajectories(
        role_df=role_df,
        period_order=period_order,
        out_path=option3_pdf,
        top_n=25,
    )

    transitions_csv = output_dir / "flow_transition_counts.csv"
    option1_transitions.to_csv(transitions_csv, index=False)

    return {
        "option1": str(option1_path),
        "option1_pdf": str(option1_pdf),
        "option2": str(option2_path),
        "option2_pdf": str(option2_pdf),
        "option3": str(option3_path),
        "option3_pdf": str(option3_pdf),
        "transition_counts": str(transitions_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prototype institution flow visualizations (Sankey-style alternatives)."
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "figures" / "flow_options"),
        help="Output directory for prototype visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(
        project_root=Path(args.project_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
