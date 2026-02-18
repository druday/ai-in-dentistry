#!/usr/bin/env python3
# ruff: noqa: E402,E501,I001
from __future__ import annotations

import argparse
import json
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


STATUS_FILL_COLORS = {
    "Newcomer": "#1f77b4",
    "Consistent": "#ff7f0e",
}

FIELD_BORDER_COLORS = {
    "Dental": "#2ca02c",
    "Medical": "#d62728",
    "Technical": "#9467bd",
    "Other": "#7f7f7f",
}

FIELD_LINK_COLORS = {
    "Dental": "rgba(44,160,44,0.28)",
    "Medical": "rgba(214,39,40,0.28)",
    "Technical": "rgba(148,103,189,0.28)",
    "Other": "rgba(127,127,127,0.25)",
}

FIELD_ORDER = ["Dental", "Medical", "Technical", "Other"]

ROLE_ORDER = ["Core", "Semi-Peripheral", "Peripheral", "Isolate"]
ROLE_ORDER_WITH_ABSENT = ROLE_ORDER + ["Absent"]
ROLE_COLORS = {
    "Core": "#1f77b4",
    "Semi-Peripheral": "#17becf",
    "Peripheral": "#ff7f0e",
    "Isolate": "#bcbd22",
    "Absent": "#bdbdbd",
}


def hex_to_rgba(color: str, alpha: float) -> str:
    raw = str(color).strip().lstrip("#")
    if len(raw) != 6:
        return f"rgba(156,163,175,{alpha})"
    try:
        r = int(raw[0:2], 16)
        g = int(raw[2:4], 16)
        b = int(raw[4:6], 16)
    except ValueError:
        return f"rgba(156,163,175,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"


def resolve_graph_dir(protocol: dict[str, Any], project_root: Path) -> Path:
    configured = protocol.get("outputs", {}).get("graph_dir", "outputs/networks")
    path = Path(str(configured))
    return path if path.is_absolute() else project_root / path


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
        raise ValueError("No period definitions found in outputs or config.")
    return [str(row["label"]) for row in fixed]


def load_institution_table_from_graphs(
    project_root: Path,
    period_order: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    protocol = load_protocol(project_root / "config" / "protocol.yaml")
    patterns = compile_category_patterns(protocol["classification"])
    classifier = make_classifier(patterns)
    graph_dir = resolve_graph_dir(protocol, project_root)

    rows: list[dict[str, Any]] = []
    available_periods: list[str] = []
    for period in period_order:
        graph_path = graph_dir / f"institutions_{period.replace('-', '_')}.graphml"
        if not graph_path.exists():
            continue

        graph = nx.read_graphml(graph_path)
        if graph.number_of_nodes() == 0:
            continue

        for node in graph.nodes():
            node_type = graph.nodes[node].get("type")
            if not node_type:
                graph.nodes[node]["type"] = classifier(str(node))

        degree = nx.degree_centrality(graph)
        available_periods.append(period)
        for inst in graph.nodes():
            rows.append(
                {
                    "institution": str(inst),
                    "period": period,
                    "field": str(graph.nodes[inst].get("type", "Other")),
                    "degree_centrality": float(degree.get(inst, 0.0)),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No GraphML period networks found. Run pipeline first.")

    order_map = {p: i for i, p in enumerate(available_periods)}
    first_seen = (
        table.assign(period_idx=table["period"].map(order_map))
        .groupby("institution", as_index=False)["period_idx"]
        .min()
        .rename(columns={"period_idx": "first_idx"})
    )
    table = table.merge(first_seen, on="institution", how="left")
    table["period_idx"] = table["period"].map(order_map)
    table["status"] = table.apply(
        lambda row: "Newcomer" if row["period_idx"] == row["first_idx"] else "Consistent",
        axis=1,
    )
    return table, available_periods


def build_institution_sankey_frames(
    table: pd.DataFrame,
    period_order: list[str],
    top_n_institutions: int,
    min_periods_present: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        table.groupby("institution", as_index=False)
        .agg(
            periods_present=("period", "nunique"),
            mean_degree=("degree_centrality", "mean"),
            field=("field", lambda s: s.mode().iloc[0] if not s.mode().empty else "Other"),
        )
        .sort_values(["periods_present", "mean_degree"], ascending=[False, False])
    )
    summary = summary[summary["periods_present"] >= min_periods_present]
    summary = summary.head(top_n_institutions)
    keep = set(summary["institution"])

    filtered = table[table["institution"].isin(keep)].copy()
    filtered = filtered[filtered["period"].isin(period_order)].copy()
    order_map = {p: i for i, p in enumerate(period_order)}
    filtered["period_idx"] = filtered["period"].map(order_map)

    if filtered.empty:
        raise ValueError("No institutions passed filter settings. Relax top_n/min_periods_present.")

    nodes = filtered[
        ["period", "period_idx", "institution", "field", "status", "degree_centrality"]
    ].drop_duplicates()
    rank_map = dict(
        zip(
            summary["institution"],
            zip(summary["periods_present"], summary["mean_degree"], strict=False),
            strict=False,
        )
    )
    nodes["rank_tuple"] = nodes["institution"].map(lambda inst: rank_map.get(inst, (0, 0.0)))
    nodes["node_id"] = nodes["period"].astype(str) + "||" + nodes["institution"].astype(str)
    nodes = nodes.sort_values(
        ["period_idx", "rank_tuple", "institution"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    x_map = {
        period: (i / (len(period_order) - 1) if len(period_order) > 1 else 0.5)
        for i, period in enumerate(period_order)
    }
    nodes["x"] = nodes["period"].map(x_map).astype(float)

    y_vals: list[float] = []
    for _, grp in nodes.groupby("period", sort=False):
        n = len(grp)
        if n == 1:
            y_vals.extend([0.5])
        else:
            y_vals.extend([0.02 + (0.96 * i / (n - 1)) for i in range(n)])
    nodes["y"] = y_vals

    idx_map = {node_id: i for i, node_id in enumerate(nodes["node_id"].tolist())}

    transitions: list[dict[str, Any]] = []
    for institution, grp in filtered.groupby("institution"):
        grp = grp.sort_values("period_idx")
        periods = grp["period"].tolist()
        fields = grp["field"].tolist()
        for i in range(len(periods) - 1):
            period_from = periods[i]
            period_to = periods[i + 1]
            src_id = f"{period_from}||{institution}"
            dst_id = f"{period_to}||{institution}"
            if src_id not in idx_map or dst_id not in idx_map:
                continue
            transitions.append(
                {
                    "source": idx_map[src_id],
                    "target": idx_map[dst_id],
                    "value": 1,
                    "institution": institution,
                    "field": fields[i] if i < len(fields) else "Other",
                    "period_from": period_from,
                    "period_to": period_to,
                }
            )

    links = pd.DataFrame(transitions)
    if links.empty:
        raise ValueError("No transitions found. Lower min_periods_present or increase top_n_institutions.")
    return nodes, links


def write_institution_sankey_html(
    out_html: Path,
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    period_order: list[str],
    top_n_institutions: int,
    min_periods_present: int,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "node": {
            "label": nodes["institution"].astype(str).tolist(),
            "color": [STATUS_FILL_COLORS.get(s, STATUS_FILL_COLORS["Consistent"]) for s in nodes["status"]],
            "line_color": [FIELD_BORDER_COLORS.get(f, FIELD_BORDER_COLORS["Other"]) for f in nodes["field"]],
            "x": [round(v, 6) for v in nodes["x"].tolist()],
            "y": [round(v, 6) for v in nodes["y"].tolist()],
            "customdata": np_stack(
                nodes["period"].astype(str).tolist(),
                nodes["status"].astype(str).tolist(),
                nodes["field"].astype(str).tolist(),
                [round(v, 6) for v in nodes["degree_centrality"].tolist()],
            ),
        },
        "link": {
            "source": [int(v) for v in links["source"].tolist()],
            "target": [int(v) for v in links["target"].tolist()],
            "value": [int(v) for v in links["value"].tolist()],
            "color": [FIELD_LINK_COLORS.get(f, FIELD_LINK_COLORS["Other"]) for f in links["field"]],
            "customdata": np_stack(
                links["institution"].astype(str).tolist(),
                links["field"].astype(str).tolist(),
                links["period_from"].astype(str).tolist(),
                links["period_to"].astype(str).tolist(),
            ),
        },
        "meta": {
            "period_order": period_order,
            "top_n_institutions": top_n_institutions,
            "min_periods_present": min_periods_present,
            "node_count": int(len(nodes)),
            "link_count": int(len(links)),
        },
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Institution Flow Sankey</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 16px;
      color: #111827;
      background: #f8fafc;
    }}
    .subtitle {{ color: #4b5563; font-size: 14px; }}
    .legend {{ display: flex; gap: 24px; flex-wrap: wrap; margin: 10px 0 14px; font-size: 13px; }}
    .legend-group {{ display: flex; align-items: center; gap: 8px; }}
    .chip {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
    #chart {{
      width: 100%;
      height: 920px;
      border: 1px solid #e5e7eb;
      background: #ffffff;
      border-radius: 8px;
    }}
    .meta {{ margin-top: 10px; color: #6b7280; font-size: 12px; }}
  </style>
</head>
<body>
  <h2 style="margin:0 0 4px;">Institution Flow Across Collaboration Periods</h2>
  <div class="subtitle">
    Node fill: newcomer vs consistent. Node border: institution field. Link: institution continuation to next observed period.
  </div>
  <div class="legend">
    <div class="legend-group"><strong>Status fill</strong></div>
    <div class="legend-group"><span class="chip" style="background:{STATUS_FILL_COLORS['Newcomer']}"></span>Newcomer</div>
    <div class="legend-group"><span class="chip" style="background:{STATUS_FILL_COLORS['Consistent']}"></span>Consistent</div>
    <div class="legend-group"><strong>Field border</strong></div>
    <div class="legend-group"><span class="chip" style="background:#fff;border:2px solid {FIELD_BORDER_COLORS['Dental']}"></span>Dental</div>
    <div class="legend-group"><span class="chip" style="background:#fff;border:2px solid {FIELD_BORDER_COLORS['Medical']}"></span>Medical</div>
    <div class="legend-group"><span class="chip" style="background:#fff;border:2px solid {FIELD_BORDER_COLORS['Technical']}"></span>Technical</div>
    <div class="legend-group"><span class="chip" style="background:#fff;border:2px solid {FIELD_BORDER_COLORS['Other']}"></span>Other</div>
  </div>
  <div id="chart"></div>
  <div class="meta">
    Periods: {', '.join(period_order)} |
    Top institutions: {top_n_institutions} |
    Min periods present: {min_periods_present}
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const trace = {{
      type: "sankey",
      arrangement: "fixed",
      node: {{
        pad: 10,
        thickness: 14,
        label: payload.node.label,
        color: payload.node.color,
        x: payload.node.x,
        y: payload.node.y,
        line: {{ color: payload.node.line_color, width: 2 }},
        customdata: payload.node.customdata,
        hovertemplate:
          "<b>%{{label}}</b><br>" +
          "Period: %{{customdata[0]}}<br>" +
          "Status: %{{customdata[1]}}<br>" +
          "Field: %{{customdata[2]}}<br>" +
          "Degree centrality: %{{customdata[3]}}<extra></extra>"
      }},
      link: {{
        source: payload.link.source,
        target: payload.link.target,
        value: payload.link.value,
        color: payload.link.color,
        customdata: payload.link.customdata,
        hovertemplate:
          "Institution: %{{customdata[0]}}<br>" +
          "Field: %{{customdata[1]}}<br>" +
          "From: %{{customdata[2]}}<br>" +
          "To: %{{customdata[3]}}<extra></extra>"
      }}
    }};
    const layout = {{
      margin: {{l: 20, r: 20, t: 24, b: 20}},
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {{size: 11, color: "#111827"}}
    }};
    Plotly.newPlot("chart", [trace], layout, {{
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"]
    }});
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def load_role_table_from_graphs(
    project_root: Path,
    period_order: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    protocol = load_protocol(project_root / "config" / "protocol.yaml")
    patterns = compile_category_patterns(protocol["classification"])
    classifier = make_classifier(patterns)
    graph_dir = resolve_graph_dir(protocol, project_root)

    rows: list[dict[str, Any]] = []
    available_periods: list[str] = []
    for period in period_order:
        graph_path = graph_dir / f"institutions_{period.replace('-', '_')}.graphml"
        if not graph_path.exists():
            continue

        graph = nx.read_graphml(graph_path)
        if graph.number_of_nodes() == 0:
            continue

        for node in graph.nodes():
            node_type = graph.nodes[node].get("type")
            if not node_type:
                graph.nodes[node]["type"] = classifier(str(node))

        degree = nx.degree_centrality(graph)
        non_zero = [float(v) for v in degree.values() if float(v) > 0]
        q50 = float(pd.Series(non_zero).quantile(0.50)) if non_zero else 0.0
        q90 = float(pd.Series(non_zero).quantile(0.90)) if non_zero else 0.0
        available_periods.append(period)

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
                    "institution": str(inst),
                    "period": period,
                    "role": role,
                    "field": str(graph.nodes[inst].get("type", "Other")),
                    "degree_centrality": deg,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No GraphML period networks found. Run pipeline first.")
    return table, available_periods


def build_role_transition_table(
    role_table: pd.DataFrame,
    period_order: list[str],
    field_filter: str | None = None,
    include_absent: bool = False,
) -> pd.DataFrame:
    filtered = role_table.copy()
    if field_filter is not None:
        filtered = filtered[filtered["field"] == field_filter].copy()

    by_period: dict[str, pd.DataFrame] = {period: grp for period, grp in filtered.groupby("period")}
    transitions: list[dict[str, Any]] = []
    pair_index = {period: idx for idx, period in enumerate(period_order)}

    for left, right in zip(period_order[:-1], period_order[1:], strict=False):
        left_df = by_period.get(left, pd.DataFrame(columns=filtered.columns))
        right_df = by_period.get(right, pd.DataFrame(columns=filtered.columns))
        left_map = dict(zip(left_df["institution"], left_df["role"], strict=False))
        right_map = dict(zip(right_df["institution"], right_df["role"], strict=False))
        institutions = sorted(set(left_map) | set(right_map))

        for inst in institutions:
            source_role = left_map.get(inst, "Absent")
            target_role = right_map.get(inst, "Absent")
            if not include_absent and ("Absent" in (source_role, target_role)):
                continue
            transitions.append(
                {
                    "period_from": left,
                    "period_to": right,
                    "state_from": source_role,
                    "state_to": target_role,
                    "value": 1,
                    "pair_index": pair_index[left],
                }
            )

    if not transitions:
        return pd.DataFrame(
            columns=["period_from", "period_to", "state_from", "state_to", "value", "pair_index"]
        )

    return (
        pd.DataFrame(transitions)
        .groupby(["period_from", "period_to", "state_from", "state_to", "pair_index"], as_index=False)["value"]
        .sum()
        .sort_values(["pair_index", "state_from", "state_to"], ascending=[True, True, True])
        .reset_index(drop=True)
    )


def build_role_sankey_frames(
    transitions: pd.DataFrame,
    period_order: list[str],
    include_absent: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if transitions.empty:
        raise ValueError("No role transitions found for selected filters.")

    states = ROLE_ORDER_WITH_ABSENT if include_absent else ROLE_ORDER
    state_index = {state: idx for idx, state in enumerate(states)}
    period_index = {period: idx for idx, period in enumerate(period_order)}

    outgoing = (
        transitions.groupby(["period_from", "state_from"], as_index=False)["value"].sum().rename(
            columns={"period_from": "period", "state_from": "state", "value": "outgoing"}
        )
    )
    incoming = (
        transitions.groupby(["period_to", "state_to"], as_index=False)["value"].sum().rename(
            columns={"period_to": "period", "state_to": "state", "value": "incoming"}
        )
    )

    node_rows: list[dict[str, Any]] = []
    for period in period_order:
        for state in states:
            out_val = float(
                outgoing[(outgoing["period"] == period) & (outgoing["state"] == state)]["outgoing"].sum()
            )
            in_val = float(
                incoming[(incoming["period"] == period) & (incoming["state"] == state)]["incoming"].sum()
            )
            if out_val <= 0 and in_val <= 0:
                continue
            x_pos = period_index[period] / (len(period_order) - 1) if len(period_order) > 1 else 0.5
            y_pos = 0.5 if len(states) == 1 else 0.08 + 0.84 * (state_index[state] / (len(states) - 1))
            node_rows.append(
                {
                    "period": period,
                    "role": state,
                    "incoming": int(round(in_val)),
                    "outgoing": int(round(out_val)),
                    "x": float(x_pos),
                    "y": float(y_pos),
                    "node_id": f"{period}||{state}",
                }
            )

    nodes = pd.DataFrame(node_rows)
    if nodes.empty:
        raise ValueError("No role nodes available for selected filters.")
    idx_map = {node_id: i for i, node_id in enumerate(nodes["node_id"].tolist())}

    source_totals = (
        transitions.groupby(["period_from", "state_from"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "source_total"})
    )
    transitions_enriched = transitions.merge(
        source_totals,
        on=["period_from", "state_from"],
        how="left",
    )

    link_rows: list[dict[str, Any]] = []
    for _, row in transitions_enriched.iterrows():
        src_id = f"{row['period_from']}||{row['state_from']}"
        dst_id = f"{row['period_to']}||{row['state_to']}"
        if src_id not in idx_map or dst_id not in idx_map:
            continue
        value = int(row["value"])
        source_total = float(row["source_total"]) if row["source_total"] else 0.0
        pct_source = (100.0 * value / source_total) if source_total > 0 else 0.0
        color = hex_to_rgba(ROLE_COLORS.get(str(row["state_from"]), "#9ca3af"), 0.34)
        link_rows.append(
            {
                "source": idx_map[src_id],
                "target": idx_map[dst_id],
                "value": value,
                "period_from": str(row["period_from"]),
                "period_to": str(row["period_to"]),
                "state_from": str(row["state_from"]),
                "state_to": str(row["state_to"]),
                "pct_source": round(pct_source, 2),
                "color": color,
            }
        )

    links = pd.DataFrame(link_rows)
    if links.empty:
        raise ValueError("No role links available for selected filters.")
    return nodes, links


def build_role_trace(nodes: pd.DataFrame, links: pd.DataFrame, domain: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "type": "sankey",
        "arrangement": "fixed",
        "domain": domain,
        "node": {
            "pad": 12,
            "thickness": 14,
            "label": nodes["role"].astype(str).tolist(),
            "color": [ROLE_COLORS.get(role, "#9ca3af") for role in nodes["role"].astype(str)],
            "line": {"color": "#ffffff", "width": 1},
            "x": [round(v, 6) for v in nodes["x"].tolist()],
            "y": [round(v, 6) for v in nodes["y"].tolist()],
            "customdata": np_stack(
                nodes["period"].astype(str).tolist(),
                nodes["role"].astype(str).tolist(),
                [int(v) for v in nodes["incoming"].tolist()],
                [int(v) for v in nodes["outgoing"].tolist()],
            ),
            "hovertemplate": (
                "<b>%{customdata[1]}</b><br>"
                "Period: %{customdata[0]}<br>"
                "Incoming: %{customdata[2]}<br>"
                "Outgoing: %{customdata[3]}<extra></extra>"
            ),
        },
        "link": {
            "source": [int(v) for v in links["source"].tolist()],
            "target": [int(v) for v in links["target"].tolist()],
            "value": [int(v) for v in links["value"].tolist()],
            "color": links["color"].astype(str).tolist(),
            "customdata": np_stack(
                links["period_from"].astype(str).tolist(),
                links["period_to"].astype(str).tolist(),
                links["state_from"].astype(str).tolist(),
                links["state_to"].astype(str).tolist(),
                [float(v) for v in links["pct_source"].tolist()],
            ),
            "hovertemplate": (
                "From: %{customdata[0]} (%{customdata[2]})<br>"
                "To: %{customdata[1]} (%{customdata[3]})<br>"
                "Institutions: %{value}<br>"
                "Share of source role: %{customdata[4]}%<extra></extra>"
            ),
        },
    }


def write_role_sankey_html(
    out_html: Path,
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    period_order: list[str],
    include_absent: bool,
    field_scope: str,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    trace = build_role_trace(nodes, links, domain={"x": [0.0, 1.0], "y": [0.0, 1.0]})

    payload = {
        "trace": trace,
        "meta": {
            "periods": period_order,
            "field_scope": field_scope,
            "include_absent": include_absent,
            "node_count": int(len(nodes)),
            "link_count": int(len(links)),
        },
    }

    role_legend_html = "".join(
        f'<div class="legend-group"><span class="chip" style="background:{ROLE_COLORS[role]}"></span>{role}</div>'
        for role in (ROLE_ORDER_WITH_ABSENT if include_absent else ROLE_ORDER)
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Role Flow Sankey</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 16px;
      color: #111827;
      background: #f8fafc;
    }}
    .subtitle {{ color: #4b5563; font-size: 14px; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 10px 0 14px; font-size: 13px; }}
    .legend-group {{ display: flex; align-items: center; gap: 8px; }}
    .chip {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
    #chart {{
      width: 100%;
      height: 860px;
      border: 1px solid #e5e7eb;
      background: #ffffff;
      border-radius: 8px;
    }}
    .meta {{ margin-top: 10px; color: #6b7280; font-size: 12px; }}
  </style>
</head>
<body>
  <h2 style="margin:0 0 4px;">Institution Role Flow Across Periods</h2>
  <div class="subtitle">
    Aggregated transitions only ({field_scope}). Nodes are role tiers by period; links are institution counts moving between tiers.
  </div>
  <div class="legend">
    <div class="legend-group"><strong>Role colors</strong></div>
    {role_legend_html}
  </div>
  <div id="chart"></div>
  <div class="meta">
    Field scope: {field_scope} |
    Include Absent transitions: {str(include_absent).lower()} |
    Periods: {', '.join(period_order)}
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const layout = {{
      margin: {{l: 20, r: 20, t: 20, b: 20}},
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {{size: 12, color: "#111827"}}
    }};
    Plotly.newPlot("chart", [payload.trace], layout, {{
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"]
    }});
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def write_role_small_multiples_html(
    out_html: Path,
    field_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    period_order: list[str],
    include_absent: bool,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    domains = {
        "Dental": {"x": [0.00, 0.48], "y": [0.54, 1.00]},
        "Medical": {"x": [0.52, 1.00], "y": [0.54, 1.00]},
        "Technical": {"x": [0.00, 0.48], "y": [0.00, 0.46]},
        "Other": {"x": [0.52, 1.00], "y": [0.00, 0.46]},
    }

    traces: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for field in FIELD_ORDER:
        if field not in field_frames:
            continue
        nodes, links = field_frames[field]
        traces.append(build_role_trace(nodes, links, domain=domains[field]))
        annotations.append(
            {
                "text": f"{field} institutions",
                "xref": "paper",
                "yref": "paper",
                "x": sum(domains[field]["x"]) / 2.0,
                "y": domains[field]["y"][1] + 0.03,
                "showarrow": False,
                "font": {"size": 13, "color": "#111827"},
            }
        )
    if not traces:
        raise ValueError("No field-specific transitions available for small-multiples Sankey.")

    role_legend_html = "".join(
        f'<div class="legend-group"><span class="chip" style="background:{ROLE_COLORS[role]}"></span>{role}</div>'
        for role in (ROLE_ORDER_WITH_ABSENT if include_absent else ROLE_ORDER)
    )

    payload = {"traces": traces, "annotations": annotations}
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Field-Specific Role Flow Sankey</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 16px;
      color: #111827;
      background: #f8fafc;
    }}
    .subtitle {{ color: #4b5563; font-size: 14px; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 10px 0 14px; font-size: 13px; }}
    .legend-group {{ display: flex; align-items: center; gap: 8px; }}
    .chip {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
    #chart {{
      width: 100%;
      height: 980px;
      border: 1px solid #e5e7eb;
      background: #ffffff;
      border-radius: 8px;
    }}
    .meta {{ margin-top: 10px; color: #6b7280; font-size: 12px; }}
  </style>
</head>
<body>
  <h2 style="margin:0 0 4px;">Field-Specific Institution Role Flows</h2>
  <div class="subtitle">
    Small multiples of aggregated role transitions across periods by institutional field.
  </div>
  <div class="legend">
    <div class="legend-group"><strong>Role colors</strong></div>
    {role_legend_html}
  </div>
  <div id="chart"></div>
  <div class="meta">
    Include Absent transitions: {str(include_absent).lower()} |
    Periods: {', '.join(period_order)}
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const layout = {{
      margin: {{l: 20, r: 20, t: 40, b: 20}},
      annotations: payload.annotations,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {{size: 11, color: "#111827"}}
    }};
    Plotly.newPlot("chart", payload.traces, layout, {{
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"]
    }});
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def np_stack(*columns: list[Any]) -> list[list[Any]]:
    if not columns:
        return []
    n = len(columns[0])
    for column in columns:
        if len(column) != n:
            raise ValueError("All columns must have equal length for stacking.")
    return [[columns[idx][row] for idx in range(len(columns))] for row in range(n)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interactive Sankey HTML visualizations for institutional flow."
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Repository root path.",
    )
    parser.add_argument(
        "--mode",
        choices=["role", "institution"],
        default="role",
        help="Sankey granularity: aggregated role flow (role) or institution-level flow (institution).",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output HTML file path. Defaults depend on mode.",
    )
    parser.add_argument(
        "--save-data-csv",
        action="store_true",
        help="Also export node/link tables next to HTML.",
    )

    parser.add_argument(
        "--top-n-institutions",
        type=int,
        default=120,
        help="Institution mode: max institutions to include after ranking by persistence/centrality.",
    )
    parser.add_argument(
        "--min-periods-present",
        type=int,
        default=2,
        help="Institution mode: minimum number of periods an institution must appear in.",
    )

    parser.add_argument(
        "--field",
        choices=["all", "dental", "medical", "technical", "other"],
        default="all",
        help="Role mode: restrict to one institutional field or use all.",
    )
    parser.add_argument(
        "--include-absent",
        action="store_true",
        help="Role mode: include explicit Absent transitions for entry/exit dynamics.",
    )
    parser.add_argument(
        "--write-field-small-multiples",
        action="store_true",
        help="Role mode: additionally generate one 2x2 field-specific Sankey HTML.",
    )
    parser.add_argument(
        "--field-small-multiples-html",
        default=None,
        help="Role mode: output path for 2x2 field-specific Sankey HTML.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    period_order = load_period_order(root)

    if args.output_html:
        out_html = Path(args.output_html).resolve()
    else:
        default_name = "role_flow_sankey.html" if args.mode == "role" else "institution_flow_sankey.html"
        out_html = (root / "outputs" / "figures" / "flow_options" / default_name).resolve()

    outputs: dict[str, Any] = {"mode": args.mode, "html": str(out_html)}

    if args.mode == "institution":
        table, available_periods = load_institution_table_from_graphs(root, period_order)
        period_order = [period for period in period_order if period in available_periods]
        if len(period_order) < 2:
            raise ValueError("Need at least two available period graphs to render Sankey.")

        nodes, links = build_institution_sankey_frames(
            table=table,
            period_order=period_order,
            top_n_institutions=args.top_n_institutions,
            min_periods_present=args.min_periods_present,
        )
        write_institution_sankey_html(
            out_html=out_html,
            nodes=nodes,
            links=links,
            period_order=period_order,
            top_n_institutions=args.top_n_institutions,
            min_periods_present=args.min_periods_present,
        )

        outputs["nodes"] = int(len(nodes))
        outputs["links"] = int(len(links))
        if args.save_data_csv:
            node_csv = out_html.with_name(out_html.stem + "_nodes.csv")
            link_csv = out_html.with_name(out_html.stem + "_links.csv")
            nodes.to_csv(node_csv, index=False)
            links.to_csv(link_csv, index=False)
            outputs["nodes_csv"] = str(node_csv)
            outputs["links_csv"] = str(link_csv)

        print(json.dumps(outputs, indent=2))
        return

    role_table, available_periods = load_role_table_from_graphs(root, period_order)
    period_order = [period for period in period_order if period in available_periods]
    if len(period_order) < 2:
        raise ValueError("Need at least two available period graphs to render Sankey.")

    field_scope = "All Fields"
    field_filter = None
    if args.field != "all":
        field_filter = args.field.title()
        field_scope = field_filter

    transitions = build_role_transition_table(
        role_table=role_table,
        period_order=period_order,
        field_filter=field_filter,
        include_absent=bool(args.include_absent),
    )
    nodes, links = build_role_sankey_frames(
        transitions=transitions,
        period_order=period_order,
        include_absent=bool(args.include_absent),
    )
    write_role_sankey_html(
        out_html=out_html,
        nodes=nodes,
        links=links,
        period_order=period_order,
        include_absent=bool(args.include_absent),
        field_scope=field_scope,
    )

    outputs["field_scope"] = field_scope
    outputs["nodes"] = int(len(nodes))
    outputs["links"] = int(len(links))

    if args.save_data_csv:
        node_csv = out_html.with_name(out_html.stem + "_nodes.csv")
        link_csv = out_html.with_name(out_html.stem + "_links.csv")
        transitions_csv = out_html.with_name(out_html.stem + "_transitions.csv")
        nodes.to_csv(node_csv, index=False)
        links.to_csv(link_csv, index=False)
        transitions.to_csv(transitions_csv, index=False)
        outputs["nodes_csv"] = str(node_csv)
        outputs["links_csv"] = str(link_csv)
        outputs["transitions_csv"] = str(transitions_csv)

    if args.write_field_small_multiples and args.field == "all":
        field_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for field in FIELD_ORDER:
            field_transitions = build_role_transition_table(
                role_table=role_table,
                period_order=period_order,
                field_filter=field,
                include_absent=bool(args.include_absent),
            )
            if field_transitions.empty:
                continue
            try:
                field_nodes, field_links = build_role_sankey_frames(
                    transitions=field_transitions,
                    period_order=period_order,
                    include_absent=bool(args.include_absent),
                )
            except ValueError:
                continue
            field_frames[field] = (field_nodes, field_links)

        if field_frames:
            if args.field_small_multiples_html:
                fields_html = Path(args.field_small_multiples_html).resolve()
            else:
                fields_html = out_html.with_name(out_html.stem + "_fields.html")
            write_role_small_multiples_html(
                out_html=fields_html,
                field_frames=field_frames,
                period_order=period_order,
                include_absent=bool(args.include_absent),
            )
            outputs["field_small_multiples_html"] = str(fields_html)
            outputs["field_small_multiples_fields"] = sorted(field_frames.keys())

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
