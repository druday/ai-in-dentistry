from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


ROLE_ORDER = ["Core", "Semi-Peripheral", "Peripheral", "Isolate"]
ROLE_ORDER_WITH_ABSENT = ROLE_ORDER + ["Absent"]
ROLE_DISPLAY = {
    "Core": "High-centrality institutions",
    "Semi-Peripheral": "Mid-centrality institutions",
    "Peripheral": "Low-centrality institutions",
    "Isolate": "Zero-centrality institutions",
    "Absent": "Absent",
}
ROLE_COLORS = {
    "Core": "#1f77b4",
    "Semi-Peripheral": "#17becf",
    "Peripheral": "#ff7f0e",
    "Isolate": "#bcbd22",
    "Absent": "#bdbdbd",
}
FIELD_ORDER = ["Dental", "Medical", "Technical", "Other"]


def _display_role(role: str) -> str:
    return ROLE_DISPLAY.get(str(role), str(role))


def role_labels_from_graph(graph: nx.Graph) -> dict[str, str]:
    if graph.number_of_nodes() == 0:
        return {}
    degree = nx.degree_centrality(graph)
    non_zero = [float(value) for value in degree.values() if float(value) > 0]
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


def build_role_transition_counts(
    period_graphs: dict[str, nx.Graph],
    period_labels: list[str],
    include_absent: bool = True,
    field_filter: str | None = None,
) -> pd.DataFrame:
    role_maps: dict[str, dict[str, str]] = {}
    filtered_nodes: dict[str, set[str]] = {}
    for period in period_labels:
        graph = period_graphs.get(period, nx.Graph())
        roles = role_labels_from_graph(graph)
        if field_filter is not None:
            keep_nodes = {
                str(node)
                for node, attrs in graph.nodes(data=True)
                if str(attrs.get("type", "Other")) == field_filter
            }
            roles = {node: role for node, role in roles.items() if node in keep_nodes}
            filtered_nodes[period] = keep_nodes
        else:
            filtered_nodes[period] = set(roles.keys())
        role_maps[period] = roles

    rows: list[dict[str, Any]] = []
    for idx, (left, right) in enumerate(zip(period_labels[:-1], period_labels[1:], strict=False)):
        left_map = role_maps.get(left, {})
        right_map = role_maps.get(right, {})
        institutions = set(left_map) | set(right_map)
        for institution in institutions:
            source = left_map.get(institution, "Absent")
            target = right_map.get(institution, "Absent")
            if not include_absent and ("Absent" in (source, target)):
                continue
            rows.append(
                {
                    "pair_index": idx,
                    "period_from": left,
                    "period_to": right,
                    "state_from": source,
                    "state_to": target,
                    "value": 1,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["pair_index", "period_from", "period_to", "state_from", "state_to", "value"]
        )

    return (
        pd.DataFrame(rows)
        .groupby(["pair_index", "period_from", "period_to", "state_from", "state_to"], as_index=False)["value"]
        .sum()
        .sort_values(["pair_index", "state_from", "state_to"], ascending=[True, True, True])
        .reset_index(drop=True)
    )


def _build_sankey_frames(
    transitions: pd.DataFrame,
    period_labels: list[str],
    include_absent: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = ROLE_ORDER_WITH_ABSENT if include_absent else ROLE_ORDER
    state_index = {state: idx for idx, state in enumerate(states)}
    period_index = {period: idx for idx, period in enumerate(period_labels)}

    outgoing = (
        transitions.groupby(["period_from", "state_from"], as_index=False)["value"]
        .sum()
        .rename(columns={"period_from": "period", "state_from": "role", "value": "outgoing"})
    )
    incoming = (
        transitions.groupby(["period_to", "state_to"], as_index=False)["value"]
        .sum()
        .rename(columns={"period_to": "period", "state_to": "role", "value": "incoming"})
    )

    node_rows: list[dict[str, Any]] = []
    for period in period_labels:
        for role in states:
            out_val = float(
                outgoing[(outgoing["period"] == period) & (outgoing["role"] == role)]["outgoing"].sum()
            )
            in_val = float(
                incoming[(incoming["period"] == period) & (incoming["role"] == role)]["incoming"].sum()
            )
            if out_val <= 0 and in_val <= 0:
                continue
            x = period_index[period] / (len(period_labels) - 1) if len(period_labels) > 1 else 0.5
            y = 0.06 + 0.82 * (state_index[role] / max(1, (len(states) - 1)))
            node_rows.append(
                {
                    "period": period,
                    "role": role,
                    "incoming": int(round(in_val)),
                    "outgoing": int(round(out_val)),
                    "x": float(x),
                    "y": float(y),
                    "node_id": f"{period}||{role}",
                }
            )

    nodes = pd.DataFrame(
        node_rows,
        columns=["period", "role", "incoming", "outgoing", "x", "y", "node_id"],
    )
    idx_map = {node_id: idx for idx, node_id in enumerate(nodes["node_id"].tolist())}

    source_totals = (
        transitions.groupby(["period_from", "state_from"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "source_total"})
    )
    enriched = transitions.merge(source_totals, on=["period_from", "state_from"], how="left")

    link_rows: list[dict[str, Any]] = []
    for _, row in enriched.iterrows():
        src_id = f"{row['period_from']}||{row['state_from']}"
        dst_id = f"{row['period_to']}||{row['state_to']}"
        if src_id not in idx_map or dst_id not in idx_map:
            continue
        source_total = float(row["source_total"]) if float(row["source_total"]) > 0 else 0.0
        pct_source = (100.0 * float(row["value"]) / source_total) if source_total > 0 else 0.0
        link_rows.append(
            {
                "source": idx_map[src_id],
                "target": idx_map[dst_id],
                "value": int(row["value"]),
                "period_from": str(row["period_from"]),
                "period_to": str(row["period_to"]),
                "state_from": str(row["state_from"]),
                "state_to": str(row["state_to"]),
                "pct_source": round(pct_source, 2),
            }
        )
    links = pd.DataFrame(
        link_rows,
        columns=[
            "source",
            "target",
            "value",
            "period_from",
            "period_to",
            "state_from",
            "state_to",
            "pct_source",
        ],
    )
    return nodes, links


def _rgba_from_hex(color: str, alpha: float) -> str:
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


def _stack(*columns: list[Any]) -> list[list[Any]]:
    if not columns:
        return []
    n_rows = len(columns[0])
    for col in columns:
        if len(col) != n_rows:
            raise ValueError("All customdata columns must have equal length.")
    return [[columns[idx][row] for idx in range(len(columns))] for row in range(n_rows)]


def write_role_sankey_html(
    transitions: pd.DataFrame,
    period_labels: list[str],
    out_html: str | Path,
    title: str,
    subtitle: str,
    include_absent: bool = True,
    include_period_annotations: bool = True,
) -> Path:
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nodes, links = _build_sankey_frames(
        transitions=transitions,
        period_labels=period_labels,
        include_absent=include_absent,
    )
    if nodes.empty:
        nodes = pd.DataFrame(
            columns=["period", "role", "incoming", "outgoing", "x", "y", "node_id"]
        )
    if links.empty:
        links = pd.DataFrame(
            columns=[
                "source",
                "target",
                "value",
                "period_from",
                "period_to",
                "state_from",
                "state_to",
                "pct_source",
            ]
        )

    annotations: list[dict[str, Any]] = []
    if include_period_annotations and period_labels:
        for idx, period in enumerate(period_labels):
            x = idx / (len(period_labels) - 1) if len(period_labels) > 1 else 0.5
            annotations.append(
                {
                    "text": period,
                    "xref": "paper",
                    "yref": "paper",
                    "x": x,
                    "y": 1.07,
                    "showarrow": False,
                    "textangle": -18,
                    "font": {"size": 12, "color": "#374151"},
                }
            )

    role_legend = "".join(
        f'<div class="legend-item"><span class="chip" style="background:{ROLE_COLORS[role]}"></span>{_display_role(role)}</div>'
        for role in (ROLE_ORDER_WITH_ABSENT if include_absent else ROLE_ORDER)
    )

    payload = {
        "nodes": {
            "label": [_display_role(role) for role in nodes["role"].astype(str).tolist()],
            "color": [ROLE_COLORS.get(role, "#9ca3af") for role in nodes["role"].astype(str)],
            "x": [round(v, 6) for v in nodes["x"].tolist()],
            "y": [round(v, 6) for v in nodes["y"].tolist()],
            "customdata": _stack(
                nodes["period"].astype(str).tolist(),
                [_display_role(role) for role in nodes["role"].astype(str).tolist()],
                [int(v) for v in nodes["incoming"].tolist()],
                [int(v) for v in nodes["outgoing"].tolist()],
            ),
        },
        "links": {
            "source": [int(v) for v in links.get("source", pd.Series(dtype=int)).tolist()],
            "target": [int(v) for v in links.get("target", pd.Series(dtype=int)).tolist()],
            "value": [int(v) for v in links.get("value", pd.Series(dtype=int)).tolist()],
            "color": [
                _rgba_from_hex(ROLE_COLORS.get(role, "#9ca3af"), 0.34)
                for role in links.get("state_from", pd.Series(dtype=str)).astype(str).tolist()
            ],
            "customdata": _stack(
                links.get("period_from", pd.Series(dtype=str)).astype(str).tolist(),
                links.get("period_to", pd.Series(dtype=str)).astype(str).tolist(),
                [_display_role(role) for role in links.get("state_from", pd.Series(dtype=str)).astype(str).tolist()],
                [_display_role(role) for role in links.get("state_to", pd.Series(dtype=str)).astype(str).tolist()],
                [float(v) for v in links.get("pct_source", pd.Series(dtype=float)).tolist()],
            ),
        },
        "annotations": annotations,
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 18px;
      color: #111827;
      background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }}
    h2 {{ margin: 0 0 6px; font-size: 40px; line-height: 1.1; letter-spacing: -0.4px; }}
    .subtitle {{ color: #475569; margin-bottom: 10px; font-size: 16px; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 7px; font-size: 14px; }}
    .chip {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
    #chart {{
      width: 100%;
      height: 1180px;
      border: 1px solid #dbe3ea;
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
    }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div class="subtitle">{subtitle}</div>
  <div class="legend">{role_legend}</div>
  <div id="chart"></div>
  <script>
    const payload = {json.dumps(payload)};
    const trace = {{
      type: "sankey",
      arrangement: "snap",
      node: {{
        pad: 12,
        thickness: 14,
        label: payload.nodes.label,
        color: payload.nodes.color,
        line: {{ color: "#ffffff", width: 1 }},
        x: payload.nodes.x,
        y: payload.nodes.y,
        customdata: payload.nodes.customdata,
        hovertemplate:
          "<b>%{{customdata[1]}}</b><br>" +
          "Period: %{{customdata[0]}}<br>" +
          "Incoming: %{{customdata[2]}}<br>" +
          "Outgoing: %{{customdata[3]}}<extra></extra>",
      }},
      link: {{
        source: payload.links.source,
        target: payload.links.target,
        value: payload.links.value,
        color: payload.links.color,
        customdata: payload.links.customdata,
        hovertemplate:
          "From: %{{customdata[0]}} (%{{customdata[2]}})<br>" +
          "To: %{{customdata[1]}} (%{{customdata[3]}})<br>" +
          "Institutions: %{{value}}<br>" +
          "Share of source role: %{{customdata[4]}}%<extra></extra>",
      }}
    }};
    const layout = {{
      margin: {{ l: 22, r: 22, t: 110, b: 56 }},
      annotations: payload.annotations,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {{ size: 12, color: "#111827" }},
    }};
    Plotly.newPlot("chart", [trace], layout, {{
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_role_sankey_small_multiples_html(
    field_transitions: dict[str, pd.DataFrame],
    period_labels: list[str],
    out_html: str | Path,
    title: str,
    include_absent: bool = True,
) -> Path:
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    domains = {
        "Dental": {"x": [0.00, 0.48], "y": [0.53, 0.97]},
        "Medical": {"x": [0.52, 1.00], "y": [0.53, 0.97]},
        "Technical": {"x": [0.00, 0.48], "y": [0.03, 0.47]},
        "Other": {"x": [0.52, 1.00], "y": [0.03, 0.47]},
    }

    traces: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for field in FIELD_ORDER:
        if field not in field_transitions:
            continue
        nodes, links = _build_sankey_frames(
            transitions=field_transitions[field],
            period_labels=period_labels,
            include_absent=include_absent,
        )
        role_labels = [_display_role(role) for role in nodes["role"].astype(str).tolist()]
        traces.append(
            {
                "type": "sankey",
                "arrangement": "snap",
                "domain": domains[field],
                "node": {
                    "pad": 12,
                    "thickness": 12,
                    "label": role_labels,
                    "color": [ROLE_COLORS.get(role, "#9ca3af") for role in nodes["role"].astype(str)],
                    "line": {"color": "#ffffff", "width": 1},
                    "x": [round(v, 6) for v in nodes["x"].tolist()],
                    "y": [round(v, 6) for v in nodes["y"].tolist()],
                },
                "link": {
                    "source": [int(v) for v in links["source"].tolist()],
                    "target": [int(v) for v in links["target"].tolist()],
                    "value": [int(v) for v in links["value"].tolist()],
                    "color": [
                        _rgba_from_hex(ROLE_COLORS.get(role, "#9ca3af"), 0.30)
                        for role in links["state_from"].astype(str).tolist()
                    ],
                },
            }
        )
        panel_x = (domains[field]["x"][0] + domains[field]["x"][1]) / 2.0
        panel_top = domains[field]["y"][1]
        annotations.append(
            {
                "text": f"{field} institutions",
                "xref": "paper",
                "yref": "paper",
                "x": panel_x,
                "y": panel_top + 0.045,
                "showarrow": False,
                "font": {"size": 14, "color": "#111827"},
            }
        )
        for idx, period in enumerate(period_labels):
            x = domains[field]["x"][0] + (
                (domains[field]["x"][1] - domains[field]["x"][0]) * idx / max(1, (len(period_labels) - 1))
            )
            annotations.append(
                {
                    "text": period,
                    "xref": "paper",
                    "yref": "paper",
                    "x": x,
                    "y": panel_top + 0.01,
                    "showarrow": False,
                    "textangle": -16,
                    "font": {"size": 10, "color": "#374151"},
                }
            )

    role_legend = "".join(
        f'<div class="legend-item"><span class="chip" style="background:{ROLE_COLORS[role]}"></span>{_display_role(role)}</div>'
        for role in (ROLE_ORDER_WITH_ABSENT if include_absent else ROLE_ORDER)
    )
    payload = {"traces": traces, "annotations": annotations}
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 18px;
      color: #111827;
      background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }}
    h2 {{ margin: 0 0 6px; font-size: 38px; line-height: 1.1; letter-spacing: -0.4px; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 7px; font-size: 14px; }}
    .chip {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
    #chart {{
      width: 100%;
      height: 1320px;
      border: 1px solid #dbe3ea;
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
    }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div class="legend">{role_legend}</div>
  <div id="chart"></div>
  <script>
    const payload = {json.dumps(payload)};
    Plotly.newPlot("chart", payload.traces, {{
      margin: {{ l: 24, r: 24, t: 110, b: 30 }},
      annotations: payload.annotations,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {{ size: 11, color: "#111827" }},
    }}, {{
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
