from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Callable, Iterable

import networkx as nx
import pandas as pd


KEY_TYPES = {"Dental", "Medical", "Technical"}


def build_collaboration_graph(publication_institutions: Iterable[list[str]]) -> nx.Graph:
    edge_counter: Counter[tuple[str, str]] = Counter()
    graph = nx.Graph()

    for institutions in publication_institutions:
        unique = list(dict.fromkeys(inst for inst in institutions if inst))
        graph.add_nodes_from(unique)
        for a, b in combinations(sorted(unique), 2):
            edge_counter[(a, b)] += 1

    for (a, b), weight in edge_counter.items():
        graph.add_edge(a, b, weight=weight)

    return graph


def annotate_node_types(graph: nx.Graph, classify: Callable[[str], str]) -> None:
    for node in graph.nodes():
        graph.nodes[node]["type"] = classify(node)


def largest_connected_component_subgraph(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    if nx.is_empty(graph):
        return graph.copy()
    component_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(component_nodes).copy()


def global_metrics(graph: nx.Graph) -> dict[str, float | int]:
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    density = nx.density(graph) if nodes > 1 else 0.0
    avg_degree = (2 * edges / nodes) if nodes else 0.0

    if edges > 0:
        avg_clustering = nx.average_clustering(graph, weight="weight")
    else:
        avg_clustering = 0.0
    largest_cc = len(max(nx.connected_components(graph), key=len)) if nodes > 0 else 0

    return {
        "nodes": nodes,
        "edges": edges,
        "density": density,
        "avg_degree": avg_degree,
        "avg_clustering_weighted": avg_clustering,
        "largest_connected_component_size": largest_cc,
    }


def top_centrality_table(
    graph: nx.Graph,
    top_n: int = 10,
    use_lcc_for_path_metrics: bool = True,
) -> pd.DataFrame:
    degree_input = graph
    path_input = largest_connected_component_subgraph(graph) if use_lcc_for_path_metrics else graph

    degree_scores = nx.degree_centrality(degree_input) if degree_input.number_of_nodes() else {}
    between_scores = (
        nx.betweenness_centrality(path_input, weight="weight", normalized=True)
        if path_input.number_of_nodes()
        else {}
    )

    top_degree = sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_between = sorted(between_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    max_len = max(len(top_degree), len(top_between), top_n)
    rows: list[dict[str, object]] = []
    for idx in range(max_len):
        row = {"rank": idx + 1}
        if idx < len(top_degree):
            row["degree_node"] = top_degree[idx][0]
            row["degree_score"] = top_degree[idx][1]
        else:
            row["degree_node"] = ""
            row["degree_score"] = 0.0

        if idx < len(top_between):
            row["betweenness_node"] = top_between[idx][0]
            row["betweenness_score"] = top_between[idx][1]
        else:
            row["betweenness_node"] = ""
            row["betweenness_score"] = 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def core_newcomer_metrics(prev_graph: nx.Graph, curr_graph: nx.Graph) -> dict[str, float | int]:
    prev_nodes = set(prev_graph.nodes())
    curr_nodes = set(curr_graph.nodes())
    returning = prev_nodes & curr_nodes
    new_nodes = curr_nodes - prev_nodes

    if len(returning) > 1:
        density_existing = nx.density(curr_graph.subgraph(returning))
    else:
        density_existing = 0.0

    if len(new_nodes) > 1:
        density_new = nx.density(curr_graph.subgraph(new_nodes))
    else:
        density_new = 0.0

    edges_with_new = sum(1 for u, v in curr_graph.edges() if u in new_nodes or v in new_nodes)
    edge_growth_per_new = edges_with_new / len(new_nodes) if new_nodes else 0.0

    return {
        "density_existing": density_existing,
        "density_new": density_new,
        "edge_growth_per_new": edge_growth_per_new,
        "new_nodes": len(new_nodes),
        "returning_nodes": len(returning),
    }


def cluster_interdisciplinarity_metrics(graph: nx.Graph) -> dict[str, int | float]:
    components = list(nx.connected_components(graph))
    single_field = 0
    two_field = 0
    three_field = 0

    for component in components:
        component_types = {graph.nodes[node].get("type", "Other") for node in component}
        key_types = component_types & KEY_TYPES
        if len(key_types) == 1:
            single_field += 1
        elif len(key_types) == 2:
            two_field += 1
        elif len(key_types) == 3:
            three_field += 1

    total = len(components) if components else 1
    return {
        "num_components": len(components),
        "single_field_components": single_field,
        "two_field_components": two_field,
        "three_field_components": three_field,
        "single_field_ratio": single_field / total,
        "two_field_ratio": two_field / total,
        "three_field_ratio": three_field / total,
    }
