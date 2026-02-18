from ai_dentistry.network_metrics import (
    build_collaboration_graph,
    core_newcomer_metrics,
    global_metrics,
)


def test_build_collaboration_graph_counts_weighted_edges() -> None:
    publications = [
        ["a", "b", "c"],
        ["a", "b"],
        ["d"],
    ]
    graph = build_collaboration_graph(publications)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert graph["a"]["b"]["weight"] == 2


def test_core_newcomer_metrics_basic_case() -> None:
    previous = build_collaboration_graph([["a", "b"], ["b", "c"]])
    current = build_collaboration_graph([["a", "b"], ["a", "d"], ["d", "e"]])
    metrics = core_newcomer_metrics(previous, current)
    assert metrics["returning_nodes"] == 2
    assert metrics["new_nodes"] == 2
    assert metrics["edge_growth_per_new"] > 0


def test_global_metrics_non_empty_graph() -> None:
    graph = build_collaboration_graph([["a", "b"], ["b", "c"]])
    metrics = global_metrics(graph)
    assert metrics["nodes"] == 3
    assert metrics["edges"] == 2
    assert metrics["largest_connected_component_size"] == 3
