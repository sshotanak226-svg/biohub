import numpy as np

from biohub_demo.tracking_variants import build_variant, graph_from_raw_prediction


def test_distance_hungarian_builds_two_consistent_tracks() -> None:
    coords = np.asarray([
        [0, 1, 10, 10], [0, 1, 30, 30],
        [1, 1, 11, 10], [1, 1, 29, 30],
        [2, 1, 12, 10], [2, 1, 28, 30],
    ], dtype=np.float32)
    scores = np.full(6, 0.99, dtype=np.float32)
    graph, candidates = graph_from_raw_prediction(
        "sample", coords, scores, [], (1.625, 0.40625, 0.40625)
    )
    result, diagnostic = build_variant(graph, candidates, {
        "tracker": "distance_hungarian",
        "detection_threshold": 0.95,
        "tight_gate_um": 6.0,
        "max_distance_um": 10.0,
        "linked_nodes_only": True,
    })
    assert len(result.nodes) == 6
    assert len(result.edges) == 4
    assert all(len(result.incoming(node.node_id)) <= 1 for node in result.nodes.values())
    assert all(len(result.outgoing(node.node_id)) <= 1 for node in result.nodes.values())
    assert diagnostic["solver"] == "distance_hungarian"
