from __future__ import annotations

import numpy as np
import pytest
import torch

from biohub_demo.optimal_transport import resolve_ot_device
from biohub_demo.tracking_variants import build_variant, graph_from_raw_prediction


def _two_tracks():
    coords = np.asarray([
        [0, 1, 10, 10], [0, 1, 30, 30],
        [1, 1, 11, 10], [1, 1, 29, 30],
        [2, 1, 12, 10], [2, 1, 28, 30],
    ], dtype=np.float32)
    scores = np.full(6, 0.99, dtype=np.float32)
    return graph_from_raw_prediction(
        "sample", coords, scores, [], (1.625, 0.40625, 0.40625)
    )


def test_distance_uot_builds_consistent_tracks_on_cpu() -> None:
    graph, candidates = _two_tracks()
    result, diagnostic = build_variant(graph, candidates, {
        "tracker": "unbalanced_ot",
        "ot_mode": "distance",
        "ot_device": "cpu",
        "detection_threshold": 0.95,
        "max_distance_um": 10.0,
        "entropy_epsilon": 0.05,
        "source_mass_penalty": 0.5,
        "target_mass_penalty": 0.5,
        "sinkhorn_iterations": 100,
        "sinkhorn_tolerance": 1e-4,
        "ot_threshold": 0.05,
        "top_k_source": 2,
        "top_k_target": 3,
        "linked_nodes_only": True,
    })

    assert len(result.nodes) == 6
    assert len(result.edges) == 4
    assert all(len(result.incoming(node.node_id)) <= 1 for node in result.nodes.values())
    assert all(len(result.outgoing(node.node_id)) <= 1 for node in result.nodes.values())
    assert diagnostic["solver"] == "unbalanced-ot:distance"
    assert diagnostic["ot_device"] == "cpu"
    assert diagnostic["gpu_accelerated"] is False
    assert diagnostic["gated_edges"] == 4


def test_explicit_cuda_request_fails_when_cuda_is_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this runner")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_ot_device("cuda")
