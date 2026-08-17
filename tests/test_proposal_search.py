from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from biohub_demo.method_search import load_method_search_config
from biohub_demo.proposal_registry import PROPOSAL_METHODS
from biohub_demo.proposal_search import (
    _build_one,
    _checkpoint_record,
    _configured_settings,
    _method_signature_payload,
    build_proposal_plan,
)
from biohub_demo.tracking_variants import graph_from_raw_prediction


ROOT = Path(__file__).resolve().parents[1]


def _proposal_names() -> set[str]:
    text = (ROOT / "OPTIMAL_TRANSPORT_MATCHING_PROPOSAL.md").read_text(encoding="utf-8")
    return set(re.findall(
        r"^#{3,5}\s+(?:(?:\d+\.\d+)|(?:[A-Z]\d+\.))\s+`([^`]+)`",
        text,
        re.MULTILINE,
    ))


def _tiny_raw_graph():
    coords = np.asarray([
        [0, 1, 10, 10], [0, 1, 30, 30],
        [1, 1, 11, 10], [1, 1, 29, 30],
        [2, 1, 12, 10], [2, 1, 28, 30],
    ], dtype=np.float32)
    scores = np.full(6, 0.99, dtype=np.float32)
    edges = [
        (0, 2, 0.95, 0.0), (1, 3, 0.95, 0.0),
        (0, 3, 0.05, 0.0), (1, 2, 0.05, 0.0),
        (2, 4, 0.95, 0.0), (3, 5, 0.95, 0.0),
        (2, 5, 0.05, 0.0), (3, 4, 0.05, 0.0),
    ]
    return graph_from_raw_prediction(
        "44b6_test", coords, scores, edges, (1.625, 0.40625, 0.40625)
    )


def test_registry_covers_every_unique_proposal_method() -> None:
    registered = {method.name for method in PROPOSAL_METHODS}
    assert registered == _proposal_names()
    assert len(registered) == 53
    plan = build_proposal_plan(
        load_method_search_config(ROOT / "configs" / "local_method_search.yaml"),
        paired_datasets=199,
    )
    assert plan["executable_now"] == 43
    assert plan["requires_additional_training"] == 10
    assert all(method.checkpoint_reuse for method in PROPOSAL_METHODS)
    assert sum(
        method.checkpoint_reuse == "direct_raw_prediction"
        for method in PROPOSAL_METHODS
    ) == 43


def test_checkpoint_record_reuses_marker_and_verifies_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "edge_predictor_best.pth"
    checkpoint.write_bytes(b"local-method-search-checkpoint")
    from biohub_demo.io import sha256_file

    marker = tmp_path / "training.complete.json"
    marker.write_text(
        '{"checkpoint": "' + str(checkpoint).replace("\\", "\\\\")
        + '", "checkpoint_sha256": "' + sha256_file(checkpoint)
        + '", "signature": "training-signature"}',
        encoding="utf-8",
    )
    record = _checkpoint_record(
        tmp_path,
        None,
        {"training_marker": str(marker), "verify_recorded_sha256": True},
    )
    assert record["source"] == "local_method_search_marker"
    assert record["training_signature"] == "training-signature"


def test_checkpoint_reuse_policy_is_recorded_in_prediction_signature() -> None:
    original = PROPOSAL_METHODS[0]
    changed = replace(original, checkpoint_reuse="different-documentation-only-policy")
    assert _method_signature_payload(original) != _method_signature_payload(changed)


def test_every_currently_executable_method_builds_a_valid_tiny_graph() -> None:
    config = load_method_search_config(ROOT / "configs" / "local_method_search.yaml")
    configured = _configured_settings(config)
    raw_graph, raw_candidates = _tiny_raw_graph()
    graphs = {}
    for original in PROPOSAL_METHODS:
        if original.fidelity == "requires_training":
            continue
        settings = dict(original.settings)
        if settings.get("tracker") == "unbalanced_ot":
            settings.update(ot_device="cpu", sinkhorn_iterations=20, sinkhorn_tolerance=0.0)
        method = replace(original, settings=settings)
        graph, _ = _build_one(method, raw_graph, raw_candidates, configured, graphs)
        graphs[method.name] = graph
        assert all(edge.source_id in graph.nodes for edge in graph.edges.values())
        assert all(edge.target_id in graph.nodes for edge in graph.edges.values())
        assert all(len(graph.incoming(node_id)) <= 1 for node_id in graph.nodes)
        assert all(len(graph.outgoing(node_id)) <= 2 for node_id in graph.nodes)
