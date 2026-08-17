from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from biohub_demo.literature_methods import PROFILE_SETTINGS, build_literature_method
from biohub_demo.literature_registry import LITERATURE_METHODS
from biohub_demo.literature_search import _node_count_stats, _numeric_delta, build_literature_plan
from biohub_demo.method_search import load_method_search_config
from biohub_demo.tracking_variants import graph_from_raw_prediction


ROOT = Path(__file__).resolve().parents[1]


def _tiny_raw_graph():
    coords = np.asarray([
        [0, 1, 10, 10], [0, 1, 30, 30],
        [1, 1, 11, 10], [1, 1, 29, 30], [1, 1, 13, 12],
        [2, 1, 12, 10], [2, 1, 28, 30], [2, 1, 15, 13],
    ], dtype=np.float32)
    scores = np.full(len(coords), 0.99, dtype=np.float32)
    edges = [
        (0, 2, 0.95, 0.0), (0, 4, 0.70, 0.0),
        (1, 3, 0.95, 0.0), (2, 5, 0.95, 0.0),
        (4, 7, 0.90, 0.0), (3, 6, 0.95, 0.0),
        (2, 7, 0.10, 0.0), (4, 5, 0.15, 0.0),
    ]
    return graph_from_raw_prediction(
        "44b6_test", coords, scores, edges, (1.625, 0.40625, 0.40625)
    )


def test_registry_has_exactly_twenty_auditable_methods() -> None:
    assert len(LITERATURE_METHODS) == 20
    assert len({method.name for method in LITERATURE_METHODS}) == 20
    assert all(method.fidelity == "screening_proxy" for method in LITERATURE_METHODS)
    assert all(method.paper_url.startswith("https://") for method in LITERATURE_METHODS)
    configured = yaml.safe_load(
        (ROOT / "configs" / "literature_20_methods.yaml").read_text(encoding="utf-8")
    )
    assert configured["methods"] == [method.name for method in LITERATURE_METHODS]


def test_plan_reports_twenty_proxies_plus_baseline() -> None:
    config = load_method_search_config(ROOT / "configs" / "local_method_search.yaml")
    plan = build_literature_plan(config)
    assert plan["screening_proxy_count"] == 20
    assert plan["comparison_rows"] == 21
    assert plan["paper_faithful_reimplementations"] is False
    assert plan["validation_datasets"] == 20


def test_component_diagnostics_are_safe_for_missing_divisions() -> None:
    assert _node_count_stats({"per_dataset": [
        {"total_node_ratio": 0.5}, {"total_node_ratio": -0.1}
    ]}) == {"mean_node_ratio": 0.2, "mean_abs_node_ratio": 0.3}
    assert _numeric_delta(None, 0.0) is None
    assert _numeric_delta(float("nan"), 0.0) is None
    assert _numeric_delta(0.7, 0.6) == pytest.approx(0.1)


def test_all_twenty_proxies_build_valid_tiny_graphs(monkeypatch) -> None:
    for settings in PROFILE_SETTINGS.values():
        if settings.get("tracker") == "unbalanced_ot":
            monkeypatch.setitem(settings, "ot_device", "cpu")
            monkeypatch.setitem(settings, "sinkhorn_iterations", 20)
            monkeypatch.setitem(settings, "sinkhorn_tolerance", 0.0)
    raw_graph, raw_candidates = _tiny_raw_graph()
    cache = {}
    for method in LITERATURE_METHODS:
        graph, diagnostic = build_literature_method(
            method, raw_graph, raw_candidates, cache
        )
        assert diagnostic["paper_faithful"] is False
        assert all(edge.source_id in graph.nodes for edge in graph.edges.values())
        assert all(edge.target_id in graph.nodes for edge in graph.edges.values())
        assert all(len(graph.incoming(node_id)) <= 1 for node_id in graph.nodes)
        assert all(len(graph.outgoing(node_id)) <= 2 for node_id in graph.nodes)
