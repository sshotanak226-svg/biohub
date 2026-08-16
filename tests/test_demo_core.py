from __future__ import annotations

import numpy as np
import pandas as pd

from biohub_demo.graph import Edge, Node, TrackGraph
from biohub_demo.io import write_submission
from biohub_demo.synthetic import generate_synthetic
from biohub_demo.validate import validate_graph, validate_submission


def test_synthetic_is_reproducible() -> None:
    first = generate_synthetic(seed=7, shape=(9, 12, 64, 64), cell_count=9)
    second = generate_synthetic(seed=7, shape=(9, 12, 64, 64), cell_count=9)
    assert np.array_equal(first.image, second.image)
    assert first.truth.nodes == second.truth.nodes
    assert first.truth.edges == second.truth.edges


def test_validator_rejects_nonconsecutive_edge() -> None:
    graph = TrackGraph("bad")
    graph.add_node(Node(0, 0, 1, 1, 1))
    graph.add_node(Node(1, 2, 1, 1, 1))
    graph.add_edge(Edge(0, 1))
    result = validate_graph(graph, (3, 4, 4, 4))
    assert not result["valid"]
    assert any("non-consecutive" in error for error in result["errors"])


def test_submission_contract(tmp_path) -> None:
    graph = TrackGraph("sample")
    graph.add_node(Node(0, 0, 1, 2, 3))
    path = write_submission([graph], tmp_path / "submission.csv")
    assert validate_submission(pd.read_csv(path))["valid"]
