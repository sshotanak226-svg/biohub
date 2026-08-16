from __future__ import annotations

import json
from pathlib import Path

from biohub_demo.graph import Edge, Node, TrackGraph
from biohub_demo.io import is_complete_geff, write_geff


def _tiny_graph() -> TrackGraph:
    graph = TrackGraph("tiny")
    graph.add_node(Node(0, 0, 1.0, 2.0, 3.0, 0.9))
    graph.add_node(Node(1, 1, 1.5, 2.5, 3.5, 0.8))
    graph.add_edge(Edge(0, 1, 0.7, 1.0))
    return graph


def test_write_geff_publishes_completion_metadata(tmp_path: Path) -> None:
    output = tmp_path / "tiny.geff"
    assert not is_complete_geff(output)
    write_geff(_tiny_graph(), output)
    assert is_complete_geff(output)
    metadata = json.loads((output / "zarr.json").read_text(encoding="utf-8"))
    assert metadata["attributes"]["geff"]["directed"] is True


def test_write_geff_quarantines_an_interrupted_artifact(tmp_path: Path) -> None:
    output = tmp_path / "tiny.geff"
    output.mkdir()
    (output / "zarr.json").write_text(
        json.dumps({"attributes": {}, "zarr_format": 3, "node_type": "group"}),
        encoding="utf-8",
    )
    assert not is_complete_geff(output)

    write_geff(_tiny_graph(), output)

    assert is_complete_geff(output)
    quarantined = list(tmp_path.glob("tiny.geff.incomplete-*"))
    assert len(quarantined) == 1
    assert not is_complete_geff(quarantined[0])
