from __future__ import annotations

from pathlib import Path

from biohub_demo.run import run_local_demo


def test_end_to_end_local_demo(tmp_path: Path) -> None:
    result = run_local_demo(Path("configs/local_synthetic_demo.yaml"), tmp_path / "run")
    metrics = result["metrics"]
    assert metrics["edge_jaccard"] >= 0.95
    assert metrics["division_jaccard"] >= 0.80
    assert metrics["node_recall"] >= 0.98
    assert result["manifest"]["validation"]["valid"]
    assert result["manifest"]["submission_validation"]["valid"]
