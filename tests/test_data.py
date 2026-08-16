from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from biohub_demo.data import discover_competition_root, inspect_split, resolve_split_dir


def test_data_layout_discovery_and_zarr_inspection(tmp_path: Path) -> None:
    root = tmp_path / "competition"
    train, test = root / "train", root / "test"
    train.mkdir(parents=True)
    test.mkdir()
    (root / "sample_submission.csv").write_text("id\n", encoding="utf-8")
    group = zarr.open_group(str(test / "tiny.zarr"), mode="w")
    group.create_array("0", data=np.zeros((2, 4, 8, 8), dtype=np.uint16))

    layout = discover_competition_root(root)
    assert layout.root == root.resolve()
    assert resolve_split_dir(root, "test") == test.resolve()
    report = inspect_split(test)
    assert report["dataset_count"] == 1
    assert report["ground_truth_count"] == 0
    assert report["missing_ground_truth"] == ["tiny"]
    assert report["datasets"][0]["shape"] == [2, 4, 8, 8]
