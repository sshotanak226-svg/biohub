from pathlib import Path

from biohub_demo.train_eval import build_split, find_baseline_root


def test_build_split_is_disjoint_and_reproducible(tmp_path: Path) -> None:
    for index in range(6):
        (tmp_path / f"sample_{index}.zarr").mkdir()
        (tmp_path / f"sample_{index}.geff").mkdir()
    first = build_split(tmp_path, seed=7, train_count=3, val_count=2)
    second = build_split(tmp_path, seed=7, train_count=3, val_count=2)
    assert first == second
    assert len(first["train"]) == 3
    assert len(first["test"]) == 2
    assert not (set(first["train"]) & set(first["test"]))


def test_build_split_stratifies_dataset_families(tmp_path: Path) -> None:
    for family in ("44b6", "6bba"):
        for index in range(5):
            (tmp_path / f"{family}_{index}.zarr").mkdir()
            (tmp_path / f"{family}_{index}.geff").mkdir()
    split = build_split(tmp_path, seed=11, train_count=4, val_count=4)
    assert {name.split("_", 1)[0] for name in split["train"]} == {"44b6", "6bba"}
    assert {name.split("_", 1)[0] for name in split["test"]} == {"44b6", "6bba"}


def test_downloaded_baseline_is_present() -> None:
    assert find_baseline_root().is_dir()
