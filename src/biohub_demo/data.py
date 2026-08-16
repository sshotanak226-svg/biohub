"""Discovery and lightweight validation of the official Biohub data layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zarr

COMPETITION = "biohub-cell-tracking-during-development"
EXPECTED_TRAIN_DATASETS = 199
EXPECTED_TEST_DATASETS = 4


@dataclass(frozen=True)
class CompetitionLayout:
    root: Path
    train: Path
    test: Path
    sample_submission: Path


def _as_layout(path: Path) -> CompetitionLayout | None:
    candidate = path.resolve()
    if candidate.name in {"train", "test"}:
        candidate = candidate.parent
    train = candidate / "train"
    test = candidate / "test"
    sample = candidate / "sample_submission.csv"
    if train.is_dir() and test.is_dir() and sample.is_file():
        return CompetitionLayout(candidate, train, test, sample)
    return None


def discover_competition_root(explicit: str | Path | None = None) -> CompetitionLayout:
    """Find an extracted competition root on this PC or in a Kaggle notebook."""
    if explicit is not None and str(explicit).strip().lower() not in {"", "auto"}:
        requested = Path(explicit).expanduser()
        layout = _as_layout(requested)
        if layout is None:
            raise FileNotFoundError(
                f"Biohub data root is incomplete: {requested}. Expected train/, test/, "
                "and sample_submission.csv."
            )
        return layout

    workspace = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    environment_root = os.environ.get("BIOHUB_DATA_ROOT")
    if environment_root:
        candidates.append(Path(environment_root))
    candidates.extend(
        [
            workspace
            / "biohub_top10_download"
            / "data"
            / "kagglehub_cache"
            / "competitions"
            / COMPETITION,
            Path("/kaggle/input") / COMPETITION,
            Path("/kaggle/input/competitions") / COMPETITION,
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(path.parent for path in kaggle_input.glob("*/test") if path.is_dir())

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        layout = _as_layout(candidate)
        if layout is not None:
            return layout
    raise FileNotFoundError(
        "Biohub competition data was not found. Checked: " + "; ".join(checked)
    )


def resolve_split_dir(value: str | Path | None, split: str = "test") -> Path:
    if split not in {"train", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if value is not None and str(value).strip().lower() not in {"", "auto"}:
        path = Path(value).expanduser()
        if path.is_dir() and path.name == split:
            return path.resolve()
        if path.is_dir() and any(path.glob("*.zarr")):
            return path.resolve()
        if (path / split).is_dir():
            return (path / split).resolve()
        # A config authored for Kaggle may use a mount convention that differs
        # from the current notebook. Fall back to bounded discovery there.
        if str(path).replace("\\", "/").startswith("/kaggle/input/"):
            return getattr(discover_competition_root(), split)
        raise FileNotFoundError(f"{split} input directory not found: {path}")
    return getattr(discover_competition_root(), split)


def inspect_split(path: Path) -> dict[str, Any]:
    datasets = sorted(path.glob("*.zarr"))
    ground_truth = sorted(path.glob("*.geff"))
    items: list[dict[str, Any]] = []
    for dataset in datasets:
        group = zarr.open_group(str(dataset), mode="r")
        if "0" not in group:
            raise ValueError(f"missing array '0': {dataset}")
        array = group["0"]
        if len(array.shape) != 4:
            raise ValueError(f"expected T,Z,Y,X at {dataset}, got {array.shape}")
        items.append(
            {
                "name": dataset.stem,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "chunks": list(array.chunks),
            }
        )
    dataset_names = {item.stem for item in datasets}
    truth_names = {item.stem for item in ground_truth}
    return {
        "path": str(path.resolve()),
        "dataset_count": len(datasets),
        "ground_truth_count": len(ground_truth),
        "all_datasets_have_ground_truth": dataset_names <= truth_names,
        "missing_ground_truth": sorted(dataset_names - truth_names),
        "orphan_ground_truth": sorted(truth_names - dataset_names),
        "datasets": items,
    }
