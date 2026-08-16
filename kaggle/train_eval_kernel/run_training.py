"""Kaggle GPU job: real training -> held-out scoring -> test submission.

All reported validation numbers are calculated from predicted GEFF graphs and
the held-out official train GEFF files.  The competition test data has no public
labels; its CSV must be submitted to obtain a leaderboard score.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORKING = Path("/kaggle/working")
SOURCE = Path(__file__).resolve().parent
COMPETITION = "biohub-cell-tracking-during-development"
SEED = 20260809
METHOD = "unet_transformer_real_3ep"


def install_dependencies() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "zarr>=3.0.10,<4",
            "geff>=1.1.3.1.1",
            "geff-spec<1.2",
            "git+https://github.com/royerlab/tracksdata@main",
        ]
    )


def find_competition_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions") / COMPETITION,
        Path("/kaggle/input") / COMPETITION,
    ]
    candidates.extend(path.parent for path in Path("/kaggle/input").glob("*/train"))
    for path in candidates:
        if (path / "train").is_dir() and (path / "test").is_dir():
            return path
    raise FileNotFoundError(f"competition data not mounted; checked {candidates}")


def prepare_baseline() -> Path:
    """Extract the official baseline source attached as a private Dataset."""
    local = SOURCE / "baseline"
    if (local / "scripts" / "train_unet_transformer.py").is_file():
        return local
    # Kaggle normally expands uploaded ZIP datasets before mounting them.
    for script in Path("/kaggle/input").glob("**/scripts/train_unet_transformer.py"):
        candidate = script.parent.parent
        if (candidate / "src" / "tracking_cellmot").is_dir():
            return candidate
    archives = list(Path("/kaggle/input").glob("**/baseline_source.zip"))
    if not archives:
        raise FileNotFoundError("attached baseline_source.zip was not found under /kaggle/input")
    destination = WORKING / "baseline_source"
    shutil.unpack_archive(archives[0], destination)
    candidates = [destination / "baseline", destination]
    for candidate in candidates:
        if (candidate / "scripts" / "train_unet_transformer.py").is_file():
            return candidate
    raise FileNotFoundError(f"baseline source missing after extracting {archives[0]}")


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    started = time.time()
    install_dependencies()
    baseline = prepare_baseline()
    sys.path.insert(0, str(baseline / "scripts"))
    sys.path.insert(0, str(baseline / "src"))

    import torch
    import dataspec
    import evaluate
    import geffs_to_csv
    import predict_unet_transformer as predictor
    import train_unet_transformer as trainer
    from tracking_cellmot.metrics import summarise

    root = find_competition_root()
    train_dir = root / "train"
    test_dir = root / "test"
    stems = sorted(
        path.stem for path in train_dir.glob("*.zarr")
        if (train_dir / f"{path.stem}.geff").exists()
    )
    random.Random(SEED).shuffle(stems)
    n_val = max(1, len(stems) // 10)
    split = {"split": 0, "train": stems[n_val:], "test": stems[:n_val]}
    splits_path = WORKING / "dataset_splits.json"
    splits_path.write_text(json.dumps([split], indent=2) + "\n")

    # Redirect every baseline artifact away from the read-only kernel source.
    dataspec.WEIGHTS_PATH = WORKING / "weights"
    dataspec.PREDICTIONS_PATH = WORKING / "predictions"
    dataspec.RESULTS_PATH = WORKING / "results"
    trainer.WEIGHTS_PATH = dataspec.WEIGHTS_PATH

    config = {
        "kind": "real-full-training",
        "train_datasets": len(split["train"]),
        "validation_datasets": len(split["test"]),
        "epochs": 3,
        "batch_size": 8,
        "num_workers": 2,
        "lr": 1e-4,
        "unet_out_channels": 32,
        "unet_layers": [32, 64, 128],
        "downsample": [1, 4, 4],
        "detection_threshold": 0.99,
        "seed": SEED,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    (WORKING / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("=== CONFIG ===", flush=True)
    print(json.dumps(config, indent=2), flush=True)

    trainer.train(
        data_dir=train_dir,
        fold=0,
        splits_file=splits_path,
        method=METHOD,
        n_epochs=3,
        lr=1e-4,
        batch_size=8,
        num_workers=2,
        unet_out_channels=32,
        unet_layers=[32, 64, 128],
        downsample=(1, 4, 4),
        det_loss_weight=1.0,
        det_neg_weight=1e-2,
        seed=SEED,
        window_size=2,
        pool_kernel_um=5.0,
        data_parallel=True,
    )

    weights_dir = dataspec.WEIGHTS_PATH / METHOD / "split_0"
    checkpoint = weights_dir / "edge_predictor_best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing after training: {checkpoint}")

    cfg = predictor.PredictConfig(det_threshold=0.99, use_ilp=False)
    print("=== HELD-OUT INFERENCE ===", flush=True)
    predictor.predict(
        data_dir=train_dir,
        fold=0,
        splits_file=splits_path,
        weights_path=checkpoint,
        cfg=cfg,
        method=METHOD,
        unet_batch_size=4,
        evaluate=False,
    )
    validation_dir = dataspec.PREDICTIONS_PATH / dataspec.USERNAME / METHOD / "split_0"
    rows, skipped = evaluate.evaluate_pairs(validation_dir, train_dir)
    metrics = {
        "kind": "held-out-real-data",
        "is_kaggle_leaderboard_score": False,
        "summary": json_ready(summarise(rows)),
        "per_dataset": json_ready(rows),
        "skipped": skipped,
    }
    (WORKING / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    print("=== HELD-OUT METRICS ===", flush=True)
    print(json.dumps(metrics["summary"], indent=2), flush=True)

    print("=== TEST INFERENCE ===", flush=True)
    test_names = sorted(path.stem for path in test_dir.glob("*.zarr"))
    test_split_path = WORKING / "test_split.json"
    test_split_path.write_text(
        json.dumps([{"split": 0, "train": [], "test": test_names}], indent=2) + "\n"
    )
    test_method = METHOD + "_test"
    predictor.predict(
        data_dir=test_dir,
        fold=0,
        splits_file=test_split_path,
        weights_path=checkpoint,
        cfg=cfg,
        method=test_method,
        unet_batch_size=4,
        evaluate=False,
    )
    test_prediction_dir = (
        dataspec.PREDICTIONS_PATH / dataspec.USERNAME / test_method / "split_0"
    )
    geffs_to_csv.geffs_to_csv(test_prediction_dir, WORKING / "submission.csv")

    manifest = {
        "config": config,
        "checkpoint": str(checkpoint),
        "validation_metrics": metrics,
        "submission": str(WORKING / "submission.csv"),
        "elapsed_seconds": time.time() - started,
    }
    (WORKING / "run_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, allow_nan=False) + "\n"
    )
    print("=== COMPLETE ===", flush=True)
    print(json.dumps(json_ready(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
