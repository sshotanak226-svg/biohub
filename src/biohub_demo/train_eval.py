"""Reproducible real-data training, held-out evaluation, and test inference.

The model, loss, prediction, and competition metric come from the downloaded
official baseline repository.  This module supplies the missing orchestration:
a persisted split, bounded/full profiles, durable metrics, and an optional run
through the portable Kaggle inference implementation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .data import discover_competition_root
from .competition_metric import (
    MAX_DISTANCE_UM,
    official_metric_contract,
    validate_evaluation_inputs,
    validate_official_metric_runtime,
)
from .io import environment_manifest, sha256_file, write_json


WORKSPACE = Path(__file__).resolve().parents[2]
BASELINE_RELATIVE = Path(
    "biohub_cell_tracking_5methods_8gb_kaggle_2026-08-06"
    "/00_共通資料/English/01_official_baseline_repository"
)
PACKAGED_BASELINE_RELATIVE = Path("kaggle/train_eval_kernel/baseline")


def find_baseline_root(explicit: Path | None = None) -> Path:
    """Locate and validate the downloaded official baseline repository."""
    candidates = (
        [explicit]
        if explicit is not None
        else [WORKSPACE / BASELINE_RELATIVE, WORKSPACE / PACKAGED_BASELINE_RELATIVE]
    )
    failures: list[str] = []
    for candidate in candidates:
        root = candidate.resolve()
        required = (
            root / "scripts" / "train_unet_transformer.py",
            root / "scripts" / "predict_unet_transformer.py",
            root / "scripts" / "evaluate.py",
            root / "src" / "tracking_cellmot",
        )
        missing = [str(path) for path in required if not path.exists()]
        if not missing:
            return root
        failures.extend(missing)
    raise FileNotFoundError("official baseline is incomplete: " + "; ".join(failures))


def _load_baseline(root: Path) -> dict[str, Any]:
    scripts = str(root / "scripts")
    source = str(root / "src")
    for value in (scripts, source):
        if value not in sys.path:
            sys.path.insert(0, value)
    modules = {
        "train": importlib.import_module("train_unet_transformer"),
        "predict": importlib.import_module("predict_unet_transformer"),
        "evaluate": importlib.import_module("evaluate"),
        "metrics": importlib.import_module("tracking_cellmot.metrics"),
        "dataspec": importlib.import_module("dataspec"),
    }
    validate_official_metric_runtime(modules)
    return modules


def build_split(
    train_dir: Path,
    *,
    seed: int,
    train_count: int | None,
    val_count: int | None,
) -> dict[str, Any]:
    """Create one deterministic train/validation split from paired Zarr/GEFF data."""
    stems = sorted(
        path.stem for path in train_dir.glob("*.zarr")
        if (train_dir / f"{path.stem}.geff").exists()
    )
    if len(stems) < 2:
        raise ValueError(f"at least two paired datasets are required in {train_dir}")
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for stem in stems:
        groups.setdefault(stem.split("_", 1)[0], []).append(stem)
    for values in groups.values():
        rng.shuffle(values)
    requested_val = val_count if val_count is not None else max(1, len(stems) // 10)
    if requested_val < 1 or requested_val >= len(stems):
        raise ValueError(f"val_count must be between 1 and {len(stems) - 1}")
    validation: list[str] = []
    if requested_val >= len(groups):
        for key in sorted(groups):
            validation.append(groups[key].pop())
    remaining_val = requested_val - len(validation)
    pooled = [stem for values in groups.values() for stem in values]
    rng.shuffle(pooled)
    validation.extend(pooled[:remaining_val])
    validation_set = set(validation)
    remaining_by_group = {
        key: [stem for stem in values if stem not in validation_set]
        for key, values in groups.items()
    }
    remaining = [stem for values in remaining_by_group.values() for stem in values]
    requested_train = train_count if train_count is not None else len(remaining)
    if requested_train < 1 or requested_train > len(remaining):
        raise ValueError(f"train_count must be between 1 and {len(remaining)}")
    training: list[str] = []
    if requested_train >= len(remaining_by_group):
        for key in sorted(remaining_by_group):
            if remaining_by_group[key]:
                training.append(remaining_by_group[key].pop())
    remaining_train = requested_train - len(training)
    pooled = [stem for values in remaining_by_group.values() for stem in values]
    rng.shuffle(pooled)
    training.extend(pooled[:remaining_train])
    if set(training) & set(validation):
        raise AssertionError("training and validation datasets overlap")
    return {"split": 0, "train": training, "test": validation}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def evaluate_prediction_dir(
    modules: dict[str, Any],
    prediction_dir: Path,
    truth_dir: Path,
    *,
    expected_datasets: list[str],
) -> dict[str, Any]:
    """Run only the Kaggle metric, failing closed on coverage or metric drift."""
    expected = validate_evaluation_inputs(
        modules["evaluate"], prediction_dir, truth_dir, expected_datasets
    )
    rows, skipped = modules["evaluate"].evaluate_pairs(
        prediction_dir, truth_dir, max_distance=MAX_DISTANCE_UM
    )
    recovered_empty: list[str] = []
    for name in list(skipped):
        predicted = modules["evaluate"]._load_graph(prediction_dir / f"{name}.geff")
        if predicted.num_nodes() != 0:
            continue
        truth = modules["evaluate"]._load_graph(truth_dir / f"{name}.geff")
        division_fn = sum(
            1 for degree in truth.out_degree(truth.node_ids()) if int(degree) > 1
        )
        result = modules["metrics"].EvaluationResult(
            edge_tp=0, edge_fp=0, edge_fn=truth.num_edges(),
            division_tp=0, division_fp=0, division_fn=division_fn,
            num_pred_nodes=0,
        )
        estimated_total = modules["evaluate"]._read_estimated_n_total(
            truth_dir / f"{name}.geff"
        )
        rows.append(modules["metrics"].per_sample_metrics(result, estimated_total, 0.0))
        recovered_empty.append(name)
    skipped = [name for name in skipped if name not in recovered_empty]
    if skipped:
        raise RuntimeError(f"official evaluation failed for datasets: {sorted(skipped)}")
    if len(rows) != len(expected):
        raise RuntimeError(
            "official evaluation row-count mismatch: "
            f"expected {len(expected)}, got {len(rows)}"
        )
    return {
        "kind": "kaggle-official-metric-held-out-real-data",
        "is_kaggle_leaderboard_score": False,
        "metric_contract": official_metric_contract(),
        "summary": _json_ready(modules["metrics"].summarise(rows)),
        "per_dataset": _json_ready(rows),
        "skipped": skipped,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute training, held-out inference/evaluation, and optional test inference."""
    started = time.time()
    layout = discover_competition_root(args.data_root)
    baseline_root = find_baseline_root(args.baseline_root)
    modules = _load_baseline(baseline_root)

    smoke = args.profile == "smoke"
    train_count = args.train_count if args.train_count is not None else (1 if smoke else None)
    val_count = args.val_count if args.val_count is not None else (1 if smoke else None)
    epochs = args.epochs if args.epochs is not None else (1 if smoke else 3)
    max_iters = args.max_iters if args.max_iters is not None else (2 if smoke else None)
    max_frames = args.max_frames if args.max_frames is not None else (8 if smoke else None)
    num_workers = args.num_workers if args.num_workers is not None else (0 if smoke else 2)
    channels = args.unet_out_channels if args.unet_out_channels is not None else (8 if smoke else 32)
    layers = args.unet_layers or ([8, 16, 32] if smoke else [32, 64, 128])
    det_threshold = args.det_threshold if args.det_threshold is not None else (0.5 if smoke else 0.99)
    method = args.method or f"real_{args.profile}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", method):
        raise ValueError("--method may contain only letters, digits, dot, underscore, and dash")

    run_dir = (args.output_dir or (WORKSPACE / "outputs" / "training" / method)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    split = build_split(
        layout.train, seed=args.seed, train_count=train_count, val_count=val_count
    )
    splits_path = run_dir / "dataset_splits.json"
    splits_path.write_text(json.dumps([split], indent=2) + "\n", encoding="utf-8")

    run_config = {
        "profile": args.profile,
        "method": method,
        "data_root": str(layout.root),
        "baseline_root": str(baseline_root),
        "split_file": str(splits_path),
        "train_datasets": len(split["train"]),
        "validation_datasets": len(split["test"]),
        "epochs": epochs,
        "max_iters_per_epoch": max_iters,
        "max_frames_per_dataset": max_frames,
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "unet_out_channels": channels,
        "unet_layers": layers,
        "downsample": args.downsample,
        "detection_threshold": det_threshold,
        "seed": args.seed,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(run_config, run_dir / "run_config.json")

    print("=== REAL TRAINING ===", flush=True)
    print(json.dumps(run_config, indent=2, ensure_ascii=False), flush=True)
    modules["train"].train(
        data_dir=layout.train,
        fold=0,
        splits_file=splits_path,
        method=method,
        n_epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=num_workers,
        unet_out_channels=channels,
        unet_layers=layers,
        downsample=tuple(args.downsample),
        det_loss_weight=args.det_loss_weight,
        det_neg_weight=args.det_neg_weight,
        max_iters=max_iters,
        seed=args.seed,
        max_frames=max_frames,
        window_size=2,
        pool_kernel_um=args.pool_kernel_um,
        data_parallel=False,
    )

    weights_dir = baseline_root / "weights" / method / "split_0"
    checkpoint = weights_dir / "edge_predictor_best.pth"
    model_config = weights_dir / "config.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"training did not create a checkpoint: {checkpoint}")

    print("=== HELD-OUT INFERENCE ===", flush=True)
    predict_config = modules["predict"].PredictConfig(
        det_threshold=det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=-1.0,
        ilp_appearance_weight=0.1,
        ilp_disappearance_weight=0.1,
        ilp_division_weight=1.0,
    )
    modules["predict"].predict(
        data_dir=layout.train,
        fold=0,
        splits_file=splits_path,
        weights_path=checkpoint,
        cfg=predict_config,
        method=method,
        unet_batch_size=args.inference_batch_size,
        evaluate=False,
    )

    prediction_dir = (
        baseline_root / "predictions" / modules["dataspec"].USERNAME / method / "split_0"
    )
    metrics = evaluate_prediction_dir(
        modules,
        prediction_dir,
        layout.train,
        expected_datasets=split["test"],
    )
    write_json(metrics, run_dir / "metrics.json")
    copied_predictions = run_dir / "validation_predictions"
    copied_predictions.mkdir(exist_ok=True)
    for source in prediction_dir.glob("*.geff"):
        target = copied_predictions / source.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)

    kaggle_manifest = None
    if args.predict_test:
        print("=== PORTABLE KAGGLE TEST INFERENCE ===", flush=True)
        from .kaggle import run_kaggle

        kaggle_manifest = run_kaggle(
            WORKSPACE / "configs" / "kaggle_inference.yaml",
            output_override=run_dir / "kaggle_test",
            input_override=layout.test,
            weight_overrides=[checkpoint],
            model_config_override=model_config,
            max_datasets=args.max_test_datasets,
        )

    manifest = {
        **environment_manifest(),
        "run_config": run_config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config": str(model_config),
        "metrics": metrics,
        "kaggle_test_manifest": kaggle_manifest,
        "elapsed_seconds": time.time() - started,
    }
    write_json(_json_ready(manifest), run_dir / "run_manifest.json")
    print("=== MEASURED RESULT ===", flush=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return manifest


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on official Biohub train data, infer an unseen validation split, and score it."
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--val-count", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unet-out-channels", type=int)
    parser.add_argument("--unet-layers", type=_parse_ints)
    parser.add_argument("--downsample", type=_parse_ints, default=[1, 4, 4])
    parser.add_argument("--det-loss-weight", type=float, default=1.0)
    parser.add_argument("--det-neg-weight", type=float, default=1e-2)
    parser.add_argument("--pool-kernel-um", type=float, default=5.0)
    parser.add_argument("--det-threshold", type=float)
    parser.add_argument("--use-ilp", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--max-test-datasets", type=int)
    args = parser.parse_args()
    if len(args.downsample) != 3:
        parser.error("--downsample must contain Z,Y,X")
    if args.unet_layers is not None and not args.unet_layers:
        parser.error("--unet-layers must not be empty")
    run(args)


if __name__ == "__main__":
    main()
