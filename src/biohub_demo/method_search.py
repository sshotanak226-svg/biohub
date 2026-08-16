"""High-level Biohub method search before fine hyperparameter tuning."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .augmentations import robust_augmentations
from .competition_metric import METRIC_CONTRACT_ID, official_metric_contract
from .data import discover_competition_root
from .early_stopping import train_with_early_stopping
from .hyper_search import _signature
from .io import is_complete_geff, sha256_file, write_geff, write_json
from .tracking_variants import build_variant, graph_from_raw_prediction
from .train_eval import (
    WORKSPACE,
    _json_ready,
    _load_baseline,
    build_split,
    evaluate_prediction_dir,
    find_baseline_root,
)


def load_method_search_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("method-search config must be a YAML mapping")
    for section in ("dataset", "training", "raw_inference", "method_variants"):
        if section not in config:
            raise ValueError(f"missing method-search section: {section}")
    variants = config["method_variants"]
    if not isinstance(variants, list) or len(variants) < 2:
        raise ValueError("method_variants must contain at least two methods")
    names = [str(item.get("name", "")) for item in variants]
    if len(set(names)) != len(names) or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names
    ):
        raise ValueError("method variant names must be unique filesystem-safe identifiers")
    allowed = {
        "greedy", "ilp", "distance_hungarian", "hybrid_hungarian",
        "unbalanced_ot",
    }
    unknown = sorted({str(item.get("tracker")) for item in variants} - allowed)
    if unknown:
        raise ValueError(f"unknown trackers: {unknown}")
    dataset = config["dataset"]
    validation_count = int(dataset["validation_count"])
    if validation_count < 2:
        raise ValueError("dataset.validation_count must be at least 2")
    training = config["training"]
    if int(training["min_epochs"]) > int(training["max_epochs"]):
        raise ValueError("training.min_epochs cannot exceed training.max_epochs")
    if int(training["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience must be positive")
    for key in ("max_iters_per_epoch", "max_validation_iters"):
        if training.get(key) is not None and int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive or null")
    raw = config["raw_inference"]
    for key in ("detection_threshold", "edge_threshold"):
        if not 0 <= float(raw[key]) <= 1:
            raise ValueError(f"raw_inference.{key} must be between 0 and 1")
    minimum_variant_detection = min(float(item["detection_threshold"]) for item in variants)
    if float(raw["detection_threshold"]) > minimum_variant_detection:
        raise ValueError("raw detection threshold must cover every method variant")
    for item in variants:
        if item.get("tracker") != "unbalanced_ot":
            continue
        if str(item.get("ot_mode", "hybrid")) not in {
            "distance", "hybrid", "hybrid_division", "consensus_ilp",
        }:
            raise ValueError(f"unknown OT mode: {item.get('ot_mode')}")
        if int(item.get("sinkhorn_iterations", 100)) < 1:
            raise ValueError("sinkhorn_iterations must be positive")
        if float(item.get("entropy_epsilon", 0.05)) <= 0:
            raise ValueError("entropy_epsilon must be positive")
    return config


def build_method_search_plan(config: dict[str, Any], paired_datasets: int) -> dict[str, Any]:
    validation_count = int(config["dataset"]["validation_count"])
    return {
        "purpose": "coarse-method-selection-before-hyperparameter-search",
        "paired_datasets": paired_datasets,
        "training_datasets": paired_datasets - validation_count,
        "validation_datasets": validation_count,
        "uses_all_available_training_sequences_except_holdout": True,
        "max_frames_per_dataset": config["dataset"].get("max_frames"),
        "max_training_epochs": int(config["training"]["max_epochs"]),
        "min_training_epochs": int(config["training"]["min_epochs"]),
        "early_stopping_patience": int(config["training"]["early_stopping_patience"]),
        "max_training_batches_per_epoch": config["training"].get("max_iters_per_epoch"),
        "max_validation_batches_per_epoch": config["training"].get("max_validation_iters"),
        "shared_raw_inference_passes": validation_count,
        "method_variants": [item["name"] for item in config["method_variants"]],
        "official_metric_ranked_variants": len(config["method_variants"]),
        "metric_contract": official_metric_contract(),
    }


def _official_score(record: dict[str, Any]) -> float:
    if record.get("status") != "complete":
        return float("-inf")
    metrics = record.get("metrics", {})
    if metrics.get("metric_contract", {}).get("id") != METRIC_CONTRACT_ID:
        return float("-inf")
    value = metrics.get("summary", {}).get("score")
    return float(value) if value is not None else float("-inf")


def _checkpoint_record(
    *,
    modules: dict[str, Any],
    baseline: Path,
    train_dir: Path,
    splits_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    split: dict[str, Any],
    checkpoint_override: Path | None,
) -> dict[str, Any]:
    if checkpoint_override is not None:
        checkpoint = checkpoint_override.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not (checkpoint.parent / "config.json").is_file():
            raise FileNotFoundError(f"checkpoint config is missing: {checkpoint.parent / 'config.json'}")
        return {
            "status": "provided", "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }

    training = config["training"]
    method = str(training.get("method", "local_method_search_robust"))
    checkpoint = baseline / "weights" / method / "split_0" / "edge_predictor_best.pth"
    marker = output_dir / "training" / "training.complete.json"
    payload = {
        "training": training,
        "split": split,
        "max_frames": config["dataset"].get("max_frames"),
        "seed": int(config.get("seed", 20260809)),
        "augmentation_recipe": "xy-dihedral+gain+gamma+bias+noise-v1",
    }
    signature = _signature(payload)
    if bool(config.get("reuse_completed", True)) and marker.is_file() and checkpoint.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8-sig"))
        if previous.get("signature") == signature:
            return previous

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train_kwargs = {
        "data_dir": train_dir,
        "fold": 0,
        "splits_file": splits_path,
        "method": method,
        "n_epochs": int(training["max_epochs"]),
        "lr": float(training["learning_rate"]),
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "unet_out_channels": int(training["unet_out_channels"]),
        "unet_layers": [int(value) for value in training["unet_layers"]],
        "downsample": tuple(int(value) for value in training["downsample"]),
        "det_loss_weight": float(training["detection_loss_weight"]),
        "det_neg_weight": float(training["detection_negative_weight"]),
        "max_iters": training.get("max_iters_per_epoch"),
        "seed": int(config.get("seed", 20260809)),
        "max_frames": config["dataset"].get("max_frames"),
        "window_size": int(training.get("window_size", 2)),
        "augmentations": robust_augmentations(),
        "pool_kernel_um": float(training.get("pool_kernel_um", 5.0)),
        "data_parallel": bool(training.get("data_parallel", False)),
    }
    result = train_with_early_stopping(
        modules["train"],
        checkpoint=checkpoint,
        min_epochs=int(training["min_epochs"]),
        patience=int(training["early_stopping_patience"]),
        min_delta=float(training.get("early_stopping_min_delta", 1e-4)),
        train_kwargs=train_kwargs,
        max_validation_iters=training.get("max_validation_iters"),
    )
    record = {
        "status": "complete",
        "signature": signature,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameters": payload,
        "early_stopping": _json_ready(result.__dict__),
        "elapsed_seconds": time.time() - started,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_json(record, marker)
    return record


def _capture_raw_prediction(
    predict_module: Any,
    *,
    model: torch.nn.Module,
    dataset_path: Path,
    device: torch.device,
    window_size: int,
    downsample: tuple[int, ...],
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    captured: dict[int, np.ndarray] = {}
    original_detector = predict_module._detect_cells_pooled

    def detector(det_logits, t, det_threshold=0.5, pool_kernel=(3, 3, 3)):
        coords = original_detector(det_logits, t, det_threshold, pool_kernel)
        if len(coords):
            indices = coords[:, 1:].astype(np.int64)
            probabilities = torch.sigmoid(det_logits[0]).detach().cpu().numpy()
            captured[int(t)] = probabilities[indices[:, 0], indices[:, 1], indices[:, 2]]
        else:
            captured[int(t)] = np.empty(0, dtype=np.float32)
        return coords

    cfg = predict_module.PredictConfig(
        det_threshold=float(settings["detection_threshold"]),
        det_tta=bool(settings.get("det_tta", True)),
        pool_kernel_um=float(settings.get("pool_kernel_um", 3.0)),
        edge_activation=str(settings.get("edge_activation", "softmax")),
        threshold=float(settings["edge_threshold"]),
        # Prevent the baseline's greedy pruning; method variants select edges later.
        use_ilp=True,
    )
    predict_module._detect_cells_pooled = detector
    try:
        coords, edges = predict_module.predict_video(
            model,
            dataset_path,
            device,
            cfg=cfg,
            window_size=window_size,
            unet_batch_size=int(settings.get("inference_batch_size", 1)),
            downsample=downsample,
        )
    finally:
        predict_module._detect_cells_pooled = original_detector
    score_parts = [captured[int(t)] for t in sorted(captured)]
    scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=np.float32)
    edge_array = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    return coords, scores, edge_array


def _load_or_predict_raw(
    *,
    modules: dict[str, Any],
    checkpoint_record: dict[str, Any],
    train_dir: Path,
    names: list[str],
    output_dir: Path,
    settings: dict[str, Any],
    reuse_completed: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    signature = _signature({
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "raw_inference": settings,
    })
    raw_dir = output_dir / "raw_candidates" / signature[:12]
    metadata = raw_dir / "metadata.json"
    if metadata.is_file():
        old = json.loads(metadata.read_text(encoding="utf-8-sig"))
        reusable = reuse_completed and old.get("signature") == signature
    else:
        reusable = False

    missing = [name for name in names if not (raw_dir / f"{name}.npz").is_file()]
    if not reusable or missing:
        raw_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, window_size, downsample = modules["predict"].load_model(
            Path(checkpoint_record["checkpoint"]), device
        )
        for index, name in enumerate(names, 1):
            target = raw_dir / f"{name}.npz"
            if reusable and target.is_file():
                continue
            print(f"RAW INFERENCE {index}/{len(names)}: {name}", flush=True)
            coords, scores, edges = _capture_raw_prediction(
                modules["predict"], model=model, dataset_path=train_dir / name,
                device=device, window_size=window_size, downsample=downsample,
                settings=settings,
            )
            np.savez_compressed(target, coords=coords, scores=scores, edges=edges)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_json({"signature": signature, "settings": settings}, metadata)

    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in names:
        with np.load(raw_dir / f"{name}.npz") as values:
            output[name] = (values["coords"], values["scores"], values["edges"])
    return output


def execute_method_search(config_path: Path, checkpoint_override: Path | None = None) -> dict[str, Any]:
    config = load_method_search_config(config_path)
    config.setdefault("seed", 20260809)
    layout = discover_competition_root(config.get("data_root"))
    baseline = find_baseline_root(None)
    modules = _load_baseline(baseline)
    output_value = Path(config.get("output_dir", "outputs/method_search"))
    output_dir = output_value if output_value.is_absolute() else WORKSPACE / output_value
    output_dir.mkdir(parents=True, exist_ok=True)

    split = build_split(
        layout.train,
        seed=int(config["seed"]),
        train_count=None,
        val_count=int(config["dataset"]["validation_count"]),
    )
    splits_path = output_dir / "dataset_splits.json"
    splits_path.write_text(json.dumps([split], indent=2) + "\n", encoding="utf-8")
    checkpoint_record = _checkpoint_record(
        modules=modules, baseline=baseline, train_dir=layout.train,
        splits_path=splits_path, output_dir=output_dir, config=config,
        split=split, checkpoint_override=checkpoint_override,
    )
    raw = _load_or_predict_raw(
        modules=modules,
        checkpoint_record=checkpoint_record,
        train_dir=layout.train,
        names=list(split["test"]),
        output_dir=output_dir,
        settings=config["raw_inference"],
        reuse_completed=bool(config.get("reuse_completed", True)),
    )

    records: list[dict[str, Any]] = []
    for index, variant in enumerate(config["method_variants"], 1):
        print(f"METHOD {index}/{len(config['method_variants'])}: {variant['name']}", flush=True)
        variant_signature = _signature({
            "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
            "raw_inference": config["raw_inference"],
            "variant": variant,
            "metric_contract_id": METRIC_CONTRACT_ID,
        })
        prediction_dir = output_dir / "predictions" / f"{variant['name']}_{variant_signature[:12]}"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        record_path = output_dir / "variants" / f"{variant['name']}_{variant_signature[:12]}.json"
        if bool(config.get("reuse_completed", True)) and record_path.is_file():
            previous = json.loads(record_path.read_text(encoding="utf-8-sig"))
            complete_files = all(
                is_complete_geff(prediction_dir / f"{name}.geff")
                for name in split["test"]
            )
            if (
                previous.get("status") == "complete"
                and previous.get("signature") == variant_signature
                and previous.get("metrics", {}).get("metric_contract", {}).get("id")
                == METRIC_CONTRACT_ID
                and complete_files
            ):
                records.append(previous)
                continue
        diagnostics: dict[str, Any] = {}
        started = time.time()
        for name in split["test"]:
            geff_path = prediction_dir / f"{name}.geff"
            if is_complete_geff(geff_path):
                diagnostics[name] = {"reused_complete_geff": True}
                continue
            coords, scores, edge_array = raw[name]
            raw_edges = [tuple(row) for row in edge_array.tolist()]
            raw_graph, raw_candidates = graph_from_raw_prediction(
                name, coords, scores, raw_edges, (1.625, 0.40625, 0.40625)
            )
            graph, diagnostic = build_variant(raw_graph, raw_candidates, variant)
            write_geff(graph, geff_path)
            diagnostics[name] = diagnostic
        metrics = evaluate_prediction_dir(
            modules, prediction_dir, layout.train, expected_datasets=list(split["test"])
        )
        record = {
            "status": "complete",
            "signature": variant_signature,
            "name": variant["name"],
            "settings": variant,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "prediction_dir": str(prediction_dir),
            "elapsed_seconds": time.time() - started,
        }
        record_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(_json_ready(record), record_path)
        records.append(record)

    records.sort(key=_official_score, reverse=True)
    if not records or _official_score(records[0]) == float("-inf"):
        raise RuntimeError("no method variant produced an official competition score")
    result = {
        "kind": "kaggle-official-metric-coarse-method-search",
        "is_kaggle_leaderboard_score": False,
        "metric_contract": official_metric_contract(),
        "selection_rule": "maximum held-out official competition score",
        "data_usage": {
            "available_paired_train": len(split["train"]) + len(split["test"]),
            "training": len(split["train"]),
            "validation": len(split["test"]),
            "max_frames": config["dataset"].get("max_frames"),
        },
        "checkpoint": checkpoint_record,
        "results": records,
        "best": records[0],
    }
    write_json(_json_ready(result), output_dir / "method_search_results.json")
    write_json(_json_ready(records[0]), output_dir / "best_method.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one strong shared model, then compare coarse tracking methods."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/local_method_search.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="skip training and reuse this checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_method_search_config(args.config)
    layout = discover_competition_root(config.get("data_root"))
    paired = sum(
        1 for path in layout.train.glob("*.zarr")
        if (layout.train / f"{path.stem}.geff").exists()
    )
    plan = build_method_search_plan(config, paired)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    result = execute_method_search(args.config, args.checkpoint)
    print(json.dumps({
        "best_method": result["best"]["name"],
        "official_cv_score": result["best"]["metrics"]["summary"]["score"],
        "result_file": str(Path(config.get("output_dir", "outputs/method_search")) / "method_search_results.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
