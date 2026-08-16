"""Two-stage local hyperparameter search with real Biohub train/GEFF data."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .competition_metric import METRIC_CONTRACT_ID, official_metric_contract
from .data import discover_competition_root
from .io import sha256_file, write_json
from .train_eval import (
    WORKSPACE,
    _json_ready,
    _load_baseline,
    build_split,
    evaluate_prediction_dir,
    find_baseline_root,
)


def load_search_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("search config must be a YAML mapping")
    required = ("dataset", "training", "training_trials", "base_inference", "inference_grid")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing search config sections: {missing}")
    trials = config["training_trials"]
    if not isinstance(trials, list) or not trials:
        raise ValueError("training_trials must be a non-empty list")
    names = [str(trial.get("name", "")) for trial in trials]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names):
        raise ValueError("training trial names must be unique filesystem-safe identifiers")
    training = config["training"]
    if int(training["top_k_checkpoints"]) < 1:
        raise ValueError("training.top_k_checkpoints must be positive")
    if int(training["batch_size"]) < 1 or int(training["epochs"]) < 1:
        raise ValueError("training batch_size and epochs must be positive")
    dataset = config["dataset"]
    if int(dataset["train_count"]) < 1 or int(dataset["validation_count"]) < 1:
        raise ValueError("dataset train_count and validation_count must be positive")
    if int(dataset["max_frames"]) < 2:
        raise ValueError("dataset.max_frames must be at least 2")
    for trial in trials:
        if float(trial["learning_rate"]) <= 0:
            raise ValueError(f"learning_rate must be positive: {trial['name']}")
        if float(trial["detection_loss_weight"]) <= 0:
            raise ValueError(f"detection_loss_weight must be positive: {trial['name']}")
        if float(trial["detection_negative_weight"]) <= 0:
            raise ValueError(f"detection_negative_weight must be positive: {trial['name']}")
    inference_values = [config["base_inference"]]
    inference_values.extend(
        {
            "detection_threshold": detection,
            "edge_threshold": edge,
            "pool_kernel_um": pool,
        }
        for detection, edge, pool in itertools.product(
            config["inference_grid"]["detection_thresholds"],
            config["inference_grid"]["edge_thresholds"],
            config["inference_grid"]["pool_kernel_um"],
        )
    )
    for values in inference_values:
        for key in ("detection_threshold", "edge_threshold"):
            if not 0.0 <= float(values[key]) <= 1.0:
                raise ValueError(f"{key} must be between 0 and 1")
        if float(values["pool_kernel_um"]) <= 0:
            raise ValueError("pool_kernel_um must be positive")
    return config


def build_search_plan(config: dict[str, Any]) -> dict[str, Any]:
    training_trials = len(config["training_trials"])
    default_epochs = int(config["training"]["epochs"])
    trial_epochs = [
        int(trial.get("epochs", default_epochs)) for trial in config["training_trials"]
    ]
    top_k = min(int(config["training"]["top_k_checkpoints"]), training_trials)
    grid = config["inference_grid"]
    grid_per_checkpoint = (
        len(grid["detection_thresholds"])
        * len(grid["edge_thresholds"])
        * len(grid["pool_kernel_um"])
    )
    ilp_trials = min(int(grid.get("compare_ilp_for_top_k", 0)), top_k * grid_per_checkpoint)
    return {
        "training_trials": training_trials,
        "epochs_per_training_trial": trial_epochs,
        "maximum_training_epochs": sum(trial_epochs),
        "base_validation_trials": training_trials,
        "selected_checkpoints": top_k,
        "grid_trials_per_checkpoint": grid_per_checkpoint,
        "grid_trials": top_k * grid_per_checkpoint,
        "additional_ilp_trials": ilp_trials,
        "maximum_inference_trials": training_trials + top_k * grid_per_checkpoint + ilp_trials,
    }


def _score(record: dict[str, Any]) -> float:
    if record.get("status") != "complete":
        return float("-inf")
    metrics = record.get("metrics", {})
    if metrics.get("metric_contract", {}).get("id") != METRIC_CONTRACT_ID:
        return float("-inf")
    summary = metrics.get("summary", {})
    value = summary.get("score")
    return float(value) if value is not None else float("-inf")


def _signature(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_matching_record(path: Path, signature: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if record.get("signature") == signature and record.get("status") == "complete":
        return record
    return None


def _training_parameters(config: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    shared = config["training"]
    return {
        "name": trial["name"],
        "learning_rate": float(trial["learning_rate"]),
        "detection_loss_weight": float(trial["detection_loss_weight"]),
        "detection_negative_weight": float(trial["detection_negative_weight"]),
        "epochs": int(trial.get("epochs", shared["epochs"])),
        "batch_size": int(trial.get("batch_size", shared["batch_size"])),
        "num_workers": int(shared["num_workers"]),
        "window_size": int(shared["window_size"]),
        "downsample": [int(value) for value in shared["downsample"]],
        "unet_out_channels": int(shared["unet_out_channels"]),
        "unet_layers": [int(value) for value in shared["unet_layers"]],
        "pool_kernel_um": float(shared["pool_kernel_um"]),
    }


def _train_one(
    *,
    modules: dict[str, Any],
    baseline: Path,
    train_dir: Path,
    splits_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    trial: dict[str, Any],
    split_signature: str,
) -> dict[str, Any]:
    params = _training_parameters(config, trial)
    method = f"local_hp_{params['name']}"
    checkpoint = baseline / "weights" / method / "split_0" / "edge_predictor_best.pth"
    trial_dir = output_dir / "training" / params["name"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    marker = trial_dir / "training.complete.json"
    signature = _signature({"parameters": params, "split": split_signature, "seed": config["seed"]})
    if bool(config.get("reuse_completed", True)):
        previous = _read_matching_record(marker, signature)
        if previous is not None and checkpoint.is_file():
            return previous

    started = time.time()
    try:
        model = modules["train"].train(
            data_dir=train_dir,
            fold=0,
            splits_file=splits_path,
            method=method,
            n_epochs=params["epochs"],
            lr=params["learning_rate"],
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            unet_out_channels=params["unet_out_channels"],
            unet_layers=params["unet_layers"],
            downsample=tuple(params["downsample"]),
            det_loss_weight=params["detection_loss_weight"],
            det_neg_weight=params["detection_negative_weight"],
            seed=int(config["seed"]),
            max_frames=int(config["dataset"]["max_frames"]),
            window_size=params["window_size"],
            pool_kernel_um=params["pool_kernel_um"],
            data_parallel=False,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint was not created: {checkpoint}")
        record = {
            "status": "complete",
            "signature": signature,
            "method": method,
            "parameters": params,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "elapsed_seconds": time.time() - started,
        }
        write_json(record, marker)
        return record
    except Exception as exc:
        record = {
            "status": "failed",
            "signature": signature,
            "method": method,
            "parameters": params,
            "checkpoint": str(checkpoint),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.time() - started,
        }
        write_json(record, trial_dir / "training.failed.json")
        if not bool(config.get("continue_on_error", True)):
            raise
        return record


def _inference_parameters(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "detection_threshold": float(values["detection_threshold"]),
        "edge_threshold": float(values["edge_threshold"]),
        "pool_kernel_um": float(values["pool_kernel_um"]),
        "use_ilp": bool(values.get("use_ilp", False)),
        "inference_batch_size": int(values.get("inference_batch_size", 1)),
    }


def _infer_one(
    *,
    modules: dict[str, Any],
    baseline: Path,
    train_dir: Path,
    splits_path: Path,
    output_dir: Path,
    training_record: dict[str, Any],
    values: dict[str, Any],
    split_signature: str,
    reuse_completed: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    params = _inference_parameters(values)
    payload = {
        "checkpoint_sha256": training_record["checkpoint_sha256"],
        "split": split_signature,
        "parameters": params,
        "metric_contract_id": METRIC_CONTRACT_ID,
    }
    signature = _signature(payload)
    short_id = signature[:12]
    training_name = training_record["parameters"]["name"]
    trial_dir = output_dir / "inference" / training_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    record_path = trial_dir / f"{short_id}.json"
    if reuse_completed:
        previous = _read_matching_record(record_path, signature)
        if previous is not None:
            return previous

    method = f"local_hp_eval_{training_name}_{short_id}"
    started = time.time()
    try:
        cfg = modules["predict"].PredictConfig(
            det_threshold=params["detection_threshold"],
            pool_kernel_um=params["pool_kernel_um"],
            threshold=params["edge_threshold"],
            use_ilp=params["use_ilp"],
            ilp_edge_weight=-1.0,
            ilp_appearance_weight=0.1,
            ilp_disappearance_weight=0.1,
            ilp_division_weight=1.0,
        )
        modules["predict"].predict(
            data_dir=train_dir,
            fold=0,
            splits_file=splits_path,
            weights_path=Path(training_record["checkpoint"]),
            cfg=cfg,
            method=method,
            unet_batch_size=params["inference_batch_size"],
            evaluate=False,
        )
        prediction_dir = baseline / "predictions" / modules["dataspec"].USERNAME / method / "split_0"
        split_payload = json.loads(splits_path.read_text(encoding="utf-8-sig"))
        expected_datasets = list(split_payload[0]["test"])
        metrics = evaluate_prediction_dir(
            modules,
            prediction_dir,
            train_dir,
            expected_datasets=expected_datasets,
        )
        record = {
            "status": "complete",
            "signature": signature,
            "training_trial": training_name,
            "training_parameters": training_record["parameters"],
            "checkpoint": training_record["checkpoint"],
            "checkpoint_sha256": training_record["checkpoint_sha256"],
            "inference_parameters": params,
            "metrics": metrics,
            "prediction_dir": str(prediction_dir),
            "elapsed_seconds": time.time() - started,
        }
        write_json(record, record_path)
        return record
    except Exception as exc:
        record = {
            "status": "failed",
            "signature": signature,
            "training_trial": training_name,
            "inference_parameters": params,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.time() - started,
        }
        write_json(record, record_path)
        if not continue_on_error:
            raise
        return record


def execute_search(config_path: Path) -> dict[str, Any]:
    config = load_search_config(config_path)
    config.setdefault("seed", 20260809)
    layout = discover_competition_root(config.get("data_root"))
    baseline = find_baseline_root(None)
    modules = _load_baseline(baseline)
    configured_output = Path(config.get("output_dir", "outputs/hyperparameter_search"))
    output_dir = (
        configured_output if configured_output.is_absolute() else WORKSPACE / configured_output
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_config = output_dir / "search_config.yaml"
    if config_path.resolve() != copied_config.resolve():
        shutil.copy2(config_path, copied_config)

    split = build_split(
        layout.train,
        seed=int(config["seed"]),
        train_count=int(config["dataset"]["train_count"]),
        val_count=int(config["dataset"]["validation_count"]),
    )
    splits_path = output_dir / "dataset_splits.json"
    splits_path.write_text(json.dumps([split], indent=2) + "\n", encoding="utf-8")
    split_signature = _signature({"split": split, "max_frames": config["dataset"]["max_frames"]})
    plan = build_search_plan(config)
    write_json(plan, output_dir / "search_plan.json")

    training_records: list[dict[str, Any]] = []
    base_records: list[dict[str, Any]] = []
    for index, trial in enumerate(config["training_trials"], 1):
        print(f"=== TRAINING {index}/{len(config['training_trials'])}: {trial['name']} ===", flush=True)
        training_record = _train_one(
            modules=modules, baseline=baseline, train_dir=layout.train,
            splits_path=splits_path, output_dir=output_dir, config=config,
            trial=trial, split_signature=split_signature,
        )
        training_records.append(training_record)
        if training_record["status"] != "complete":
            continue
        base_values = {**config["base_inference"], "use_ilp": False}
        base_records.append(_infer_one(
            modules=modules, baseline=baseline, train_dir=layout.train,
            splits_path=splits_path, output_dir=output_dir,
            training_record=training_record, values=base_values,
            split_signature=split_signature,
            reuse_completed=bool(config.get("reuse_completed", True)),
            continue_on_error=bool(config.get("continue_on_error", True)),
        ))

    top_k = int(config["training"]["top_k_checkpoints"])
    ranked_base = sorted(base_records, key=_score, reverse=True)
    eligible_base = [record for record in ranked_base if _score(record) != float("-inf")]
    selected_names = {record["training_trial"] for record in eligible_base[:top_k]}
    selected = [
        record for record in training_records
        if record.get("status") == "complete" and record["parameters"]["name"] in selected_names
    ]

    grid = config["inference_grid"]
    grid_records: list[dict[str, Any]] = []
    for training_record in selected:
        for detection, edge, pool in itertools.product(
            grid["detection_thresholds"], grid["edge_thresholds"], grid["pool_kernel_um"]
        ):
            values = {
                "detection_threshold": detection,
                "edge_threshold": edge,
                "pool_kernel_um": pool,
                "use_ilp": False,
                "inference_batch_size": grid.get("inference_batch_size", 1),
            }
            grid_records.append(_infer_one(
                modules=modules, baseline=baseline, train_dir=layout.train,
                splits_path=splits_path, output_dir=output_dir,
                training_record=training_record, values=values,
                split_signature=split_signature,
                reuse_completed=bool(config.get("reuse_completed", True)),
                continue_on_error=bool(config.get("continue_on_error", True)),
            ))

    ranked_grid = sorted(grid_records, key=_score, reverse=True)
    ilp_records: list[dict[str, Any]] = []
    for source in ranked_grid[: int(grid.get("compare_ilp_for_top_k", 0))]:
        if source.get("status") != "complete":
            continue
        training_record = next(
            record for record in selected
            if record["parameters"]["name"] == source["training_trial"]
        )
        values = {**source["inference_parameters"], "use_ilp": True}
        ilp_records.append(_infer_one(
            modules=modules, baseline=baseline, train_dir=layout.train,
            splits_path=splits_path, output_dir=output_dir,
            training_record=training_record, values=values,
            split_signature=split_signature,
            reuse_completed=bool(config.get("reuse_completed", True)),
            continue_on_error=bool(config.get("continue_on_error", True)),
        ))

    all_inference = base_records + grid_records + ilp_records
    ranked = sorted(all_inference, key=_score, reverse=True)
    best = ranked[0] if ranked and _score(ranked[0]) != float("-inf") else None
    result = {
        "kind": "kaggle-official-metric-held-out-hyperparameter-search",
        "is_kaggle_leaderboard_score": False,
        "metric_contract": official_metric_contract(),
        "config": str(config_path.resolve()),
        "plan": plan,
        "split_file": str(splits_path),
        "training_records": training_records,
        "base_ranked": ranked_base,
        "selected_training_trials": sorted(selected_names),
        "all_ranked": ranked,
        "best": best,
    }
    write_json(_json_ready(result), output_dir / "search_results.json")
    best_payload = None if best is None else {
        "training_trial": best["training_trial"],
        "training_parameters": best["training_parameters"],
        "inference_parameters": best["inference_parameters"],
        "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "held_out_metrics": best["metrics"]["summary"],
        "metric_contract": official_metric_contract(),
        "is_kaggle_leaderboard_score": False,
    }
    write_json(_json_ready(best_payload), output_dir / "best_hyperparameters.json")
    print(json.dumps(_json_ready(best_payload), indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Local two-stage Biohub hyperparameter search")
    parser.add_argument("--config", type=Path, default=Path("configs/local_hyperparameter_search.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan without training")
    args = parser.parse_args()
    config = load_search_config(args.config)
    plan = build_search_plan(config)
    if args.dry_run:
        print(json.dumps({"config": str(args.config.resolve()), "plan": plan}, indent=2))
        return
    execute_search(args.config)


if __name__ == "__main__":
    main()
