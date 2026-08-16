"""Offline Kaggle inference runner using trained detector/transformer checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr

from .candidates import Candidate, sparse_candidates
from .config import load_config, write_resolved_config
from .data import EXPECTED_TEST_DATASETS, inspect_split, resolve_split_dir
from .detect import decode_probability
from .graph import Node, TrackGraph
from .io import (
    environment_manifest, parse_zarr_scale, read_graph_json, sha256_file,
    write_geff, write_graph_json, write_json, write_submission,
)
from .models import BiohubModel, position_embedding, select_features
from .optimizer import solve_forest
from .postprocess import add_safe_divisions, close_one_frame_gaps, smooth_non_branching
from .validate import validate_graph, validate_submission


def _required(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if key not in value:
            raise ValueError(f"missing config key: {'.'.join(keys)}")
        value = value[key]
    return value


def check_config(config: dict[str, Any]) -> list[str]:
    for keys in (
        ("input_dir",), ("output_dir",), ("weights",), ("subsample",), ("point_threshold",),
        ("edges", "top_k_parents"), ("ilp", "disappearance_weight"), ("runtime", "stop_hours"),
    ):
        _required(config, *keys)
    if len(config["subsample"]) != 3 or any(int(v) < 1 for v in config["subsample"]):
        raise ValueError("subsample must contain three positive integers")
    if int(config.get("tta", 4)) not in (1, 4):
        raise ValueError("this promoted runner supports tta: 1 or 4")
    warnings: list[str] = []
    try:
        input_dir = resolve_split_dir(config["input_dir"], "test")
        dataset_count = len(list(input_dir.glob("*.zarr")))
        if dataset_count != EXPECTED_TEST_DATASETS:
            warnings.append(
                f"expected {EXPECTED_TEST_DATASETS} test datasets, found {dataset_count}: {input_dir}"
            )
    except (FileNotFoundError, ValueError) as exc:
        warnings.append(str(exc))
    for weight in config["weights"]:
        if not Path(weight).is_file():
            warnings.append(f"weight does not exist in this environment: {weight}")
    return warnings


def load_models(config: dict[str, Any], device: torch.device) -> list[BiohubModel]:
    models: list[BiohubModel] = []
    for weight_value in config["weights"]:
        weight_path = Path(weight_value)
        if not weight_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {weight_path}")
        model_config_path = Path(config.get("model_config") or weight_path.with_name("config.json"))
        model_config = json.loads(model_config_path.read_text(encoding="utf-8")) if model_config_path.is_file() else {}
        model = BiohubModel(model_config)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("model."): value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        critical_missing = [key for key in missing if key.startswith(("unet.", "detect_head.", "transformer."))]
        if critical_missing or unexpected:
            raise RuntimeError(
                f"incompatible checkpoint {weight_path}; missing={critical_missing[:8]}, unexpected={unexpected[:8]}"
            )
        model.to(device).eval()
        models.append(model)
    if not models:
        raise ValueError("at least one checkpoint is required")
    return models


def _flips(tta: int) -> list[tuple[int, ...]]:
    return [(), (-1,), (-2,), (-2, -1)] if tta == 4 else [()]


@torch.inference_mode()
def infer_pair(
    models: list[BiohubModel], image: torch.Tensor, precision: str, tta: int
) -> tuple[np.ndarray, list[tuple[torch.Tensor, torch.Tensor]]]:
    logits_per_model: list[torch.Tensor] = []
    features_per_model: list[tuple[torch.Tensor, torch.Tensor]] = []
    autocast_enabled = image.device.type == "cuda" and precision == "fp16"
    for model in models:
        restored_logits: list[torch.Tensor] = []
        restored_features: list[tuple[torch.Tensor, torch.Tensor]] = []
        for dims in _flips(tta):
            augmented = torch.flip(image, dims) if dims else image
            with torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=autocast_enabled):
                features, logits = model.forward_unet(augmented)
            if dims:
                features = [torch.flip(value, dims) for value in features]
                logits = [torch.flip(value, dims) for value in logits]
            restored_logits.append(torch.stack([value[0, 0].float() for value in logits]))
            restored_features.append((features[0][0].float(), features[1][0].float()))
        logits_per_model.append(torch.stack(restored_logits).mean(0))
        features_per_model.append((
            torch.stack([pair[0] for pair in restored_features]).mean(0),
            torch.stack([pair[1] for pair in restored_features]).mean(0),
        ))
    probability = torch.sigmoid(torch.stack(logits_per_model).mean(0)).cpu().numpy()
    return probability, features_per_model


@torch.inference_mode()
def transformer_probabilities(
    models: list[BiohubModel], features: list[tuple[torch.Tensor, torch.Tensor]],
    source_coords: np.ndarray, target_coords: np.ndarray, t: int, time_length: int,
) -> np.ndarray:
    if not len(source_coords) or not len(target_coords):
        return np.zeros((len(source_coords), len(target_coords)), dtype=np.float32)
    device = next(models[0].parameters()).device
    source = torch.as_tensor(source_coords, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_coords, dtype=torch.float32, device=device)
    image_shape = features[0][0].shape[-3:]
    source_pos = position_embedding(source, t, image_shape, time_length)
    target_pos = position_embedding(target, t + 1, image_shape, time_length)
    logits: list[torch.Tensor] = []
    for model, (feature0, feature1) in zip(models, features):
        selected0, selected1 = select_features(feature0, source), select_features(feature1, target)
        logits.append(model.forward_transformer(selected0, selected1, source, target, source_pos, target_pos).float())
    return torch.softmax(torch.stack(logits).mean(0), dim=0).cpu().numpy()


def infer_dataset(
    dataset_path: Path, models: list[BiohubModel], device: torch.device, config: dict[str, Any]
) -> tuple[TrackGraph, dict[str, Any], Any]:
    root = zarr.open_group(str(dataset_path), mode="r")
    array = root["0"]
    if len(array.shape) != 4:
        raise ValueError(f"expected T,Z,Y,X at {dataset_path}, got {array.shape}")
    scale = parse_zarr_scale(dict(root.attrs))
    subsample = tuple(int(value) for value in config["subsample"])
    model_scale = tuple(s * d for s, d in zip(scale, subsample))
    quantiles = dict(root.attrs).get("image_statistics", {}).get("quantiles", {})
    q_low = _quantile_value(quantiles, 0.001)
    q_high = _quantile_value(quantiles, 0.999)
    if q_low is None or q_high is None or q_high <= q_low:
        sample = np.asarray(array[:: max(1, array.shape[0] // 4), ::subsample[0], ::subsample[1], ::subsample[2]])
        q_low, q_high = (float(value) for value in np.quantile(sample.ravel()[::50], (0.001, 0.999)))
    if not np.isfinite((q_low, q_high)).all() or q_high <= q_low:
        raise ValueError(f"invalid quantiles for {dataset_path.stem}: {q_low}, {q_high}")

    graph = TrackGraph(dataset_path.stem)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    candidates: list[Candidate] = []
    next_node_id = 0
    for t in range(array.shape[0] - 1):
        raw = np.asarray(array[t:t + 2, ::subsample[0], ::subsample[1], ::subsample[2]], dtype=np.float32)
        normalized = np.clip((raw - q_low) / (q_high - q_low), 0.0, 4.0)
        image = torch.from_numpy(normalized[None]).to(device, non_blocking=True)
        probability, features = infer_pair(models, image, str(config["precision"]), int(config["tta"]))
        if t == 0:
            coord0, score0 = decode_probability(
                probability[0], float(config["point_threshold"]), model_scale, float(config["nms_um"])
            )
            for coord, score in zip(coord0 * np.asarray(subsample), score0):
                graph.add_node(Node(next_node_id, 0, *coord, confidence=float(score)))
                next_node_id += 1
        sources = graph.nodes_at(t)
        source_coords = np.stack([node.zyx / np.asarray(subsample) for node in sources]) if sources else np.empty((0, 3))
        coord1, score1 = decode_probability(
            probability[1], float(config["point_threshold"]), model_scale, float(config["nms_um"])
        )
        targets: list[Node] = []
        for coord, score in zip(coord1 * np.asarray(subsample), score1):
            node = Node(next_node_id, t + 1, *coord, confidence=float(score))
            graph.add_node(node)
            targets.append(node)
            next_node_id += 1
        edge_probability = transformer_probabilities(models, features, source_coords, coord1, t, array.shape[0])
        edge_cfg = config["edges"]
        candidates.extend(sparse_candidates(
            sources, targets, edge_probability, scale,
            strong_threshold=float(edge_cfg["strong_threshold"]),
            min_probability=float(edge_cfg["min_probability"]),
            top_k_parents=int(edge_cfg["top_k_parents"]),
            max_distance_um=float(edge_cfg["max_distance_um"]),
        ))
        del image, probability, features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ilp = config["ilp"]
    graph, solver = solve_forest(
        graph, candidates, edge_weight=float(ilp["edge_weight"]),
        appearance_weight=float(ilp["appearance_weight"]),
        disappearance_weight=float(ilp["disappearance_weight"]), division_weight=float(ilp["division_weight"]),
    )
    graph, division_added = add_safe_divisions(
        graph, candidates, scale, parent_child_um=float(config["division"]["parent_child_um"]),
        child_child_um=float(config["division"]["child_child_um"]),
        min_probability=float(config["division"]["min_probability"]),
    )
    graph, gap_added = close_one_frame_gaps(
        graph, array, scale, max_added_fraction=float(config["gap"]["max_added_fraction"]),
        max_added_absolute=int(config["gap"]["max_added_absolute"]), max_link_um=float(config["gap"]["max_link_um"]),
    )
    graph, smoothed = smooth_non_branching(
        graph, scale, weight=float(config["smooth"]["weight"]), max_shift_um=float(config["smooth"]["max_shift_um"]),
    )
    stats = {
        "dataset": dataset_path.stem, "shape": list(array.shape), "scale_um": list(scale),
        "nodes": len(graph.nodes), "edges": len(graph.edges), "candidates": len(candidates),
        "solver": solver, "gap_nodes_added": gap_added, "division_edges_added": division_added,
        "smoothed_nodes": smoothed,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
    }
    return graph, stats, array


def _quantile_value(values: dict[str, Any], requested: float) -> float | None:
    for key, value in values.items():
        try:
            if abs(float(key) - requested) <= 1e-7:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def run_kaggle(
    config_path: Path, shard_index: int = 0, shard_count: int = 1,
    output_override: Path | None = None, input_override: Path | None = None,
    weight_overrides: list[Path] | None = None,
    model_config_override: Path | None = None,
    max_datasets: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if input_override is not None:
        config["input_dir"] = str(input_override)
    if weight_overrides is not None:
        config["weights"] = [str(path) for path in weight_overrides]
    if model_config_override is not None:
        config["model_config"] = str(model_config_override)
    warnings = check_config(config)
    input_dir = resolve_split_dir(config["input_dir"], "test")
    config["input_dir"] = str(input_dir)
    output_dir = Path(output_override or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir, geff_dir = output_dir / "predictions", output_dir / "geff"
    prediction_dir.mkdir(exist_ok=True)
    geff_dir.mkdir(exist_ok=True)
    datasets = sorted(input_dir.glob("*.zarr"))[shard_index::shard_count]
    if max_datasets is not None:
        if max_datasets < 1:
            raise ValueError("max_datasets must be positive")
        datasets = datasets[:max_datasets]
    if not datasets:
        raise FileNotFoundError(f"no .zarr datasets found for shard {shard_index}/{shard_count}")
    if not torch.cuda.is_available() and config.get("device", "cuda") == "cuda":
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(config.get("device", "cuda"))
    np.random.seed(int(config.get("seed", 20260809)))
    torch.manual_seed(int(config.get("seed", 20260809)))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.get("seed", 20260809)))
    models = load_models(config, device)
    start = time.perf_counter()
    timings: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    graphs: list[TrackGraph] = []
    stop_seconds = float(config["runtime"]["stop_hours"]) * 3600
    warning_seconds = float(config["runtime"]["warning_hours"]) * 3600

    for index, dataset_path in enumerate(datasets):
        marker = output_dir / f"{dataset_path.stem}.complete"
        graph_json = prediction_dir / f"{dataset_path.stem}.geff.json"
        if marker.is_file() and graph_json.is_file():
            graphs.append(read_graph_json(graph_json))
            continue
        elapsed = time.perf_counter() - start
        if elapsed >= stop_seconds:
            raise TimeoutError("11-hour safety limit reached before starting a new dataset")
        item_start = time.perf_counter()
        graph, dataset_stats, _ = infer_dataset(dataset_path, models, device, config)
        validation = validate_graph(graph, tuple(dataset_stats["shape"]))
        if not validation["valid"]:
            raise RuntimeError(f"{dataset_path.stem}: validation failed: {validation['errors'][:5]}")
        write_graph_json(graph, graph_json)
        write_geff(graph, geff_dir / f"{dataset_path.stem}.geff")
        marker.write_text("complete\n", encoding="ascii")
        graphs.append(graph)
        seconds = time.perf_counter() - item_start
        timings.append({"dataset": dataset_path.stem, "seconds": seconds})
        stats.append({**dataset_stats, "seconds": seconds, "validation": validation})
        max_vram = float(config["runtime"].get("max_vram_gb", 14.5)) * 1024**3
        if dataset_stats["peak_vram_reserved_bytes"] > max_vram:
            raise MemoryError(
                f"{dataset_path.stem}: peak reserved VRAM exceeded {max_vram / 1024**3:.1f} GB"
            )
        elapsed = time.perf_counter() - start
        estimate = elapsed / (index + 1) * len(datasets)
        if estimate > warning_seconds:
            print(f"WARNING: projected runtime is {estimate / 3600:.2f} h", flush=True)

    submission = write_submission(graphs, output_dir / "submission.csv")
    validation = validate_submission(pd.read_csv(submission))
    if not validation["valid"]:
        raise RuntimeError(f"submission validation failed: {validation['errors']}")
    with (output_dir / "timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "seconds"))
        writer.writeheader(); writer.writerows(timings)
    write_json({"datasets": stats}, output_dir / "metrics.json")
    write_json(validation, output_dir / "validation.json")
    write_resolved_config(config, output_dir / "config.resolved.yaml")
    weight_hashes = {str(path): sha256_file(path) for path in map(Path, config["weights"])}
    write_json(weight_hashes, output_dir / "weights.sha256")
    manifest = {
        **environment_manifest(), "config": str(config_path.resolve()), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "shard_index": shard_index, "shard_count": shard_count, "datasets": len(graphs),
        "input_dir": str(input_dir), "input_validation": inspect_split(input_dir),
        "weights": weight_hashes, "warnings": warnings, "submission_sha256": sha256_file(submission),
        "validation": validation, "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(manifest, output_dir / "run_manifest.json")
    (output_dir / "submission.sha256").write_text(manifest["submission_sha256"] + "\n", encoding="ascii")
    (output_dir / "environment.txt").write_text(json.dumps(environment_manifest(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "source_commit.txt").write_text(str(manifest["source_commit"]) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Biohub offline Kaggle inference")
    parser.add_argument("--config", type=Path, default=Path("configs/kaggle_inference.yaml"))
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-dir", type=Path, help="test directory or extracted competition root")
    parser.add_argument("--weight", type=Path, action="append", dest="weights")
    parser.add_argument("--model-config", type=Path, help="config.json saved beside the trained checkpoint")
    parser.add_argument("--max-datasets", type=int, help="bounded local smoke run; omit for Kaggle submission")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.input_dir is not None:
        config["input_dir"] = str(args.input_dir)
    if args.weights is not None:
        config["weights"] = [str(path) for path in args.weights]
    warnings = check_config(config)
    if args.check_config:
        payload: dict[str, Any] = {"valid": True, "warnings": warnings}
        try:
            input_dir = resolve_split_dir(config["input_dir"], "test")
            payload["input"] = inspect_split(input_dir)
        except (FileNotFoundError, ValueError) as exc:
            payload["valid"] = False
            payload["input_error"] = str(exc)
        print(json.dumps(payload, indent=2))
        return
    print(json.dumps(run_kaggle(
        args.config, args.shard_index, args.shard_count, args.output_dir,
        args.input_dir, args.weights, args.model_config, args.max_datasets,
    ), indent=2))


if __name__ == "__main__":
    main()
