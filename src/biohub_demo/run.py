"""Local Biohub demonstration on official data, with a synthetic fallback."""

from __future__ import annotations

import argparse
import csv
import json
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from .candidates import build_distance_candidates
from .config import load_config, write_resolved_config
from .data import discover_competition_root, inspect_split
from .detect import detect_classical
from .graph import Edge, Node, TrackGraph
from .io import (
    environment_manifest, parse_zarr_scale, peak_rss_bytes, sha256_file,
    write_graph_json, write_json, write_submission,
)
from .metrics import evaluate_graph
from .optimizer import solve_forest
from .postprocess import add_safe_divisions, close_one_frame_gaps, smooth_non_branching
from .synthetic import generate_synthetic
from .validate import validate_graph, validate_submission


def run_local_demo(config_path: str | Path, output_override: str | Path | None = None) -> dict:
    config = load_config(config_path)
    mode = str(config.get("mode", "real-mini"))
    if mode not in {"real-mini", "synthetic"}:
        raise ValueError("mode must be real-mini or synthetic")
    output_dir = Path(output_override or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    timings: list[dict[str, float | str]] = []
    started = time.perf_counter()
    tracemalloc.start()

    def timed(name: str, operation):
        begin = time.perf_counter()
        result = operation()
        timings.append({"stage": name, "seconds": time.perf_counter() - begin})
        return result

    data_context: dict[str, object]
    if mode == "synthetic":
        synthetic = timed("generate_synthetic", lambda: generate_synthetic(
            seed=int(config["seed"]), shape=tuple(config["shape"]),
            cell_count=int(config.get("cell_count", 24)), scale=tuple(config["voxel_size_um"]),
        ))
        image, truth, scale = synthetic.image, synthetic.truth, synthetic.scale
        missing = truth.nodes[synthetic.missing_truth_node]
        inject_missing = (missing.t, missing.zyx)
        output_offset = (0, 0, 0, 0)
        full_shape = tuple(image.shape)
        data_context = {"source": "deterministic synthetic fixture"}
    else:
        real = timed("load_real_mini", lambda: _load_real_mini(config))
        image = real["image"]
        truth = real["truth"]
        scale = real["scale"]
        inject_missing = None
        output_offset = real["offset"]
        full_shape = real["full_shape"]
        data_context = real["context"]

    detected = timed("detect", lambda: detect_classical(
        image, truth.dataset, scale,
        threshold=float(config["point_threshold"]), nms_um=float(config["nms_um"]),
        inject_missing_zyxt=inject_missing,
    ))
    edge_cfg = config["edges"]
    candidates = timed("candidates", lambda: build_distance_candidates(
        detected, scale, strong_threshold=float(edge_cfg["strong_threshold"]),
        min_probability=float(edge_cfg["min_probability"]), top_k_parents=int(edge_cfg["top_k_parents"]),
        max_distance_um=float(edge_cfg["max_distance_um"]),
    ))
    ilp_cfg = config["ilp"]
    optimized, solver = timed("ilp", lambda: solve_forest(
        detected, candidates, edge_weight=float(ilp_cfg["edge_weight"]),
        appearance_weight=float(ilp_cfg["appearance_weight"]),
        disappearance_weight=float(ilp_cfg["disappearance_weight"]),
        division_weight=float(ilp_cfg["division_weight"]),
    ))
    repaired, division_added = timed("safe_division", lambda: add_safe_divisions(
        optimized, candidates, scale,
        parent_child_um=float(config["division"]["parent_child_um"]),
        child_child_um=float(config["division"]["child_child_um"]),
        min_probability=float(config["division"]["min_probability"]),
    ))
    repaired, gap_added = timed("gap_close", lambda: close_one_frame_gaps(
        repaired, image, scale,
        max_added_fraction=float(config["gap"]["max_added_fraction"]),
        max_added_absolute=int(config["gap"]["max_added_absolute"]),
        max_link_um=float(config["gap"]["max_link_um"]),
    ))
    final_graph, smoothed = timed("smooth", lambda: smooth_non_branching(
        repaired, scale, weight=float(config["smooth"]["weight"]),
        max_shift_um=float(config["smooth"]["max_shift_um"]),
    ))

    graph_validation = validate_graph(final_graph, image.shape)
    if not graph_validation["valid"]:
        raise RuntimeError(f"graph validation failed: {graph_validation['errors'][:5]}")
    metrics = evaluate_graph(final_graph, truth, scale)
    output_graph = _offset_graph(final_graph, output_offset)
    output_validation = validate_graph(output_graph, full_shape)
    if not output_validation["valid"]:
        raise RuntimeError(f"global graph validation failed: {output_validation['errors'][:5]}")
    submission_path = write_submission([output_graph], output_dir / "submission.csv")
    submission_validation = validate_submission(pd.read_csv(submission_path))
    if not submission_validation["valid"]:
        raise RuntimeError(f"submission validation failed: {submission_validation['errors']}")

    write_graph_json(output_graph, output_dir / "predictions" / f"{output_graph.dataset}.geff.json")
    write_json(metrics, output_dir / "metrics.json")
    _write_preview(image, truth, final_graph, output_dir / "preview.png", mode)
    peak_python_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    timings.append({"stage": "total", "seconds": time.perf_counter() - started})
    with (output_dir / "timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stage", "seconds"))
        writer.writeheader()
        writer.writerows(timings)
    write_resolved_config(config, output_dir / "config.resolved.yaml")
    manifest = {
        **environment_manifest(), "config": str(Path(config_path).resolve()), "seed": config["seed"],
        "mode": mode, "solver": solver, "candidate_count": len(candidates),
        "node_count": len(final_graph.nodes), "edge_count": len(final_graph.edges),
        "gap_nodes_added": gap_added, "division_edges_added": division_added,
        "smoothed_nodes": smoothed, "peak_python_heap_bytes": peak_python_bytes,
        "peak_ram_bytes": peak_rss_bytes(), "peak_vram_bytes": 0, "device": "cpu",
        "validation": graph_validation, "global_validation": output_validation,
        "submission_validation": submission_validation, "data": data_context,
        "submission_sha256": sha256_file(submission_path),
    }
    if mode == "real-mini":
        split_validation = inspect_split(Path(str(data_context["split_path"])))
        manifest["data_validation"] = {
            key: value for key, value in split_validation.items() if key != "datasets"
        }
        write_json(split_validation, output_dir / "data_validation.json")
    write_json(manifest, output_dir / "run_manifest.json")
    (output_dir / "submission.sha256").write_text(manifest["submission_sha256"] + "\n", encoding="ascii")
    return {"output_dir": str(output_dir), "metrics": metrics, "manifest": manifest}


def _load_real_mini(config: dict) -> dict[str, object]:
    import geff

    layout = discover_competition_root(config.get("data_root"))
    split_name = str(config.get("split", "train"))
    if split_name != "train":
        raise ValueError("real-mini evaluation requires split: train because test has no ground truth")
    split_path = layout.train
    requested = config.get("dataset")
    dataset_path = split_path / f"{requested}.zarr" if requested else next(iter(sorted(split_path.glob("*.zarr"))), None)
    if dataset_path is None or not dataset_path.is_dir():
        raise FileNotFoundError(f"real dataset not found: {dataset_path}")
    truth_path = dataset_path.with_suffix(".geff")
    if not truth_path.is_dir():
        raise FileNotFoundError(f"ground-truth GEFF not found: {truth_path}")

    root = zarr.open_group(str(dataset_path), mode="r")
    array = root["0"]
    full_shape = tuple(int(value) for value in array.shape)
    scale = parse_zarr_scale(dict(root.attrs))
    network, _ = geff.read(truth_path, node_props=["t", "z", "y", "x"], edge_props=[])
    time_start = max(0, int(config.get("time_start", 0)))
    time_stop = min(full_shape[0], time_start + int(config.get("time_frames", 4)))
    if time_stop <= time_start:
        raise ValueError("the real-mini time window is empty")
    selected_nodes = [
        (int(node_id), values)
        for node_id, values in network.nodes(data=True)
        if time_start <= int(values["t"]) < time_stop
    ]
    if not selected_nodes:
        raise ValueError(f"no ground-truth nodes in time window {time_start}:{time_stop}")

    requested_crop = tuple(int(value) for value in config.get("crop_shape", (16, 128, 128)))
    crop_shape = tuple(min(size, full) for size, full in zip(requested_crop, full_shape[1:]))
    centers = np.asarray([[values[axis] for axis in ("z", "y", "x")] for _, values in selected_nodes])
    center = np.median(centers, axis=0)
    spatial_start = tuple(
        max(0, min(full - size, int(round(value - size / 2))))
        for value, size, full in zip(center, crop_shape, full_shape[1:])
    )
    z0, y0, x0 = spatial_start
    dz, dy, dx = crop_shape
    image = np.asarray(array[time_start:time_stop, z0:z0 + dz, y0:y0 + dy, x0:x0 + dx])

    truth = TrackGraph(dataset_path.stem)
    for node_id, values in selected_nodes:
        local = (
            int(values["t"]) - time_start,
            float(values["z"]) - z0,
            float(values["y"]) - y0,
            float(values["x"]) - x0,
        )
        if 0 <= local[1] < dz and 0 <= local[2] < dy and 0 <= local[3] < dx:
            truth.add_node(Node(node_id, *local))
    for source, target in network.edges:
        if int(source) in truth.nodes and int(target) in truth.nodes:
            truth.add_edge(Edge(int(source), int(target)))
    if not truth.nodes:
        raise ValueError("the selected real-mini crop contains no ground-truth nodes")

    return {
        "image": image,
        "truth": truth,
        "scale": scale,
        "offset": (time_start, z0, y0, x0),
        "full_shape": full_shape,
        "context": {
            "competition_root": str(layout.root),
            "split_path": str(split_path),
            "dataset_path": str(dataset_path),
            "truth_path": str(truth_path),
            "full_shape": list(full_shape),
            "crop_shape": list(image.shape),
            "crop_offset_tzyx": [time_start, z0, y0, x0],
            "scale_um": list(scale),
            "truth_nodes_in_crop": len(truth.nodes),
            "truth_edges_in_crop": len(truth.edges),
        },
    }


def _offset_graph(graph: TrackGraph, offset: tuple[int, int, int, int]) -> TrackGraph:
    dt, dz, dy, dx = offset
    shifted = TrackGraph(graph.dataset)
    for node in graph.nodes.values():
        shifted.add_node(Node(
            node.node_id, node.t + dt, node.z + dz, node.y + dy, node.x + dx, node.confidence
        ))
    for edge in graph.edges.values():
        shifted.add_edge(edge)
    return shifted


def _write_preview(image, truth, prediction, path: Path, mode: str) -> None:
    t = image.shape[0] // 2
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(image[t].max(axis=0), cmap="gray")
    truth_nodes, pred_nodes = truth.nodes_at(t), prediction.nodes_at(t)
    ax.scatter([n.x for n in truth_nodes], [n.y for n in truth_nodes], s=45, facecolors="none", edgecolors="lime", label="truth")
    ax.scatter([n.x for n in pred_nodes], [n.y for n in pred_nodes], s=12, c="magenta", label="prediction")
    ax.set(title=f"{mode} frame t={t}: XY maximum projection", xlabel="x", ylabel="y")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Biohub local demo on official or synthetic data")
    parser.add_argument("--config", type=Path, default=Path("configs/local_demo.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_local_demo(args.config, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
