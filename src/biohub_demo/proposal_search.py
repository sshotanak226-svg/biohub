"""Evaluate every unique method named in OPTIMAL_TRANSPORT_MATCHING_PROPOSAL.md."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .competition_metric import METRIC_CONTRACT_ID, official_metric_contract
from .data import discover_competition_root
from .hyper_search import _signature
from .io import is_complete_geff, sha256_file, write_geff, write_json
from .method_search import _load_or_predict_raw, load_method_search_config
from .proposal_postprocess import (
    cascade_graphs,
    division_specialist,
    division_value_gate,
    ensemble_graphs,
    expected_jaccard_prune,
    future_supported_divisions,
    metric_aware_track_prune,
    motion_refine,
    node_count_calibrate,
)
from .proposal_registry import PROPOSAL_METHODS, ProposalMethod
from .tracking_variants import build_variant, graph_from_raw_prediction
from .train_eval import (
    WORKSPACE,
    _json_ready,
    _load_baseline,
    build_split,
    evaluate_prediction_dir,
    find_baseline_root,
)


def _score(record: dict[str, Any]) -> float:
    if record.get("status") != "complete":
        return float("-inf")
    value = record.get("metrics", {}).get("summary", {}).get("score")
    return float(value) if value is not None else float("-inf")


def _method_signature_payload(method: ProposalMethod) -> dict[str, Any]:
    """Centralize the method payload used by the current prediction cache."""
    return asdict(method)


def _write_scoreboard(records: list[dict[str, Any]], path: Path) -> None:
    ranked = {
        record["name"]: rank
        for rank, record in enumerate(
            sorted((item for item in records if item.get("status") == "complete"), key=_score, reverse=True),
            1,
        )
    }
    columns = [
        "rank", "name", "group", "status", "fidelity", "score",
        "adj_edge_jaccard", "edge_jaccard", "division_jaccard",
        "elapsed_seconds", "checkpoint_reuse", "requirements", "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (ranked.get(item["name"], 10_000), item["name"])):
            summary = record.get("metrics", {}).get("summary", {})
            writer.writerow({
                "rank": ranked.get(record["name"], ""),
                "name": record["name"],
                "group": record.get("group", ""),
                "status": record.get("status", ""),
                "fidelity": record.get("fidelity", ""),
                "score": summary.get("score", ""),
                "adj_edge_jaccard": summary.get("adj_edge_jaccard", ""),
                "edge_jaccard": summary.get("edge_jaccard", ""),
                "division_jaccard": summary.get("division_jaccard", ""),
                "elapsed_seconds": record.get("elapsed_seconds", ""),
                "checkpoint_reuse": record.get("checkpoint_reuse", ""),
                "requirements": ";".join(record.get("requirements", [])),
                "note": record.get("note", ""),
            })


def _checkpoint_record(
    base_output: Path,
    checkpoint: Path | None,
    reuse_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reuse_config = reuse_config or {}
    marker_value = Path(reuse_config.get(
        "training_marker", base_output / "training" / "training.complete.json"
    ))
    marker = marker_value if marker_value.is_absolute() else WORKSPACE / marker_value
    previous: dict[str, Any] | None = None
    marker_checkpoint: Path | None = None
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8-sig"))
        marker_checkpoint = Path(previous["checkpoint"]).resolve()
    if checkpoint is None:
        if not marker.is_file():
            raise FileNotFoundError(
                f"training marker is missing: {marker}; run local method search first or pass --checkpoint"
            )
        checkpoint = marker_checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha256 = sha256_file(checkpoint)
    uses_local_marker = marker_checkpoint is not None and checkpoint == marker_checkpoint
    if (
        uses_local_marker
        and bool(reuse_config.get("verify_recorded_sha256", True))
        and previous
        and previous.get("checkpoint_sha256")
        and previous["checkpoint_sha256"] != actual_sha256
    ):
        raise RuntimeError(
            "local method-search checkpoint SHA-256 does not match its training marker: "
            f"{checkpoint}"
        )
    return {
        "status": "provided",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha256,
        "source": "local_method_search_marker" if uses_local_marker else "explicit_override",
        "training_marker": str(marker) if uses_local_marker else None,
        "training_signature": previous.get("signature") if uses_local_marker and previous else None,
        "raw_cache_policy": reuse_config.get("raw_cache", "reuse_or_fill_missing"),
    }


def _configured_settings(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): dict(item) for item in config["method_variants"]}


def _family_settings(settings: dict[str, Any], dataset: str) -> dict[str, Any]:
    values = dict(settings)
    if dataset.startswith("44b6"):
        values.update(max_distance_um=9.0, entropy_epsilon=0.04, source_mass_penalty=0.65, target_mass_penalty=0.65)
    elif dataset.startswith("6bba"):
        values.update(max_distance_um=11.0, entropy_epsilon=0.06, source_mass_penalty=0.45, target_mass_penalty=0.45)
    return values


def _phase_settings(settings: dict[str, Any], raw_graph) -> dict[str, Any]:
    values = dict(settings)
    counts = [len(raw_graph.nodes_at(t)) for t in range(max((n.t for n in raw_graph.nodes.values()), default=-1) + 1)]
    density = float(np.median(counts)) if counts else 0.0
    if density >= 500:
        values.update(max_distance_um=8.5, entropy_epsilon=0.035, structure_weight=0.30)
    elif density <= 150:
        values.update(max_distance_um=11.5, entropy_epsilon=0.07)
    return values


def _calibrated_settings(settings: dict[str, Any], raw_graph) -> dict[str, Any]:
    values = dict(settings)
    confidences = np.asarray([node.confidence for node in raw_graph.nodes.values()], dtype=np.float64)
    if confidences.size:
        values["detection_threshold"] = float(np.clip(np.quantile(confidences, 0.55), 0.92, 0.985))
    return values


def _apply_named_postprocess(name: str, graph, candidates):
    diagnostics: dict[str, Any] = {}
    if name in {"three_frame_mmot", "windowed_dynamic_ot", "structurally_constrained_dynamic_ot", "full_movie_transductive_ot"}:
        graph, changed = motion_refine(graph, candidates, residual_gate_um=7.0)
        diagnostics["motion_edge_delta"] = changed
    if name in {"counterfactual_division_ot", "reaction_division_ot"}:
        graph, removed = future_supported_divisions(graph)
        diagnostics["unsupported_divisions_removed"] = removed
    if name in {"metric_aware_ot", "expected_jaccard_edge_selection"}:
        graph, removed = expected_jaccard_prune(graph)
        diagnostics["expected_jaccard_edges_removed"] = removed
    if name == "division_value_gate":
        graph, removed = division_value_gate(graph)
        diagnostics["negative_value_divisions_removed"] = removed
    if name in {"family_time_node_count_prior", "test_distribution_calibrated_detection"}:
        graph, removed = node_count_calibrate(graph)
        diagnostics["node_count_calibration_removed"] = removed
    if name == "metric_aware_track_pruning":
        graph, removed = metric_aware_track_prune(graph)
        diagnostics["components_removed"] = removed
    return graph, diagnostics


def _build_one(
    method: ProposalMethod,
    raw_graph,
    raw_candidates,
    configured: dict[str, dict[str, Any]],
    dependencies: dict[str, Any],
    single_cache: dict[str, tuple[Any, dict[str, Any], list[Any]]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    if method.builder == "configured":
        settings = configured[method.name]
        return build_variant(raw_graph, raw_candidates, settings)
    if method.builder in {"single", "single_post", "family_single", "phase_single", "calibrated_single"}:
        settings = dict(method.settings)
        if method.builder == "family_single":
            settings = _family_settings(settings, raw_graph.dataset)
        elif method.builder == "phase_single":
            settings = _phase_settings(settings, raw_graph)
        elif method.builder == "calibrated_single":
            settings = _calibrated_settings(settings, raw_graph)
        cache_key = _signature({"settings": settings, "dataset": raw_graph.dataset})
        if single_cache is not None and cache_key in single_cache:
            cached_graph, cached_diagnostic, _ = single_cache[cache_key]
            graph = cached_graph.copy()
            diagnostic = {**cached_diagnostic, "reused_shared_postprocess": True}
        else:
            graph, diagnostic = build_variant(raw_graph, raw_candidates, settings)
            if single_cache is not None:
                single_cache[cache_key] = (graph.copy(), dict(diagnostic), raw_candidates)
        if method.builder == "single_post":
            graph, post = _apply_named_postprocess(method.name, graph, raw_candidates)
            diagnostic.update(post)
        return graph, diagnostic

    members = [dependencies[name] for name in method.dependencies]
    if method.builder == "ensemble":
        mode = "hard"
        weights = None
        if method.name == "current6_weighted_edge_vote":
            mode = "weighted"
            weights = [0.60, 0.63, 0.65, 0.69, 0.65, 0.66]
        elif method.name == "hungarian_uot_consensus":
            mode = "agreement"
        elif method.name == "uncertainty_barycenter_ensemble":
            mode = "weighted"
            weights = [0.45, 0.25, 0.30]
        elif method.name == "score_frontier_ensemble":
            mode = "union_ilp"
        return ensemble_graphs(raw_graph, members, mode=mode, weights=weights)
    if method.builder == "ensemble_ilp":
        return ensemble_graphs(raw_graph, members, mode="union_ilp")
    if method.builder == "abstention":
        return ensemble_graphs(raw_graph, members, mode="abstention")
    if method.builder == "cascade":
        return cascade_graphs(raw_graph, members)
    if method.builder == "division_ensemble":
        return division_specialist(raw_graph, members[0], members[1:])
    raise ValueError(f"unsupported proposal builder: {method.builder}")


def build_proposal_plan(
    config: dict[str, Any],
    paired_datasets: int,
    checkpoint_reuse_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    executable = [method for method in PROPOSAL_METHODS if method.fidelity != "requires_training"]
    training = [method for method in PROPOSAL_METHODS if method.fidelity == "requires_training"]
    return {
        "purpose": "compare-every-method-in-optimal-transport-proposal",
        "registered_methods": len(PROPOSAL_METHODS),
        "executable_now": len(executable),
        "requires_additional_training": len(training),
        "paired_datasets": paired_datasets,
        "validation_datasets": int(config["dataset"]["validation_count"]),
        "shared_raw_inference": True,
        "checkpoint_reuse": checkpoint_reuse_config or {
            "policy": "prefer_local_method_search",
            "training_marker": "outputs/method_search/training/training.complete.json",
            "raw_cache": "reuse_or_fill_missing",
        },
        "gpu_postprocessing": "auto",
        "methods": [
            {
                "name": method.name,
                "group": method.group,
                "fidelity": method.fidelity,
                "requirements": list(method.requirements),
                "checkpoint_reuse": method.checkpoint_reuse,
            }
            for method in PROPOSAL_METHODS
        ],
        "metric_contract": official_metric_contract(),
    }


def execute_proposal_search(
    config_path: Path,
    output_dir: Path,
    checkpoint: Path | None = None,
    external_prediction_dirs: dict[str, str | None] | None = None,
    strict_all_methods: bool = False,
    checkpoint_reuse_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_method_search_config(config_path)
    config.setdefault("seed", 20260809)
    external_prediction_dirs = external_prediction_dirs or {}
    if strict_all_methods:
        missing_external = [
            method.name for method in PROPOSAL_METHODS
            if method.fidelity == "requires_training"
            and not external_prediction_dirs.get(method.name)
        ]
        if missing_external:
            raise RuntimeError(
                "strict all-method comparison requires prediction directories for: "
                + ", ".join(missing_external)
            )
    layout = discover_competition_root(config.get("data_root"))
    baseline = find_baseline_root(None)
    modules = _load_baseline(baseline)
    base_output_value = Path(config.get("output_dir", "outputs/method_search"))
    base_output = base_output_value if base_output_value.is_absolute() else WORKSPACE / base_output_value
    output_dir = output_dir if output_dir.is_absolute() else WORKSPACE / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variants").mkdir(parents=True, exist_ok=True)
    checkpoint_record = _checkpoint_record(base_output, checkpoint, checkpoint_reuse_config)
    split = build_split(
        layout.train,
        seed=int(config["seed"]),
        train_count=None,
        val_count=int(config["dataset"]["validation_count"]),
    )
    raw = _load_or_predict_raw(
        modules=modules,
        checkpoint_record=checkpoint_record,
        train_dir=layout.train,
        names=list(split["test"]),
        output_dir=base_output,
        settings=config["raw_inference"],
        reuse_completed=True,
    )
    configured = _configured_settings(config)
    dependency_names = {
        dependency for method in PROPOSAL_METHODS for dependency in method.dependencies
    }
    executable = [method for method in PROPOSAL_METHODS if method.fidelity != "requires_training"]
    signatures = {
        method.name: _signature({
            "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
            "raw_inference": config["raw_inference"],
            "proposal_method": _method_signature_payload(method),
            "metric_contract_id": METRIC_CONTRACT_ID,
        })
        for method in executable
    }
    prediction_dirs = {
        method.name: output_dir / "predictions" / f"{method.name}_{signatures[method.name][:12]}"
        for method in executable
    }
    for path in prediction_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, dict[str, Any]] = {method.name: {} for method in executable}
    elapsed: dict[str, float] = {method.name: 0.0 for method in executable}

    for dataset_index, name in enumerate(split["test"], 1):
        print(f"PROPOSAL DATASET {dataset_index}/{len(split['test'])}: {name}", flush=True)
        coords, scores, edge_array = raw[name]
        raw_graph, raw_candidates = graph_from_raw_prediction(
            name, coords, scores, [tuple(row) for row in edge_array.tolist()],
            (1.625, 0.40625, 0.40625),
        )
        retained_graphs: dict[str, Any] = {}
        single_cache: dict[str, tuple[Any, dict[str, Any], list[Any]]] = {}
        for method_index, method in enumerate(executable, 1):
            geff_path = prediction_dirs[method.name] / f"{name}.geff"
            complete = is_complete_geff(geff_path)
            if complete and method.name not in dependency_names:
                diagnostics[method.name][name] = {"resumed_from_complete_geff": True}
                print(
                    f"  METHOD {method_index}/{len(executable)}: {method.name} [resume skip]",
                    flush=True,
                )
                continue
            print(
                f"  METHOD {method_index}/{len(executable)}: {method.name}"
                + (" [rebuild dependency]" if complete else ""),
                flush=True,
            )
            started = time.perf_counter()
            graph, diagnostic = _build_one(
                method, raw_graph, raw_candidates, configured, retained_graphs, single_cache
            )
            method_seconds = time.perf_counter() - started
            elapsed[method.name] += method_seconds
            if not complete:
                write_geff(graph, geff_path)
            diagnostics[method.name][name] = diagnostic
            if method.name in dependency_names:
                retained_graphs[method.name] = graph
            print(
                f"    completed in {method_seconds:.1f}s; "
                f"nodes={len(graph.nodes)} edges={len(graph.edges)}",
                flush=True,
            )

    records: list[dict[str, Any]] = []
    for method in PROPOSAL_METHODS:
        if method.fidelity == "requires_training":
            external_value = external_prediction_dirs.get(method.name)
            external_dir = Path(external_value).resolve() if external_value else None
            external_complete = bool(external_dir) and all(
                is_complete_geff(external_dir / f"{name}.geff") for name in split["test"]
            )
            if external_complete:
                print(f"OFFICIAL METRIC EXTERNAL: {method.name}", flush=True)
                metrics = evaluate_prediction_dir(
                    modules, external_dir, layout.train, expected_datasets=list(split["test"])
                )
                records.append({
                    "status": "complete",
                    "name": method.name,
                    "group": method.group,
                    "fidelity": method.fidelity,
                    "execution": "external-trained-predictions",
                    "requirements": list(method.requirements),
                    "checkpoint_reuse": method.checkpoint_reuse,
                    "note": method.note,
                    "metrics": metrics,
                    "prediction_dir": str(external_dir),
                })
                continue
            records.append({
                "status": "requires_training",
                "name": method.name,
                "group": method.group,
                "fidelity": method.fidelity,
                "requirements": list(method.requirements),
                "checkpoint_reuse": method.checkpoint_reuse,
                "note": method.note,
            })
            continue
        print(f"OFFICIAL METRIC: {method.name}", flush=True)
        metrics = evaluate_prediction_dir(
            modules,
            prediction_dirs[method.name],
            layout.train,
            expected_datasets=list(split["test"]),
        )
        record = {
            "status": "complete",
            "signature": signatures[method.name],
            "name": method.name,
            "group": method.group,
            "fidelity": method.fidelity,
            "settings": method.settings,
            "requirements": list(method.requirements),
            "checkpoint_reuse": method.checkpoint_reuse,
            "note": method.note,
            "metrics": metrics,
            "diagnostics": diagnostics[method.name],
            "prediction_dir": str(prediction_dirs[method.name]),
            "elapsed_seconds": elapsed[method.name],
        }
        write_json(_json_ready(record), output_dir / "variants" / f"{method.name}_{signatures[method.name][:12]}.json")
        records.append(record)

    completed = sorted(
        (record for record in records if record["status"] == "complete"),
        key=_score,
        reverse=True,
    )
    result = {
        "kind": "kaggle-official-metric-all-proposal-method-search",
        "is_kaggle_leaderboard_score": False,
        "metric_contract": official_metric_contract(),
        "selection_rule": "maximum held-out official competition score among executable methods",
        "method_count": len(PROPOSAL_METHODS),
        "completed_count": len(completed),
        "requires_training_count": len(records) - len(completed),
        "checkpoint": checkpoint_record,
        "data_usage": {
            "training": len(split["train"]),
            "validation": len(split["test"]),
            "max_frames": config["dataset"].get("max_frames"),
        },
        "results": completed + [record for record in records if record["status"] != "complete"],
        "best": completed[0] if completed else None,
    }
    write_json(_json_ready(result), output_dir / "proposal_search_results.json")
    _write_scoreboard(records, output_dir / "proposal_scoreboard.csv")
    write_json(
        {"methods": [_json_ready(asdict(method)) for method in PROPOSAL_METHODS]},
        output_dir / "proposal_method_manifest.json",
    )
    if completed:
        write_json(_json_ready(completed[0]), output_dir / "best_proposal_method.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all methods named in the OT proposal.")
    parser.add_argument("--proposal-config", type=Path, default=Path("configs/proposal_all_methods.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/local_method_search.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/proposal_all_methods"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--strict-all-methods", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    proposal_config: dict[str, Any] = {}
    if args.proposal_config.is_file():
        proposal_config = yaml.safe_load(args.proposal_config.read_text(encoding="utf-8")) or {}
        if args.config == Path("configs/local_method_search.yaml") and proposal_config.get("base_config"):
            args.config = Path(proposal_config["base_config"])
        if args.output_dir == Path("outputs/proposal_all_methods") and proposal_config.get("output_dir"):
            args.output_dir = Path(proposal_config["output_dir"])
    config = load_method_search_config(args.config)
    layout = discover_competition_root(config.get("data_root"))
    paired = sum(
        1 for path in layout.train.glob("*.zarr")
        if (layout.train / f"{path.stem}.geff").exists()
    )
    checkpoint_reuse_config = dict(proposal_config.get("checkpoint_reuse", {}))
    plan = build_proposal_plan(config, paired, checkpoint_reuse_config)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    result = execute_proposal_search(
        args.config,
        args.output_dir,
        args.checkpoint,
        external_prediction_dirs=dict(proposal_config.get("external_prediction_dirs", {})),
        strict_all_methods=(
            args.strict_all_methods or bool(proposal_config.get("strict_all_methods", False))
        ),
        checkpoint_reuse_config=checkpoint_reuse_config,
    )
    print(json.dumps({
        "completed": result["completed_count"],
        "requires_training": result["requires_training_count"],
        "best_method": result["best"]["name"] if result["best"] else None,
        "official_cv_score": result["best"]["metrics"]["summary"]["score"] if result["best"] else None,
        "result_file": str(args.output_dir / "proposal_search_results.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
