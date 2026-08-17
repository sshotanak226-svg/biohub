"""Compare twenty literature-inspired recipes against the current best UOT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .competition_metric import METRIC_CONTRACT_ID, official_metric_contract
from .data import discover_competition_root
from .hyper_search import _signature
from .io import is_complete_geff, write_geff, write_json
from .literature_methods import build_current_baseline, build_literature_method
from .literature_registry import LITERATURE_METHODS, LITERATURE_METHOD_BY_NAME
from .method_search import _load_or_predict_raw, load_method_search_config
from .proposal_search import _checkpoint_record
from .tracking_variants import graph_from_raw_prediction
from .train_eval import (
    WORKSPACE,
    _json_ready,
    _load_baseline,
    build_split,
    evaluate_prediction_dir,
    find_baseline_root,
)

BASELINE_NAME = "current_uot_mutual_top"
IMPLEMENTATION_VERSION = "literature-screen-v1"


def _score(record: dict[str, Any]) -> float:
    value = record.get("metrics", {}).get("summary", {}).get("score")
    return float(value) if value is not None else float("-inf")


def _node_count_stats(metrics: dict[str, Any]) -> dict[str, float | None]:
    ratios = [
        float(row["total_node_ratio"])
        for row in metrics.get("per_dataset", [])
        if row.get("total_node_ratio") is not None
    ]
    if not ratios:
        return {"mean_node_ratio": None, "mean_abs_node_ratio": None}
    return {
        "mean_node_ratio": sum(ratios) / len(ratios),
        "mean_abs_node_ratio": sum(abs(value) for value in ratios) / len(ratios),
    }


def _numeric_delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    current_value, baseline_value = float(value), float(baseline)
    if not math.isfinite(current_value) or not math.isfinite(baseline_value):
        return None
    return current_value - baseline_value


def _write_scoreboard(records: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "rank", "name", "role", "fidelity", "score", "adj_edge_jaccard",
        "edge_jaccard", "division_jaccard", "division_tp", "division_fp",
        "division_fn", "node_recall", "mean_node_ratio", "mean_abs_node_ratio",
        "score_delta_vs_baseline", "adj_edge_delta_vs_baseline",
        "division_delta_vs_baseline", "elapsed_seconds", "focus", "paper",
        "paper_url", "note",
    ]
    ordered = sorted(records, key=_score, reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for rank, record in enumerate(ordered, 1):
            summary = record["metrics"]["summary"]
            writer.writerow({
                "rank": rank,
                "name": record["name"],
                "role": record["role"],
                "fidelity": record["fidelity"],
                "score": summary.get("score"),
                "adj_edge_jaccard": summary.get("adj_edge_jaccard"),
                "edge_jaccard": summary.get("edge_jaccard"),
                "division_jaccard": summary.get("division_jaccard"),
                "division_tp": summary.get("division_tp"),
                "division_fp": summary.get("division_fp"),
                "division_fn": summary.get("division_fn"),
                "node_recall": summary.get("node_recall"),
                "mean_node_ratio": record.get("mean_node_ratio"),
                "mean_abs_node_ratio": record.get("mean_abs_node_ratio"),
                "score_delta_vs_baseline": record.get("score_delta_vs_baseline"),
                "adj_edge_delta_vs_baseline": record.get("adj_edge_delta_vs_baseline"),
                "division_delta_vs_baseline": record.get("division_delta_vs_baseline"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "focus": record.get("focus", ""),
                "paper": record.get("paper", ""),
                "paper_url": record.get("paper_url", ""),
                "note": record.get("note", ""),
            })


def build_literature_plan(
    config: dict[str, Any],
    selected_names: list[str] | None = None,
    max_datasets: int | None = None,
) -> dict[str, Any]:
    methods = (
        [LITERATURE_METHOD_BY_NAME[name] for name in selected_names]
        if selected_names else list(LITERATURE_METHODS)
    )
    validation_count = int(config["dataset"]["validation_count"])
    used_count = min(validation_count, max_datasets) if max_datasets else validation_count
    return {
        "purpose": "coarse local screening of twenty literature-backed ideas",
        "paper_faithful_reimplementations": False,
        "screening_proxy_count": len(methods),
        "comparison_rows": len(methods) + 1,
        "baseline": BASELINE_NAME,
        "implementation_version": IMPLEMENTATION_VERSION,
        "shared_checkpoint": True,
        "shared_raw_predictions": True,
        "validation_datasets": used_count,
        "full_validation_datasets": validation_count,
        "metric_contract": official_metric_contract(),
        "methods": [asdict(method) for method in methods],
        "warning": (
            "Architecture-specific papers are represented by executable ablations over the "
            "current checkpoint. A winning proxy is evidence to prioritize a faithful retraining, "
            "not evidence that the cited model itself achieved this score."
        ),
    }


def execute_literature_search(
    config_path: Path,
    literature_config: dict[str, Any],
    output_dir: Path,
    checkpoint: Path | None = None,
    selected_names: list[str] | None = None,
    max_datasets: int | None = None,
) -> dict[str, Any]:
    config = load_method_search_config(config_path)
    config.setdefault("seed", 20260809)
    methods = (
        [LITERATURE_METHOD_BY_NAME[name] for name in selected_names]
        if selected_names else list(LITERATURE_METHODS)
    )
    layout = discover_competition_root(config.get("data_root"))
    baseline_root = find_baseline_root(None)
    modules = _load_baseline(baseline_root)
    base_output_value = Path(config.get("output_dir", "outputs/method_search"))
    base_output = base_output_value if base_output_value.is_absolute() else WORKSPACE / base_output_value
    output_dir = output_dir if output_dir.is_absolute() else WORKSPACE / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variants").mkdir(parents=True, exist_ok=True)
    checkpoint_record = _checkpoint_record(
        base_output,
        checkpoint,
        dict(literature_config.get("checkpoint_reuse", {})),
    )
    split = build_split(
        layout.train,
        seed=int(config["seed"]),
        train_count=None,
        val_count=int(config["dataset"]["validation_count"]),
    )
    validation_names = list(split["test"])
    if max_datasets:
        validation_names = validation_names[:max_datasets]
    raw = _load_or_predict_raw(
        modules=modules,
        checkpoint_record=checkpoint_record,
        train_dir=layout.train,
        names=validation_names,
        output_dir=base_output,
        settings=config["raw_inference"],
        reuse_completed=True,
    )

    entries: list[dict[str, Any]] = [{
        "name": BASELINE_NAME,
        "role": "current_baseline",
        "fidelity": "exact_current_pipeline",
        "focus": "nodes+links",
        "paper": "Current local best",
        "paper_url": "",
        "note": "Exact current uot_mutual_top anchor.",
        "method": None,
    }]
    entries.extend({
        "name": method.name,
        "role": "literature_screening_candidate",
        "fidelity": method.fidelity,
        "focus": method.focus,
        "paper": method.paper,
        "paper_url": method.paper_url,
        "note": method.note,
        "method": method,
    } for method in methods)
    signatures = {
        entry["name"]: _signature({
            "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
            "raw_inference": config["raw_inference"],
            "metric_contract_id": METRIC_CONTRACT_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "entry": (
                {"baseline": BASELINE_NAME}
                if entry["method"] is None else asdict(entry["method"])
            ),
        })
        for entry in entries
    }
    prediction_dirs = {
        entry["name"]: output_dir / "predictions" / f"{entry['name']}_{signatures[entry['name']][:12]}"
        for entry in entries
    }
    for path in prediction_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, dict[str, Any]] = {entry["name"]: {} for entry in entries}
    elapsed: dict[str, float] = {entry["name"]: 0.0 for entry in entries}

    for dataset_index, name in enumerate(validation_names, 1):
        print(f"LITERATURE DATASET {dataset_index}/{len(validation_names)}: {name}", flush=True)
        coords, scores, edge_array = raw[name]
        raw_graph, raw_candidates = graph_from_raw_prediction(
            name,
            coords,
            scores,
            [tuple(row) for row in edge_array.tolist()],
            (1.625, 0.40625, 0.40625),
        )
        profile_cache: dict[str, tuple[Any, dict[str, Any]]] = {}
        for method_index, entry in enumerate(entries, 1):
            geff_path = prediction_dirs[entry["name"]] / f"{name}.geff"
            if is_complete_geff(geff_path):
                diagnostics[entry["name"]][name] = {"resumed_from_complete_geff": True}
                print(f"  METHOD {method_index}/{len(entries)}: {entry['name']} [resume skip]", flush=True)
                continue
            print(f"  METHOD {method_index}/{len(entries)}: {entry['name']}", flush=True)
            started = time.perf_counter()
            if entry["method"] is None:
                graph, diagnostic = build_current_baseline(raw_graph, raw_candidates, profile_cache)
            else:
                graph, diagnostic = build_literature_method(
                    entry["method"], raw_graph, raw_candidates, profile_cache
                )
            method_seconds = time.perf_counter() - started
            elapsed[entry["name"]] += method_seconds
            write_geff(graph, geff_path)
            diagnostics[entry["name"]][name] = diagnostic
            print(
                f"    completed in {method_seconds:.1f}s; nodes={len(graph.nodes)} "
                f"edges={len(graph.edges)} divisions="
                f"{sum(len(graph.outgoing(node_id)) >= 2 for node_id in graph.nodes)}",
                flush=True,
            )

    records: list[dict[str, Any]] = []
    for entry in entries:
        print(f"OFFICIAL METRIC: {entry['name']}", flush=True)
        metrics = evaluate_prediction_dir(
            modules,
            prediction_dirs[entry["name"]],
            layout.train,
            expected_datasets=validation_names,
        )
        record = {
            "status": "complete",
            "name": entry["name"],
            "role": entry["role"],
            "fidelity": entry["fidelity"],
            "paper_faithful": None if entry["method"] is None else False,
            "focus": entry["focus"],
            "paper": entry["paper"],
            "paper_url": entry["paper_url"],
            "note": entry["note"],
            "signature": signatures[entry["name"]],
            "metrics": metrics,
            "diagnostics": diagnostics[entry["name"]],
            "prediction_dir": str(prediction_dirs[entry["name"]]),
            "elapsed_seconds": elapsed[entry["name"]],
            **_node_count_stats(metrics),
        }
        write_json(
            _json_ready(record),
            output_dir / "variants" / f"{entry['name']}_{signatures[entry['name']][:12]}.json",
        )
        records.append(record)

    baseline_record = next(record for record in records if record["name"] == BASELINE_NAME)
    baseline_summary = baseline_record["metrics"]["summary"]
    for record in records:
        summary = record["metrics"]["summary"]
        record["score_delta_vs_baseline"] = _numeric_delta(
            summary.get("score"), baseline_summary.get("score")
        )
        record["adj_edge_delta_vs_baseline"] = _numeric_delta(
            summary.get("adj_edge_jaccard"), baseline_summary.get("adj_edge_jaccard")
        )
        record["division_delta_vs_baseline"] = _numeric_delta(
            summary.get("division_jaccard"), baseline_summary.get("division_jaccard")
        )
    ordered = sorted(records, key=_score, reverse=True)
    result = {
        "kind": "kaggle-official-metric-literature-screening",
        "is_kaggle_leaderboard_score": False,
        "paper_faithful_reimplementations": False,
        "metric_contract": official_metric_contract(),
        "selection_rule": "maximum held-out official score among coarse executable proxies",
        "method_count": len(methods),
        "comparison_rows": len(records),
        "checkpoint": checkpoint_record,
        "data_usage": {
            "training_sequences": len(split["train"]),
            "validation_sequences": len(validation_names),
            "full_holdout_sequences": len(split["test"]),
            "max_frames": config["dataset"].get("max_frames"),
        },
        "limitations": [
            "The twenty candidates are executable screening proxies, not paper-faithful retrained models.",
            "A proxy win prioritizes the corresponding faithful implementation; it does not reproduce a paper score.",
            "Scores use the official formula on local held-out data and are not Kaggle leaderboard scores.",
        ],
        "results": ordered,
        "best": ordered[0] if ordered else None,
        "baseline": baseline_record,
    }
    write_json(_json_ready(result), output_dir / "literature_search_results.json")
    _write_scoreboard(records, output_dir / "literature_scoreboard.csv")
    write_json(
        {"baseline": BASELINE_NAME, "methods": [asdict(method) for method in methods]},
        output_dir / "literature_method_manifest.json",
    )
    return result


def _parse_names(value: str | None) -> list[str] | None:
    if not value:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(LITERATURE_METHOD_BY_NAME))
    if unknown:
        raise ValueError("unknown literature methods: " + ", ".join(unknown))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen twenty literature-backed Biohub recipes.")
    parser.add_argument("--literature-config", type=Path, default=Path("configs/literature_20_methods.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/local_method_search.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/literature_20_methods"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--methods", help="Comma-separated subset of the twenty method names.")
    parser.add_argument("--max-datasets", type=int, help="Use only the first N fixed holdout movies for a quick screen.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    literature_config: dict[str, Any] = {}
    if args.literature_config.is_file():
        literature_config = yaml.safe_load(args.literature_config.read_text(encoding="utf-8")) or {}
    if args.config == Path("configs/local_method_search.yaml") and literature_config.get("base_config"):
        args.config = Path(literature_config["base_config"])
    if args.output_dir == Path("outputs/literature_20_methods") and literature_config.get("output_dir"):
        args.output_dir = Path(literature_config["output_dir"])
    selected_names = _parse_names(args.methods)
    if selected_names is None and literature_config.get("methods"):
        selected_names = list(literature_config["methods"])
        _parse_names(",".join(selected_names))
    max_datasets = args.max_datasets
    if max_datasets is not None and max_datasets < 1:
        parser.error("--max-datasets must be positive")
    config = load_method_search_config(args.config)
    plan = build_literature_plan(config, selected_names, max_datasets)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    result = execute_literature_search(
        args.config,
        literature_config,
        args.output_dir,
        args.checkpoint,
        selected_names,
        max_datasets,
    )
    print(json.dumps({
        "best_method": result["best"]["name"],
        "best_official_cv_score": result["best"]["metrics"]["summary"]["score"],
        "baseline_score": result["baseline"]["metrics"]["summary"]["score"],
        "result_file": str(args.output_dir / "literature_search_results.json"),
        "scoreboard": str(args.output_dir / "literature_scoreboard.csv"),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
