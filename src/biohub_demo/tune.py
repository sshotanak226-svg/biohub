"""Local inference-parameter search using one real trained checkpoint."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from .competition_metric import METRIC_CONTRACT_ID, official_metric_contract
from .data import discover_competition_root
from .io import write_json
from .train_eval import (
    WORKSPACE,
    _load_baseline,
    evaluate_prediction_dir,
    find_baseline_root,
)


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    layout = discover_competition_root(args.data_root)
    baseline = find_baseline_root(args.baseline_root)
    modules = _load_baseline(baseline)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    splits_path = args.splits.resolve()
    folds = json.loads(splits_path.read_text(encoding="utf-8"))
    validation_names = folds[0]["test"]
    if not validation_names:
        raise ValueError("validation split is empty")

    output = (args.output_dir or (WORKSPACE / "outputs" / "parameter_search")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ilp_values = [False, True] if args.compare_ilp else [False]
    trials = list(itertools.product(args.det_thresholds, args.edge_thresholds, ilp_values))
    results: list[dict[str, Any]] = []
    for index, (det_threshold, edge_threshold, use_ilp) in enumerate(trials, 1):
        trial_name = (
            f"local_tune_d{det_threshold:.5f}_e{edge_threshold:.3f}_"
            f"ilp{int(use_ilp)}"
        ).replace(".", "p")
        print(
            f"=== TRIAL {index}/{len(trials)} det={det_threshold} "
            f"edge={edge_threshold} ilp={use_ilp} ===",
            flush=True,
        )
        cfg = modules["predict"].PredictConfig(
            det_threshold=det_threshold,
            threshold=edge_threshold,
            use_ilp=use_ilp,
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
            cfg=cfg,
            method=trial_name,
            unet_batch_size=args.inference_batch_size,
            evaluate=False,
        )
        prediction_dir = (
            baseline / "predictions" / modules["dataspec"].USERNAME / trial_name / "split_0"
        )
        metrics = evaluate_prediction_dir(
            modules,
            prediction_dir,
            layout.train,
            expected_datasets=validation_names,
        )
        record = {
            "det_threshold": det_threshold,
            "edge_threshold": edge_threshold,
            "use_ilp": use_ilp,
            "validation_datasets": validation_names,
            "prediction_dir": str(prediction_dir),
            "metrics": metrics,
        }
        results.append(record)
        write_json(record, output / f"{trial_name}.json")

    ranked = sorted(
        results,
        key=lambda item: (
            item["metrics"].get("metric_contract", {}).get("id")
            == METRIC_CONTRACT_ID,
            item["metrics"]["summary"].get("score") is not None,
            item["metrics"]["summary"].get("score") or float("-inf"),
        ),
        reverse=True,
    )
    payload = {
        "kind": "kaggle-official-metric-held-out-parameter-search",
        "is_kaggle_leaderboard_score": False,
        "metric_contract": official_metric_contract(),
        "checkpoint": str(checkpoint),
        "split_file": str(splits_path),
        "trial_count": len(results),
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
    }
    write_json(payload, output / "search_results.json")
    print(json.dumps(payload["best"], indent=2, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep detection/link thresholds on a fixed real checkpoint and held-out split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--det-thresholds", type=_floats, default=[0.95, 0.97, 0.98, 0.99, 0.995]
    )
    parser.add_argument("--edge-thresholds", type=_floats, default=[0.3, 0.5, 0.7])
    parser.add_argument("--compare-ilp", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=1)
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
