"""Immutable contract for the official Kaggle Biohub evaluation metric.

The values in this module come from the competition Evaluation page.  Local
validation may use held-out training samples, but it must use this exact metric
contract so hyperparameter ordering remains aligned with the Kaggle objective.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any, Iterable


COMPETITION_ID = 136605
COMPETITION_SLUG = "biohub-cell-tracking-during-development"
EVALUATION_URL = (
    "https://www.kaggle.com/competitions/"
    f"{COMPETITION_SLUG}/overview/evaluation"
)
METRIC_DETAILS_URL = (
    "https://github.com/royerlab/kaggle-cell-tracking-competition/"
    "blob/main/metrics.md"
)

# Increment this only when the competition owner changes the official metric.
METRIC_CONTRACT_ID = "kaggle-biohub-136605-official-v1"
MAX_DISTANCE_UM = 7.0
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
ADJUSTMENT_ALPHA = 0.1
DIVISION_WEIGHT = 0.1


def official_metric_contract() -> dict[str, Any]:
    """Return JSON-serialisable provenance for every measured local score."""
    return {
        "id": METRIC_CONTRACT_ID,
        "competition_id": COMPETITION_ID,
        "competition_slug": COMPETITION_SLUG,
        "evaluation_url": EVALUATION_URL,
        "metric_details_url": METRIC_DETAILS_URL,
        "score_formula": "adjusted_edge_jaccard + 0.1 * division_jaccard",
        "max_distance_um": MAX_DISTANCE_UM,
        "voxel_scale_um": list(VOXEL_SCALE_UM),
        "adjustment_alpha": ADJUSTMENT_ALPHA,
        "division_weight": DIVISION_WEIGHT,
        "edge_aggregation": "per-sample adjusted Jaccard weighted by TP+FP+FN",
        "division_aggregation": "micro-averaged TP/FP/FN across all samples",
    }


def validate_official_metric_runtime(modules: dict[str, Any]) -> None:
    """Fail closed if the imported baseline no longer matches Kaggle's metric."""
    metrics = modules["metrics"]
    evaluate = modules["evaluate"]

    observed_alpha = float(getattr(metrics, "ADJUSTMENT_ALPHA", float("nan")))
    observed_division = float(
        getattr(metrics, "SCORE_DIVISION_WEIGHT", float("nan"))
    )
    if not math.isclose(observed_alpha, ADJUSTMENT_ALPHA, abs_tol=1e-12):
        raise RuntimeError(
            "official metric mismatch: ADJUSTMENT_ALPHA "
            f"must be {ADJUSTMENT_ALPHA}, got {observed_alpha}"
        )
    if not math.isclose(observed_division, DIVISION_WEIGHT, abs_tol=1e-12):
        raise RuntimeError(
            "official metric mismatch: SCORE_DIVISION_WEIGHT "
            f"must be {DIVISION_WEIGHT}, got {observed_division}"
        )

    signature = inspect.signature(evaluate.evaluate_pairs)
    default_distance = signature.parameters["max_distance"].default
    if not math.isclose(float(default_distance), MAX_DISTANCE_UM, abs_tol=1e-12):
        raise RuntimeError(
            "official metric mismatch: evaluate_pairs max_distance default "
            f"must be {MAX_DISTANCE_UM}, got {default_distance}"
        )

    for name in ("evaluate", "per_sample_metrics", "summarise"):
        if not callable(getattr(metrics, name, None)):
            raise RuntimeError(f"official metric mismatch: metrics.{name} is missing")


def validate_evaluation_inputs(
    evaluate_module: Any,
    prediction_dir: Path,
    truth_dir: Path,
    expected_datasets: Iterable[str],
) -> list[str]:
    """Require exact dataset coverage and the official physical voxel scale."""
    expected = sorted(set(expected_datasets))
    if not expected:
        raise ValueError("official evaluation requires at least one expected dataset")

    predicted = {path.stem for path in prediction_dir.glob("*.geff")}
    truth = {path.stem for path in truth_dir.glob("*.geff")}
    expected_set = set(expected)
    missing_predictions = sorted(expected_set - predicted)
    unexpected_predictions = sorted(predicted - expected_set)
    missing_truth = sorted(expected_set - truth)
    if missing_predictions or unexpected_predictions or missing_truth:
        raise RuntimeError(
            "official evaluation dataset coverage mismatch: "
            f"missing_predictions={missing_predictions}, "
            f"unexpected_predictions={unexpected_predictions}, "
            f"missing_truth={missing_truth}"
        )

    for name in expected:
        scale = tuple(float(value) for value in evaluate_module._read_scale(truth_dir, name))
        if len(scale) != len(VOXEL_SCALE_UM) or any(
            not math.isclose(actual, required, abs_tol=1e-12)
            for actual, required in zip(scale, VOXEL_SCALE_UM)
        ):
            raise RuntimeError(
                f"official metric voxel scale mismatch for {name}: "
                f"expected {VOXEL_SCALE_UM}, got {scale}"
            )
    return expected
