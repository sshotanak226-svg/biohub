from pathlib import Path
from types import SimpleNamespace

import pytest

from biohub_demo.competition_metric import (
    ADJUSTMENT_ALPHA,
    DIVISION_WEIGHT,
    MAX_DISTANCE_UM,
    METRIC_CONTRACT_ID,
    VOXEL_SCALE_UM,
    official_metric_contract,
    validate_evaluation_inputs,
    validate_official_metric_runtime,
)


def _evaluate_pairs(_pred, _truth, max_distance=MAX_DISTANCE_UM):
    return [], []


def _modules(*, alpha=ADJUSTMENT_ALPHA, division=DIVISION_WEIGHT):
    metrics = SimpleNamespace(
        ADJUSTMENT_ALPHA=alpha,
        SCORE_DIVISION_WEIGHT=division,
        evaluate=lambda *_args, **_kwargs: None,
        per_sample_metrics=lambda *_args, **_kwargs: None,
        summarise=lambda rows: rows,
    )
    evaluate = SimpleNamespace(evaluate_pairs=_evaluate_pairs)
    return {"metrics": metrics, "evaluate": evaluate}


def test_official_metric_contract_is_fixed_to_kaggle() -> None:
    contract = official_metric_contract()
    assert contract["id"] == METRIC_CONTRACT_ID
    assert contract["max_distance_um"] == 7.0
    assert contract["voxel_scale_um"] == [1.625, 0.40625, 0.40625]
    assert contract["adjustment_alpha"] == 0.1
    assert contract["division_weight"] == 0.1


def test_runtime_validation_rejects_metric_weight_drift() -> None:
    validate_official_metric_runtime(_modules())
    with pytest.raises(RuntimeError, match="ADJUSTMENT_ALPHA"):
        validate_official_metric_runtime(_modules(alpha=0.2))
    with pytest.raises(RuntimeError, match="SCORE_DIVISION_WEIGHT"):
        validate_official_metric_runtime(_modules(division=0.2))


def test_evaluation_inputs_require_exact_coverage_and_scale(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "pred"
    truth_dir = tmp_path / "truth"
    prediction_dir.mkdir()
    truth_dir.mkdir()
    (prediction_dir / "sample.geff").mkdir()
    (truth_dir / "sample.geff").mkdir()
    evaluate = SimpleNamespace(_read_scale=lambda *_args: VOXEL_SCALE_UM)

    assert validate_evaluation_inputs(
        evaluate, prediction_dir, truth_dir, ["sample"]
    ) == ["sample"]

    with pytest.raises(RuntimeError, match="missing_predictions"):
        validate_evaluation_inputs(
            evaluate, prediction_dir, truth_dir, ["sample", "missing"]
        )
