from pathlib import Path

from biohub_demo.competition_metric import METRIC_CONTRACT_ID
from biohub_demo.method_search import (
    _official_score,
    build_method_search_plan,
    load_method_search_config,
)


def test_method_search_uses_all_non_holdout_data_and_shared_inference() -> None:
    path = Path(__file__).parents[1] / "configs" / "local_method_search.yaml"
    config = load_method_search_config(path)
    plan = build_method_search_plan(config, paired_datasets=199)
    assert plan["training_datasets"] == 179
    assert plan["validation_datasets"] == 20
    assert plan["max_frames_per_dataset"] is None
    assert plan["shared_raw_inference_passes"] == 20
    assert plan["official_metric_ranked_variants"] == 6
    assert plan["uses_all_available_training_sequences_except_holdout"] is True


def test_method_ranking_rejects_non_official_scores() -> None:
    assert _official_score({"status": "complete", "metrics": {"summary": {"score": 0.99}}}) == float("-inf")
    record = {
        "status": "complete",
        "metrics": {
            "metric_contract": {"id": METRIC_CONTRACT_ID},
            "summary": {"score": 0.81},
        },
    }
    assert _official_score(record) == 0.81
