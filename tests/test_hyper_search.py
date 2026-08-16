from pathlib import Path

from biohub_demo.competition_metric import METRIC_CONTRACT_ID
from biohub_demo.hyper_search import _score, build_search_plan, load_search_config


def test_default_hyperparameter_search_plan() -> None:
    path = Path(__file__).parents[1] / "configs" / "local_hyperparameter_search.yaml"
    config = load_search_config(path)
    plan = build_search_plan(config)
    assert plan["training_trials"] == 6
    assert plan["epochs_per_training_trial"] == [10, 10, 10, 10, 10, 10]
    assert plan["maximum_training_epochs"] == 60
    assert plan["selected_checkpoints"] == 2
    assert plan["grid_trials_per_checkpoint"] == 30
    assert plan["grid_trials"] == 60
    assert plan["additional_ilp_trials"] == 3
    assert plan["maximum_inference_trials"] == 69


def test_failed_or_missing_score_is_not_rankable() -> None:
    assert _score({"status": "failed"}) == float("-inf")
    assert _score({"status": "complete", "metrics": {"summary": {"score": None}}}) == float("-inf")
    assert _score({"status": "complete", "metrics": {"summary": {"score": 0.25}}}) == float("-inf")
    assert _score({
        "status": "complete",
        "metrics": {
            "metric_contract": {"id": METRIC_CONTRACT_ID},
            "summary": {"score": 0.25},
        },
    }) == 0.25
