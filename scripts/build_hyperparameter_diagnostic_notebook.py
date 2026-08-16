"""Build the read-only notebook used to audit the local hyperparameter search."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUTPUT = Path("outputs/hyperparameter_search/diagnostics/hyperparameter_diagnosis.ipynb")


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Biohub local hyperparameter search diagnosis

## tl;dr

- The selected held-out score is **0.03150**, but the selected ILP result has only **0.0171 node recall** and is not a usable high-accuracy model.
- The strongest non-ILP result recovers only **56 of 2,040 GT edges**; edge candidate/scoring coverage is the dominant verified bottleneck.
- Training used **16/199 series** and only **30/100 frames**, with a `44b6:6bba` training split of **3:13**. Generalization evidence is therefore weak.
- The search needs recall/node-ratio guardrails, lower or top-k edge candidates, distinct physical pooling kernels, broader balanced training data, and OOF validation before Kaggle promotion.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

This diagnostic reads the completed local search artifacts produced on 2026-08-16. The unit of evaluation is one held-out biological series; four series were used. The official local score is computed from predicted and ground-truth GEFF graphs, but it is **not** a Kaggle Public/Private Leaderboard score.

### Key Assumptions

- `search_results.json` is the controlling source for measured candidate metrics.
- The design reference score `0.8789296` is context only because it uses a different fixed-eight-series population.
- Recommendations are diagnostic hypotheses unless the proposed ablation has actually been run.
"""
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Load completed search artifacts"),
        nbf.v4.new_code_cell(
            """import json
from pathlib import Path

import pandas as pd

WORKSPACE = next(
    candidate for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "pyproject.toml").is_file()
)
SEARCH_ROOT = WORKSPACE / "outputs/hyperparameter_search"
DATA_ROOT = WORKSPACE / "biohub_top10_download/data/kagglehub_cache/competitions/biohub-cell-tracking-during-development/train"

results = json.loads((SEARCH_ROOT / "search_results.json").read_text(encoding="utf-8"))
best = json.loads((SEARCH_ROOT / "best_hyperparameters.json").read_text(encoding="utf-8"))
split = json.loads((SEARCH_ROOT / "dataset_splits.json").read_text(encoding="utf-8"))[0]
records = results["all_ranked"]

print({
    "training_records": len(results["training_records"]),
    "ranked_entries": len(records),
    "unique_inference_signatures": len({row["signature"] for row in records}),
    "missing_scores": sum(row["metrics"]["summary"]["score"] is None for row in records),
    "best_score": best["held_out_metrics"]["score"],
})
"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 2. Compare the selected result with usable alternatives"),
        nbf.v4.new_code_cell(
            """def aggregate_record(label, record):
    per_dataset = record["metrics"]["per_dataset"]
    return {
        "case": label,
        "trial": record["training_trial"],
        "score": record["metrics"]["summary"]["score"],
        "edge_jaccard": record["metrics"]["summary"]["edge_jaccard"],
        "node_recall": record["metrics"]["summary"]["node_recall"],
        "edge_tp": sum(row["edge_tp"] for row in per_dataset),
        "edge_fp": sum(row["edge_fp"] for row in per_dataset),
        "edge_fn": sum(row["edge_fn"] for row in per_dataset),
        "division_tp": sum(row["division_tp"] for row in per_dataset),
        "division_fn": sum(row["division_fn"] for row in per_dataset),
        "pred_nodes": sum(row["num_pred_nodes"] for row in per_dataset),
        "det_threshold": record["inference_parameters"]["detection_threshold"],
        "edge_threshold": record["inference_parameters"]["edge_threshold"],
        "use_ilp": record["inference_parameters"]["use_ilp"],
    }

best_overall = records[0]
best_non_ilp = next(row for row in records if not row["inference_parameters"]["use_ilp"])
highest_recall = max(records, key=lambda row: row["metrics"]["summary"]["node_recall"])

case_table = pd.DataFrame([
    aggregate_record("Selected by score", best_overall),
    aggregate_record("Best non-ILP", best_non_ilp),
    aggregate_record("Highest node recall", highest_recall),
])
case_table
"""
        ),
        nbf.v4.new_markdown_cell("### 3. Verify that edge success is concentrated in one series"),
        nbf.v4.new_code_cell(
            """validation_names = sorted(split["test"])
per_series = []
for name, row in zip(validation_names, best_non_ilp["metrics"]["per_dataset"]):
    per_series.append({
        "dataset": name,
        "family": name.split("_", 1)[0],
        "edge_tp": row["edge_tp"],
        "edge_fp": row["edge_fp"],
        "edge_fn": row["edge_fn"],
        "node_recall": row["node_recall"],
        "pred_nodes": row["num_pred_nodes"],
        "node_ratio_delta": row["total_node_ratio"],
    })
per_series_table = pd.DataFrame(per_series)
per_series_table
"""
        ),
        nbf.v4.new_markdown_cell("### 4. Quantify training coverage and family balance"),
        nbf.v4.new_code_cell(
            """all_names = sorted(path.stem for path in DATA_ROOT.glob("*.zarr"))

def family_counts(names):
    return pd.Series([name.split("_", 1)[0] for name in names]).value_counts().to_dict()

coverage = pd.DataFrame([
    {"population": "All official train series", "series": len(all_names), **family_counts(all_names)},
    {"population": "Search training split", "series": len(split["train"]), **family_counts(split["train"])},
    {"population": "Search validation split", "series": len(split["test"]), **family_counts(split["test"])},
])
coverage["series_share_of_official"] = coverage["series"] / len(all_names)
coverage["frames_used_for_training"] = [None, 30, None]
coverage
"""
        ),
        nbf.v4.new_markdown_cell("### 5. Check search boundaries and redundant pool settings"),
        nbf.v4.new_code_cell(
            """selected_trials = {"baseline_lr1e4", "lr3e4_det05"}
non_ilp = [
    row for row in records
    if row["training_trial"] in selected_trials and not row["inference_parameters"]["use_ilp"]
]

edge_rows = []
for trial in sorted(selected_trials):
    for edge_threshold in (0.3, 0.5, 0.7):
        candidates = [
            row for row in non_ilp
            if row["training_trial"] == trial
            and row["inference_parameters"]["edge_threshold"] == edge_threshold
        ]
        chosen = max(candidates, key=lambda row: row["metrics"]["summary"]["score"])
        edge_rows.append({
            "trial": trial,
            "edge_threshold": edge_threshold,
            "best_score": chosen["metrics"]["summary"]["score"],
            "node_recall": chosen["metrics"]["summary"]["node_recall"],
        })

def effective_pool_kernel(um, voxel_size=1.625):
    kernel = max(1, round(um / voxel_size))
    return kernel + 1 if kernel % 2 == 0 else kernel

edge_boundary_table = pd.DataFrame(edge_rows)
pool_table = pd.DataFrame([
    {"pool_kernel_um": value, "effective_kernel_per_axis": effective_pool_kernel(value)}
    for value in (3.0, 5.0)
])

display(edge_boundary_table)
display(pool_table)
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. **Fix edge candidate coverage first.** The best non-ILP result has 56 TP and 1,984 FN; threshold `0.3` is already the lower search boundary. Add lower thresholds and a distance-gated target top-k path, then log candidate recall before ILP.
2. **Do not promote the current ILP winner.** It wins only without the design guardrails; its node recall is 0.0171. Enforce node recall and predicted-node-ratio eligibility before ranking.
3. **Retrain on broader, balanced temporal coverage.** The current model sees 16/199 series and 30/100 frames, with only 3 `44b6` training series. This is insufficient to claim generalization.
4. **Align checkpoint selection with the competition metric.** Training `acc` is dominated by negative edge pairs and is not an official score. Track positive-edge recall/AP and periodically score complete validation graphs.
5. **Use distinct physical pooling kernels.** At the effective 1.625 µm voxel size, 3 µm and 5 µm both become a 3-voxel kernel and therefore do not constitute two experiments.
6. **Expand validation before tuning division.** Four validation series contain only two division events; division TP remains zero and cannot be tuned reliably from this split.
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
