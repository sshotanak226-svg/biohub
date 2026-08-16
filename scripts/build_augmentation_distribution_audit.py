"""Build the reproducible train/test intensity-distribution audit notebook."""

from pathlib import Path

import nbformat as nbf


OUTPUT = (
    Path.cwd()
    if Path.cwd().name == "augmentation_distribution_audit"
    else Path("outputs/augmentation_distribution_audit")
)
NOTEBOOK = OUTPUT / "augmentation_distribution_audit.ipynb"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Biohub train/test intensity-distribution audit

## tl;dr

- The four downloaded `test/*.zarr` datasets are byte-identical to datasets with the same IDs under `train/`; this notebook verifies every Zarr file by SHA-256.
- Consequently, the supplied test images do not exhibit a directional intensity shift relative to those training images.
- The model also normalizes each dataset with its own 0.1% and 99.9% quantiles, removing most global gain and offset differences before augmentation.
- Strong gain/gamma/bias augmentation therefore has no test-distribution justification. It should be treated only as a CV ablation, not as the default recipe.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

The decision is whether the default training pipeline should deliberately shift grayscale intensity. The audit uses two independent checks:

1. exact SHA-256 comparison of every file in each same-ID train/test Zarr tree;
2. full-volume quantiles stored in each Zarr's authoritative `image_statistics` metadata.

The unit of comparison is one dataset, avoiding an artificial sample-size advantage for the 199 training datasets over the four test datasets.

### Key assumptions

- The local KaggleHub competition directory is the data that will be mounted for local testing.
- Stored quantiles describe the full image volume and are the same values used by the official baseline normalization path.
- The augmentation simulation excludes additive pixel noise; its out-of-envelope estimate is therefore conservative.
"""
        ),
        nbf.v4.new_code_cell(
            """from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import percentileofscore

from biohub_demo.data import discover_competition_root

OUTPUT = (
    Path.cwd()
    if Path.cwd().name == "augmentation_distribution_audit"
    else Path("outputs/augmentation_distribution_audit")
)
OUTPUT.mkdir(parents=True, exist_ok=True)
layout = discover_competition_root()
train_paths = sorted(layout.train.glob("*.zarr"))
test_paths = sorted(layout.test.glob("*.zarr"))
test_names = [path.stem for path in test_paths]
print({"root": str(layout.root), "train": len(train_paths), "test": len(test_paths)})
print("test IDs:", test_names)
"""
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Verify exact train/test identity"),
        nbf.v4.new_code_cell(
            """def tree_sha256(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                total_bytes += len(chunk)
                digest.update(chunk)
    return digest.hexdigest(), len(files), total_bytes


identity_rows = []
for name in test_names:
    train_hash, train_files, train_bytes = tree_sha256(layout.train / f"{name}.zarr")
    test_hash, test_files, test_bytes = tree_sha256(layout.test / f"{name}.zarr")
    identity_rows.append({
        "dataset": name,
        "train_files": train_files,
        "test_files": test_files,
        "train_bytes": train_bytes,
        "test_bytes": test_bytes,
        "sha256_equal": train_hash == test_hash,
        "train_geff_exists": (layout.train / f"{name}.geff").exists(),
    })
identity = pd.DataFrame(identity_rows)
identity
"""
        ),
        nbf.v4.new_markdown_cell("### 2. Read full-volume intensity quantiles"),
        nbf.v4.new_code_cell(
            """def intensity_profile(path: Path, split: str) -> dict[str, float | str]:
    group = zarr.open_group(str(path), mode="r")
    quantiles = group.attrs["image_statistics"]["quantiles"]
    q001 = float(quantiles["0.001"])
    q999 = float(quantiles["0.999"])
    span = q999 - q001
    return {
        "dataset": path.stem,
        "split": split,
        "family": path.stem.split("_", 1)[0],
        "raw_q001": q001,
        "raw_q01": float(quantiles["0.01"]),
        "raw_q10": float(quantiles["0.1"]),
        "raw_q90": float(quantiles["0.9"]),
        "raw_q99": float(quantiles["0.99"]),
        "raw_q999": q999,
        "raw_span_001_999": span,
        "norm_q01": (float(quantiles["0.01"]) - q001) / span,
        "norm_q10": (float(quantiles["0.1"]) - q001) / span,
        "norm_q90": (float(quantiles["0.9"]) - q001) / span,
        "norm_q99": (float(quantiles["0.99"]) - q001) / span,
    }


profiles = pd.DataFrame(
    [intensity_profile(path, "train") for path in train_paths]
    + [intensity_profile(path, "test") for path in test_paths]
)
train_profile = profiles[profiles.split == "train"].reset_index(drop=True)
test_profile = profiles[profiles.split == "test"].reset_index(drop=True)

metrics = ["raw_span_001_999", "norm_q01", "norm_q10", "norm_q90", "norm_q99"]
summary_rows = []
for metric in metrics:
    values = train_profile[metric].to_numpy()
    summary_rows.append({
        "metric": metric,
        "train_p01": np.quantile(values, 0.01),
        "train_p25": np.quantile(values, 0.25),
        "train_median": np.median(values),
        "train_p75": np.quantile(values, 0.75),
        "train_p99": np.quantile(values, 0.99),
        "test_min": test_profile[metric].min(),
        "test_median": test_profile[metric].median(),
        "test_max": test_profile[metric].max(),
    })
distribution_summary = pd.DataFrame(summary_rows)
distribution_summary.to_csv(OUTPUT / "distribution_summary.csv", index=False)
distribution_summary
"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 3. Locate each test dataset within the training distribution"),
        nbf.v4.new_code_cell(
            """percentile_rows = []
for _, row in test_profile.iterrows():
    output = {"dataset": row.dataset, "family": row.family}
    for metric in metrics:
        output[f"{metric}_train_percentile"] = percentileofscore(
            train_profile[metric], row[metric], kind="mean"
        )
    percentile_rows.append(output)
test_percentiles = pd.DataFrame(percentile_rows)
test_percentiles.to_csv(OUTPUT / "test_percentiles.csv", index=False)
test_percentiles
"""
        ),
        nbf.v4.new_markdown_cell("### 4. Visual comparison after the model's per-dataset normalization"),
        nbf.v4.new_code_cell(
            """chart_metrics = ["norm_q01", "norm_q10", "norm_q90", "norm_q99"]
labels = ["1%", "10%", "90%", "99%"]
fig, ax = plt.subplots(figsize=(10, 5.8))
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.82)
positions = np.arange(1, len(chart_metrics) + 1)
ax.boxplot(
    [train_profile[column] for column in chart_metrics],
    positions=positions,
    widths=0.52,
    patch_artist=True,
    boxprops={"facecolor": "#DCE8F5", "edgecolor": "#24527A"},
    medianprops={"color": "#1D2733", "linewidth": 1.8},
    whiskerprops={"color": "#6C7785"},
    capprops={"color": "#6C7785"},
    flierprops={"marker": ".", "markersize": 3, "markerfacecolor": "#A7B2BF", "markeredgecolor": "#A7B2BF"},
)
offsets = np.linspace(-0.12, 0.12, len(test_profile))
for offset, (_, row) in zip(offsets, test_profile.iterrows()):
    ax.scatter(
        positions + offset,
        [row[column] for column in chart_metrics],
        s=42,
        facecolor="#D99A2B",
        edgecolor="#5B431B",
        linewidth=0.7,
        zorder=4,
    )
ax.set_xticks(positions, labels)
ax.set_ylabel("Normalized intensity")
ax.set_xlabel("Full-volume quantile")
fig.suptitle(
    "Normalized intensity quantiles: train distribution and test datasets",
    x=0.09, y=0.97, ha="left", fontsize=16,
)
fig.text(
    0.09, 0.91,
    "Boxes: 199 train datasets; gold points: 4 test datasets; normalization uses each dataset's 0.1% and 99.9% quantiles",
    fontsize=9,
    color="#505A66",
)
ax.grid(axis="y", color="#E5E8EB", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
chart_path = OUTPUT / "normalized_intensity_distribution.png"
fig.savefig(chart_path, dpi=180, bbox_inches="tight")
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell("### 5. Check whether the current strong augmentation leaves the observed envelope"),
        nbf.v4.new_code_cell(
            """def simulate_augmentation(
    *,
    gain_range: tuple[float, float],
    gamma_range: tuple[float, float],
    bias_range: tuple[float, float],
    draws: int = 100_000,
    seed: int = 20260816,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = ["norm_q10", "norm_q90", "norm_q99"]
    source = train_profile[columns].to_numpy()[rng.integers(0, len(train_profile), draws)]
    gain = rng.uniform(*gain_range, size=(draws, 1))
    gamma = rng.uniform(*gamma_range, size=(draws, 1))
    bias = rng.uniform(*bias_range, size=(draws, 1))
    transformed = np.clip(source**gamma * gain + bias, 0.0, 4.0)
    result = pd.DataFrame(transformed, columns=columns)
    result["outside_any_train_p01_p99"] = False
    for column in columns:
        low, high = train_profile[column].quantile([0.01, 0.99])
        outside = (result[column] < low) | (result[column] > high)
        result[f"{column}_outside"] = outside
        result["outside_any_train_p01_p99"] |= outside
    return result


identity_recipe = simulate_augmentation(
    gain_range=(1.0, 1.0), gamma_range=(1.0, 1.0), bias_range=(0.0, 0.0)
)
current = simulate_augmentation(
    gain_range=(0.75, 1.30), gamma_range=(0.70, 1.45), bias_range=(-0.10, 0.10)
)
modest = simulate_augmentation(
    gain_range=(0.90, 1.10), gamma_range=(0.90, 1.10), bias_range=(-0.03, 0.03)
)
coverage = pd.DataFrame([
    {
        "recipe": "no_intensity_augmentation",
        "outside_any_rate": identity_recipe.outside_any_train_p01_p99.mean(),
        **{column: identity_recipe[f"{column}_outside"].mean() for column in ["norm_q10", "norm_q90", "norm_q99"]},
    },
    {
        "recipe": "current_strong_without_noise",
        "outside_any_rate": current.outside_any_train_p01_p99.mean(),
        **{column: current[f"{column}_outside"].mean() for column in ["norm_q10", "norm_q90", "norm_q99"]},
    },
    {
        "recipe": "modest_candidate_without_noise",
        "outside_any_rate": modest.outside_any_train_p01_p99.mean(),
        **{column: modest[f"{column}_outside"].mean() for column in ["norm_q10", "norm_q90", "norm_q99"]},
    },
])
coverage.to_csv(OUTPUT / "augmentation_coverage.csv", index=False)
coverage
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. **There is no observed train-to-test intensity direction to chase.** The four local test Zarr trees are exact duplicates of same-ID train Zarr trees.
2. **Per-dataset quantile normalization already aligns global brightness and scale.** Any remaining differences are shape/contrast differences after normalization, not raw camera gain alone.
3. **Strong intensity augmentation is unsupported as a default.** The coverage table quantifies how often the current gain/gamma/bias ranges create normalized quantiles outside the empirical 1st–99th percentile training envelope. Additive noise would broaden this further.
4. **XY rotations/reflections are a separate invariance assumption.** They do not attempt to repair intensity drift and should be evaluated by official held-out score.
5. **Recommended experiment:** keep `no intensity augmentation` as the reference, compare it with a modest intensity recipe, and retain augmentation only if the official held-out competition score improves across more than one split.

The four-test-dataset sample is too small to justify fitting a directional transform, even without the exact-duplicate finding.
"""
        ),
    ]
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
