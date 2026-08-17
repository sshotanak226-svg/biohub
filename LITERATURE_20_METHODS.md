# Biohub literature-backed 20-method local screen

This workflow compares twenty executable **screening proxies** against the
current `uot_mutual_top` baseline with the official competition metric on the
same fixed holdout movies. It reuses the checkpoint and raw `.npz` predictions
created by `scripts/run_local_method_search.ps1`.

It does not claim to reproduce twenty papers. Architectures such as HOCT,
Trackastra, StarDist3D, nnU-Net, Swin UNETR, Cellpose, and Cell-TRACTR require
new model code, annotations, weights, or training. Their entries isolate an
executable part of the published idea using the current detector: temporal
context, multiple hypotheses, structured flow, division gating, confidence
consensus, or uncertainty abstention. A winning proxy is a reason to implement
and train the corresponding paper faithfully next.

## Commands

Inspect all recipes without running inference:

```powershell
.\scripts\run_literature_20_methods.ps1 -DryRun
```

Fast provisional comparison on three fixed holdout movies:

```powershell
.\scripts\run_literature_20_methods.ps1 -Quick
```

Full comparison on all twenty fixed holdout movies:

```powershell
.\scripts\run_literature_20_methods.ps1
```

Run a selected subset:

```powershell
.\scripts\run_literature_20_methods.ps1 `
  -Methods "hoct_edge_attention_proxy,calibrated_probabilistic_flow_proxy,four_frame_division_detector_proxy"
```

Interrupted runs are resumable because a method/movie pair is skipped only
after its GEFF root metadata is complete.

## Outputs

- `outputs/literature_20_methods/literature_scoreboard.csv`
- `outputs/literature_20_methods/literature_search_results.json`
- `outputs/literature_20_methods/literature_method_manifest.json`
- `outputs/literature_20_methods/variants/*.json`
- `outputs/literature_20_methods/predictions/*/*.geff`

The scoreboard contains 21 rows: twenty literature proxies plus the exact
current baseline. `score`, `adj_edge_jaccard`, `edge_jaccard`,
`division_jaccard`, Division TP/FP/FN, `node_recall`, mean node-count ratio,
mean absolute node-count ratio, and every component's delta from the baseline
are reported separately.

## Selection rule

Do not select a method from `-Quick` alone. Use quick mode to remove clearly
inferior recipes, then run all twenty holdout movies. Promote a paper to a
faithful training implementation only if its proxy improves the full official
CV score without collapsing either adjusted edge Jaccard or division Jaccard.
