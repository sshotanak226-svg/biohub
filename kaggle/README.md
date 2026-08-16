# Kaggle execution notes

1. `dist/biohub-0.1.0-py3-none-any.whl`、`configs/kaggle_inference.yaml`、checkpoint、必要なoffline wheelsをKaggle Datasetへ添付する。
2. YAML内の `weights` と `model_config` を添付先へ合わせる。
3. internet offでwheelを `pip --no-index` インストールする。
4. `python -m biohub_demo.kaggle --config ...` を実行する。
5. `/kaggle/working/biohub_run/validation.json` の `valid=true` を確認してから `submission.csv` を提出する。

`--check-config` は推論せず設定構造だけを検査します。完了済み系列は `.complete` と `predictions/*.geff.json` から再開されます。
