# Biohub cell tracking implementation

## 推奨: ハイパーパラメータ探索前の粗い手法探索

従来の `scripts/run_local_hyperparameter_search.ps1` はそのまま残してあります。
これは同じTransformer方式の学習率・loss weight・閾値を細かく比較する旧経路です。
方式自体を選ぶ前に実行すると時間がかかるため、先に次の新しい経路を使います。

```powershell
# 計画だけを表示。学習・推論・ファイル生成は行わない
.\scripts\run_local_method_search.ps1 -DryRun

# 全データ利用の学習、early stopping、6方式の公式CV比較
.\scripts\run_local_method_search.ps1
```

デフォルト設定は [configs/local_method_search.yaml](configs/local_method_search.yaml) です。
取得済みtrain 199系列のうち20系列を固定holdoutとし、残る179系列の全100フレームを学習に使用します。
holdoutを学習へ混ぜないのはlocal scoreの漏洩を防ぐためです。方式決定後のKaggle用最終学習では、選択epoch数を使って199系列すべてへrefitします。

学習は最大30 epoch、最低8 epoch、validation lossが4 epoch改善しなければ停止します。
early stopping中に表示されるaccuracy/recallは停止判定の補助値にすぎません。
手法の順位は必ず公式式 `adjusted_edge_jaccard + 0.1 * division_jaccard` のheld-out scoreだけで決まります。

追加したaugmentationはXY回転・反転、gain、gamma、bias、noiseです。
元画像は1チャネルなので彩度augmentationは定義できません。代わりにtrain/test間で問題になる輝度・コントラスト分布を直接ずらします。

比較する方式は次の6系統です。

1. Transformer確率のgreedy forest
2. Transformer確率のILP forest
3. 上位Notebook型の物理距離二段Hungarian（6 µm / 10 µm）
4. Hungarian＋4ノード未満の短軌跡除去＋平滑化
5. Transformer確率と物理距離を併用するHungarian
6. 5に短軌跡除去・平滑化・安全なdivision候補を追加

検出TTAと低閾値candidate抽出はholdout系列ごとに1回だけ行い、その結果を6方式で共有します。
結果は `outputs/method_search/method_search_results.json`、最良方式だけは
`outputs/method_search/best_method.json` で確認できます。途中停止後も学習checkpointとraw candidateを再利用します。

## GPU対応OT後処理比較

[OPTIMAL_TRANSPORT_MATCHING_PROPOSAL.md](OPTIMAL_TRANSPORT_MATCHING_PROPOSAL.md) の
4方式 `uot_distance`、`uot_hybrid`、`uot_hybrid_division`、
`uot_consensus_ilp` は既存6方式と同じraw prediction・同じ20系列holdoutで比較します。
学習済みcheckpointとraw cacheを再利用し、再学習せず実行するコマンドは次です。

```powershell
.\scripts\run_local_ot_postprocess_search.ps1 -DryRun
.\scripts\run_local_ot_postprocess_search.ps1
```

`ot_device: auto` はCUDAが利用可能なら、物理距離行列、candidate gating、
cost構築、log-domain Unbalanced Sinkhorn、threshold/top-k抽出をGPUで実行します。
Hungarian/ILP、グラフ整形、GEFF書込み、公式metric評価はCPU処理です。
各系列で実際に使ったdeviceとSinkhorn時間はvariant JSONの
`diagnostics.<dataset>.ot_device`、`gpu_accelerated`、`tensor_seconds`で確認できます。

既存checkpointだけでtracking方式を先に比較する場合は、学習を省略できます。

```powershell
.\scripts\run_local_method_search.ps1 `
  -Checkpoint "<weights>\edge_predictor_best.pth"
```

checkpointと同じフォルダに対応する `config.json` が必要です。旧16系列checkpointでもコード確認はできますが、最終判断には179系列で学習したcheckpointを使用してください。

## 0. 実学習・保持評価・Kaggle test推論

`local_demo.py`は固定ルールを確認する小領域デモであり、ニューラルネットの学習ではありません。
実データで学習から評価まで行う場合は次を使用します。

```powershell
# 数分で配線だけ確認する実学習。値は本番精度として使用しない
uv run python train_and_evaluate.py --profile smoke

# 公式trainを固定seedで90%学習・10%保持検証し、3 epoch学習して公式指標を計算
uv run python train_and_evaluate.py --profile full --epochs 3

# 上記に加えてKaggle testを推論しsubmission.csvを生成
uv run python train_and_evaluate.py --profile full --epochs 3 --predict-test
```

結果は`outputs/training/<method>/`に保存されます。

- `dataset_splits.json`: 重複のない学習・保持検証系列
- `run_config.json`: epoch、モデル、閾値、GPUなどの実行条件
- `metrics.json`: 保持検証GEFFに対する公式指標とTP/FP/FN
- `run_manifest.json`: checkpoint SHA-256、環境、所要時間
- `kaggle_test/submission.csv`: `--predict-test`指定時の提出候補

`metrics.json`は保持trainデータに対するCV値で、Kaggle Leaderboard値ではありません。
testの正解は配布されないため、Public/Private Leaderboard値は生成したCSVをKaggleへ提出した後にのみ確定します。

学習済みcheckpointを固定して、ローカル保持系列上で検出・edge閾値を探索できます。

```powershell
uv run python local_parameter_search.py `
  --checkpoint "<weights>/edge_predictor_best.pth" `
  --splits "outputs/training/<method>/dataset_splits.json" `
  --det-thresholds 0.95,0.97,0.98,0.99,0.995 `
  --edge-thresholds 0.3,0.5,0.7
```

`outputs/parameter_search/search_results.json`に全候補のTP/FP/FNと順位を保存します。

## ローカル・ハイパーパラメータ探索

`hyperparameter_search.py`は、取得済みの公式train画像と正解GEFFを使い、学習から保持検証までをローカルで比較します。Kaggleへの送信やKaggle Notebookの起動は行いません。また、スクリプトを明示的に実行するまで学習は開始されません。

既定設定では、全候補を同一seed・同一の16学習系列／4保持系列で比較します。

1. 学習率、検出loss重み、negative重みの6学習候補を各10 epoch学習（合計最大60 epoch）
2. 固定推論条件の実測保持スコアで上位2 checkpointを選択
3. 上位checkpointについて、検出閾値5種 × edge閾値3種 × pool径2種を探索
4. 上位3条件のみILPあり／なしを比較
5. 公式評価コードが計算した最良条件をJSONへ保存

探索条件は [configs/local_hyperparameter_search.yaml](configs/local_hyperparameter_search.yaml) で変更できます。実行前に計画だけを確認するには次を使います。これは学習も推論も行いません。

```powershell
.\scripts\run_local_hyperparameter_search.ps1 -DryRun
```

実際の探索を開始するときだけ、次を手動実行してください。フォアグラウンド実行なので、`Ctrl+C`で中止できます。

```powershell
.\scripts\run_local_hyperparameter_search.ps1
```

PowerShellラッパーを使わない場合は次と同じです。

```powershell
uv run --offline python hyperparameter_search.py `
  --config configs/local_hyperparameter_search.yaml
```

結果は`outputs/hyperparameter_search/`へ保存されます。

- `search_plan.json`: 候補数と探索予定
- `dataset_splits.json`: 全候補で共用する固定学習／保持分割
- `training/`: 学習候補ごとの完了記録またはエラー記録
- `inference/`: 推論候補ごとの実測指標
- `search_results.json`: 全結果と順位
- `best_hyperparameters.json`: 実測上位の学習・推論条件、checkpoint、保持指標

完了済み候補は署名とcheckpointが一致すると再利用されるため、中断後に同じコマンドで再開できます。ここで得る値はローカル保持データのスコアであり、Kaggle Leaderboardスコアではありません。

10 epochは候補を絞るためのローカル粗探索です。最終的に採用した条件をKaggle用checkpointとして確定するときは、その条件だけを50 epoch前後で再学習してください。

`outputs/biohub_strategy_2026-08-09` の設計を、現PC向け実データデモとKaggle向け推論コードとして実装したプロジェクトです。

## 取得済み公式データ

既定では次の完全検証済みデータを自動検出します。

```text
biohub_top10_download/data/kagglehub_cache/competitions/
└── biohub-cell-tracking-during-development/
    ├── train/                 # 199 .zarr + 199 正解 .geff
    ├── test/                  # 4 .zarr
    └── sample_submission.csv
```

別の場所を使う場合は、コンペティション直下を環境変数`BIOHUB_DATA_ROOT`または設定の`data_root`で指定できます。

## 1. 現PC向けデモ

`configs/local_demo.yaml`の既定値は`real-mini`です。公式学習データ`44b6_0113de3b`から4時点・`16×128×128`の小領域だけを読み、古典検出、物理距離候補、ILP、gap closing、division、平滑化、正解GEFF評価、CSV検証を実行します。全81 GiBをメモリへ読み込むことはありません。

```powershell
uv sync
uv run python local_demo.py
```

または次を使用できます。

```powershell
uv run biohub-local --config configs/local_demo.yaml
```

成果物は`outputs/local_demo/`に保存されます。

- `preview.png`: 実画像の最大値投影と正解・予測点
- `submission.csv`: 元データ座標へ戻した予測
- `metrics.json`: 正解GEFFとの比較
- `data_validation.json`: 公式trainデータの全199系列メタデータ
- `run_manifest.json`: 入力パス、クロップ、環境、実行統計

データ非依存の回帰確認には合成デモを使用します。

```powershell
uv run python local_demo.py --config configs/local_synthetic_demo.yaml
```

## 2. Kaggle向け推論

`configs/kaggle_inference.yaml`は`input_dir: auto`です。以下を順に探索します。

1. `BIOHUB_DATA_ROOT`
2. 現PCの上記`kagglehub_cache`
3. `/kaggle/input/biohub-cell-tracking-during-development`
4. `/kaggle/input/competitions/biohub-cell-tracking-during-development`

現PCで入力4系列と形状だけを検査できます。推論重みがなくてもこの検査は実行可能です。

```powershell
uv run python kaggle_inference.py --check-config
```

Kaggleではwheel、設定、学習済みcheckpointを入力Datasetとして添付して実行します。

```bash
python -m pip install --no-index --no-deps /kaggle/input/<bundle>/biohub-0.1.0-py3-none-any.whl
python -m biohub_demo.kaggle --config /kaggle/input/<bundle>/configs/kaggle_inference.yaml
```

設定済みパスと異なる重みを使う場合は`--weight`を繰り返し指定できます。

```bash
python -m biohub_demo.kaggle \
  --config /kaggle/input/<bundle>/configs/kaggle_inference.yaml \
  --weight /kaggle/input/<weights>/fold0.pth \
  --weight /kaggle/input/<weights>/fold1.pth
```

ローカルの限定動作確認には`--input-dir`、`--output-dir`、`--max-datasets 1`を使用できます。ただし、本PCには学習済みcheckpointがないため、実推論にはcheckpointの配置が必要です。

2 GPUで分割する場合は、GPUごとに別プロセスと別出力先を使用します。

```bash
python -m biohub_demo.kaggle --shard-index 0 --shard-count 2 --output-dir /kaggle/working/run0
python -m biohub_demo.kaggle --shard-index 1 --shard-count 2 --output-dir /kaggle/working/run1
```

## 3. モデル

`TemporalUNet3D`は連続2時点の3D画像特徴を抽出し、`SimpleNodeTransformer`が時点間ノード候補を評価します。実データでノード数が増えた場合に備え、pair MLPはsource・target両方向をチャンク処理し、一時テンソルのピークを抑えています。

## 4. Kaggle bundle

```powershell
scripts/build_kaggle_bundle.ps1
```

生成されたwheel、`configs/kaggle_inference.yaml`、checkpointとモデル設定JSONをKaggle Datasetへ添付してください。本番NotebookではInternetを無効にできます。

## 5. 検証

```powershell
uv run pytest -q
uv run python kaggle_inference.py --check-config
```

Kaggleのテスト正解ラベルは配布されていません。ローカル評価は`train/*.geff`、提出生成は`test/*.zarr`を使用します。
