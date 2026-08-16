# Biohub細胞追跡：最適輸送マッチング改良案

## 1. 目的

現在の手法探索では、連続フレーム間の細胞対応をGreedy、Hungarian、ILPで決定している。本提案では、各細胞候補間の対応を0/1で直ちに決めず、まず「どの対応へどれだけの確率質量を輸送するか」を最適輸送（Optimal Transport; OT）で求める。

主目的は次の4点である。

- 細胞が密集し、距離だけでは対応が曖昧な場合に周辺候補を同時に比較する
- Transformerのedge確率と物理距離を一つのコストとして統合する
- 新規出現、消失、検出漏れ、偽陽性による細胞数の不一致を扱う
- 通常の1対1対応と、親1個から娘2個へのdivisionを同じ候補グラフ上で扱う

これはKaggle公式評価式を変更するものではない。最終順位は従来どおり、固定holdoutに対する次の公式スコアで決める。

\[
\text{Score}
=
\text{Adjusted Edge Jaccard}
+0.1\times\text{Division Jaccard}
\]

## 2. 推奨する方式

最初に実装すべき方式は、次の構成である。

> **Division-aware Unbalanced Sinkhorn + ILP projection**

処理の流れは次のとおり。

```text
Temporal U-Net + Node Transformer
  ↓
各時刻のnode候補、検出確率、edge確率
  ↓
物理距離によるcandidate gating
  ↓
Unbalanced Sinkhornでsoft transport planを計算
  ↓
通常edge候補とdivision edge候補へ変換
  ↓
ILPで次数・系譜整合性を満たす離散グラフへ射影
  ↓
短軌跡除去・平滑化
  ↓
GEFF出力・Kaggle公式評価
```

Balanced OTだけでは、前後フレームで細胞数が異なる場合や、細胞の出現・消失を自然に扱いにくい。このデータでは検出漏れ、偽陽性、画面外への移動、divisionがあるため、周辺質量の保存を緩和できるUnbalanced OTを基本とする。

## 3. 入力

時刻 \(t\) の細胞候補を \(X_t=\{x_i\}_{i=1}^{N}\)、時刻 \(t+1\) の候補を \(X_{t+1}=\{y_j\}_{j=1}^{M}\) とする。

現在のraw inferenceから次を再利用する。

- node座標：\((z,y,x)\)
- node検出確率：\(q_i, q_j\)
- Transformer edge確率：\(p_{ij}\)
- voxel scale：\((1.625, 0.40625, 0.40625)\,\mu m\)
- 物理距離：\(d_{ij}\)

初期実装では新しいニューラルネットワークは不要である。現在保存しているraw node・edge predictionからOT追跡だけを追加できる。

## 4. Candidate gating

全node間に輸送を許可すると計算量が増え、遠距離の誤対応も生まれる。そこで、次の条件を満たす組だけを候補とする。

\[
d_{ij}\le d_{\max}
\]

初期値は現在の探索と同じ \(d_{\max}=10\,\mu m\) とする。公式評価時のnode matching距離 \(7\,\mu m\) と、追跡候補を作る際の最大移動距離は別の値である。

Transformer候補が存在する方式では、さらに次を適用する。

\[
p_{ij}\ge p_{\min}
\]

初期値は現在と同じ \(p_{\min}=0.01\) とし、OTへ渡す前には候補を広めに残す。

## 5. 輸送コスト

各候補edgeのコストを次で定義する。

\[
C_{ij}
=
\lambda_d C^{dist}_{ij}
+\lambda_p C^{prob}_{ij}
+\lambda_q C^{det}_{ij}
+\lambda_v C^{motion}_{ij}
\]

### 5.1 物理距離コスト

\[
C^{dist}_{ij}
=
\left(\frac{d_{ij}}{d_{\max}}\right)^2
\]

Z/Y/Xのvoxel数ではなく、voxel scaleを掛けた物理距離を使用する。

### 5.2 Transformer確率コスト

\[
C^{prob}_{ij}
=
-\log(p_{ij}+\delta)
\]

\(\delta\) はゼロ除算防止用の小さい値で、初期値を \(10^{-6}\) とする。Transformerを使わない距離OTでは、この項の重みを0にする。

### 5.3 検出信頼度コスト

\[
C^{det}_{ij}
=
-\frac{1}{2}\left[\log(q_i+\delta)+\log(q_j+\delta)\right]
\]

低信頼node同士のedgeが選ばれにくくなる。

### 5.4 運動予測コスト（第2段階）

直前2フレームから速度 \(v_i\) を推定できる場合、予測位置との残差を使う。

\[
C^{motion}_{ij}
=
\left(
\frac{\lVert y_j-(x_i+v_i)\rVert}{d_{\max}}
\right)^2
\]

初期実装では \(\lambda_v=0\) とし、OT単体の効果を確認してから追加する。

## 6. Unbalanced Optimal Transport

時刻 \(t\) と \(t+1\) のnode質量を次で表す。

\[
a_i=\operatorname{clip}(q_i,q_{min},1),\qquad
b_j=\operatorname{clip}(q_j,q_{min},1)
\]

輸送行列 \(P\in\mathbb{R}_{+}^{N\times M}\) を、次の目的関数で求める。

\[
\min_{P\ge0}
\langle C,P\rangle
+\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1)
+\tau_s\operatorname{KL}(P\mathbf{1}\,\|\,a)
+\tau_t\operatorname{KL}(P^T\mathbf{1}\,\|\,b)
\]

- \(\langle C,P\rangle\)：対応コスト
- \(\varepsilon\)：entropy正則化。大きいほど対応が滑らかになる
- \(\tau_s,\tau_t\)：質量保存をどれだけ強制するか
- 行方向の余り：親細胞の消失、検出漏れ候補
- 列方向の不足：娘細胞の出現、偽陰性候補

Sinkhorn反復はGPU上の行列演算で実装できる。数値安定性のため、実装はlog-domain Sinkhornを使用する。

## 7. Dustbinとの比較

もう一つの方式は、行列に「対応なし」を表すdustbin行・列を追加するBalanced OTである。

```text
通常の行・列：実在node間の対応
最終列：source nodeの消失
最終行：target nodeの新規出現
```

この方式は対応なしを明示的に表現できる。一方、dustbin costの調整に敏感であり、divisionでは追加設計が必要になる。本提案では次の順で比較する。

1. Unbalanced OT
2. Dustbin付きBalanced OT
3. 両者を公式スコアで比較

## 8. Soft transport planからedgeへの変換

Sinkhornの出力 \(P_{ij}\) はsoft assignmentであり、そのままGEFF edgeにはできない。次の順序で離散化する。

### 8.1 候補抽出

- \(P_{ij}\ge\theta_{ot}\) を満たす
- sourceごとに上位 \(k_s\) 候補を残す
- targetごとに上位 \(k_t\) 候補を残す
- \(d_{ij}\le d_{\max}\) を再確認する

初期値：

```yaml
ot_threshold: 0.05
top_k_source: 2
top_k_target: 3
```

### 8.2 通常edge

通常edgeでは次の制約を課す。

\[
\operatorname{indegree}(j)\le1
\]

\[
\operatorname{outdegree}(i)\le1
\]

相互最大、すなわちsource側とtarget側の双方で最良になった対応は、高信頼edgeとして優先する。

## 9. Division-aware OT

標準的な1対1 OTだけでは、親1個から娘2個へのdivisionを正しく表しにくい。単に輸送質量が2か所へ分かれただけでは、曖昧な1対多対応と生物学的divisionを区別できない。

そこで各親nodeに、通常スロットとdivision用の第2スロットを用意する。

```text
parent i / primary slot   → daughter j1
parent i / division slot  → daughter j2
```

第2スロットには追加コストを与える。

\[
C^{div}_{ij}=C_{ij}+\lambda_{div}-\lambda_{prior}\log(p^{div}_i+\delta)
\]

現モデルに専用division headはないため、初期実装では \(p^{div}_i\) を次から作る。

- Transformerが2本の高確率edgeを出している
- 親娘距離が上限以内
- 娘同士の距離が上限以内
- 2本の輸送確率がともに閾値以上

現在の安全条件を初期値として再利用する。

```yaml
division_parent_child_um: 7.5
division_child_child_um: 8.5
division_min_probability: 0.20
```

## 10. ILPによる最終射影

OTはsoft assignmentを得る部分に使用し、最終グラフの整合性はILPで保証する。

edge選択変数を \(z_{ij}\in\{0,1\}\) とし、OT輸送量を含むedge costを次とする。

\[
E_{ij}
=
-w_{ot}\log(P_{ij}+\delta)
-w_p\log(p_{ij}+\delta)
+w_d C^{dist}_{ij}
-w_vote V_{ij}
\]

ここで \(V_{ij}\) は既存6方式のうち何方式がedgeを採用したかを表すensemble投票値である。

ILPでは次を制約する。

- targetの親は最大1個
- sourceの子は通常最大1個、division時のみ最大2個
- 3個以上の娘は禁止
- 時刻を逆行するedgeは禁止
- appearance/disappearanceを許可
- division用第2edgeには追加penaltyを課す

これにより「OTによる全体的なsoft matching」と「ILPによる有効な系譜グラフ」を分担できる。

## 11. 手法探索へ追加する4候補

既存6方式に、次の4方式を追加して比較する。

### 11.1 `uot_distance`

- コスト：物理距離＋検出信頼度
- Transformer edge確率を使用しない
- Unbalanced Sinkhorn後に1対1へ離散化
- Hungarianに対するOT単体の効果を確認する対照群

### 11.2 `uot_hybrid`

- コスト：物理距離＋検出信頼度＋Transformer edge確率
- Unbalanced Sinkhorn後に1対1へ離散化
- 現在の`hybrid_hungarian`との直接比較候補

### 11.3 `uot_hybrid_division`

- `uot_hybrid`にdivision用第2親スロットを追加
- 親娘・娘間距離と確率条件を使用
- Division Jaccard改善を狙う

### 11.4 `uot_consensus_ilp`

- OT輸送量
- Transformer edge確率
- 距離スコア
- 既存6方式のedge投票

を統合し、最後にILPで離散グラフを作る。精度上の本命候補だが、他方式より計算量と調整項目が多い。

追加後の比較数は次のとおり。

```text
既存方式：6
OT方式：4
合計：10方式
```

## 12. 初期パラメータ案

以下は細かいハイパーパラメータ探索前の開始点であり、最終値ではない。

```yaml
- name: uot_hybrid
  tracker: unbalanced_ot
  detection_threshold: 0.95
  edge_threshold: 0.01
  max_distance_um: 10.0
  distance_weight: 1.0
  transformer_weight: 1.0
  detection_weight: 0.25
  motion_weight: 0.0
  entropy_epsilon: 0.05
  source_mass_penalty: 0.5
  target_mass_penalty: 0.5
  sinkhorn_iterations: 100
  ot_threshold: 0.05
  top_k_source: 2
  top_k_target: 3
  linked_nodes_only: true
```

値のスケールが異なるため、距離・確率コストはholdoutから学習する前に中央値またはrobust scaleで正規化する。

## 13. 評価計画

方式選択では、全方式が同じcheckpoint、同じraw predictions、同じ固定holdout 20系列を使用する。

記録する値：

- 公式combined score
- Adjusted Edge Jaccard
- Division Jaccard
- edge TP / FP / FN
- division TP / FP / FN
- node recall
- 1系列あたりのOT反復時間
- 候補edge数と採用edge数
- appearance / disappearance数
- OT輸送量のentropy

最初は上記4方式を粗い固定値で比較する。OT方式が既存最高方式を上回った後に限り、\(\varepsilon\)、質量penalty、各cost weight、OT閾値を細かく探索する。

OTパラメータを最終holdout 20系列へ直接合わせ続けると過適合するため、重み調整には学習179系列内のinner splitまたはOOF予測を使用し、20系列は最終方式選択用として保持する。

## 14. 計算量と実装上の注意

フレーム間のsource数を \(N\)、target数を \(M\)、Sinkhorn反復数を \(K\) とすると、dense実装の概算計算量は次である。

\[
O(KNM)
\]

対策：

- 10 µm gatingで遠距離edgeを除外する
- nodeを空間blockに分割する
- log-domainで計算してunderflowを防ぐ
- costが禁止値の箇所は大きな有限値ではなくmaskする
- 小規模行列はCPU、大規模行列はGPUで処理する
- 同じraw inferenceを全方式で共有する

entropy正則化が大きすぎると輸送が広がり、偽edgeが増える。小さすぎるとSinkhornが不安定になり、Hungarianに近い硬い対応になる。したがって、輸送entropyと公式edge FP/FNを同時に記録する。

## 15. 期待される利点と失敗条件

期待される利点：

- 密集領域で局所Greedyより全体整合的な対応を得られる
- 前後フレームのnode数が違っても対応できる
- Transformer確率と距離を確率的に統合できる
- soft assignmentをdivision候補の抽出に使える
- 既存方式のedge votingを同じコストへ統合できる

失敗しやすい条件：

- entropyが大きく、輸送量が多数の候補へ拡散する
- 検出確率が未校正で、質量設定が不適切になる
- divisionと単なる曖昧な1対多対応を混同する
- 高密度領域でdense cost matrixが大きくなる
- 同じ20系列でOT重みを過剰に調整する

## 16. 実装優先順位

1. `uot_distance`を実装し、Hungarianとの差だけを検証する
2. Transformer確率を加えた`uot_hybrid`を比較する
3. division第2スロットを追加する
4. OT結果をILPへ入力する
5. 既存6方式の投票を追加する
6. OOFでcost weightを調整する

この順序なら、どの追加要素が公式スコアへ寄与したかを切り分けられる。

## 17. 比較候補の拡張分類

以下では、2026年8月時点で考えられる比較を「シンプル」「複雑」「最新」「独創的」に分ける。新しい手法ほど高精度とは限らないため、各群には必ず既存Hungarian・ILPを対照として含める。

### 17.0 実装済みbaseline群：現在のプログラムで比較中

次の6方式は提案段階ではなく、`configs/local_method_search.yaml`と`src/biohub_demo/tracking_variants.py`に実装済みである。すべて同じTemporal 3D U-Net + Node Transformer checkpointと同じraw inferenceを共有する。

#### B0. `transformer_greedy`

- tracker：Greedy
- Transformer edge確率：使用
- detection threshold：0.95
- edge threshold：0.03
- strong edge threshold：0.30
- 最大親候補：targetあたり3
- 最大距離：10 µm
- 高確率edgeから順に採用
- targetの親は最大1、sourceの子は最大2

#### B1. `transformer_ilp`

- tracker：ILP forest solver
- Transformer edge確率：使用
- detection threshold：0.95
- edge threshold：0.03
- strong edge threshold：0.30
- 最大距離：10 µm
- appearance cost：0.1
- disappearance cost：0.1
- division cost：1.0
- グラフ全体の制約下でedgeを選択

#### B2. `distance_hungarian_top_style`

- tracker：2-pass Hungarian
- Transformer edge確率：不使用
- detection threshold：0.95
- tight gate：6 µm
- 最大距離：10 µm
- 第1passで近距離対応、第2passで未対応nodeを補完
- 短軌跡除去・平滑化なし

#### B3. `distance_hungarian_filtered`

- tracker：2-pass Hungarian
- Transformer edge確率：不使用
- tight gate：6 µm
- 最大距離：10 µm
- 4 node未満の短いcomponentを除去
- smoothing weight：0.15
- 最大平滑化移動：1.5 µm

#### B4. `hybrid_hungarian`

- tracker：2-pass Hungarian
- 物理距離とTransformer edge確率を併用
- detection threshold：0.95
- edge threshold：0.03
- tight gate：6 µm
- 最大距離：10 µm
- probability weight：4.0
- 短軌跡除去・平滑化なし

#### B5. `hybrid_repair_division`

- tracker：Hybrid Hungarian
- 物理距離とTransformer edge確率を併用
- 4 node未満の短いcomponentを除去
- smoothing weight：0.15
- safe division repair：有効
- division minimum probability：0.20
- 通常trackの後に安全条件を満たす2本目の娘edgeを追加

実装済み6方式の比較軸：

| ID | 方式 | Transformer確率 | 全体solver | 短軌跡除去 | 平滑化 | division修復 |
|---|---|---:|---|---:|---:|---:|
| B0 | Transformer Greedy | 使用 | Greedy | なし | なし | Greedy内で最大2子 |
| B1 | Transformer ILP | 使用 | ILP | なし | なし | ILP制約 |
| B2 | Distance Hungarian | 不使用 | Hungarian | なし | なし | なし |
| B3 | Filtered Distance Hungarian | 不使用 | Hungarian | 4 node未満 | あり | なし |
| B4 | Hybrid Hungarian | 使用 | Hungarian | なし | なし | なし |
| B5 | Hybrid + Division Repair | 使用 | Hungarian | 4 node未満 | あり | あり |

これら6方式が、以降のOT方式に対する必須baselineになる。

### 17.1 シンプル群：同じraw predictionだけで比較可能

#### S0. `balanced_sinkhorn_dustbin`

- 通常のentropy正則化OT
- appearance/disappearanceをdustbin行・列で表現
- 距離とTransformer確率だけをコストに使用
- 最終対応は相互最大または閾値で離散化
- 目的：Hungarianをsoftな全体割当へ置換するだけで改善するか

#### S1. `partial_ot_hybrid`

- 全質量を対応させず、高信頼な一定割合だけを輸送
- 残りはappearance/disappearanceとして扱う
- 目的：低信頼nodeを無理に接続しないことがFP削減に効くか

#### S2. `uot_distance`

- 距離と検出信頼度だけを使用するUnbalanced OT
- Transformer確率を使用しない
- 目的：OT自体の効果を距離Hungarianと比較する

#### S3. `uot_hybrid`

- 距離、検出信頼度、Transformer edge確率を統合
- 追加学習なし
- 目的：`hybrid_hungarian`をsoft mass matchingへ変える価値があるか

#### S4. `uot_mutual_top`

- `uot_hybrid`の輸送行列から相互top-1だけを確定
- 未確定nodeを第2passで補完
- 目的：diffuseな輸送計画を最小限の規則で安全に離散化できるか

シンプル群は現在のraw `.npz`を再利用できるため、比較コストが最も低い。

### 17.2 複雑群：時系列・構造制約を追加

#### C0. `uot_hybrid_division`

- 親ごとにprimary slotとdivision slotを用意
- 1対1対応と1対2対応を同時に候補化
- 目的：Edge Jaccardを維持しながらDivision Jaccardを改善できるか

#### C1. `uot_consensus_ilp`

- OT輸送量、Transformer確率、距離、既存方式の投票を統合
- 最終決定はILP
- 目的：softな候補評価と厳密な系譜制約を両立できるか

#### C2. `fused_unbalanced_gw`

- node間の直接コストに加え、各フレーム内の近傍構造も比較
- Fused Unbalanced Gromov-Wassersteinを使用
- 細胞密集領域で、個々の細胞が似ていても周辺配置の保存を利用
- 目的：点単位の距離・特徴だけでなく局所組織構造が識別に効くか

#### C3. `three_frame_mmot`

- \(t-1,t,t+1\) の3フレームをMulti-Marginal OTで同時対応
- 速度または加速度の急変にpenaltyを付与
- 目的：フレームごとの独立matchingよりidentity switchを減らせるか

#### C4. `windowed_dynamic_ot`

- 5〜10フレームの短いwindow全体で輸送を最適化
- windowの重なり部分で軌跡IDを接続
- 目的：長期整合性とローカル計算量のバランスを調べる

#### C5. `optimal_flow_transport`

- 二部matchingではなく、時間方向DAG全体にflow balance制約を設定
- appearance、disappearance、divisionをflowの生成・消滅として扱う
- 目的：pairwise OT後のILPではなく、最初から全時系列flowとして解く価値があるか

### 17.3 最新群：2025〜2026年の研究方向を取り込む

ここでの「最新」は発表年を指し、本コンペでの有効性を保証しない。異なる応用分野のアイデアは、必ず単純baselineとのablationで検証する。

#### N0. `mesh_uot_hybrid`

- Sinkhorn後の輸送entropyが小さくなるようコスト行列を反復更新
- 曖昧なsoft transportを、より疎で解釈可能な対応へ集中させる
- 2025年のOT-MESHによる細胞型matchingを、時系列細胞matchingへ転用
- 追加モデル学習なしでも比較可能

#### N1. `oft_sinkhorn_lineage`

- ICLR 2025のOptimal Flow Transport型のflow balanceを時間グラフへ適用
- 通常OTのsource/target marginal制約を、系譜グラフのflow保存制約へ置換
- division箇所だけflow増加を許可
- `optimal_flow_transport`のGPU向けSinkhorn版

#### N2. `ot_localization_loss`

- ICLR 2026のOT set-matching lossの考え方を検出head学習へ導入
- voxel単位BCEだけでなく、予測node集合とGT node集合の輸送距離をlossに追加
- NMS前の検出分布をend-to-endで学習
- 新しいモデル学習が必要

#### N3. `ot_triplet_edge_embedding`

- 2025年のOTベースmulti-object trackingで使われる、OTを組み込んだ特徴距離学習を参考にする
- 同一細胞edgeを近づけ、異なる細胞候補を離すembedding lossを追加
- 最終matchingはUOTまたはHungarianで比較
- 新しいappearance embedding学習が必要

#### N4. `structurally_constrained_dynamic_ot`

- 2026年の時系列single-cell解析に見られる構造制約付きOTの考え方を使用
- 近傍構造、時系列連続性、familyごとの運動統計を制約として追加
- 画像上の同一細胞追跡へ直接適用された手法ではないため、研究的比較枠とする

#### N5. `parallel_time_sinkhorn`

- 2026年のparallel-in-time Sinkhornの考え方で複数フレーム対を並列に解く
- matching精度を変える方式ではなく、Dynamic OTの実行時間を短縮する方式
- 精度比較ではなく、同一解に対する速度・GPU使用量比較とする

### 17.4 独創的群：本コンペ固有の仮説

以下は独創性候補であり、既存研究との差分調査を完了するまでは「新規手法」と断定しない。

#### O0. `metric_aware_ot`

- OT cost weightをedge accuracyではなく、公式Adjusted Edge Jaccardの近似値で最適化
- FP、FN、過剰node数に異なるpenaltyを設定
- Division Jaccard近似項も追加
- 目的：学習・matching目的とKaggle評価値のずれを減らす

#### O1. `reaction_division_ot`

- divisionを第2edge追加ではなく、親質量が2個の娘質量へ反応・分裂する輸送として定式化
- 通常のmass creationとは別のdivision reaction costを持つ
- 目的：appearanceとdivisionを明示的に区別する

#### O2. `forward_backward_cycle_ot`

- \(t\rightarrow t+1\) と \(t+1\rightarrow t\) の輸送を両方計算
- 往復対応が一致しないedgeにpenalty
- divisionだけは1対2と2対1の対応規則でcycle consistencyを判定
- 目的：一方向だけで生じる曖昧なedgeを除去する

#### O3. `family_conditioned_ot`

- `44b6`と`6bba`で距離、mass penalty、entropy、division costを変える
- family名だけでなく、細胞密度・移動量・輝度統計からgateを選択可能
- 目的：異なるデータ分布に単一コストを強制しない

#### O4. `uncertainty_barycenter_ensemble`

- 複数seed・checkpoint・tracking方式から複数の輸送行列を生成
- OT barycenterまたは重み付きconsensus planを作る
- 方式間で一致しないedgeは低信頼としてILPへ渡す
- 目的：モデルとtracker両方の不確実性を利用する

#### O5. `counterfactual_division_ot`

- divisionあり／なしの2つの局所グラフを作る
- 各仮説で次の2〜3フレームの輸送コストを再計算
- 将来フレームまで整合する仮説だけを採用
- 目的：単一フレーム対では判断しにくいdivisionを将来整合性で決める

### 17.5 アンサンブル群：モデル・tracker・OT planの統合

アンサンブルは完成後のGEFFだけでなく、できるだけ離散化前の共通node・edge候補上で行う。完成グラフでは方式ごとにnode削除、座標平滑化、ID付与が異なり、単純投票の前にnode alignmentが必要になるためである。

#### E0. `current6_hard_edge_vote`

- 現在のB0〜B5が採用したedgeを同じraw node ID上で投票
- 6方式中3方式以上が採用したedgeを残す
- 同票時はTransformer確率、次に物理距離で決定
- 最も単純なensemble baseline

#### E1. `current6_weighted_edge_vote`

- B0〜B5のOOF公式スコアから方式重みを決める
- edgeごとに採用方式の重みを合計
- edge weightが閾値以上の場合だけ採用
- 重みは最終holdoutではなくinner CVで固定する

\[
S_{vote}(e)=\sum_m w_m\mathbf{1}[e\in G_m]
\]

#### E2. `current6_union_ilp`

- B0〜B5が提案したedgeの和集合を作る
- 投票数、Transformer確率、距離をILP costへ変換
- indegree、outdegree、division制約下で最終edgeを選択
- OTを使わないensemble対照群

#### E3. `hungarian_uot_consensus`

- `hybrid_hungarian`と`uot_hybrid`の輸送結果を統合
- 両方式一致edgeを固定
- 片方だけが提案したedgeをILPまたは第2passへ渡す
- hard assignmentとsoft assignmentの相補性を検証

#### E4. `precision_recall_cascade`

- 高precision方式を第1段として確定
- 未接続nodeだけ高recall方式で補完
- 初期候補：`transformer_ilp`でcore edgeを作り、`uot_hybrid`または距離Hungarianでorphanを接続
- unionによるFP増加を避ける段階型ensemble

#### E5. `division_specialist_ensemble`

- 通常edgeはEdge Jaccardが最も高い方式から取得
- division候補だけ`hybrid_repair_division`、`uot_hybrid_division`、`counterfactual_division_ot`から投票
- 通常追跡とdivision追跡のexpertを分離
- combined scoreの期待増分が正の場合だけdivision edgeを追加

#### E6. `multi_seed_probability_ensemble`

- 同一モデルを複数seedまたはfoldで学習
- node検出logitとedge logitを平均または幾何平均
- 平均raw predictionへ同じtrackerを適用
- tracker ensembleではなくニューラルモデル不確実性を低減する対照群

#### E7. `checkpoint_tracker_cross_ensemble`

- 複数checkpoint × 複数trackerの直積からgraph候補を作る
- 例：3 checkpoint × 4 tracker = 12 graph
- edgeごとのモデル一致度とtracker一致度を別々の特徴として保持
- model diversityとsolver diversityの寄与を分離する

#### E8. `family_conditioned_mixture_of_experts`

- `44b6`と`6bba`でensemble構成と重みを変える
- family内でも密度、速度、edge entropyからrouterが重みを出す
- graphを1個選ぶhard routingと、edge重みを混ぜるsoft routingを比較
- train OOFだけでrouterを学習する

#### E9. `stacked_edge_meta_model`

各candidate edgeについて次を特徴とする小型meta modelを学習する。

- B0〜B5の採否6値
- UOT輸送確率
- Transformer edge確率
- 物理距離
- source/target検出確率
- forward/backward一致
- family、時刻、局所密度

出力は最終edge確率とし、最後にILPでグラフ制約を適用する。Logistic Regressionを最小baselineとし、LightGBMまたは小型MLPと比較する。

#### E10. `uncertainty_abstention_ensemble`

- 方式間のedge採否entropyを不確実性とする
- 高一致edgeは自動確定
- 高不一致edgeは接続しない、または将来3フレームで再判定
- recallよりFP抑制が有利な領域をOOFで特定
- 過剰接続を避けるconservative ensemble

アンサンブル群の比較表：

| ID | 統合対象 | 追加学習 | 最終solver | 主な狙い |
|---|---|---:|---|---|
| E0 | 現行6 tracker | 不要 | Vote/Greedy | 最小多数決baseline |
| E1 | 現行6 tracker | 重みのみ | Weighted vote | 方式信頼度の反映 |
| E2 | 現行6 edge union | 不要 | ILP | 投票と系譜制約 |
| E3 | Hungarian + UOT | 不要 | ILP/2-pass | hard/soft対応統合 |
| E4 | precision + recall expert | 不要 | Cascade | FPを抑えた補完 |
| E5 | 通常edge + division experts | 場合による | ILP | division専用統合 |
| E6 | 複数seed checkpoint | 必要 | 共通tracker | モデル分散低減 |
| E7 | checkpoint × tracker | 必要 | Vote/ILP | 二種類の多様性 |
| E8 | family別experts | routerのみ | Mixture | データ分布別統合 |
| E9 | 全edge特徴 | 必要 | Meta model + ILP | 学習型stacking |
| E10 | 全方式の不一致 | 不要 | Abstention + repair | 不確実edge抑制 |

## 18. 比較マトリクス

| ID | 群 | 追加学習 | 時間範囲 | division | 実装難度 | 主な比較目的 |
|---|---|---:|---|---|---|---|
| S0 | シンプル | 不要 | 2フレーム | なし | 低 | Hungarian対Sinkhorn |
| S1 | シンプル | 不要 | 2フレーム | なし | 低 | 強制matching対partial matching |
| S2 | シンプル | 不要 | 2フレーム | なし | 低 | 距離Hungarian対距離UOT |
| S3 | シンプル | 不要 | 2フレーム | なし | 低 | Hybrid Hungarian対Hybrid UOT |
| S4 | シンプル | 不要 | 2フレーム | なし | 低〜中 | OT離散化方式 |
| C0 | 複雑 | 不要 | 2フレーム | あり | 中 | division slotの効果 |
| C1 | 複雑 | 不要 | 2フレーム | あり | 中〜高 | OT単独対OT+ILP |
| C2 | 複雑 | 不要 | 2フレーム | 間接 | 高 | 点特徴対近傍構造 |
| C3 | 複雑 | 不要 | 3フレーム | あり | 高 | pairwise対multi-marginal |
| C4 | 複雑 | 不要 | 5〜10 | あり | 高 | 短期対長期整合性 |
| C5 | 複雑 | 不要 | 全時系列 | あり | 非常に高 | pairwise対global flow |
| N0 | 最新 | 不要 | 2フレーム | 任意 | 中 | 通常Sinkhorn対entropy-minimized Sinkhorn |
| N1 | 最新 | 不要 | 全時系列 | あり | 非常に高 | ILP対OFT-Sinkhorn |
| N2 | 最新 | 必要 | 2フレーム | なし | 高 | voxel BCE対OT localization loss |
| N3 | 最新 | 必要 | 2フレーム | 間接 | 高 | edge分類対OT embedding学習 |
| N4 | 最新 | 場合による | 複数 | あり | 非常に高 | unconstrained対structurally constrained OT |
| N5 | 最新 | 不要 | 複数 | 任意 | 非常に高 | 逐次対並列時間Sinkhorn |
| O0 | 独創的 | 場合による | 2フレーム | あり | 高 | 汎用cost対公式metric-aware cost |
| O1 | 独創的 | 不要 | 2フレーム | あり | 非常に高 | division slot対reaction OT |
| O2 | 独創的 | 不要 | 3フレーム | あり | 中〜高 | 一方向対cycle consistency |
| O3 | 独創的 | 場合による | 2フレーム | あり | 中 | 共通cost対family別cost |
| O4 | 独創的 | 複数checkpoint | 2フレーム | あり | 高 | 単独plan対ensemble plan |
| O5 | 独創的 | 不要 | 3〜4フレーム | あり | 高 | 局所division対将来仮説検証 |

## 19. 必須ablation

比較数を増やすだけでは原因が分からなくなるため、次の1要素差比較を固定する。

- Hungarian vs Balanced Sinkhorn：solverだけの差
- Balanced Sinkhorn vs Unbalanced Sinkhorn：mass保存だけの差
- UOT distance vs UOT hybrid：Transformer確率の寄与
- UOT hybrid vs MESH-UOT：entropy最小化の寄与
- UOT hybrid vs UOT hybrid + division：division処理の寄与
- UOT hybrid + division vs UOT + ILP：離散化・系譜制約の寄与
- pairwise UOT vs 3-frame MMOT：時間範囲の寄与
- UOT vs Fused UGW：近傍構造の寄与
- 共通cost vs family-conditioned cost：データfamily適応の寄与
- 単独OT plan vs OT ensemble：不確実性統合の寄与
- BCE detection loss vs BCE + OT localization loss：学習lossの寄与

各比較では、checkpoint、raw node候補、holdout、公式metric contractを固定する。追加学習が必要な比較ではseedと学習データ順も固定する。

## 20. 段階的な比較計画

全候補を一度に探索すると時間と多重比較による過適合が増えるため、次の段階に分ける。

### Round 1：低コストscreening

```text
transformer_greedy
transformer_ilp
distance_hungarian_top_style
distance_hungarian_filtered
hybrid_hungarian
hybrid_repair_division
balanced_sinkhorn_dustbin
partial_ot_hybrid
uot_distance
uot_hybrid
uot_mutual_top
```

追加学習なし。実装済み6方式と簡易OT 5方式の計11方式で、同じraw predictionを使用し、公式スコアと処理時間を比較する。

### Round 1.5：実装済み方式のアンサンブル

Round 1で生成済みのgraphとraw predictionを再利用する。

```text
current6_hard_edge_vote
current6_weighted_edge_vote
current6_union_ilp
hungarian_uot_consensus
precision_recall_cascade
division_specialist_ensemble
uncertainty_abstention_ensemble
```

追加モデル学習が必要なmulti-seed、cross ensemble、stackingはRound 4へ回す。

### Round 2：構造・division

```text
Round 1上位2方式
uot_hybrid_division
uot_consensus_ilp
mesh_uot_hybrid
forward_backward_cycle_ot
family_conditioned_ot
```

Edge JaccardとDivision Jaccardを別々に確認する。combined scoreだけではdivision改善とedge悪化を区別できない。

### Round 3：長時間・高複雑度

```text
Round 2上位2方式
fused_unbalanced_gw
three_frame_mmot
windowed_dynamic_ot
optimal_flow_transport / oft_sinkhorn_lineage
counterfactual_division_ot
```

まずholdoutの一部で計算量を測定し、実行可能な方式だけを全holdoutへ進める。

### Round 4：追加学習

```text
従来checkpoint
OT localization loss checkpoint
OT edge embedding checkpoint
metric-aware OT checkpoint
multi_seed_probability_ensemble
checkpoint_tracker_cross_ensemble
family_conditioned_mixture_of_experts
stacked_edge_meta_model
```

モデル差とmatching差を混同しないよう、各checkpointについて同じtracking方式も評価する。

### 最終確認

- 開発中のcost weight調整：179系列内のinner CVまたはOOF
- 方式選択：未使用の固定holdout 20系列
- Kaggle提出候補：選択後に全199系列で再学習
- 報告値：公式combined score、各構成要素、実行時間、GPUメモリ

## 21. 推奨優先順位

精度、実装時間、比較の明瞭さを合わせた優先順位は次のとおり。

1. `uot_hybrid`
2. `balanced_sinkhorn_dustbin`
3. `partial_ot_hybrid`
4. `uot_hybrid_division`
5. `mesh_uot_hybrid`
6. `uot_consensus_ilp`
7. `forward_backward_cycle_ot`
8. `family_conditioned_ot`
9. `three_frame_mmot`
10. `fused_unbalanced_gw`
11. `ot_localization_loss`
12. `optimal_flow_transport`
13. `metric_aware_ot`
14. `reaction_division_ot`

最初の6方式までは現在のraw inferenceを共有できる。7以降は時系列状態、追加特徴、または追加学習が必要になるため、前段が改善した場合だけ進める。

重複する名称を一方式として数えた比較候補の全体像は次のとおり。

```text
実装済みbaseline：6方式
シンプル・複雑・最新・独創的OT：23方式
コンペ専用最適化：13方式
アンサンブル実験構成：11方式
合計：53方式
```

53方式を同時に実行するのではなく、Round 1の11方式と、その出力を再利用するRound 1.5から段階的に絞り込む。

## 22. コンペ専用最適化群

汎用的な細胞追跡としての再利用性を捨て、Biohub competitionの公式metricだけを最大化する比較も許容する。この群は論文上の汎用性より、データ仕様・annotation方針・評価式への適合を優先する。

### K0. `annotation_propensity_pruning`

このデータのGTは全細胞を密にannotateしたものではなく、subsetのtrack graphである。したがって、「画像内に見える全細胞」を予測するモデルと、「公式GTに含まれやすい細胞」を予測するモデルを分ける。

- 画像上のcellness
- 時刻
- Z位置
- 胚内の相対位置
- 軌跡長
- 速度
- family
- 周辺細胞密度

から、GT graphへ含まれる傾向をOOFで学習し、低propensity trackをsubmissionから除外する。

目的：生物学的な全細胞recallではなく、Adjusted Edge Jaccardと過剰node penaltyを直接改善する。

### K1. `family_expert_44b6_6bba`

`44b6`と`6bba`で完全に別の設定またはcheckpointを使用する。

- detection threshold
- UOT mass penalty
- 最大移動距離
- entropy
- minimum track length
- division threshold
- augmentation

をfamily別に最適化する。単一の汎用モデルよりも、コンペ内の2分布へ特化したexpertを優先する。

### K2. `development_phase_conditioned_ot`

100フレームを一定分布とみなさず、発生段階ごとにOT costを変える。

```text
early frames：低密度・長距離移動を許容
middle frames：通常設定
late frames：高密度・短距離・division prior増加
```

時刻 \(t/99\) をcost gateへ直接入力し、距離scale、appearance/disappearance、division penaltyを時間依存にする。

### K3. `family_time_node_count_prior`

train GTからfamily別・時刻別のannotation node数曲線を推定する。

\[
\hat N(f,t)=\operatorname{median}_{s\in family(f)}N_s(t)
\]

submission graphのnode数がこの曲線から大きく外れる場合、低信頼nodeまたは短軌跡を削減する。公式metricの過剰node penaltyへ特化した処理である。

### K4. `expected_jaccard_edge_selection`

edge確率0.5を固定閾値にせず、候補edgeを追加したときの期待Jaccard変化を近似する。

現在の予測TP・FP・FN推定値を \(\widehat{TP},\widehat{FP},\widehat{FN}\)、候補edgeの正解確率を \(p_e\) とし、追加前後の期待Jaccard差が正の場合だけ採用する。

目的：分類accuracyではなく、公式Jaccardに適したedge数を選ぶ。

### K5. `division_value_gate`

division項の重みは0.1であるため、Division Jaccard改善よりEdge Jaccard悪化が大きいdivision候補は追加しない。

\[
\Delta \widehat{Score}
=
\Delta \widehat{J}_{edge,adj}
+0.1\Delta \widehat{J}_{division}
\]

この期待差が正の候補だけを追加する。division recall最大化ではなく、combined score最大化に特化する。

### K6. `full_movie_transductive_ot`

推論時に100フレーム全体を一度読み、未来情報を使用して各edgeを決める。

- forward OT
- backward OT
- 時刻別node count曲線
- 全movieの速度分布
- 全movieの輝度分布
- 後続3フレームでのdivision整合性

を使う。リアルタイムtrackingを捨て、提出用offline inferenceへ特化する。

### K7. `per_sequence_policy_router`

一つの方式を全test系列へ適用せず、系列特徴からtrackerを選ぶ。

入力特徴：

- family
- 平均・分散・percentile輝度
- frameごとの検出数
- 細胞密度
- median displacement
- edge確率entropy
- division候補率

出力：

```text
greedy / ILP / Hungarian / UOT / UOT+ILP / division-on/off
```

学習はleave-one-sequence-outで行い、同じ系列をrouter学習と評価へ同時使用しない。

### K8. `test_distribution_calibrated_detection`

test画像のラベルを使わず、画像強度・検出score・node密度の分布だけを使って閾値を調整する。

- family別quantile normalization
- train/test検出score分布のunsupervised alignment
- frame間で急変しないthreshold smoothing
- test movie内でのself-calibration

train/testの画像分布が同じ場合は何も変更しないidentity calibrationを選べるようにし、不要なaugmentation補正を避ける。

### K9. `sparse_gt_positive_unlabeled`

GTにない候補をすべてnegativeとみなさず、positive-unlabeled（PU）として扱う。

- GTに一致するnode・edge：positive
- GTにない候補：negativeではなくunlabeled
- non-negative PU riskまたは信頼度付きnegative sampling

を使用する。疎なGTを通常BCEの完全negativeとして扱うことによる誤学習を減らす。

### K10. `score_frontier_ensemble`

単独最高scoreだけでなく、誤り方が異なるPareto frontier方式をensembleする。

```text
高precision方式
高recall方式
高division方式
低node-overprediction方式
```

各方式のedge unionを作り、期待公式scoreで再選択する。単純な平均score上位だけをensembleしない。

### K11. `metric_aware_track_pruning`

固定`minimum_track_nodes`ではなく、trackごとの期待寄与で削除する。

特徴：

- track長
- 平均edge確率
- 最小edge確率
- 平均検出確率
- 移動の滑らかさ
- family
- 時刻範囲
- graph component size

trackを削除した場合のnode penalty減少とedge TP/FN変化をOOFで推定し、期待scoreが上がるtrackだけを残す。

### K12. `four_test_movie_specialist_router`

最終testが少数movieで固定される場合、movieごとに異なる事前学習済みpolicyを選ぶ。ただし選択にはtest label、手動正解、対応するGTを使用せず、画像とモデル出力から得られるunsupervised特徴だけを使う。

```text
test movie A → family expert + conservative division
test movie B → UOT + longer motion gate
test movie C → high-density policy
test movie D → low-node-count policy
```

ID文字列だけを根拠に恣意的な設定を割り当てず、train OOFで学習したrouter規則を固定して適用する。

## 23. コンペ専用比較の優先順位

最初に比較する価値が高い順：

1. `family_expert_44b6_6bba`
2. `sparse_gt_positive_unlabeled`
3. `annotation_propensity_pruning`
4. `metric_aware_track_pruning`
5. `development_phase_conditioned_ot`
6. `division_value_gate`
7. `full_movie_transductive_ot`
8. `per_sequence_policy_router`
9. `score_frontier_ensemble`
10. `expected_jaccard_edge_selection`
11. `family_time_node_count_prior`
12. `test_distribution_calibrated_detection`
13. `four_test_movie_specialist_router`

推奨する比較軸：

| 比較 | 固定するもの | 変更するもの | 検証目的 |
|---|---|---|---|
| 共通モデル vs family expert | tracking、metric | checkpoint | family分離の価値 |
| BCE vs PU loss | backbone、split | label risk | sparse GT対応 |
| 全node vs propensity pruning | raw prediction | node選択 | 過剰node penalty |
| 固定track長 vs metric-aware pruning | edge graph | track選択 | Jaccard最適化 |
| 時刻共通OT vs phase OT | checkpoint、候補 | 時間依存cost | 発生段階適応 |
| division常時on vs value gate | division候補 | 採用規則 | combined score最適化 |
| pairwise vs full-movie OT | raw node | 時間範囲 | offline推論の価値 |
| 単独best vs frontier ensemble | candidate graphs | ensemble | 誤りの多様性利用 |

## 24. 規約・リーク境界

Kaggle Rules本文を機械的に取得できなかったため、次の案は規約確認が完了するまで実装・提出に使用しない。

- test画像と同一または近似するtrain画像を検索し、対応するtrain GEFFをコピーする
- test movieを人手で追跡・修正する
- 非公開ラベル、他参加者の非公開出力、許可されていない外部データを使う
- test IDに対して、正解を推測して手作業で座標やedgeをハードコードする
- leaderboardへ大量提出してラベル情報を逆算する

以前のローカルSHA監査では、test Zarrと同名train Zarrの一致が示唆されている。この場合でも、train GEFFをtest予測として直接流用する処理は、競技主催者から明示的な許可を得るまで採用しない。

コンペ特化であっても、原則として次の境界に保つ。

```text
使用する教師情報：competition trainで公開されたGTのみ
testで使う情報：画像、公開metadata、モデル自身の予測のみ
調整方法：train内OOFで固定した規則
最終評価：未使用holdoutとKaggle公式metric
```

Rules確認先：<https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules>

## 25. 理論付録A：追跡問題を測度輸送として定式化する

### 25.1 記号と三つの異なる対象

時刻 \(t\) と \(t+1\) の検出候補を、離散測度

\[
\mu_t=\sum_{i=1}^{N}a_i\delta_{x_i},
\qquad
\mu_{t+1}=\sum_{j=1}^{M}b_j\delta_{y_j}
\]

で表す。\(x_i,y_j\in\mathbb{R}^3\) は物理座標、\(a_i,b_j\ge0\) は検出信頼度等から作る質量、
\(\delta_x\) は位置 \(x\) のDirac測度である。本設計では、次の三つを混同しない。

1. **輸送計画** \(P_{ij}\ge0\)：質量がどの候補へどの程度対応するかを表す連続値
2. **追跡グラフ** \(z_{ij}\in\{0,1\}\)：GEFFへ書く離散edge
3. **Kaggle評価値**：7 µm以内のnode matching後に得る非微分可能な集合指標

OTが直接保証するのは1の最適性であり、2の生物学的妥当性や3の最大化ではない。そのため本提案は、OTを候補の全体整合的な再採点に使い、ILPで1を2へ射影し、最後に公式実装で3を測る構成とする。

### 25.2 Balanced OT

総質量が等しい \(\sum_i a_i=\sum_j b_j\) とき、許容輸送計画の集合は

\[
\mathcal U(a,b)
=
\left\{
P\in\mathbb R_+^{N\times M}
\mid
P\mathbf 1_M=a,\quad P^\top\mathbf 1_N=b
\right\}
\]

である。Kantorovich型の離散OTは

\[
\min_{P\in\mathcal U(a,b)}\langle C,P\rangle
\]

を解く。これは各行・列が必ず所定量を輸送するため、検出数が違う細胞追跡へそのまま使うと、偽陽性や消失にも無理に対応を割り当てる。

Cuturi (2013) のentropy正則化では

\[
\min_{P\in\mathcal U(a,b)}
\langle C,P\rangle
+\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1)
\tag{25.1}
\]

とする。双対は定数項を除き

\[
\max_{f,g}
\langle f,a\rangle+\langle g,b\rangle
-\varepsilon\sum_{ij}
\exp\!\left(\frac{f_i+g_j-C_{ij}}{\varepsilon}\right)
\tag{25.2}
\]

で、最適計画は

\[
P^\star=\operatorname{diag}(u)K\operatorname{diag}(v),
\qquad K_{ij}=\exp(-C_{ij}/\varepsilon)
\]

と因数分解できる。したがってSinkhorn反復は要素ごとの除算

\[
u\leftarrow a\oslash(Kv),
\qquad
v\leftarrow b\oslash(K^\top u)
\tag{25.3}
\]

で実行できる。dense行列なら1反復は \(O(NM)\) である。

\(\varepsilon\) は単なる数値パラメータではない。小さいほど低コストedgeへ集中して非正則化OTへ近づくが、\(K_{ij}\) のunderflowと反復停滞が起きやすい。大きいほど滑らかで安定する一方、密集領域で質量が多数の娘候補へ広がる。実装では

\[
\log u_i
\leftarrow
\log a_i-\operatorname{LSE}_j
\left(\frac{-C_{ij}}{\varepsilon}+\log v_j\right)
\]

のようなlog-domain更新を用いる。なお \(\varepsilon\to0\) でOT解には近づくが、非一様質量、dustbin、UOTを含む場合に「Hungarianと必ず同じ割当になる」とは限らない。

## 26. 理論付録B：Partial OT、dustbin、Unbalanced OT

### 26.1 三方式の違い

対応なしを扱う代表的な方法は次の三つである。

- **dustbin**：未対応専用の行・列を加え、拡張行列では質量を保存する。SuperGlue (Sarlin et al., 2020) が部分的特徴対応に用いた考え方である。
- **Partial OT**：輸送する総量 \(m\) を固定し、残りを輸送しない。
- **Unbalanced OT (UOT)**：周辺分布の不一致を発散で連続的に罰し、輸送量自体を最適化する。

Partial OTの一例は

\[
\min_{P\ge0}\langle C,P\rangle,
\quad
P\mathbf1\le a,
\quad
P^\top\mathbf1\le b,
\quad
\mathbf1^\top P\mathbf1=m
\tag{26.1}
\]

である。\(m\) が妥当なら不要な対応を捨てられるが、フレームごとに真の対応質量を知る必要がある。

### 26.2 KL緩和UOT

一般化KLを

\[
\operatorname{KL}(r\|s)
=\sum_k\left[r_k\log\frac{r_k}{s_k}-r_k+s_k\right]
\]

とする。Chizat et al. (2018) のスケーリング法に対応する離散UOTは

\[
\min_{P\ge0}
\langle C,P\rangle
+\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1)
+\tau_s\operatorname{KL}(P\mathbf1\|a)
+\tau_t\operatorname{KL}(P^\top\mathbf1\|b)
\tag{26.2}
\]

である。KL項の場合の一般化Sinkhorn更新は

\[
u\leftarrow
\left(a\oslash Kv\right)^{\rho_s},
\qquad
v\leftarrow
\left(b\oslash K^\top u\right)^{\rho_t},
\qquad
\rho_s=\frac{\tau_s}{\tau_s+\varepsilon},
\quad
\rho_t=\frac{\tau_t}{\tau_t+\varepsilon}
\tag{26.3}
\]

となる。極限の意味は明確である。

- \(\tau_s,\tau_t\to\infty\)：周辺制約が硬くなりBalanced OTへ近づく
- \(\tau_s,\tau_t\to0\)：質量の生成・消滅が安価になり、低コスト箇所だけを結びやすい
- \(\varepsilon\) 増加：計画が拡散する

したがって `source_mass_penalty` と `target_mass_penalty` は、単なる探索値ではなく、appearance/disappearanceをどれほど許すかというモデル仮定である。ただし「余った質量＝細胞死」「不足質量＝誕生」という対応は本コンペへの解釈であり、UOTの定理が生物学的事象を識別してくれるわけではない。Wasserstein--Fisher--Rao/Hellinger--Kantorovich系の動的UOTも質量変化を反応項として扱えるが、離散的な親娘関係は別途必要である（Chizat et al., 2018）。

### 26.3 dustbinとUOTを両方比較する理由

dustbinは「対応なし」のコストを明示でき、UOTは総質量差を滑らかに吸収する。前者はdustbin cost、後者は \(\tau_s,\tau_t\) に感度を持つ。どちらが優れるかは検出誤差の生成機構に依存し、理論だけでは決まらない。このため同じraw prediction・同じholdoutで比較する。

## 27. 理論付録C：コストを確率として解釈できる条件

edge分類器の出力を温度 \(T>0\) で

\[
\tilde p_{ij}=\sigma(\ell_{ij}/T)
\]

と校正し、\(C^{prob}_{ij}=-\log(\tilde p_{ij}+\delta)\) とすれば、これは対応edgeの負の対数尤度として解釈できる。Guo et al. (2017) は、現代的ニューラルネットが未校正になり得ることと、temperature scalingが有効な単一パラメータ校正法であることを示している。したがってraw sigmoid値をそのままOT質量や期待Jaccardへ使う前に、OOF reliability diagram、NLL、Brier score等を確認する。

複合コスト

\[
C_{ij}=\sum_k\lambda_k C^{(k)}_{ij}
\]

は、各項が負の対数尤度なら

\[
\exp(-C_{ij})
=\prod_k p_k(i,j)^{\lambda_k}
\]

というweighted product-of-expertsになる。しかし、距離・Transformer・検出信頼度は同じ画像から生じ相関するため、条件付き独立な確率モデルとは限らない。よって本設計の \(\lambda_k\) は確率論的に自動決定される値ではなく、OOFで選ぶenergy weightである。

距離項と負対数確率項はスケールが違う。各familyまたは学習splitで

\[
\widehat C^{(k)}
=
\frac{C^{(k)}-\operatorname{median}(C^{(k)})}
{\operatorname{IQR}(C^{(k)})+\delta}
\tag{27.1}
\]

のようにrobust標準化すると、一項だけが数値スケールで支配するのを避けられる。gating外のedgeは巨大な有限コストではなく \(P_{ij}=0\) のmaskにする。有限値では \(\varepsilon\) が大きいと禁止edgeへも微小質量が漏れるためである。

## 28. 理論付録D：構造OT、Gromov--Wasserstein、SCOTT

通常のWassersteinコストは \(x_i\) と \(y_j\) の直接比較に依存する。Gromov--Wasserstein (GW) は、各フレーム内の関係行列 \(D^X,D^Y\) を比較し、代表的な二乗損失では

\[
\min_{P\in\mathcal U(a,b)}
\sum_{i,k,j,l}
\left|D^X_{ik}-D^Y_{jl}\right|^2
P_{ij}P_{kl}
\tag{28.1}
\]

を解く。全体が平行移動しても内部距離が保たれる場合に有利だが、(P) の二次式なので一般に非凸で、初期値によって局所解が変わる。

Fused Gromov--Wasserstein (FGW; Vayer et al., 2019) はfeatureと構造を

\[
\min_{P\in\mathcal U(a,b)}
\alpha\sum_{ij}M_{ij}P_{ij}
+(1-\alpha)
\sum_{i,k,j,l}L(D^X_{ik},D^Y_{jl})P_{ij}P_{kl}
\tag{28.2}
\]

で統合する。ここで \(M_{ij}\) はappearance/位置featureの直接コストである。SCOTT (Gao et al.) は細胞追跡でshapeとlocationをOTへ統合した先行研究であり、本提案の構造項を導入する根拠になる。細胞数も違う場合はUnbalanced GW（Séjourné et al., 2021）へ拡張できる。

本コンペでは、まずUOTで得た \(P\) をFGW/UGWの初期値にし、\(D^X,D^Y\) を近傍グラフ距離または局所座標距離として比較する。これは計算量と局所解リスクを抑える工学的選択であり、原論文の性能保証を本データへ移すものではない。

## 29. 理論付録E：multi-frame OTとflow保存

### 29.1 Multi-Marginal OT

(L) フレームを同時に扱うとき、輸送計画は行列でなくテンソル

\[
\Pi_{i_0i_1\ldots i_{L-1}}\ge0
\]

となる。各軸の周辺分布を各フレームの \(a^{(t)}\) に一致させ、

\[
\min_{\Pi}
\sum_{i_0,\ldots,i_{L-1}}
\mathcal C_{i_0\ldots i_{L-1}}
\Pi_{i_0\ldots i_{L-1}}
+\varepsilon H(\Pi)
\tag{29.1}
\]

を解く。追跡用コストの一例は

\[
\mathcal C
=
\sum_{t=0}^{L-2}
\lambda_d\|x^{t+1}_{i_{t+1}}-x^t_{i_t}\|^2
+\sum_{t=1}^{L-2}
\lambda_a
\|x^{t+1}_{i_{t+1}}-2x^t_{i_t}+x^{t-1}_{i_{t-1}}\|^2
\tag{29.2}
\]

で、第二項が離散加速度を罰する。Elvander et al. (2020) はentropy正則化MMOTで複数時刻の情報とdynamicsを統合できることを示した。Beier et al. (2023) はunbalanced multi-marginal Sinkhornの収束を、一定の仮定の下で証明している。

ただし各フレームに \(n\) 候補がある素朴なテンソルは \(n^L\) 要素である。Le et al. (2022) のcomplexity結果もmulti-marginal partial OTの指数依存を明示する。一方、pairwise/tree構造を持つコストはmessage passing的な縮約が可能である。したがって実験では3フレームから始め、100フレーム全体のdense tensorは作らない。

### 29.2 Optimal Flow Transportとの関係

候補edgeを有向グラフとし、incidence行列を \(B\)、edge flowを \(f\ge0\) とすれば、通常のflow保存は

\[
Bf=s
\tag{29.3}
\]

と書ける。OFT-Sinkhorn (Shi et al., 2025) は、このflow balanceを満たすentropy正則化輸送をGPU向け反復で扱う。pairwise OTを独立に解く方式と違い、中間時刻で「前フレームから入った質量」と「次フレームへ出す質量」を結合できる。

本コンペへ適用する際、\(B\) の符号を `outflow - inflow` と定めると、通常継続は \(s_v=0\)、appearanceは \(s_v=+1\)、disappearanceは \(s_v=-1\)、1親2娘のdivisionは親側で \(s_v=+1\) となる。ただしこれはOFTのflow式を系譜へ写した**本提案の拡張**であり、OFT論文が細胞分裂の離散意味論を解決したわけではない。

## 30. 理論付録F：divisionはなぜ標準OTだけでは不十分か

標準OTは一つのsource質量を複数targetへ分割できる。しかし、その分割は「曖昧な対応」「密集によるentropy拡散」「本当の細胞分裂」を区別しない。divisionには、親のoutdegreeが2、各娘のindegreeが1、親娘・娘間の形状が妥当、という離散構造が必要である。

edge変数 \(z_{ij}\in\{0,1\}\)、division変数 \(d_i\in\{0,1\}\) を導入し、

\[
\sum_i z_{ij}\le1,
\qquad
\sum_j z_{ij}\le1+d_i,
\qquad
\sum_j z_{ij}\ge2d_i
\tag{30.1}
\]

とすれば、\(d_i=0\) のとき子は最大1、\(d_i=1\) のとき子はちょうど2になる。娘pair \((j,k)\) の変数 \(h_{ijk}\) は

\[
h_{ijk}\le z_{ij},\quad
h_{ijk}\le z_{ik},\quad
h_{ijk}\ge z_{ij}+z_{ik}-1
\tag{30.2}
\]

で線形化できる。最終目的を

\[
\min_{z,d,h}
\sum_{ij}E_{ij}z_{ij}
+\lambda_{div}\sum_i d_i
+\sum_{i,j<k}S_{ijk}h_{ijk}
\tag{30.3}
\]

とし、\(S_{ijk}\) に娘間距離、体積比、方向、後続フレームでの生存を入れる。これは既知のOTそのものではなく、OT planを候補scoreとして用いるcompetition-specificな整数最適化である。ゆえに「division-aware OT」という名称は、OT単独が分裂を認識するという意味ではない。

## 31. 理論付録G：アンサンブルをどの空間で行うか

### 31.1 確率平均とplan平均

\(R\) 個のモデルの校正済みedge確率を \(p^{(r)}_{ij}\) とすると、単純平均は

\[
\bar p_{ij}=\sum_{r=1}^{R}w_r p^{(r)}_{ij},
\qquad w_r\ge0,\quad\sum_r w_r=1
\tag{31.1}
\]

である。Deep Ensembles (Lakshminarayanan et al., 2017) は独立学習した確率モデルの平均が予測性能と不確実性推定に有効な強いbaselineであることを示した。ただし同じcheckpointの閾値違いは誤りの相関が高いため、seed・backbone・tracker・augmentationの多様性を持つ方が意味がある。

同一の周辺分布 \(a,b\) に対し \(P^{(r)}\in\mathcal U(a,b)\) なら、

\[
\bar P=\sum_r w_rP^{(r)}
\]

も

\[
\bar P\mathbf1=a,
\qquad
\bar P^\top\mathbf1=b
\]

を満たす。したがってbalanced OT planの凸平均は厳密にfeasibleである。一方、各方式でnode集合・周辺質量・gatingが違う場合はこの性質を使えず、共通候補グラフへ揃えてから再射影する必要がある。

Wasserstein barycenter

\[
\nu^\star
=\arg\min_\nu\sum_r w_r W_\varepsilon(\nu,\mu_r)
\tag{31.2}
\]

は分布の幾何学的平均であり（Cuturi and Doucet, 2014）、同一座標上のplan要素平均とは別物である。tracker ensembleでbarycenterを使うなら、各予測が異なるsupportを持つ場合に限って追加価値を検証する。

### 31.2 stackingと不一致

OOF特徴 \(r_{ij}=(p^{(1)}_{ij},\ldots,p^{(R)}_{ij},d_{ij},q_i,q_j)\) からmeta-model \(g(r_{ij})\) を学習する。ただしbase modelが学習したsample自身の予測をstackerへ渡すとリークするので、必ずOOF予測だけを使う。

方式間不一致は

\[
U_{ij}
=-\bar p_{ij}\log\bar p_{ij}
-(1-\bar p_{ij})\log(1-\bar p_{ij})
\tag{31.3}
\]

またはmodel間分散で測る。高不一致edgeを一律除去するのではなく、ILPへ高い不確実性penaltyとして渡す。多数決は多数の同型モデルが同じ誤りをする場合に改善保証がない。

## 32. 理論付録H：疎なGTとPositive--Unlabeled学習

真のedgeを \(Y=+1\)、非edgeを \(Y=-1\) とし、unlabeled分布が

\[
p_U(x)=\pi_p p_P(x)+(1-\pi_p)p_N(x)
\tag{32.1}
\]

という混合であると仮定する。positive priorを \(\pi_p=P(Y=+1)\)、損失を \(\ell(g(x),y)\) とすると、PN riskは

\[
R(g)
=\pi_p R_P^+(g)+(1-\pi_p)R_N^-(g)
\]

である。\(R_U^-=\pi_pR_P^-+(1-\pi_p)R_N^-\) より

\[
R_{PU}(g)
=\pi_p R_P^+(g)+R_U^-(g)-\pi_pR_P^-(g)
\tag{32.2}
\]

とnegativeデータなしで書ける。ただし高容量モデルでは経験riskの負部分が負へ発散し過学習し得る。Kiryo et al. (2017) のnon-negative PU estimatorは

\[
\widehat R_{nnPU}(g)
=\pi_p\widehat R_P^+(g)
+\max\left\{0,
\widehat R_U^-(g)-\pi_p\widehat R_P^-(g)
\right\}
\tag{32.3}
\]

でこの問題を抑える。

重要な制限は、通常のPU理論が「ラベル付きpositiveがpositive母集団を代表する」選択機構を仮定する点である。GTが特定時刻・明瞭な細胞だけに偏るならSCAR（selected completely at random）が破れ、式 (32.3) をそのまま正当化できない。2025年のbiased PU研究も、instance-dependent propensityまたは追加の校正が必要になることを示している。したがって本コンペでは次を必須とする。

- GTにない候補edgeを直ちにnegativeとしない
- family・時刻・輝度・密度ごとのannotation propensityを推定する
- \(\pi_p\) の感度分析を行う
- 通常BCE、confidence-weighted BCE、nnPUを同一OOFで比較する
- PU lossが公式scoreを改善しなければ採用しない

## 33. 理論付録I：公式評価式とOT目的のずれ

### 33.1 公式実装と同じ集約式

sample \(s\) のnode matchingは物理距離7 µm、voxel scale
\((1.625,0.40625,0.40625)\) µmで行う。matching後のedge countを
\(T_s=TP_s,F_s=FP_s,N_s=FN_s\) とし、

\[
J_s=\frac{T_s}{T_s+F_s+N_s}
\tag{33.1}
\]

を得る。予測node数を \(N_s^{pred}\)、公式metadataのtarget node数を \(N_s^{total}\) とすると

\[
r_s=\frac{N_s^{pred}-N_s^{total}}{N_s^{total}},
\qquad
A_s=\max\{0,J_s(1-0.1r_s)\}
\tag{33.2}
\]

である。run-level Adjusted Edge Jaccardは単純平均でもglobal micro Jaccardでもなく、

\[
J_{edge,adj}
=
\frac{\sum_s w_s A_s}{\sum_s w_s},
\qquad
w_s=T_s+F_s+N_s
\tag{33.3}
\]

である。divisionは全valid sampleでmicro集計し、

\[
J_{div}
=
\frac{\sum_s TP_s^{div}}
{\sum_s(TP_s^{div}+FP_s^{div}+FN_s^{div})}
\tag{33.4}
\]

最終値は

\[
\boxed{\mathrm{Score}=J_{edge,adj}+0.1J_{div}}
\tag{33.5}
\]

となる。divisionがsplit全体に一件もない場合、公式baseline実装はdivision項を落とす。この式はKaggle evaluation pageと公式baselineの `tracking_cellmot.metrics` に固定し、accuracy、training loss、通常Jaccardを方式選択値にしない。

式 (33.2) は正の \(r_s\)、すなわちnode過剰予測を直接下げる。node過少予測では係数だけを見ると増えるが、通常はedge FNとnode recall悪化を伴うため、nodeを削れば必ず得になるわけではない。

### 33.2 expected Jaccardは近似である

説明のためnode matchingとnode-count補正を固定し、現在の \(D=T+F+N\) に候補edgeを一つ足す。候補が確率 \(p\) で「既存FNをTPへ直す正解」、確率 \(1-p\) で「FPを一つ増やす誤り」だと仮定すると、期待差は

\[
\mathbb E[\Delta J]
=\frac{p}{D}
-\frac{(1-p)T}{D(D+1)}
\tag{33.6}
\]

であり、正になる条件は

\[
p>\frac{T}{D+1+T}
\tag{33.7}
\]

となる。したがってJaccard最適な閾値は一般に0.5固定ではない。Jaccardを含むlinear-fractional utilityに校正surrogateを用いる理論的研究も存在する（Bao and Sugiyama, 2020）。

ただし実際にはedge追加がnode matching、次数制約、division、後続edge、\(r_s\) を同時に変える。式 (33.6) は局所surrogateにすぎず、公式scoreの厳密な微分でも最適性保証でもない。最終採否は必ず完全なGEFFを作って式 (33.1)--(33.5) で再評価する。

## 34. 理論付録J：既知理論と本提案の境界

| 要素 | 根拠 | この文書で言えること | 言えないこと |
|---|---|---|---|
| Entropic Sinkhorn | Cuturi 2013 | 式 (25.1) を高速な行列反復で近似可能 | 公式scoreがHungarianより高い保証 |
| KL-UOT | Chizat et al. 2018 | 周辺質量制約を連続的に緩和可能 | 生成質量が生物学的appearanceである保証 |
| Dustbin matching | Sarlin et al. 2020 | unmatched候補を明示的slotで扱える | 細胞divisionを自動認識する保証 |
| FGW/UGW | Vayer et al. 2019; Séjourné et al. 2021 | featureと内部構造を同時比較可能 | 非凸最適化の大域解保証 |
| MMOT/UMOT | Elvander et al. 2020; Beier et al. 2023 | 複数時刻の整合性を一つのplanで結合可能 | 100 frame dense計算が現実的という保証 |
| Flow OT | Shi et al. 2025 | 中間時刻のflow balanceを明示可能 | 1対2分裂の意味論を自動獲得する保証 |
| nnPU | Kiryo et al. 2017 | P/U混合仮定下で非負riskを構成可能 | annotation bias下でも無条件に正しい保証 |
| Deep ensemble | Lakshminarayanan et al. 2017 | 独立モデル平均は強い不確実性baseline | 相関したtrackerの多数決が改善する保証 |
| division-slot ILP | 本コンペ向け提案 | 1親2娘の次数とpair geometryを明示可能 | 論文で本データ上の有効性が実証済みとの主張 |
| metric-aware pruning/router | 本コンペ向け提案 | 公式式に沿った仮説としてOOF検証可能 | 未使用test labelなしで最適policyを知る保証 |

### 34.1 計算量の上限と実験停止条件

| 方法 | 代表的な時間量 | 主要メモリ | 停止・棄却条件 |
|---|---:|---:|---|
| Hungarian（正方 \(n\times n\)） | \(O(n^3)\) | \(O(n^2)\) | division/UOT効果の対照として残す |
| dense Sinkhorn | \(O(KNM)\) | \(O(NM)\) | marginal residualが許容値以下、または反復上限 |
| sparse gated Sinkhorn | \(O(K|E|)\) | \(O(|E|)\) | dense版との小規模一致を確認できない場合は棄却 |
| naive GW objective | \(O(N^2M^2)\) | 実装依存 | UOT初期値を変えるとscoreが大幅変動する場合は不安定扱い |
| naive \(L\)-frame MMOT | \(O(n^L)\) 要素 | \(O(n^L)\) | 3-frameで既存法を超えなければ拡張しない |
| ILP | 最悪指数時間 | 候補数依存 | time limit時はincumbent・gapを必ず記録 |

すべての新方式は「同じraw node/edge predictionを使うtracker-only比較」と「必要なら再学習するmodel比較」を分ける。これにより、改善が検出器、edge確率、matching、後処理、ensembleのどこから生じたかを識別する。

### 34.2 検証可能な仮説

本設計の主張は次の反証可能な形に限定する。

1. UOTがHungarianを上回るなら、主因はnode数不一致または局所曖昧性であるはずで、appearance/disappearance数とedge FP/FNに差が出る。
2. FGWがUOTを上回るなら、高密度領域で内部近傍構造を使ったsampleの改善が大きいはずである。
3. multi-frame法がpairwise法を上回るなら、加速度外れ値、one-frame gap、ID switchが減るはずである。
4. division-slotが有効なら、\(J_{div}\) の上昇がedge FP増加による \(J_{edge,adj}\) 低下を0.1倍重み込みで上回るはずである。
5. ensembleが有効なら、単体最高方式と異なる誤りを持つmemberを加えたときのみOOF scoreが上がるはずである。
6. nnPUが有効なら、未注釈候補を全negativeとしたBCEよりedge recallを上げ、かつFP増加を公式score上で相殺できるはずである。

このどれも論文から本コンペでの成功を演繹できるものではない。論文は方式の数学的妥当性を支え、採用判断は固定OOFと公式評価実装が行う。

## 35. 参考文献

- Marco Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transport,” NeurIPS 2013. <https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html>
- Lénaïc Chizat, Gabriel Peyré, Bernhard Schmitzer, François-Xavier Vialard, “Scaling Algorithms for Unbalanced Optimal Transport Problems,” Mathematics of Computation 87(314), 2018. <https://www.ams.org/mcom/2018-87-314/S0025-5718-2018-03303-8/>
- Paul-Edouard Sarlin et al., “SuperGlue: Learning Feature Matching with Graph Neural Networks,” CVPR 2020. <https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html>
- Tingran Gao et al., “SCOTT: Shape-Location Combined Tracking with Optimal Transport,” SIAM Journal on Mathematics of Data Science. <https://epubs.siam.org/doi/10.1137/19M1253976>
- Thibault Séjourné, François-Xavier Vialard, Gabriel Peyré, “The Unbalanced Gromov-Wasserstein Distance: Conic Formulation and Relaxation,” NeurIPS 2021. <https://proceedings.neurips.cc/paper_files/paper/2021/hash/4990974d150d0de5e6e15a1454fe6b0f-Abstract.html>
- Yan Zhang et al., “Unlocking Slot Attention by Changing Optimal Transport Costs,” ICML 2023. <https://openreview.net/forum?id=FMomWFNh5d>
- Mu Qiao, “Unsupervised Evolutionary Cell Type Matching via Entropy-Minimized Optimal Transport,” PMLR 2025. <https://proceedings.mlr.press/v311/qiao25a.html>
- Liangliang Shi et al., “Optimal Flow Transport and its Entropic Regularization: a GPU-friendly Matrix Iterative Algorithm for Flow Balance Satisfaction,” ICLR 2025. <https://openreview.net/forum?id=NtSlKEJ2DS>
- Chenrui Wang, Yixuan Qiu, “The Sparse-Plus-Low-Rank Quasi-Newton Method for Entropic-Regularized Optimal Transport,” ICML 2025. <https://openreview.net/forum?id=WCkMkMcqpb>
- Wenjuan Shi et al., “Multi-Object Tracking based on Optimal Transport and Coordinate Attention Mechanism,” Signal Processing 236, 2025. <https://doi.org/10.1016/j.sigpro.2025.110058>
- Romain Seailles et al., “Optimal Transport Unlocks End-to-End Learning for Single-Molecule Localization,” ICLR 2026. <https://openreview.net/forum?id=V1i58pZmp3>
- John P. Bryan, Samouil L. Farhi, Brian Cleary, “Accurate trajectory inference in time-series spatial transcriptomics with structurally-constrained optimal transport,” Nature Communications, 2026. <https://www.nature.com/articles/s41467-026-74927-8>
- “Certified Parallel-in-Time Sinkhorn for Dynamic Entropic Optimal Transport,” arXiv preprint, 2026. <https://arxiv.org/abs/2607.24741>
- Marco Cuturi, Arnaud Doucet, “Fast Computation of Wasserstein Barycenters,” ICML 2014. <https://proceedings.mlr.press/v32/cuturi14.html>
- Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, “On Calibration of Modern Neural Networks,” ICML 2017. <https://proceedings.mlr.press/v70/guo17a.html>
- Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell, “Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles,” NeurIPS 2017. <https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html>
- Lénaïc Chizat, Gabriel Peyré, Bernhard Schmitzer, François-Xavier Vialard, “Unbalanced Optimal Transport: Dynamic and Kantorovich Formulations,” Journal of Functional Analysis 274(11), 2018. <https://doi.org/10.1016/j.jfa.2018.03.008>
- Vayer Titouan, Nicolas Courty, Romain Tavenard, Laetitia Chapel, Rémi Flamary, “Optimal Transport for Structured Data with Application on Graphs,” ICML 2019. <https://proceedings.mlr.press/v97/titouan19a.html>
- Filip Elvander, Isabel Haasler, Andreas Jakobsson, Johan Karlsson, “Multi-marginal Optimal Transport Using Partial Information with Applications in Robust Localization and Sensor Fusion,” Signal Processing 171, 2020. <https://doi.org/10.1016/j.sigpro.2020.107474>
- Han Bao, Masashi Sugiyama, “Calibrated Surrogate Maximization of Linear-fractional Utility in Binary Classification,” AISTATS 2020. <https://proceedings.mlr.press/v108/bao20a.html>
- Khang Le, Huy Nguyen, Khai Nguyen, Tung Pham, Nhat Ho, “On Multimarginal Partial Optimal Transport: Equivalent Forms and Computational Complexity,” AISTATS 2022. <https://proceedings.mlr.press/v151/le22a.html>
- Florian Beier, Johannes von Lindheim, Sebastian Neumayer, Gabriele Steidl, “Unbalanced Multi-marginal Optimal Transport,” Journal of Mathematical Imaging and Vision 65, 2023. <https://doi.org/10.1007/s10851-022-01126-7>
- Ryuichi Kiryo, Gang Niu, Marthinus C. du Plessis, Masashi Sugiyama, “Positive-Unlabeled Learning with Non-Negative Risk Estimator,” NeurIPS 2017. <https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html>
- Paweł Teisseyre, Timo Martens, Jessa Bekker, Jesse Davis, “Learning from Biased Positive-Unlabeled Data via Threshold Calibration,” AISTATS 2025. <https://proceedings.mlr.press/v258/teisseyre25a.html>
- Kaggle, “Biohub Cell Tracking During Development — Evaluation.” <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation>
- Royer Lab, “Kaggle Cell Tracking Competition — Official Baseline and Metric Implementation.” <https://github.com/royerlab/kaggle-cell-tracking-competition>
