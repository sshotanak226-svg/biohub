#!/usr/bin/env bash
set -euo pipefail

# Biohub - Cell Tracking During Development
# macOS / Linux 用セットアップ
# 事前に Kaggle の競技ページで Join Competition / ルール同意を完了してください。

COMP="biohub-cell-tracking-during-development"
TOP_KERNEL="boristown/dark-agi-biohub-cell-tracking-solution/9"
ORIGINAL_KERNEL="kaiwalyaatulraut/biohub-cell-tracking-solution/5"
SUPPORT_DATASET="pilkwang/biohub-tracking-support-pack-50ep-v1"
ROOT="${1:-biohub-cell-tracking}"

mkdir -p \
  "$ROOT/notebooks/dark_agi_v9" \
  "$ROOT/notebooks/original_v5" \
  "$ROOT/data/competition" \
  "$ROOT/data/support_pack" \
  "$ROOT/manifests"

python3 -m pip install --upgrade pip kaggle

# Kaggle CLI 2.x の OAuth 認証。既に認証済みなら何も変更しません。
if ! kaggle config view >/dev/null 2>&1; then
  echo "Kaggle authentication is required. A browser-based login will start."
  kaggle auth login
fi

echo "[1/7] 競技ファイル一覧を保存"
kaggle competitions files "$COMP" -v \
  > "$ROOT/manifests/competition_files.csv"

echo "[2/7] 公開Notebookをスコア降順で保存（現在の最高候補を再確認可能）"
kaggle kernels list \
  --competition "$COMP" \
  --sort-by scoreDescending \
  --page-size 50 \
  -v > "$ROOT/manifests/kernels_by_score.csv"

echo "[3/7] Discussion一覧を保存"
kaggle competitions topics list "$COMP" \
  --sort-by active \
  --page-size 100 \
  -v > "$ROOT/manifests/discussions_active.csv" || true

echo "[4/7] 公開LB最高候補 V9 のNotebookを取得"
kaggle kernels pull "$TOP_KERNEL" \
  -p "$ROOT/notebooks/dark_agi_v9" -m

echo "[5/7] 派生元のNotebook V5も取得"
kaggle kernels pull "$ORIGINAL_KERNEL" \
  -p "$ROOT/notebooks/original_v5" -m

echo "[6/7] 競技データを取得"
kaggle competitions download "$COMP" \
  -p "$ROOT/data/competition" -o

echo "[7/7] Notebook依存のSupport Packを取得・展開"
kaggle datasets download "$SUPPORT_DATASET" \
  -p "$ROOT/data/support_pack" --unzip -o

# competition download はZIPのままの場合があるため展開する。
while IFS= read -r -d '' zipfile; do
  echo "Extracting: $zipfile"
  if command -v ditto >/dev/null 2>&1; then
    # macOS標準
    ditto -x -k "$zipfile" "$ROOT/data/competition"
  elif command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$zipfile" -d "$ROOT/data/competition"
  else
    echo "ZIP展開ツールがありません。macOSでは ditto、Linuxでは unzip を用意してください。" >&2
    exit 1
  fi
done < <(find "$ROOT/data/competition" -maxdepth 2 -type f -name '*.zip' -print0)

cat <<MSG

完了しました: $ROOT

主な場所:
  最高候補Notebook V9 : $ROOT/notebooks/dark_agi_v9
  派生元Notebook V5   : $ROOT/notebooks/original_v5
  競技データ           : $ROOT/data/competition
  Support Pack         : $ROOT/data/support_pack
  Notebookスコア一覧   : $ROOT/manifests/kernels_by_score.csv
  Discussion一覧       : $ROOT/manifests/discussions_active.csv

注意:
  - Notebook内の /kaggle/input/... はローカルパスへ置換が必要です。
  - Kaggle上で Copy & Edit して実行する場合、入力データのパス変更は通常不要です。
  - 公開LBスコアだけでなく、公式metricによるローカルCVも確認してください。
MSG
