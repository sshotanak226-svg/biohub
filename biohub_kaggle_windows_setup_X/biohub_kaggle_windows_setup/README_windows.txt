Biohub Kaggle Windows Setup v5
================================

1. ZIPを展開します。
2. run_setup_biohub_kaggle.cmd をダブルクリックします。
3. Kaggle認証画面が開いたらログインします。

ダブルクリック時の既定動作:
- Kaggleスコア降順の公開Python Notebook Top 10を取得します。
- 競技本体データとSupport Packは取得しません。
- 10件すべての.ipynbを確認できなければエラー終了します。

重要:
- Kaggle CLIの現行版にはPython 3.11以上が必要です。
- この版はPython 3.13, 3.12, 3.11の順に自動検出します。
- pip自体のアップグレードは行いません。
- Kaggle CLIが既に動く場合は再インストールしません。
- インストール失敗時の詳細は <保存先>\manifests\setup.log に残ります。

Notebookだけ取得（件数を変更する場合）:
  run_setup_biohub_kaggle.cmd -TopCount 20 -SkipCompetitionData -SkipSupportPack

Dドライブへ保存:
  run_setup_biohub_kaggle.cmd -Root "D:\biohub-cell-tracking" -SkipCompetitionData -SkipSupportPack

既定保存先:
  C:\Users\<ユーザー名>\biohub-cell-tracking
