$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
uv run python -m biohub_demo.archive "biohub_top10_download/data/competition/biohub-cell-tracking-during-development.zip"
