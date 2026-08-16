$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
uv build --wheel
Write-Host "Wheel created under dist/. Attach it, the config, and checkpoints as Kaggle datasets."
