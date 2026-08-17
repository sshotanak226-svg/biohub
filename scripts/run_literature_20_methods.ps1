param(
    [string]$LiteratureConfig = "configs\literature_20_methods.yaml",
    [string]$Checkpoint,
    [switch]$Quick,
    [int]$MaxDatasets = 0,
    [switch]$DryRun,
    [string]$Methods
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$arguments = @(
    "run", "--offline", "python", "literature_method_search.py",
    "--literature-config", $LiteratureConfig
)
if ($Checkpoint) {
    $arguments += @("--checkpoint", $Checkpoint)
}
if ($Methods) {
    $arguments += @("--methods", $Methods)
}
if ($DryRun) {
    $arguments += "--dry-run"
}
if ($MaxDatasets -gt 0) {
    $arguments += @("--max-datasets", $MaxDatasets)
} elseif ($Quick) {
    $arguments += @("--max-datasets", "3")
}

Write-Host "Literature screen: 20 executable proxies + current UOT baseline"
if ($Quick -and $MaxDatasets -le 0) {
    Write-Host "Quick mode: first 3 fixed holdout movies (ranking is provisional)"
} elseif ($MaxDatasets -gt 0) {
    Write-Host "Subset mode: first $MaxDatasets fixed holdout movies (ranking is provisional)"
} else {
    Write-Host "Full mode: all 20 fixed holdout movies"
}
Write-Host "Paper-faithful models: false; shared checkpoint/raw predictions: true"
& uv @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
