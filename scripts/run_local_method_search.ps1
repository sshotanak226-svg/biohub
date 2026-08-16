param(
    [string]$Config = "configs\local_method_search.yaml",
    [string]$Checkpoint,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$arguments = @(
    "run", "--offline", "python", "method_search.py",
    "--config", $Config
)
if ($Checkpoint) {
    $arguments += @("--checkpoint", $Checkpoint)
}
if ($DryRun) {
    $arguments += "--dry-run"
}

# Foreground-only. Ctrl+C stops training/inference immediately.
& uv @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
