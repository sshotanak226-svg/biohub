param(
    [string]$Config = "configs\local_hyperparameter_search.yaml",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$arguments = @(
    "run", "--offline", "python", "hyperparameter_search.py",
    "--config", $Config
)
if ($DryRun) {
    $arguments += "--dry-run"
}

# Deliberately foreground-only: this script runs only when invoked explicitly,
# and Ctrl+C stops the active search.
& uv @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
