param(
    [string]$ProposalConfig = "configs\proposal_all_methods.yaml",
    [string]$Checkpoint,
    [switch]$DryRun,
    [switch]$StrictAllMethods
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$arguments = @(
    "run", "--offline", "python", "proposal_search.py",
    "--proposal-config", $ProposalConfig
)
if ($Checkpoint) {
    $arguments += @("--checkpoint", $Checkpoint)
}
if ($DryRun) {
    $arguments += "--dry-run"
}
if ($StrictAllMethods) {
    $arguments += "--strict-all-methods"
}

Write-Host "Proposal methods: 53 total; GPU OT device: auto"
if ($Checkpoint) {
    Write-Host "Checkpoint: explicit override ($Checkpoint)"
} else {
    Write-Host "Checkpoint: local method-search marker from proposal config"
}
& uv @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
