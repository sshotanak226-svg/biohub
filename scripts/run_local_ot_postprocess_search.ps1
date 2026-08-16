param(
    [string]$Config = "configs\local_method_search.yaml",
    [string]$Checkpoint,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if (-not $Checkpoint -and -not $DryRun) {
    $trainingMarker = Join-Path $repoRoot "outputs\method_search\training\training.complete.json"
    if (-not (Test-Path -LiteralPath $trainingMarker -PathType Leaf)) {
        throw "学習済みcheckpoint記録がありません。先に .\scripts\run_local_method_search.ps1 を実行するか、-Checkpoint を指定してください。"
    }
    $training = Get-Content -LiteralPath $trainingMarker -Raw -Encoding UTF8 | ConvertFrom-Json
    $Checkpoint = [string]$training.checkpoint
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "記録されたcheckpointが存在しません: $Checkpoint"
    }
}

$arguments = @(
    "run", "--offline", "python", "method_search.py",
    "--config", $Config
)
if ($Checkpoint) {
    # Providing the existing checkpoint guarantees that this command performs
    # raw-cache reuse and post-processing comparison, never model training.
    $arguments += @("--checkpoint", $Checkpoint)
}
if ($DryRun) {
    $arguments += "--dry-run"
}

Write-Host "OT device: auto (CUDA when torch.cuda.is_available(), otherwise CPU)"
Write-Host "Checkpoint: $Checkpoint"
& uv @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
