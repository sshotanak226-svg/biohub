[CmdletBinding()]
param(
    [string]$Destination,

    [ValidateRange(1, 1000)]
    [int]$MaxAttempts = 50,

    [ValidateRange(1, 3600)]
    [int]$RetryDelaySeconds = 10,

    [switch]$SkipExtraction,
    [switch]$ForceExtraction
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Competition = "biohub-cell-tracking-during-development"
$ArchiveName = "$Competition.zip"
$ExpectedArchiveSize = [Int64]87393127165
$WorkspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$ValidatorPath = Join-Path $PSScriptRoot "validate_biohub_data.py"

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $WorkspaceRoot "biohub_top10_download\data\competition"
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
$ArchivePath = Join-Path $Destination $ArchiveName
$PartialMarkerPath = "$ArchivePath.kaggle-partial"
$ValidationReportPath = Join-Path $Destination "data_download_validation.json"
$script:ValidatedArchiveLength = $null
$script:ValidatedArchiveWriteTimeUtc = $null

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install uv and add it to PATH first."
}
if (-not (Test-Path -LiteralPath (Join-Path $WorkspaceRoot "pyproject.toml"))) {
    throw "The uv project was not found: $WorkspaceRoot"
}
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Invoke-Kaggle {
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$DiscardOutput
    )

    if ($DiscardOutput) {
        & uv run --project $WorkspaceRoot python -m kaggle @Arguments | Out-Null
    }
    else {
        & uv run --project $WorkspaceRoot python -m kaggle @Arguments | Out-Host
    }
    $ExitCode = $LASTEXITCODE
    return $ExitCode
}

function Test-BiohubArchive {
    if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        return $false
    }

    $Candidate = Get-Item -LiteralPath $ArchivePath
    if ($Candidate.Length -ne $ExpectedArchiveSize) {
        return $false
    }
    if ($script:ValidatedArchiveLength -eq $Candidate.Length -and
        $script:ValidatedArchiveWriteTimeUtc -eq $Candidate.LastWriteTimeUtc) {
        return $true
    }

    # A readable central directory or one readable sentinel is insufficient.
    # Validate the official size/inventory and decompress every member so that
    # zipfile checks every CRC before the archive is accepted.
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & uv run --offline --project $WorkspaceRoot python $ValidatorPath `
            $ArchivePath --crc 2>$null | Out-Null
        $ArchiveCheckExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ArchiveCheckExitCode -eq 0) {
        $script:ValidatedArchiveLength = $Candidate.Length
        $script:ValidatedArchiveWriteTimeUtc = $Candidate.LastWriteTimeUtc
        return $true
    }
    return $false
}

function Get-ExpectedPartialSize {
    if (-not (Test-Path -LiteralPath $PartialMarkerPath -PathType Leaf)) {
        return $null
    }

    try {
        $Marker = Get-Content -LiteralPath $PartialMarkerPath -Raw | ConvertFrom-Json
        if ($null -ne $Marker.size) {
            return [Int64]$Marker.size
        }
    }
    catch {
        return $null
    }
    return $null
}

function Move-InvalidArchiveAside {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Suffix = ".corrupt.$Stamp"

    if (Test-Path -LiteralPath $ArchivePath -PathType Leaf) {
        $QuarantinePath = "$ArchivePath$Suffix"
        Move-Item -LiteralPath $ArchivePath -Destination $QuarantinePath
        Write-Warning "Invalid archive moved to: $QuarantinePath"
    }
    if (Test-Path -LiteralPath $PartialMarkerPath -PathType Leaf) {
        Move-Item -LiteralPath $PartialMarkerPath -Destination "$PartialMarkerPath$Suffix"
    }
}

function Repair-InvalidDownloadState {
    if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        return
    }
    if (Test-BiohubArchive) {
        return
    }

    $LocalSize = (Get-Item -LiteralPath $ArchivePath).Length
    $ExpectedSize = Get-ExpectedPartialSize

    if ($null -ne $ExpectedSize -and $LocalSize -lt $ExpectedSize) {
        Write-Host ("Resumable partial archive found: {0:N2}/{1:N2} GiB" -f ($LocalSize / 1GB), ($ExpectedSize / 1GB))
        return
    }

    # A complete-size/oversized invalid file cannot be resumed safely. Keep it
    # as a recoverable quarantine copy and let Kaggle create a clean archive.
    Move-InvalidArchiveAside
}

Write-Host "Checking Kaggle authentication and competition-rule acceptance..."
$PreflightExitCode = Invoke-Kaggle -Arguments @(
    "competitions", "files", $Competition,
    "--page-size", "1",
    "-q"
) -DiscardOutput
if ($PreflightExitCode -ne 0) {
    throw "Kaggle authentication or competition-rule acceptance could not be verified. Join the competition in a browser and finish Kaggle CLI authentication."
}

Repair-InvalidDownloadState

$Downloaded = Test-BiohubArchive
for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
    if ($Downloaded) {
        break
    }

    Write-Host ("[{0}/{1}] Downloading competition data..." -f $Attempt, $MaxAttempts)

    # Do not add -o/--force. Kaggle CLI uses the existing partial file and
    # its .kaggle-partial marker to resume this large download safely.
    $DownloadExitCode = Invoke-Kaggle -Arguments @(
        "competitions", "download", $Competition,
        "-p", $Destination
    )

    if ($DownloadExitCode -eq 0 -and (Test-BiohubArchive)) {
        $Downloaded = $true
        break
    }

    Repair-InvalidDownloadState

    if ($Attempt -lt $MaxAttempts) {
        Write-Warning "The download failed. Resuming in $RetryDelaySeconds seconds."
        Start-Sleep -Seconds $RetryDelaySeconds
    }
}

if (-not $Downloaded) {
    throw "Competition data could not be downloaded after $MaxAttempts attempts."
}

$Archive = Get-Item -LiteralPath $ArchivePath
Write-Host ("Download complete: {0} ({1:N2} GiB)" -f $Archive.FullName, ($Archive.Length / 1GB))

if ($SkipExtraction) {
    Write-Host "ZIP extraction was skipped."
    exit 0
}

$TrainPath = Join-Path $Destination "train"
$TestPath = Join-Path $Destination "test"
$AlreadyExtracted = (Test-Path -LiteralPath $TrainPath -PathType Container) -and
                    (Test-Path -LiteralPath $TestPath -PathType Container)

if ($AlreadyExtracted -and -not $ForceExtraction) {
    Write-Host "train/test already exist. Extraction was skipped."
    exit 0
}

Write-Host "Extracting the ZIP archive with Python zipfile. This can take a long time..."
& uv run --offline --project $WorkspaceRoot python -m zipfile -e $ArchivePath $Destination
if ($LASTEXITCODE -ne 0) {
    throw "Python zipfile could not extract the competition archive."
}

if (-not (Test-Path -LiteralPath $TrainPath -PathType Container) -or
    -not (Test-Path -LiteralPath $TestPath -PathType Container)) {
    throw "The train/test directories were not found after extraction."
}

Write-Host "Verifying every extracted file against the official ZIP inventory..."
& uv run --offline --project $WorkspaceRoot python $ValidatorPath `
    $ArchivePath --destination $Destination --report $ValidationReportPath
if ($LASTEXITCODE -ne 0) {
    throw "The extracted train/test/evaluation files did not match the official ZIP."
}

Write-Host "Extraction complete: $Destination"
Write-Host "Validation report: $ValidationReportPath"
