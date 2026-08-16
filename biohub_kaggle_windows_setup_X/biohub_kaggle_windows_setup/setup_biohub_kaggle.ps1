#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Root = (Join-Path $HOME "biohub-cell-tracking"),

    [ValidateRange(1, 100)]
    [int]$TopCount = 10,

    [switch]$SkipCompetitionData,
    [switch]$SkipExtraction,
    [switch]$SkipSupportPack,
    [switch]$KeepExistingNotebooks
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Kaggle titles commonly contain emoji and non-ASCII characters. Force child
# Python processes to use UTF-8 even when Windows PowerShell is running under
# the Japanese CP932 code page.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

trap {
    $Line = $_.InvocationInfo.ScriptLineNumber
    $Position = $_.InvocationInfo.PositionMessage
    $Message = $_.Exception.Message
    Write-Host ""
    Write-Host "=== SETUP ERROR ===" -ForegroundColor Red
    Write-Host ("Line: {0}" -f $Line) -ForegroundColor Red
    Write-Host $Position -ForegroundColor Red
    Write-Host $Message -ForegroundColor Red
    if (-not [string]::IsNullOrWhiteSpace($script:SetupLogPath)) {
        Write-Host ("Log: {0}" -f $script:SetupLogPath) -ForegroundColor Yellow
    }
    Write-Host "===================" -ForegroundColor Red
    exit 1
}

# Biohub - Cell Tracking During Development
# Windows PowerShell 5.1 / PowerShell 7 対応
# 修正版: 2026-08-06 v5
# - Python 3.11以上を自動選択
# - pip自体はアップグレードしない
# - Kaggleが既に使える場合は再インストールしない
# - 通常インストール失敗時は --user で再試行
# - pip/Kaggleの標準出力・標準エラーをログに保存
# - python -m kaggle を使いPATH依存を回避
# - kernels listのNext Page Token行を除外してJSONを確実に解析
# - TopCount件すべての.ipynbを確認できた場合だけ成功
#
# Kaggle CLI 2.x の公式要件は Python 3.11+ です。

$Competition    = "biohub-cell-tracking-during-development"
$SupportDataset = "pilkwang/biohub-tracking-support-pack-50ep-v1"
$script:SetupLogPath = $null

function New-CommandResult {
    param(
        [int]$ExitCode,
        [string]$Text,
        [string]$DisplayCommand
    )

    return [PSCustomObject]@{
        ExitCode      = $ExitCode
        Text          = $Text
        DisplayCommand = $DisplayCommand
    }
}

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [string[]]$PrefixArguments = @(),

        [string[]]$Arguments = @(),

        [switch]$ShowOutput,

        [switch]$AppendLog
    )

    $AllArguments = @()
    $AllArguments += @($PrefixArguments)
    $AllArguments += @($Arguments)
    $DisplayCommand = (($Executable) + " " + ($AllArguments -join " ")).Trim()

    # Windows PowerShell 5.1 can promote a native process' stderr to a
    # terminating NativeCommandError when the script-wide preference is Stop.
    # A failed probe (for example, an unavailable `py -3.13`) must instead be
    # captured normally so runtime detection can continue to the next choice.
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $RawOutput = & $Executable @AllArguments 2>&1
        $ExitCode = $LASTEXITCODE
    }
    catch {
        $RawOutput = @($_.Exception.Message)
        $ExitCode = 1
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($null -eq $ExitCode) {
        $ExitCode = 1
    }

    $OutputLines = @($RawOutput | ForEach-Object { $_.ToString() })
    $Text = $OutputLines -join [Environment]::NewLine

    if ($ShowOutput -and -not [string]::IsNullOrWhiteSpace($Text)) {
        Write-Host $Text
    }

    if ($AppendLog -and -not [string]::IsNullOrWhiteSpace($script:SetupLogPath)) {
        Add-Content -LiteralPath $script:SetupLogPath -Encoding UTF8 -Value ""
        Add-Content -LiteralPath $script:SetupLogPath -Encoding UTF8 -Value ("> " + $DisplayCommand)
        Add-Content -LiteralPath $script:SetupLogPath -Encoding UTF8 -Value ("ExitCode: " + $ExitCode)
        if (-not [string]::IsNullOrWhiteSpace($Text)) {
            Add-Content -LiteralPath $script:SetupLogPath -Encoding UTF8 -Value $Text
        }
    }

    return New-CommandResult -ExitCode ([int]$ExitCode) -Text $Text -DisplayCommand $DisplayCommand
}

function Get-PythonRuntime {
    $Candidates = New-Object System.Collections.Generic.List[object]

    # Prefer an existing uv/project environment near the extracted setup
    # folder. This avoids requiring python.exe on PATH when the repository is
    # already managed by uv.
    $SearchRoots = @(
        $PSScriptRoot,
        (Split-Path -Parent $PSScriptRoot),
        (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
    )
    $SeenPythonPaths = @{}
    foreach ($SearchRoot in $SearchRoots) {
        if ([string]::IsNullOrWhiteSpace("$SearchRoot")) {
            continue
        }
        $ProjectPython = Join-Path $SearchRoot ".venv\Scripts\python.exe"
        if ((Test-Path -LiteralPath $ProjectPython) -and -not $SeenPythonPaths.ContainsKey($ProjectPython)) {
            $SeenPythonPaths[$ProjectPython] = $true
            $Candidates.Add([PSCustomObject]@{
                Executable = $ProjectPython
                Prefix     = @()
            })
        }
    }

    if ($null -ne (Get-Command py -ErrorAction SilentlyContinue)) {
        foreach ($VersionSelector in @("-3.13", "-3.12", "-3.11", "-3")) {
            $Candidates.Add([PSCustomObject]@{
                Executable = "py"
                Prefix     = @($VersionSelector)
            })
        }
    }

    foreach ($ExecutableName in @("python", "python3")) {
        if ($null -ne (Get-Command $ExecutableName -ErrorAction SilentlyContinue)) {
            $Candidates.Add([PSCustomObject]@{
                Executable = $ExecutableName
                Prefix     = @()
            })
        }
    }

    $Detected = New-Object System.Collections.Generic.List[string]

    foreach ($Candidate in $Candidates) {
        $ProbeCode = "import sys; print(str(sys.version_info.major)+'|'+str(sys.version_info.minor)+'|'+sys.version.split()[0]+'|'+sys.executable)"
        $Probe = Invoke-NativeCapture `
            -Executable ([string]$Candidate.Executable) `
            -PrefixArguments @($Candidate.Prefix) `
            -Arguments @("-c", $ProbeCode)

        if ($Probe.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($Probe.Text)) {
            continue
        }

        $ProbeLines = @($Probe.Text -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($ProbeLines.Length -eq 0) {
            continue
        }

        $Parts = $ProbeLines[$ProbeLines.Length - 1].Split('|')
        if ($Parts.Length -lt 4) {
            continue
        }

        $Major = 0
        $Minor = 0
        if (-not [int]::TryParse($Parts[0], [ref]$Major)) {
            continue
        }
        if (-not [int]::TryParse($Parts[1], [ref]$Minor)) {
            continue
        }

        $Version = [string]$Parts[2]
        $ExecutablePath = [string]$Parts[3]
        $Detected.Add(("{0} {1} -> Python {2}" -f $Candidate.Executable, (@($Candidate.Prefix) -join " "), $Version))

        if (($Major -gt 3) -or (($Major -eq 3) -and ($Minor -ge 11))) {
            return [PSCustomObject]@{
                Executable     = [string]$Candidate.Executable
                Prefix         = @($Candidate.Prefix)
                Version        = $Version
                ExecutablePath = $ExecutablePath
            }
        }
    }

    $DetectedText = "見つかりませんでした。"
    if ($Detected.ToArray().Length -gt 0) {
        $DetectedText = $Detected.ToArray() -join [Environment]::NewLine
    }

    throw @"
Python 3.11以上が見つかりません。
現在のKaggle CLIはPython 3.11以上が必要です。
検出結果:
$DetectedText

Python 3.11/3.12/3.13をインストールし、インストーラーで
'Add python.exe to PATH' と 'Install launcher for all users' を有効にしてください。
"@
}

$PythonRuntime = Get-PythonRuntime
$PythonExe = [string]$PythonRuntime.Executable
$PythonPrefixArgs = @($PythonRuntime.Prefix)

function Invoke-PythonCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [switch]$ShowOutput,

        [switch]$AppendLog
    )

    return Invoke-NativeCapture `
        -Executable $PythonExe `
        -PrefixArguments $PythonPrefixArgs `
        -Arguments $Arguments `
        -ShowOutput:$ShowOutput `
        -AppendLog:$AppendLog
}

function Ensure-Pip {
    $PipCheck = Invoke-PythonCapture -Arguments @("-m", "pip", "--version") -AppendLog
    if ($PipCheck.ExitCode -eq 0) {
        Write-Host ("pip: {0}" -f (($PipCheck.Text -split "`r?`n")[0]))
        return
    }

    Write-Warning "pipが利用できないため、Python標準のensurepipで復旧します。"
    $Ensure = Invoke-PythonCapture -Arguments @("-m", "ensurepip", "--upgrade") -ShowOutput -AppendLog
    if ($Ensure.ExitCode -ne 0) {
        throw "pipの準備に失敗しました。詳細: $script:SetupLogPath"
    }

    $PipCheck = Invoke-PythonCapture -Arguments @("-m", "pip", "--version") -ShowOutput -AppendLog
    if ($PipCheck.ExitCode -ne 0) {
        throw "pipを実行できません。詳細: $script:SetupLogPath"
    }
}

function Test-KaggleCli {
    $Probe = Invoke-PythonCapture -Arguments @("-m", "kaggle", "--version") -AppendLog
    if ($Probe.ExitCode -eq 0) {
        if (-not [string]::IsNullOrWhiteSpace($Probe.Text)) {
            Write-Host ("Kaggle CLI: {0}" -f (($Probe.Text -split "`r?`n")[0]))
        }
        return $true
    }
    return $false
}

function Ensure-KaggleCli {
    if (Test-KaggleCli) {
        return
    }

    Write-Host "Kaggle CLIをインストールしています..."
    $InstallArguments = @(
        "-m", "pip", "install",
        "--disable-pip-version-check",
        "--upgrade",
        "kaggle"
    )

    $Install = Invoke-PythonCapture -Arguments $InstallArguments -ShowOutput -AppendLog
    if ($Install.ExitCode -ne 0) {
        Write-Warning "通常インストールに失敗したため、ユーザー領域へ再試行します。"
        $UserInstallArguments = @(
            "-m", "pip", "install",
            "--disable-pip-version-check",
            "--user",
            "--upgrade",
            "kaggle"
        )
        $Install = Invoke-PythonCapture -Arguments $UserInstallArguments -ShowOutput -AppendLog
    }

    if ($Install.ExitCode -ne 0) {
        throw @"
Kaggle CLIのインストールに失敗しました。
実際のpip出力は次のログに保存しました:
$script:SetupLogPath
"@
    }

    if (-not (Test-KaggleCli)) {
        throw "Kaggle CLIをインストールしましたが起動確認に失敗しました。詳細: $script:SetupLogPath"
    }
}

function Invoke-Kaggle {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [ValidateRange(1, 10)]
        [int]$MaxAttempts = 3
    )

    for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
        $Result = Invoke-PythonCapture `
            -Arguments (@("-m", "kaggle") + $Arguments) `
            -ShowOutput `
            -AppendLog

        if ($Result.ExitCode -eq 0) {
            return
        }

        if ($Attempt -lt $MaxAttempts) {
            $Delay = [int][Math]::Min(30, [Math]::Pow(2, $Attempt))
            Write-Warning "Kaggleコマンドに失敗しました。$Delay 秒後に再試行します ($Attempt/$MaxAttempts)。"
            Start-Sleep -Seconds $Delay
        }
    }

    throw "Kaggleコマンドが失敗しました: kaggle $($Arguments -join ' ')。詳細: $script:SetupLogPath"
}

function Invoke-KaggleCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $Result = Invoke-PythonCapture `
        -Arguments (@("-m", "kaggle") + $Arguments) `
        -AppendLog

    if ($Result.ExitCode -ne 0) {
        throw "Kaggleコマンドが失敗しました: kaggle $($Arguments -join ' ')`n$($Result.Text)"
    }

    return $Result.Text
}

function Get-SafePathPart {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $Safe = $Value -replace '[<>:"/\\|?*]', '_'
    $Safe = $Safe.Trim().TrimEnd('.')
    if ([string]::IsNullOrWhiteSpace($Safe)) {
        return "unnamed"
    }
    return $Safe
}

function Get-PropertyValue {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,

        [Parameter(Mandatory = $true)]
        [string[]]$Names
    )

    foreach ($Name in $Names) {
        $Property = $Object.PSObject.Properties[$Name]
        if ($null -ne $Property -and $null -ne $Property.Value -and "$($Property.Value)" -ne "") {
            return $Property.Value
        }
    }
    return $null
}

function Get-UniqueTopRows {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Rows,

        [Parameter(Mandatory = $true)]
        [int]$Limit
    )

    $Seen = @{}
    $Selected = New-Object System.Collections.Generic.List[object]

    foreach ($Row in $Rows) {
        $Ref = Get-PropertyValue -Object $Row -Names @("ref", "Ref")
        if ([string]::IsNullOrWhiteSpace("$Ref")) {
            continue
        }
        if ("$Ref" -notmatch '^[^/]+/[^/]+$') {
            continue
        }
        if ($Seen.ContainsKey("$Ref")) {
            continue
        }

        $Seen["$Ref"] = $true
        $Selected.Add($Row)
        if ($Selected.ToArray().Length -ge $Limit) {
            break
        }
    }

    return $Selected.ToArray()
}

function Get-KaggleJsonArrayText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    # Kaggle CLI 2.x may print "Next Page Token = ..." before the requested
    # JSON. Extract only the outer JSON array so ConvertFrom-Json remains
    # stable across paginated and non-paginated responses.
    $Start = $Text.IndexOf('[')
    $End = $Text.LastIndexOf(']')
    if ($Start -lt 0 -or $End -lt $Start) {
        throw "$Description のJSON配列をKaggle CLI出力から抽出できませんでした。"
    }

    return $Text.Substring($Start, $End - $Start + 1)
}

$TopNotebookRoot = Join-Path $Root "notebooks\top_by_score"
$CompetitionDir = Join-Path $Root "data\competition"
$SupportDir = Join-Path $Root "data\support_pack"
$ManifestDir = Join-Path $Root "manifests"

if ((Test-Path -LiteralPath $TopNotebookRoot) -and -not $KeepExistingNotebooks) {
    Remove-Item -LiteralPath $TopNotebookRoot -Recurse -Force
}

$Directories = @(
    $TopNotebookRoot,
    $CompetitionDir,
    $SupportDir,
    $ManifestDir
)
foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}

$script:SetupLogPath = Join-Path $ManifestDir "setup.log"
Set-Content -LiteralPath $script:SetupLogPath -Encoding UTF8 -Value @(
    "Biohub Kaggle setup v5",
    ("Started: " + (Get-Date).ToString("o")),
    ("PowerShell: " + $PSVersionTable.PSVersion.ToString()),
    ("Root: " + $Root)
)

Write-Host "PythonとKaggle CLIを準備しています..."
Write-Host ("使用Python: {0} ({1})" -f $PythonRuntime.Version, $PythonRuntime.ExecutablePath)
Write-Host ("セットアップログ: {0}" -f $SetupLogPath)

Ensure-Pip
Ensure-KaggleCli

# 認証状態を実際のAPI呼び出しで確認し、未認証ならOAuthログインを開始します。
try {
    $null = Invoke-KaggleCapture -Arguments @("kernels", "list", "--page-size", "1", "-v")
}
catch {
    Write-Host "Kaggle認証が必要です。ブラウザでログインしてください。"
    Invoke-Kaggle -Arguments @("auth", "login")
    $null = Invoke-KaggleCapture -Arguments @("kernels", "list", "--page-size", "1", "-v")
}

Write-Host "[1/8] 競技ファイル一覧を保存"
$CompetitionFilesText = Invoke-KaggleCapture -Arguments @(
    "competitions", "files", $Competition,
    "--page-size", "200",
    "-v"
)
$CompetitionFilesText | Set-Content -Path (Join-Path $ManifestDir "competition_files.csv") -Encoding UTF8

Write-Host "[2/8] 公開Python NotebookをKaggleスコア降順で取得"
$KernelOutputText = Invoke-KaggleCapture -Arguments @(
    "kernels", "list",
    "--competition", $Competition,
    "--sort-by", "scoreDescending",
    "--language", "python",
    "--kernel-type", "notebook",
    "--page-size", "$TopCount",
    "--format", "json"
)
$KernelListPath = Join-Path $ManifestDir "kernels_by_score.json"
$KernelJsonText = Get-KaggleJsonArrayText -Text $KernelOutputText -Description "Notebook一覧"
$KernelJsonText | Set-Content -Path $KernelListPath -Encoding UTF8

try {
    $ParsedKernelRows = ConvertFrom-Json -InputObject $KernelJsonText
    $KernelRowList = New-Object System.Collections.Generic.List[object]
    foreach ($ParsedKernelRow in $ParsedKernelRows) {
        $KernelRowList.Add($ParsedKernelRow)
    }
    $KernelRows = $KernelRowList.ToArray()
}
catch {
    throw "Notebook一覧JSONを解析できませんでした。内容を確認してください: $KernelListPath"
}

$TopRows = @(Get-UniqueTopRows -Rows $KernelRows -Limit $TopCount)
if ($TopRows.Length -eq 0) {
    throw "競技に関連付けられた公開Python Notebookを取得できませんでした。Kaggle認証、競技参加、CLI出力を確認してください。"
}
if ($TopRows.Length -lt $TopCount) {
    throw "取得可能な公開Python Notebookは $($TopRows.Length) 件で、指定数 $TopCount 件に届きません。"
}

Write-Host "[3/8] Top $($TopRows.Length) Notebookを個別フォルダーへダウンロード"
$DownloadResults = New-Object System.Collections.Generic.List[object]
$Rank = 0

foreach ($Row in $TopRows) {
    $Rank++
    $Ref = "$(Get-PropertyValue -Object $Row -Names @('ref', 'Ref'))"
    $Title = Get-PropertyValue -Object $Row -Names @("title", "Title")
    $Author = Get-PropertyValue -Object $Row -Names @("author", "Author")
    $LastRunTime = Get-PropertyValue -Object $Row -Names @("lastRunTime", "LastRunTime")
    $TotalVotes = Get-PropertyValue -Object $Row -Names @("totalVotes", "TotalVotes")

    $RefParts = $Ref.Split('/', 2)
    $OwnerPart = Get-SafePathPart -Value $RefParts[0]
    $SlugPart = Get-SafePathPart -Value $RefParts[1]
    $FolderName = "{0:D2}_{1}_{2}" -f $Rank, $OwnerPart, $SlugPart
    $NotebookDir = Join-Path $TopNotebookRoot $FolderName

    if ((Test-Path -LiteralPath $NotebookDir) -and -not $KeepExistingNotebooks) {
        Remove-Item -LiteralPath $NotebookDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $NotebookDir -Force | Out-Null

    Write-Host ("  [{0}/{1}] {2}" -f $Rank, $TopRows.Length, $Ref)

    $Status = "downloaded"
    $NotebookFiles = @()
    $ErrorMessage = ""
    try {
        Invoke-Kaggle -Arguments @("kernels", "pull", $Ref, "-p", $NotebookDir, "-m")
        $NotebookFiles = @(Get-ChildItem -LiteralPath $NotebookDir -Filter "*.ipynb" -File -Recurse)
        if ($NotebookFiles.Length -eq 0) {
            $Status = "no_ipynb_found"
            $ErrorMessage = "ダウンロードは完了しましたが、.ipynbファイルが見つかりませんでした。"
            Write-Warning "$Ref : $ErrorMessage"
        }
    }
    catch {
        $Status = "failed"
        $ErrorMessage = $_.Exception.Message
        Write-Warning "$Ref の取得に失敗しました。残りを続行します。"
    }

    $DownloadResults.Add([PSCustomObject]@{
        rank          = $Rank
        ref           = $Ref
        title         = $Title
        author        = $Author
        lastRunTime   = $LastRunTime
        totalVotes    = $TotalVotes
        sourceOrder   = "scoreDescending"
        downloadedAt = (Get-Date).ToString("o")
        status        = $Status
        ipynbCount    = $NotebookFiles.Length
        folder        = $NotebookDir
        error         = $ErrorMessage
    })
}

$TopManifestCsv = Join-Path $ManifestDir "top_notebooks_downloaded.csv"
$TopManifestJson = Join-Path $ManifestDir "top_notebooks_downloaded.json"
$DownloadResults | Export-Csv -Path $TopManifestCsv -NoTypeInformation -Encoding UTF8
$DownloadResults | ConvertTo-Json -Depth 5 | Set-Content -Path $TopManifestJson -Encoding UTF8

Write-Host "[4/8] Discussion一覧を保存"
try {
    $DiscussionText = Invoke-KaggleCapture -Arguments @(
        "competitions", "topics", "list", $Competition,
        "--sort-by", "recent",
        "--page-size", "100",
        "-v"
    )
    $DiscussionText | Set-Content -Path (Join-Path $ManifestDir "discussions_recent.csv") -Encoding UTF8
}
catch {
    Write-Warning "Discussion一覧は取得できませんでした。処理を続行します。"
}

if ($SkipCompetitionData) {
    Write-Host "[5/8] 競技データの取得をスキップ"
}
else {
    Write-Host "[5/8] 競技データを取得（大容量）"
    # Do not use --force/-o here. The competition archive is ~81 GB and the
    # Kaggle CLI can resume a partial archive after transient SSL failures only
    # when the existing file is preserved.
    Invoke-Kaggle -Arguments @("competitions", "download", $Competition, "-p", $CompetitionDir)
}

if ($SkipSupportPack) {
    Write-Host "[6/8] Support Packの取得をスキップ"
}
else {
    Write-Host "[6/8] Support Packを取得・展開"
    Invoke-Kaggle -Arguments @("datasets", "download", $SupportDataset, "-p", $SupportDir, "--unzip", "-o")
}

if (-not $SkipCompetitionData -and -not $SkipExtraction) {
    Write-Host "[7/8] 競技ZIPを展開"
    $ZipFiles = @(Get-ChildItem -Path $CompetitionDir -Filter "*.zip" -File -Recurse)
    foreach ($ZipFile in $ZipFiles) {
        Write-Host "  展開中: $($ZipFile.FullName)"
        Expand-Archive -LiteralPath $ZipFile.FullName -DestinationPath $CompetitionDir -Force
    }
}
else {
    Write-Host "[7/8] 競技ZIPの展開をスキップ"
}

Write-Host "[8/8] 結果を確認"
$Successful = @($DownloadResults | Where-Object { $_.status -eq "downloaded" }).Length
$WithIpynb = @($DownloadResults | Where-Object { $_.ipynbCount -gt 0 }).Length
$Failed = @($DownloadResults | Where-Object { $_.status -eq "failed" }).Length

Write-Host ""
Write-Host "完了しました: $Root"
Write-Host ""
Write-Host "Top Notebook保存先 : $TopNotebookRoot"
Write-Host "取得対象数           : $($TopRows.Length)"
Write-Host "取得成功             : $Successful"
Write-Host ".ipynb確認済み       : $WithIpynb"
Write-Host "取得失敗             : $Failed"
Write-Host "ランキング一覧       : $KernelListPath"
Write-Host "取得結果CSV          : $TopManifestCsv"
Write-Host "取得結果JSON         : $TopManifestJson"
Write-Host "競技データ           : $CompetitionDir"
Write-Host "Support Pack         : $SupportDir"
Write-Host ""
Write-Host "注意:"
Write-Host "  - 順位はKaggle CLIの scoreDescending が返す公開Notebook順です。"
Write-Host "  - 各Notebookは取得時点の最新公開バージョンです。"
Write-Host "  - Notebook内の /kaggle/input/... は、ローカル実行時にパス変更が必要です。"
Write-Host "  - 競技データは非常に大きいため、十分な空き容量を確保してください。"

if ($Failed -gt 0 -or $WithIpynb -ne $TopCount) {
    throw "Top $TopCount の.ipynbをすべて取得できませんでした。詳細は $TopManifestCsv を確認してください。"
}
