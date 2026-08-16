[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Source,
    [Parameter(Mandatory)][string]$Destination,
    [Int64]$LocalDataLength = 87390335718,
    [Int64]$CentralDirectoryStart = 149941648867,
    [Int64]$ExpectedOutputLength = 87393127165
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Source = [System.IO.Path]::GetFullPath($Source)
$Destination = [System.IO.Path]::GetFullPath($Destination)
$Temporary = "$Destination.repairing"

if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    throw "Source archive was not found: $Source"
}
if ($Source -eq $Destination) {
    throw "Source and destination must differ."
}

$SourceLength = (Get-Item -LiteralPath $Source).Length
$TailLength = $SourceLength - $CentralDirectoryStart
if ($SourceLength -ne 149944440314 -or $TailLength -le 0) {
    throw "The source size does not match the known recoverable archive layout: $SourceLength"
}
if (($LocalDataLength + $TailLength) -ne $ExpectedOutputLength) {
    throw "Repair ranges do not produce the expected archive size."
}

function Copy-Range {
    param(
        [Parameter(Mandatory)][System.IO.FileStream]$InputStream,
        [Parameter(Mandatory)][System.IO.FileStream]$OutputStream,
        [Parameter(Mandatory)][Int64]$Length,
        [Parameter(Mandatory)][string]$Label
    )

    $Buffer = New-Object byte[] (16MB)
    [Int64]$Copied = 0
    [Int64]$NextReport = 5GB
    while ($Copied -lt $Length) {
        $Wanted = [int][Math]::Min([Int64]$Buffer.Length, ($Length - $Copied))
        $Read = $InputStream.Read($Buffer, 0, $Wanted)
        if ($Read -le 0) {
            throw "Unexpected end of source while copying $Label."
        }
        $OutputStream.Write($Buffer, 0, $Read)
        $Copied += $Read
        if ($Copied -ge $NextReport) {
            Write-Host ("{0}: {1:N2}/{2:N2} GiB" -f $Label, ($Copied / 1GB), ($Length / 1GB))
            $NextReport += 5GB
        }
    }
}

$Input = [System.IO.FileStream]::new(
    $Source, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
    [System.IO.FileShare]::Read, 16MB, [System.IO.FileOptions]::SequentialScan
)
$Output = [System.IO.FileStream]::new(
    $Temporary, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None, 16MB, [System.IO.FileOptions]::SequentialScan
)
try {
    Copy-Range -InputStream $Input -OutputStream $Output -Length $LocalDataLength -Label "Local file data"
    [void]$Input.Seek($CentralDirectoryStart, [System.IO.SeekOrigin]::Begin)
    Copy-Range -InputStream $Input -OutputStream $Output -Length $TailLength -Label "Central directory"
    $Output.Flush($true)
}
finally {
    $Output.Dispose()
    $Input.Dispose()
}

$ActualLength = (Get-Item -LiteralPath $Temporary).Length
if ($ActualLength -ne $ExpectedOutputLength) {
    throw "Repaired archive has an unexpected size: $ActualLength"
}
Move-Item -LiteralPath $Temporary -Destination $Destination
Write-Host "Repaired archive created: $Destination ($ActualLength bytes)"
