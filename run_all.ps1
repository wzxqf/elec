#Requires -Version 5.1

[CmdletBinding()]
param(
    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($null -eq $ExtraArgs) {
    $ExtraArgs = @()
}

if ($ExtraArgs.Count -eq 1 -and $ExtraArgs[0] -eq "--dry-run") {
    $DryRun = $true
    $ExtraArgs = @()
}

if ($ExtraArgs.Count -gt 0) {
    Write-Error "Unsupported arguments: $($ExtraArgs -join ' ')`nUsage: powershell -ExecutionPolicy Bypass -File .\run_all.ps1 [-DryRun]"
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir
$ConfigPath = Join-Path $ProjectRoot "experiment_config.yaml"

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    Write-Error "Configuration file not found: $ConfigPath"
    exit 1
}

$Version = $null
$InProjectSection = $false
foreach ($Line in Get-Content -LiteralPath $ConfigPath) {
    if ($Line -match '^\s*#') {
        continue
    }
    if ($Line -match '^project\s*:\s*$') {
        $InProjectSection = $true
        continue
    }
    if ($InProjectSection -and $Line -match '^\S') {
        break
    }
    if ($InProjectSection -and $Line -match '^\s+version\s*:\s*(.+?)\s*(?:#.*)?$') {
        $Version = $Matches[1].Trim().Trim('"').Trim("'")
        break
    }
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    Write-Error "Unable to parse project.version from $ConfigPath"
    exit 1
}

$env:MPLCONFIGDIR = Join-Path $ProjectRoot ".cache\matplotlib"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

function Add-CandidatePython {
    param(
        [System.Collections.Generic.List[string]]$Candidates,
        [string]$Path
    )

    if (-not [string]::IsNullOrWhiteSpace($Path) -and -not $Candidates.Contains($Path)) {
        $Candidates.Add($Path)
    }
}

$EnvName = "torch311"
$PythonCandidates = [System.Collections.Generic.List[string]]::new()

if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
    $CondaEnvName = Split-Path -Leaf $env:CONDA_PREFIX
    if ($CondaEnvName -eq $EnvName) {
        Add-CandidatePython $PythonCandidates (Join-Path $env:CONDA_PREFIX "python.exe")
    }
}

if (-not [string]::IsNullOrWhiteSpace($env:MAMBA_ROOT_PREFIX)) {
    Add-CandidatePython $PythonCandidates (Join-Path $env:MAMBA_ROOT_PREFIX "envs\$EnvName\python.exe")
}

if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE)) {
    $CondaRoot = Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE)
    Add-CandidatePython $PythonCandidates (Join-Path $CondaRoot "envs\$EnvName\python.exe")
}

if (-not [string]::IsNullOrWhiteSpace($env:MAMBA_EXE)) {
    $MambaRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $env:MAMBA_EXE))
    Add-CandidatePython $PythonCandidates (Join-Path $MambaRoot "envs\$EnvName\python.exe")
}

Add-CandidatePython $PythonCandidates "D:\miniforge\envs\$EnvName\python.exe"
Add-CandidatePython $PythonCandidates "$env:USERPROFILE\miniforge3\envs\$EnvName\python.exe"
Add-CandidatePython $PythonCandidates "$env:USERPROFILE\mambaforge\envs\$EnvName\python.exe"

$PythonExe = $null
foreach ($Candidate in $PythonCandidates) {
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        $PythonExe = $Candidate
        break
    }
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    Write-Error "Unable to find python.exe for environment '$EnvName'. Activate torch311 first, or install it under a standard Miniforge/Mambaforge envs directory."
    exit 1
}

$RunArgs = @("-m", "src.scripts.run_pipeline")
$DisplayCommand = "$PythonExe $($RunArgs -join ' ')"
$OutputDir = Join-Path (Join-Path $ProjectRoot "outputs") $Version

Write-Host "Project root: $ProjectRoot"
Write-Host "Experiment version: $Version"
Write-Host "Output dir: $OutputDir"
Write-Host "Command: $DisplayCommand"

if ($DryRun) {
    exit 0
}

Push-Location $ProjectRoot
try {
    & $PythonExe @RunArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
