[CmdletBinding()]
param(
    [switch]$ProbeOnly,
    [switch]$SkipProbe,
    [switch]$DryRun,
    [string]$RunId,
    [string]$RemotePython,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemoteRunnerArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Add-Candidate {
    param(
        [System.Collections.Generic.List[string]]$Candidates,
        [string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return
    }
    if (-not $Candidates.Contains($Value)) {
        $Candidates.Add($Value) | Out-Null
    }
}

function Add-Env-PythonCandidate {
    param(
        [System.Collections.Generic.List[string]]$Candidates,
        [string]$EnvPath
    )
    if ([string]::IsNullOrWhiteSpace($EnvPath)) {
        return
    }
    Add-Candidate $Candidates (Join-Path $EnvPath "python.exe")
    Add-Candidate $Candidates (Join-Path $EnvPath "bin/python")
}

function Test-PythonCandidate {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }
    $command = Get-Command $Path -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return $false
    }
    & $command.Source --version *> $null
    return $LASTEXITCODE -eq 0
}

function Resolve-LocalPython {
    $candidates = [System.Collections.Generic.List[string]]::new()

    Add-Candidate $candidates $env:ELEC_LOCAL_PYTHON
    Add-Candidate $candidates $env:PYTHON
    Add-Env-PythonCandidate $candidates $env:CONDA_PREFIX
    if (-not [string]::IsNullOrWhiteSpace($env:MAMBA_ROOT_PREFIX)) {
        Add-Env-PythonCandidate $candidates (Join-Path $env:MAMBA_ROOT_PREFIX "envs\torch311")
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_ROOT)) {
        Add-Env-PythonCandidate $candidates (Join-Path $env:CONDA_ROOT "envs\torch311")
    }
    Add-Candidate $candidates "D:\miniforge\envs\torch311\python.exe"
    Add-Candidate $candidates "D:\miniforge3\envs\torch311\python.exe"
    Add-Candidate $candidates "D:\mambaforge\envs\torch311\python.exe"
    Add-Candidate $candidates "D:\miniforge\python.exe"

    foreach ($candidate in $candidates) {
        if (Test-PythonCandidate $candidate) {
            return (Get-Command $candidate).Source
        }
    }

    throw "Unable to find a runnable local Python. Set ELEC_LOCAL_PYTHON to D:\miniforge\envs\torch311\python.exe or another project-compatible interpreter."
}

function Invoke-RemoteRunner {
    param(
        [string]$PythonPath,
        [string]$RunnerPath,
        [string[]]$Arguments
    )
    & $PythonPath $RunnerPath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Remote Jupyter runner failed with exit code $exitCode."
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runnerPath = Join-Path $projectRoot "run_remote_jupyter.py"

if (-not (Test-Path -LiteralPath $runnerPath)) {
    throw "Missing runner script: $runnerPath"
}

if ([string]::IsNullOrWhiteSpace($env:ELEC_JUPYTER_URL)) {
    $env:ELEC_JUPYTER_URL = "http://10.26.27.72:9007/"
}
if ([string]::IsNullOrWhiteSpace($env:ELEC_JUPYTER_KERNEL)) {
    $env:ELEC_JUPYTER_KERNEL = "python3"
}
if ([string]::IsNullOrWhiteSpace($env:ELEC_REMOTE_ENV)) {
    $env:ELEC_REMOTE_ENV = "torch311"
}

$pythonPath = Resolve-LocalPython
$baseArgs = [System.Collections.Generic.List[string]]::new()
if (-not [string]::IsNullOrWhiteSpace($RunId)) {
    $baseArgs.Add("--run-id") | Out-Null
    $baseArgs.Add($RunId) | Out-Null
}
if (-not [string]::IsNullOrWhiteSpace($RemotePython)) {
    $baseArgs.Add("--remote-python") | Out-Null
    $baseArgs.Add($RemotePython) | Out-Null
}
foreach ($arg in $RemoteRunnerArgs) {
    $baseArgs.Add($arg) | Out-Null
}

Write-Host "Local Python: $pythonPath"
Write-Host "Jupyter URL: $env:ELEC_JUPYTER_URL"
Write-Host "Jupyter kernel: $env:ELEC_JUPYTER_KERNEL"
Write-Host "Remote project env: $env:ELEC_REMOTE_ENV"

if ($DryRun) {
    Invoke-RemoteRunner $pythonPath $runnerPath (@("--dry-run") + $baseArgs.ToArray())
    exit 0
}

if ([string]::IsNullOrWhiteSpace($env:ELEC_JUPYTER_PASSWORD) -and [string]::IsNullOrWhiteSpace($env:ELEC_JUPYTER_TOKEN)) {
    throw "Set ELEC_JUPYTER_PASSWORD or ELEC_JUPYTER_TOKEN in the local environment before running remote validation."
}

if (-not $SkipProbe) {
    Invoke-RemoteRunner $pythonPath $runnerPath (@("--probe") + $baseArgs.ToArray())
}
if ($ProbeOnly) {
    exit 0
}

Invoke-RemoteRunner $pythonPath $runnerPath $baseArgs.ToArray()
