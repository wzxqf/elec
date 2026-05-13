param(
    [string]$Url = $env:ELEC_JUPYTER_URL,
    [string]$Kernel = $env:ELEC_JUPYTER_KERNEL,
    [string]$RemoteEnv = $env:ELEC_REMOTE_ENV,
    [string]$RemotePython = $env:ELEC_REMOTE_PYTHON,
    [string]$RunId = "",
    [switch]$ProbeOnly,
    [switch]$SkipProbe,
    [switch]$DryRun,
    [switch]$NoSyncLocalOutput
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonCandidates = @(
    "D:\miniforge\envs\torch311\python.exe",
    "D:\miniforge\python.exe"
)

$Python = $PythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Python) {
    $Python = (Get-Command "python" -ErrorAction Stop).Source
}

if (-not $Url) { $Url = "http://10.26.27.72:9007/" }
if (-not $Kernel) { $Kernel = "python3" }
if (-not $RemoteEnv) { $RemoteEnv = "torch311" }

$BaseArgs = @(
    "run_remote_jupyter.py",
    "--url", $Url,
    "--kernel", $Kernel,
    "--remote-env", $RemoteEnv
)
if ($RemotePython) { $BaseArgs += @("--remote-python", $RemotePython) }
if ($RunId) { $BaseArgs += @("--run-id", $RunId) }
if ($DryRun) { $BaseArgs += "--dry-run" }
if ($NoSyncLocalOutput) { $BaseArgs += "--no-sync-local-output" }

if ($ProbeOnly) {
    & $Python @BaseArgs "--probe"
    exit $LASTEXITCODE
}

if (-not $SkipProbe) {
    & $Python @BaseArgs "--probe"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $Python @BaseArgs
exit $LASTEXITCODE
