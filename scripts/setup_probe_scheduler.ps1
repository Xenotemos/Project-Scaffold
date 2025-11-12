# Creates or removes a Windows Scheduled Task that runs the continuous probe injector on a cadence.
param(
    [string]$TaskName = "ContinuousProbesNightly",
    [int]$IntervalMinutes = 30,
    [Nullable[datetime]]$StartTime = $null,
    [switch]$Remove
)

if ($Remove) {
    schtasks /Delete /TN $TaskName /F | Out-Null
    Write-Host "Removed scheduled task '$TaskName' (if it existed)."
    return
}

if ($IntervalMinutes -lt 1) {
    throw "IntervalMinutes must be >= 1."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv-win\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    throw "Expected Python executable at $pythonPath. Activate or create the virtualenv before scheduling."
}

$start = if ($StartTime) { [datetime]$StartTime } else { (Get-Date).AddMinutes(5) }
$startString = $start.ToString("HH:mm")

$projectRootQuoted = '"' + $projectRoot + '"'
$pythonQuoted = '"' + $pythonPath + '"'
$command = "cmd.exe /c cd /d $projectRootQuoted && $pythonQuoted -m scripts.continuous_probes --log-dir logs\probe_runs --profiles instruct base --no-retain-rows --reset-session"

schtasks /Create `
    /SC MINUTE `
    /MO $IntervalMinutes `
    /TN $TaskName `
    /TR $command `
    /ST $startString `
    /RL LIMITED `
    /RU $env:USERNAME `
    /F | Out-Null

Write-Host "Scheduled task '$TaskName' configured to run every $IntervalMinutes minute(s) starting at $startString."
