# Register (or refresh) a Windows Scheduled Task that re-runs the AgriRisk
# pipeline - the M4 "scheduled pipeline runs" entry point for local/Windows
# hosting. On Linux/cron the equivalent is simply:
#     0 6 * * * cd /path/to/AgriRisk && .venv/bin/python -m agrik
#
# Usage (from the repo root; -WhatIf supported to preview):
#     pwsh -File scripts/schedule_pipeline.ps1               # daily 06:00
#     pwsh -File scripts/schedule_pipeline.ps1 -AtHour 3 -AtMinute 30
#     pwsh -File scripts/schedule_pipeline.ps1 -Remove
#
# Notes:
#  * Runs interactively as the current user (StartWhenAvailable: catches up if
#    the machine was asleep at the trigger time).
#  * The pipeline is idempotent and offline once feeds are cached, so a daily
#    run only refreshes derived artefacts (data/, models/).
#  * Add `--set external.openmeteo.enabled=true` to ArgumentList if you want
#    the ERA5 temperature enrichment refreshed too.
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$TaskName = "AgriRiskPipelineRun",
    [ValidateRange(0, 23)][int]$AtHour = 6,
    [ValidateRange(0, 59)][int]$AtMinute = 0,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if ($Remove) {
    if ($PSCmdlet.ShouldProcess($TaskName, "Unregister scheduled task")) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Output "Removed scheduled task '$TaskName'."
    }
    return
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "venv not found at $python - create it first (uv venv .venv && pip install -e .)"
}

$action  = New-ScheduledTaskAction -Execute $python `
    -Argument "-m agrik" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At ("{0:d2}:{1:d2}" -f $AtHour, $AtMinute)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

if ($PSCmdlet.ShouldProcess($TaskName, "Register daily scheduled task")) {
    Register-ScheduledTask -TaskName $TaskName -Action $action `
        -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Output "Scheduled '$TaskName': daily $("{0:d2}:{1:d2}" -f $AtHour, $AtMinute) in $root"
}
