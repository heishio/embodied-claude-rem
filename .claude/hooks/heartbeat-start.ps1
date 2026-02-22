# heartbeat-start.ps1 - 心拍デーモン起動 (PowerShell版)
# 使い方: .\heartbeat-start.ps1

$GitBash = "C:\Program Files\Git\bin\bash.exe"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Daemon = "$ScriptDir\heartbeat-start.sh"

if (-not (Test-Path $GitBash)) {
    Write-Error "Git Bash not found at $GitBash"
    exit 1
}

Write-Host "Starting heartbeat daemon via Git Bash..."
& $GitBash $Daemon
