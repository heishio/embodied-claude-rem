# heartbeat-stop.ps1 - 心拍デーモン停止 (PowerShell版)
# 使い方: .\heartbeat-stop.ps1

$procs = Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like '*heartbeat-start.sh*' }

if ($procs) {
    foreach ($p in $procs) {
        Write-Host "Stopping heartbeat daemon (PID: $($p.ProcessId))..."
        Stop-Process -Id $p.ProcessId -Force
    }
    Write-Host "Done."
} else {
    Write-Host "Heartbeat daemon is not running."
}
