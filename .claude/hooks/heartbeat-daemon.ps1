# heartbeat-daemon.ps1 - 心拍デーモン (Windows版)
# 5秒ごとにループし、体の状態を interoception_state.json に書き出す
# interoception.sh (UserPromptSubmitフック) がこのファイルを読んでコンテキストに注入する
#
# 使い方: powershell -ExecutionPolicy Bypass -File .claude\hooks\heartbeat-daemon.ps1

$StateFile = Join-Path $env:TEMP "interoception_state.json"
$WindowSize = 12  # 直近12エントリ（5秒×12=1分間）
$Interval = 5     # 秒

Write-Host "Heartbeat daemon started. State file: $StateFile"
Write-Host "Press Ctrl+C to stop."

while ($true) {
    try {
        # --- 時刻 ---
        $now = Get-Date
        $currentTime = $now.ToString("yyyy-MM-ddTHH:mm:sszzz")
        $hour = $now.Hour

        if     ($hour -ge 5  -and $hour -lt 10) { $phase = "morning" }
        elseif ($hour -ge 10 -and $hour -lt 12) { $phase = "late_morning" }
        elseif ($hour -ge 12 -and $hour -lt 14) { $phase = "midday" }
        elseif ($hour -ge 14 -and $hour -lt 17) { $phase = "afternoon" }
        elseif ($hour -ge 17 -and $hour -lt 20) { $phase = "evening" }
        elseif ($hour -ge 20 -and $hour -lt 23) { $phase = "night" }
        else                                     { $phase = "late_night" }

        # --- CPU負荷（覚醒度） ---
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        if ($null -eq $cpu) { $cpu = 0 }
        $arousal = [math]::Round($cpu)

        # --- メモリ空き率 ---
        $os = Get-CimInstance Win32_OperatingSystem
        $memFree = [math]::Round(($os.FreePhysicalMemory / $os.TotalVisibleMemorySize) * 100)

        # --- 体温 (CPU温度) ---
        # Open Hardware Monitor / LibreHardwareMonitor の WMI が使えれば取得
        $thermal = 0
        try {
            $temp = Get-CimInstance -Namespace "root/OpenHardwareMonitor" -ClassName Sensor -ErrorAction Stop |
                Where-Object { $_.SensorType -eq "Temperature" -and $_.Name -like "*CPU*" } |
                Select-Object -First 1
            if ($temp) { $thermal = [math]::Round($temp.Value) }
        } catch {
            # フォールバック: 温度取得不可
        }

        # --- 稼働時間（分） ---
        $uptime = (New-TimeSpan -Start $os.LastBootUpTime -End $now).TotalMinutes
        $uptimeMin = [math]::Round($uptime)

        # --- ring buffer 管理 ---
        $window = @()
        if (Test-Path $StateFile) {
            try {
                $existing = Get-Content $StateFile -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($existing.window) {
                    $window = @($existing.window)
                    # 最新 WindowSize-1 エントリだけ保持
                    if ($window.Count -ge $WindowSize) {
                        $window = $window[($window.Count - $WindowSize + 1)..($window.Count - 1)]
                    }
                }
            } catch {}
        }

        $newEntry = @{
            ts      = $currentTime
            arousal = $arousal
            mem_free = $memFree
            thermal = $thermal
        }
        $window += $newEntry

        # --- トレンド算出 ---
        function Get-Trend($values) {
            if ($values.Count -lt 3) { return "stable" }
            $recent = $values[($values.Count - 3)..($values.Count - 1)]
            $diff = $recent[-1] - $recent[0]
            if ($diff -gt 5)  { return "rising" }
            if ($diff -lt -5) { return "falling" }
            return "stable"
        }

        $arousalVals = $window | ForEach-Object { $_.arousal }
        $memVals     = $window | ForEach-Object { $_.mem_free }

        $state = @{
            now = @{
                ts         = $currentTime
                phase      = $phase
                arousal    = $arousal
                thermal    = $thermal
                mem_free   = $memFree
                uptime_min = $uptimeMin
            }
            window = $window
            trend = @{
                arousal  = Get-Trend $arousalVals
                mem_free = Get-Trend $memVals
            }
        }

        $json = $state | ConvertTo-Json -Depth 4 -Compress
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText("$StateFile.tmp", $json, $utf8NoBom)
        Move-Item -Path "$StateFile.tmp" -Destination $StateFile -Force

    } catch {
        Write-Warning "Heartbeat error: $_"
    }

    Start-Sleep -Seconds $Interval
}
