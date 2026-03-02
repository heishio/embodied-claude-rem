# autonomous-action.ps1 - Claude 自律行動スクリプト (Windows版・好奇心システム対応)
# 10分ごとにタスクスケジューラで実行される
#
# desires.json の好奇心レベルに応じたプロンプトを生成して Claude CLI に渡す。
# 好奇心が高ければ調査、低ければ部屋観察を実行。
# 観察中に気になったことがあれば add_curiosity で好奇心の種を植える。
#
# 手動実行:   .\autonomous-action.ps1
# ループ実行: .\autonomous-action.ps1 -Loop

param(
    [switch]$Loop,
    [int]$IntervalMinutes = 20
)

# コンソールとパイプの出力をUTF-8に統一
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:CLAUDE_AUTONOMOUS = "1"

# BOMなしUTF-8エンコーディング（ログ書き込み用）
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Append-Log([string]$Path, [string]$Text) {
    [System.IO.File]::AppendAllText($Path, "$Text`r`n", $script:Utf8NoBom)
}

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = "$env:USERPROFILE\.claude\autonomous-logs"
$Claude = "$env:APPDATA\npm\claude.cmd"
$StateFile = "$ProjectDir\autonomous-state.json"
$DesiresFile = "$env:USERPROFILE\.claude\desires.json"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# MCP設定（autonomous用）
$McpConfig = "$ProjectDir\autonomous-mcp.json"

# --- ツールセット ---

$BaseMemoryTools = @(
    "mcp__memory__diary",
    "mcp__memory__recall",
    "mcp__memory__recall_divergent",
    "mcp__memory__recall_experience",
    "mcp__memory__recall_by_verb",
    "mcp__memory__crystallize",
    "mcp__memory__dream",
    "mcp__memory__list_recent_memories",
    "mcp__memory__link_memories",
    "mcp__memory__tom"
)

$CameraTools = @(
    "mcp__wifi-cam__see",
    "mcp__wifi-cam__look_left",
    "mcp__wifi-cam__look_right",
    "mcp__wifi-cam__look_up",
    "mcp__wifi-cam__look_down",
    "mcp__wifi-cam__look_around",
    "mcp__wifi-cam__look_front",
    "mcp__wifi-cam__get_position"
)

$DesireTools = @(
    "mcp__desire-system__get_desires",
    "mcp__desire-system__add_curiosity",
    "mcp__desire-system__list_curiosities",
    "mcp__desire-system__resolve_curiosity"
)

$CommonTools = @(
    "mcp__system-temperature__get_current_time"
)

function Get-AllowedTools {
    $tools = @()
    $tools += $BaseMemoryTools
    $tools += $DesireTools
    $tools += $CommonTools
    $tools += $CameraTools
    return ($tools | Select-Object -Unique) -join ","
}

# --- 状態管理（セッションID保持のみ） ---

function Get-AutonomousState {
    if (Test-Path $StateFile) {
        try {
            return Get-Content -Raw $StateFile | ConvertFrom-Json
        } catch {
            Write-Host "Warning: Failed to read state file, using defaults."
        }
    }
    return @{
        session_id = ""
        last_consolidate_time = ""
        activity_since_consolidate = 0
    }
}

function Save-State($State) {
    $Json = @{
        session_id = $State.session_id
        last_consolidate_time = $State.last_consolidate_time
        activity_since_consolidate = [int]$State.activity_since_consolidate
    } | ConvertTo-Json
    [System.IO.File]::WriteAllText($StateFile, $Json, $script:Utf8NoBom)
}

# --- コンソリデート判定（睡眠サイクル） ---

function Test-NeedConsolidate($State) {
    $activity = [int]($State.activity_since_consolidate)
    $lastTime = $State.last_consolidate_time
    if ($activity -lt 3) { return $false }
    if (-not $lastTime) { return $true }
    $elapsed = (Get-Date) - [DateTime]::Parse($lastTime)
    return $elapsed.TotalMinutes -ge 60
}

# --- 欲求システム ---

function Update-Desires {
    try {
        $output = & uv run --directory "$ProjectDir\desire-system" desire-updater 2>&1
        Write-Host "[desire-updater] $output"
        return $true
    } catch {
        Write-Host "[desire-updater] Error: $_"
        return $false
    }
}

function Get-CuriosityLevel {
    if (-not (Test-Path $DesiresFile)) {
        return 0.0
    }
    try {
        $data = Get-Content -Raw $DesiresFile | ConvertFrom-Json
        return [float]$data.desires.browse_curiosity
    } catch {
        return 0.0
    }
}

# --- プロンプト生成 ---

function Build-ActionPrompt([float]$CuriosityLevel) {
    if ($CuriosityLevel -ge 0.7) {
        return @"
自律行動タイム！気になることがある。以下を実行して：
1. list_curiosities で未解決の好奇心を確認
2. 気になるトピックについて調べる：
   - recall_experience や recall_by_verb で関連する体験を思い出す
   - カメラで周りを見て手がかりを探す
3. 調べた内容を記憶に保存
4. 解決したものは resolve_curiosity で解決済みにする
5. 新しく気になったことがあれば add_curiosity で種を植える
6. 最後に [CHANGE:yes] か [CHANGE:no] を必ず出力して

簡潔に報告して。
"@
    }

    return @"
自律行動タイム！以下を実行して：
1. see で部屋を見る
2. 前回と比べて変化があるか確認（人がいる/いない、明るさ、物の位置など）
3. 気づいたことがあれば記憶に保存（category: observation）
4. 気になったこと・不思議に思ったことがあれば add_curiosity で好奇心の種を植えて
5. 最後に [CHANGE:yes] か [CHANGE:no] を必ず出力して

簡潔に報告して。
"@
}

# --- メイン ---

function Ensure-SessionId($State) {
    if (-not ($State.PSObject.Properties.Name -contains 'session_id')) {
        $State | Add-Member -NotePropertyName session_id -NotePropertyValue ""
    }
    if (-not ($State.PSObject.Properties.Name -contains 'last_consolidate_time')) {
        $State | Add-Member -NotePropertyName last_consolidate_time -NotePropertyValue ""
    }
    if (-not ($State.PSObject.Properties.Name -contains 'activity_since_consolidate')) {
        $State | Add-Member -NotePropertyName activity_since_consolidate -NotePropertyValue 0
    }
    $State | Add-Member -NotePropertyName _session_is_new -NotePropertyValue $false -Force
    if (-not $State.session_id) {
        $State.session_id = [guid]::NewGuid().ToString()
        $State._session_is_new = $true
        Save-State $State
        Write-Host "New session created: $($State.session_id)"
    }
    return $State
}

function Reset-SessionId($State) {
    if (-not ($State.PSObject.Properties.Name -contains 'session_id')) {
        $State | Add-Member -NotePropertyName session_id -NotePropertyValue ""
    }
    $State.session_id = [guid]::NewGuid().ToString()
    Write-Host "Session reset: $($State.session_id)"
    return $State
}

function Invoke-AutonomousAction {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = "$LogDir\$Timestamp.log"

    Append-Log $LogFile "=== 自律行動開始: $(Get-Date) ==="

    # 1. 欲求レベルを更新
    Append-Log $LogFile "--- Updating desires ---"
    $DesireUpdated = Update-Desires
    if (-not $DesireUpdated) {
        Append-Log $LogFile "Warning: desire-updater failed, falling back to observe mode"
    }

    # 2. 好奇心レベルを取得
    $CuriosityLevel = Get-CuriosityLevel
    Append-Log $LogFile "--- Curiosity: level=$CuriosityLevel ---"

    # 3. 状態読み込み & セッションID確保
    $State = Get-AutonomousState
    $State = Ensure-SessionId $State

    # 4. プロンプトとツールセットを選択
    $Prompt = Build-ActionPrompt -CuriosityLevel $CuriosityLevel
    $AllowedTools = Get-AllowedTools

    # 4.5. コンソリデート判定（睡眠サイクル）
    $NeedConsolidate = Test-NeedConsolidate $State
    if ($NeedConsolidate) {
        $ConsolidatePrefix = @"
【最優先】記憶の整理を実行すること。観察より先に必ずやって：
1. crystallize を実行（batch_size=0, clear_buffer=true）。結果を報告して
2. 完了してから通常行動に進んで

"@
        $Prompt = $ConsolidatePrefix + $Prompt
        Append-Log $LogFile "--- Consolidate triggered (activity=$($State.activity_since_consolidate)) ---"
        # コンソリデート時は新規セッションにする（-c継続だと指示が無視される）
        $State = Reset-SessionId $State
        Save-State $State
    }

    Append-Log $LogFile "--- Session: $($State.session_id) ---"
    $Mode = if ($CuriosityLevel -ge 0.7) { "curiosity" } else { "observe" }
    Append-Log $LogFile "--- Prompt (mode=$Mode, curiosity=$CuriosityLevel) ---"
    Append-Log $LogFile $Prompt
    Append-Log $LogFile "--- AllowedTools ---"
    Append-Log $LogFile $AllowedTools
    Append-Log $LogFile "--- Output ---"

    # 5. Claude実行（初回は新規セッション、2回目以降は -c で継続）
    $TempOut = [System.IO.Path]::GetTempFileName()
    $PromptFile = [System.IO.Path]::GetTempFileName()
    [System.IO.File]::WriteAllText($PromptFile, $Prompt, $Utf8NoBom)

    try {
        if ($State._session_is_new) {
            Append-Log $LogFile "--- New session ---"
            cmd /c "chcp 65001 >nul & type `"$PromptFile`" | `"$Claude`" -p --model sonnet --mcp-config `"$McpConfig`" --allowedTools `"$AllowedTools`" > `"$TempOut`" 2>&1"
        } else {
            Append-Log $LogFile "--- Continuing session ---"
            cmd /c "chcp 65001 >nul & type `"$PromptFile`" | `"$Claude`" -p -c --model sonnet --mcp-config `"$McpConfig`" --allowedTools `"$AllowedTools`" > `"$TempOut`" 2>&1"
        }

        $ExitCode = $LASTEXITCODE
        $OutputText = [System.IO.File]::ReadAllText($TempOut, [System.Text.Encoding]::UTF8)
        Append-Log $LogFile $OutputText

        # -c が失敗したら新規セッションで再試行
        if ($ExitCode -ne 0 -and -not $State._session_is_new) {
            Append-Log $LogFile "--- Continue failed (exit=$ExitCode), starting new session ---"
            $State = Reset-SessionId $State
            Save-State $State
            cmd /c "chcp 65001 >nul & type `"$PromptFile`" | `"$Claude`" -p --model sonnet --mcp-config `"$McpConfig`" --allowedTools `"$AllowedTools`" > `"$TempOut`" 2>&1"
            $OutputText = [System.IO.File]::ReadAllText($TempOut, [System.Text.Encoding]::UTF8)
            Append-Log $LogFile $OutputText
        }
    } finally {
        Remove-Item $TempOut -ErrorAction SilentlyContinue
        Remove-Item $PromptFile -ErrorAction SilentlyContinue
    }

    # 6. コンソリデート状態の更新
    $State.activity_since_consolidate = [int]($State.activity_since_consolidate) + 1
    if ($NeedConsolidate) {
        $State.activity_since_consolidate = 0
        $State.last_consolidate_time = (Get-Date).ToString("o")
    }
    Save-State $State

    Append-Log $LogFile "--- Result ---"
    Append-Log $LogFile "Mode=$Mode Curiosity=$CuriosityLevel Session=$($State.session_id) Activity=$($State.activity_since_consolidate)"
    Append-Log $LogFile "=== 自律行動終了: $(Get-Date) ==="

    Write-Host "[$Timestamp] Done. Mode=$Mode Curiosity=$CuriosityLevel Session=$($State.session_id) Log: $LogFile"
}

if ($Loop) {
    Write-Host "Autonomous action loop started (every $IntervalMinutes min). Ctrl+C to stop."
    while ($true) {
        Invoke-AutonomousAction
        Start-Sleep -Seconds ($IntervalMinutes * 60)
    }
} else {
    Invoke-AutonomousAction
}


