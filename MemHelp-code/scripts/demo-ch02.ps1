# 第二章「Function Calling 工具链」验收演示脚本。
#
# 用法（先起库与服务，再在另一个终端跑本脚本）：
#   $ docker compose -f MemHelp-code/docker-compose.yml up -d --wait
#   $ uv run --directory MemHelp-code uvicorn app.main:app --port 8000
#   $ powershell -ExecutionPolicy Bypass -File scripts/demo-ch02.ps1
#   $ powershell -ExecutionPolicy Bypass -File scripts/demo-ch02.ps1 -Out dev-notes/ch02-acceptance-output.txt
#
# 开关：
#   -Reset   先 down -v 再 up -d --wait，让 sql/ 里的 initdb 脚本重新执行。
#            **改过 DDL 或种子数据之后必须加这个**，否则 MySQL 只在数据目录为空时
#            才跑 initdb，改动不会生效（表现为「明明改了 DDL 却还是旧表结构」）。
#
# 本脚本只打本机服务，不访问外网。服务需要真实 .env（上游模型 + MySQL 都配好）。
#
# 注意：本文件必须存成 UTF-8 with BOM —— PowerShell 5.1 会按 GBK 读 .ps1，
# 中文会撑破字符串字面量（见 agent.md 第 4 节）。

[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$Out,
    [switch]$Reset
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$composeFile = Join-Path $PSScriptRoot "..\docker-compose.yml"

$script:lines = New-Object System.Collections.Generic.List[string]
$script:failures = 0

function Write-Both {
    param([string]$Text = "")
    Write-Host $Text
    $script:lines.Add($Text)
}

function Section {
    param([string]$Title)
    Write-Both ""
    Write-Both ("=" * 72)
    Write-Both $Title
    Write-Both ("=" * 72)
}

function Check {
    param([string]$What, [bool]$Ok, [string]$Detail = "")
    if ($Ok) {
        Write-Both ("  [PASS] " + $What + $(if ($Detail) { "  -> " + $Detail } else { "" }))
    } else {
        Write-Both ("  [FAIL] " + $What + $(if ($Detail) { "  -> " + $Detail } else { "" }))
        $script:failures++
    }
}

# 打一段 SSE，把帧解析成 (event, data-json) 列表返回。
#
# 两个刻意的做法：
#  1. 用 curl.exe 而不是 Invoke-WebRequest：后者会把整段流缓冲起来，看不到逐帧。
#  2. **请求体走临时文件（--data-binary "@file"）**，不要用 -d '<json>' 直接传：
#     PowerShell 会吃掉内联 JSON 里的引号，curl 收到的是坏 JSON，
#     表现在服务端就是一个**空的 SSE 响应**（连 error 帧都没有），极难判断。
#     实测踩过：三条验收全部「一个帧都没有」。
function Invoke-Sse {
    param([string]$Json)
    $tmp = Join-Path $env:TEMP ("mewhelp-ch02-" + [guid]::NewGuid().ToString("N") + ".json")
    # UTF-8 **无 BOM**：带 BOM 时 curl 会把 BOM 当成 body 的一部分，服务端 JSON 解析失败。
    [System.IO.File]::WriteAllText($tmp, $Json, (New-Object System.Text.UTF8Encoding($false)))
    try {
        $raw = & curl.exe -sS -N --max-time 90 -X POST "$BaseUrl/api/chat/completions" `
            -H "Content-Type: application/json" --data-binary "@$tmp" 2>&1
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
    $text = ($raw | Out-String)
    $frames = @()
    foreach ($block in ($text -split "(\r?\n){2,}")) {
        if (-not $block.Trim()) { continue }
        $name = ""; $data = ""
        foreach ($line in ($block -split "\r?\n")) {
            if ($line.StartsWith("event:")) { $name = $line.Substring(6).Trim() }
            elseif ($line.StartsWith("data:")) { $data = $line.Substring(5).Trim() }
        }
        if ($name) {
            $obj = $null
            try { $obj = $data | ConvertFrom-Json } catch { }
            $frames += [pscustomobject]@{ Event = $name; Data = $obj }
        }
    }
    if ($frames.Count -eq 0) {
        # 一帧都没有时把原始响应打出来 —— 否则「全 FAIL 但没有任何线索」
        Write-Both "  [调试] 未解析到任何 SSE 帧，原始响应前 300 字："
        Write-Both ("    " + $text.Substring(0, [Math]::Min(300, $text.Length)))
    }
    return $frames
}

function Get-EventNames {
    param($Frames) 
    return (($Frames | ForEach-Object { $_.Event }) -join " -> ")
}

function Get-TokenText {
    param($Frames)
    return (($Frames | Where-Object { $_.Event -eq "token" } | ForEach-Object { $_.Data.delta }) -join "")
}

# ---------------------------------------------------------------------------
# 0. 起库（可选重置）与服务健康检查
# ---------------------------------------------------------------------------
Section "0. 环境检查"

if ($Reset) {
    Write-Both "  -Reset 指定：重建数据库卷（让 sql/ 的 initdb 脚本重新执行）"
    & docker compose -f $composeFile down -v 2>&1 | ForEach-Object { Write-Both ("    " + $_) }
    & docker compose -f $composeFile up -d --wait 2>&1 | ForEach-Object { Write-Both ("    " + $_) }
}

$healthRaw = & curl.exe -sS --max-time 10 "$BaseUrl/healthz" 2>&1
$healthText = ($healthRaw | Out-String).Trim()
Write-Both ("  /healthz -> " + $healthText)
try { $health = $healthText | ConvertFrom-Json } catch { $health = $null }
Check "服务可用（status=ok）" ($null -ne $health -and $health.status -eq "ok") $(if ($health) { $health.status } else { "无法解析" })
Check "数据库可用（database=ok）" ($null -ne $health -and $health.database -eq "ok") $(if ($health) { $health.database } else { "" })

if ($script:failures -gt 0) {
    Write-Both ""
    Write-Both "环境未就绪，后续验收无法进行。请先："
    Write-Both "  docker compose -f MemHelp-code/docker-compose.yml up -d --wait"
    Write-Both "  uv run --directory MemHelp-code uvicorn app.main:app --port 8000"
    if ($Out) { $script:lines -join [Environment]::NewLine | Set-Content -Path $Out -Encoding UTF8 }
    exit 1
}

# ---------------------------------------------------------------------------
# 验收 1：问物流 —— 模型自己选中 query_logistics，气泡带工具徽章
# ---------------------------------------------------------------------------
$s1 = "acc1-" + (Get-Random -Maximum 999999)
Section "验收 1 · 订单 1001 的物流到哪了（期望：选中 query_logistics，带工具徽章）"

$frames1 = Invoke-Sse (@{ session_id = $s1; message = "订单 1001 的物流到哪了"; stream = $true } | ConvertTo-Json -Compress)
Write-Both ("  事件序列：" + (Get-EventNames $frames1))

$done1 = $frames1 | Where-Object { $_.Event -eq "done" } | Select-Object -Last 1
# 用 @() 强制成数组：单元素时 Where-Object / if 返回的是**对象**而不是数组，
# 直接取 .Count 会得到 $null，于是「其实有」被误判成「没有」（实测踩过）。
$tools1 = @(if ($done1 -and $done1.Data.tools) { $done1.Data.tools } else { @() })
$names1 = ($tools1 | ForEach-Object { $_.name }) -join ", "
$answer1 = Get-TokenText $frames1
$statusCount1 = @($frames1 | Where-Object { $_.Event -eq "status" }).Count

Check "有 status 帧（工具执行前先推状态）" ($statusCount1 -gt 0) ("$statusCount1 个")
Check "调用了工具（done.tools 非空）" ($tools1.Count -gt 0) $names1
Check "选中的是物流工具" ($names1 -match "query_logistics")
# ConvertFrom-Json 把 JSON 的 true 转成 [bool]，所以必须与 $true 比较；
# 写成 { $_.ok -eq "true" } 会静默恒假（字符串 vs 布尔）。
Check "工具执行成功" (@($tools1 | Where-Object { $_.ok -eq $true }).Count -gt 0)
Check "有最终回答文本" ($answer1.Trim().Length -gt 0)
Write-Both ("  最终回答：" + $answer1)
Write-Both ("  工具徽章数据：" + ($tools1 | ConvertTo-Json -Compress))

# ---------------------------------------------------------------------------
# 验收 2：问退货政策 —— query_faq 查得到
# ---------------------------------------------------------------------------
$s2 = "acc2-" + (Get-Random -Maximum 999999)
Section "验收 2 · 退货政策是什么（期望：query_faq 命中 faq 表并作答）"

$frames2 = Invoke-Sse (@{ session_id = $s2; message = "退货政策是什么"; stream = $true } | ConvertTo-Json -Compress)
Write-Both ("  事件序列：" + (Get-EventNames $frames2))

$done2 = $frames2 | Where-Object { $_.Event -eq "done" } | Select-Object -Last 1
$tools2 = @(if ($done2 -and $done2.Data.tools) { $done2.Data.tools } else { @() })
$names2 = ($tools2 | ForEach-Object { $_.name }) -join ", "
$answer2 = Get-TokenText $frames2

Check "调用了 query_faq" ($names2 -match "query_faq") $names2
Check "有最终回答文本" ($answer2.Trim().Length -gt 0)
# 答对的关键：回答里要出现种子语料里的特征词，说明它真的用了查到的内容
Check "回答基于查到的政策（含 7 天/无理由/签收 之类特征）" ($answer2 -match "7\s*天|七天|无理由|签收")
Check "回答里没有 coroutine 对象（包装层忘了 await 的特征）" ($answer2 -notmatch "coroutine object")
Write-Both ("  最终回答：" + $answer2)

# ---------------------------------------------------------------------------
# 验收 3：换个说法问邮费 —— 关键词查不出来（**预期漏召回**）
# ---------------------------------------------------------------------------
$s3 = "acc3-" + (Get-Random -Maximum 999999)
Section "验收 3 · 邮费是多少（**预期漏召回**，这是留给 ch03 的基线）"

$frames3 = Invoke-Sse (@{ session_id = $s3; message = "邮费是多少"; stream = $true } | ConvertTo-Json -Compress)
Write-Both ("  事件序列：" + (Get-EventNames $frames3))

$done3 = $frames3 | Where-Object { $_.Event -eq "done" } | Select-Object -Last 1
$tools3 = if ($done3 -and $done3.Data.tools) { $done3.Data.tools } else { @() }
$names3 = ($tools3 | ForEach-Object { $_.name }) -join ", "
$answer3 = Get-TokenText $frames3

Write-Both ("  选中的工具：" + $names3)
Write-Both ("  最终回答：" + $answer3)
Check "模型仍然选了 query_faq（工具选择层没错）" ($names3 -match "query_faq") $names3
Write-Both "  [预期] 工具在**数据层**返回 count=0 —— 关键词「邮费」与语料用词"
Write-Both "         （「退货运费」「满多少包邮」）对不上。这是故意保留的缺口，"
Write-Both "         不是 bug；ch03 换向量检索后这一条应当变成能答。"
Check "回答没有编造具体运费数字" ($answer3 -notmatch "\d+\s*元\s*包邮|邮费\s*\d+")

# ---------------------------------------------------------------------------
# 落库核对：四行完整痕迹
# ---------------------------------------------------------------------------
Section "落库核对 · 验收 1 的会话在 messages 表里的形状"

$msgs = (& curl.exe -sS -X POST "$BaseUrl/api/chat/sessions/$s1/messages" 2>&1 | Out-String) | ConvertFrom-Json
$rows = $msgs.messages
foreach ($r in $rows) {
    $toolInfo = if ($r.tools) { " tools=[" + (($r.tools | ForEach-Object { $_.name + ":" + $_.ok }) -join ",") + "]" } else { "" }
    Write-Both ("  " + $r.role.PadRight(10) + ("" + $r.content).Substring(0, [Math]::Min(48, ("" + $r.content).Length)) + $toolInfo)
}
Check "落库为 user -> assistant -> tool -> assistant" (($rows | ForEach-Object { $_.role }) -join "," -eq "human,ai,tool,ai")
Check "assistant 行带回了工具轨迹（徽章刷新后仍在）" (($rows | Where-Object { $_.tools }).Count -gt 0)

# ---------------------------------------------------------------------------
Section "汇总"
if ($script:failures -eq 0) {
    Write-Both "  三条验收标准的**可自动判定部分**全部通过。"
    Write-Both "  仍需人工确认的一项：浏览器打开 $BaseUrl/chat/ 提问，"
    Write-Both "  确认气泡里出现 🔧 工具徽章（脚本无法验 DOM）。"
    Write-Both "  验收 3 的「查不到」是预期结果，已在上文标注。"
} else {
    Write-Both ("  有 " + $script:failures + " 项未通过，见上方 [FAIL]。")
}

if ($Out) {
    $outPath = Join-Path $repoRoot $Out
    $dir = Split-Path -Parent $outPath
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $script:lines -join [Environment]::NewLine | Set-Content -Path $outPath -Encoding UTF8
    Write-Host ""
    Write-Host ("  输出已写入：" + $outPath)
}

exit $script:failures
