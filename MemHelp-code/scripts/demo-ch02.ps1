# 第二章「Function Calling 工具链」验收演示脚本。
#
# 用法（先起服务，再在另一个终端跑本脚本）：
#   # 服务连的库见 .env（默认接**已有的** mewhelp-mysql，宿主 3306）
#   uv run --directory MemHelp-code uvicorn app.main:app --port 8000
#   powershell -ExecutionPolicy Bypass -File scripts/demo-ch02.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/demo-ch02.ps1 -Out dev-notes/ch02-acceptance-output.txt
#
# 本脚本只打本机服务，不访问外网。服务需要真实 .env（上游模型 + MySQL 都配好）。
#
# 注意：本文件必须存成 UTF-8 with BOM —— PowerShell 5.1 会按 GBK 读 .ps1，
# 中文会撑破字符串字面量（见 agent.md 第 4 节）。tests/test_repo_conventions.py 会检查这一点。

[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$Out
)

$ErrorActionPreference = "Continue"
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
#  1. 用 curl.exe 而不是 Invoke-WebRequest：后者把整段流缓冲起来，看不到逐帧。
#  2. 请求体走**临时文件**（--data-binary "@file"），不要用 -d '<json>' 直接传：
#     PowerShell 会吃掉内联 JSON 里的引号，curl 收到坏 JSON，
#     服务端的表现是「一个 SSE 帧都没有」，极难判断。实测踩过。
function Invoke-Sse {
    param([string]$Json)
    $tmp = Join-Path $env:TEMP ("mewhelp-ch02-" + [guid]::NewGuid().ToString("N") + ".json")
    # UTF-8 无 BOM：带 BOM 时 curl 会把 BOM 当成 body 的一部分，服务端 JSON 解析失败。
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
# 0. 环境检查
# ---------------------------------------------------------------------------
Section "0. 环境检查"

$healthRaw = & curl.exe -sS --max-time 10 "$BaseUrl/healthz" 2>&1
$healthText = ($healthRaw | Out-String).Trim()
Write-Both ("  /healthz -> " + $healthText)
try { $health = $healthText | ConvertFrom-Json } catch { $health = $null }
Check "服务可用（status=ok）" ($null -ne $health -and $health.status -eq "ok") $(if ($health) { $health.status } else { "无法解析" })
Check "数据库可用（database=ok）" ($null -ne $health -and $health.database -eq "ok") $(if ($health) { $health.database } else { "" })

if ($script:failures -gt 0) {
    Write-Both ""
    Write-Both "环境未就绪，后续验收无法进行。请先起服务（库见 .env）："
    Write-Both "  uv run --directory MemHelp-code uvicorn app.main:app --port 8000"
    if ($Out) {
        $outPath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) $Out
        $script:lines -join [Environment]::NewLine | Set-Content -Path $outPath -Encoding UTF8
    }
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
# @() 强制成数组：单元素时 Select-Object/if 返回**对象**而不是数组，
# 直接取 .Count 得到 $null，于是「其实有」被误判成「没有」。实测踩过。
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
Check "回答里没有 coroutine 对象（包装层忘了 await 的特征）" ($answer1 -notmatch "coroutine object")
Write-Both ("  最终回答：" + $answer1)
Write-Both ("  工具徽章数据：" + ($tools1 | ConvertTo-Json -Compress))

# ---------------------------------------------------------------------------
# 验收 2：问退货政策 —— query_faq 查得到并作答
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
Check "回答里没有 coroutine 对象" ($answer2 -notmatch "coroutine object")
# 答对的关键：回答里要出现语料里的特征词，说明它真的用了查到的内容
Check "回答基于查到的政策（含 7 天/无理由/签收 之类特征）" ($answer2 -match "7\s*天|七天|无理由|签收")
Write-Both ("  最终回答：" + $answer2)

# ---------------------------------------------------------------------------
# 验收 3：换个说法问邮费 —— 实测结论与 spec 原先预测不同
# ---------------------------------------------------------------------------
$s3 = "acc3-" + (Get-Random -Maximum 999999)
Section "验收 3 · 邮费是多少（期望：关键词漏召回；实测结论见下）"

$frames3 = Invoke-Sse (@{ session_id = $s3; message = "邮费是多少"; stream = $true } | ConvertTo-Json -Compress)
Write-Both ("  事件序列：" + (Get-EventNames $frames3))

$done3 = $frames3 | Where-Object { $_.Event -eq "done" } | Select-Object -Last 1
$tools3 = @(if ($done3 -and $done3.Data.tools) { $done3.Data.tools } else { @() })
$names3 = ($tools3 | ForEach-Object { $_.name }) -join ", "
$keywords3 = (($tools3 | ForEach-Object { $_.args.keyword }) -join ",")
$answer3 = Get-TokenText $frames3

Write-Both ("  选中的工具：" + $names3)
Write-Both ("  模型实际传的关键词：" + $keywords3)
Write-Both ("  最终回答：" + $answer3)
Check "模型选的是 query_faq（工具选择层没错）" ($names3 -match "query_faq") $names3
Check "回答里没有 coroutine 对象" ($answer3 -notmatch "coroutine object")

# ---------------------------------------------------------------------------
# 验收 3 的**实测结论**（与 spec 原先的预测不同，必须如实记）
#
# spec 原先预测：「邮费是多少」关键词查表查不出来，模型应承认没查到。
#
# 实测（deepseek-flash + 真实语料）分两层，结论不同：
#
#   数据层：query_faq('邮费') 稳定返回 ok=true / count=0 —— 基线成立
#           （语料含「运费」不含「邮费」，两者不是子串关系）。
#   用户可见层：**模型把用户的「邮费」自行归一化成「运费」再传参**，
#           于是命中「运费怎么算」，答出满 99 包邮 / 未满 10 元运费。
#
# 所以「用户问邮费会查不出来」这条预期前提**不成立**：召回失败在
# 「组织工具参数」这一步被模型补偿掉了。这不是工具或 SQL 的缺陷。
# 已知由模型侧的同义归一化引起，用户已确认留待后续章节修复，本章不处理。
#
# 下面**不对用户可见层的措辞做断言**：它会随模型改写而变，断言它等于把
# 「模型当时怎么想的」当成契约。数据层的基线由 tests/test_db_mysql.py 硬钉住。
# ---------------------------------------------------------------------------
Write-Both "  [实测结论] 数据层：query_faq('邮费') 稳定 count=0（基线成立）"
Write-Both "  [实测结论] 用户可见层：模型把「邮费」归一化成「运费」再传参，"
Write-Both "             于是命中「运费怎么算」并答对 —— spec 原先预测的"
Write-Both "             「用户会看到查不出来」不成立。已知原因在模型侧归一化，"
Write-Both "             用户已确认留待后续章节修复，本章不处理。"

# ---------------------------------------------------------------------------
# 落库核对：四行完整痕迹 + 徽章可回读
# ---------------------------------------------------------------------------
Section "落库核对 · 验收 1 的会话在 messages 表里的形状"

$msgs = (& curl.exe -sS -X POST "$BaseUrl/api/chat/sessions/$s1/messages" 2>&1 | Out-String) | ConvertFrom-Json
$rows = @($msgs.messages)
foreach ($r in $rows) {
    $toolInfo = if ($r.tools) { " tools=[" + ((@($r.tools) | ForEach-Object { $_.name + ":" + $_.ok }) -join ",") + "]" } else { "" }
    $content = "" + $r.content
    Write-Both ("  " + $r.role.PadRight(10) + $content.Substring(0, [Math]::Min(48, $content.Length)) + $toolInfo)
}
Check "落库为 human -> ai -> tool -> ai" ((($rows | ForEach-Object { $_.role }) -join ",") -eq "human,ai,tool,ai")
Check "assistant 行带回了工具轨迹（徽章刷新后仍在）" (@($rows | Where-Object { $_.tools }).Count -gt 0)

# ---------------------------------------------------------------------------
Section "汇总"
if ($script:failures -eq 0) {
    Write-Both "  三条验收标准的**可自动判定部分**全部通过。"
    Write-Both "  仍需人工确认的一项：浏览器打开 $BaseUrl/chat/ 提问，"
    Write-Both "  确认气泡里出现 🔧 工具徽章（脚本无法验 DOM）。"
} else {
    Write-Both ("  有 " + $script:failures + " 项未通过，见上方 [FAIL]。")
}

if ($Out) {
    $outPath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) $Out
    $dir = Split-Path -Parent $outPath
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $script:lines -join [Environment]::NewLine | Set-Content -Path $outPath -Encoding UTF8
    Write-Host ""
    Write-Host ("  输出已写入：" + $outPath)
}

exit $script:failures
