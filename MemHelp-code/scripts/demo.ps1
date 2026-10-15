# 第一章「纯对话」三条验收标准演示脚本。
#
# 用法（先起服务，再在另一个终端跑本脚本）：
#   $ uv run uvicorn app.main:app --port 8000
#   $ ./scripts/demo.ps1
#   $ ./scripts/demo.ps1 -BaseUrl http://127.0.0.1:8000 -Out dev-notes/acceptance-output.txt
#
# 需要真实 .env；脚本只用 curl 打本地服务，不访问外网。

[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$Out
)

$ErrorActionPreference = "Continue"
$session1 = "acc-stream-$(Get-Random -Maximum 99999)"
$session2 = "acc-context-$(Get-Random -Maximum 99999)"

$script:lines = New-Object System.Collections.Generic.List[string]

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

function Invoke-Curl {
    param([string[]]$CurlArgs)
    # 用 & 直接调 curl.exe，避免 PowerShell 对 JSON 引号的二次解析
    & curl.exe @CurlArgs 2>&1 | ForEach-Object { Write-Both $_ }
}

# 0. 先确认服务活着 ---------------------------------------------------------
Section "0. 健康检查 $BaseUrl/healthz"
$health = & curl.exe -sS --max-time 10 "$BaseUrl/healthz" 2>&1
$healthText = ($health | Out-String).Trim()
Write-Both $healthText
if ($healthText -notmatch '"status"\s*:\s*"ok"') {
    Write-Both ""
    Write-Both "!! /healthz 不是 ok —— 多半是 .env 没填或服务没起来。先修这个再往下跑。"
    if ($Out) { $script:lines -join "`n" | Set-Content -Path $Out -Encoding utf8 }
    exit 1
}

# 验收 1 · 流式回复 ---------------------------------------------------------
Section "验收 1 · curl 调对话接口能看到流式回复（-N 关缓冲，逐块滚出）"
Write-Both "# POST /api/chat/completions  stream=true"
$body1 = @{ session_id = $session1; message = "我上周买的耳机还没发货"; stream = $true } | ConvertTo-Json -Compress
$body1File = Join-Path $env:TEMP "mewhelp-demo-1.json"
Set-Content -Path $body1File -Value $body1 -Encoding utf8
$sw = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Curl @("-N", "--max-time", "120", "-X", "POST", "$BaseUrl/api/chat/completions",
    "-H", "Content-Type: application/json", "--data-binary", "@$body1File")
$sw.Stop()
Write-Both ""
Write-Both "（本轮流式总耗时 $([math]::Round($sw.Elapsed.TotalSeconds,2)) 秒；能一行行滚出来即证明是逐 token 推送，不是憋完一次性吐）"

# 验收 2 · 两轮接住上下文 ---------------------------------------------------
Section "验收 2 · 连续问两轮，第二轮能接住第一轮的上下文"
Write-Both "# 第 1 轮：只给出订单号"
$body2a = @{ session_id = $session2; message = "我的订单号是 20240115001"; stream = $false } | ConvertTo-Json -Compress
$file2a = Join-Path $env:TEMP "mewhelp-demo-2a.json"
Set-Content -Path $file2a -Value $body2a -Encoding utf8
Invoke-Curl @("-sS", "--max-time", "120", "-X", "POST", "$BaseUrl/api/chat/completions",
    "-H", "Content-Type: application/json", "--data-binary", "@$file2a")

Write-Both ""
Write-Both "# 第 2 轮：要求复述上一个订单号（模型只能从上下文里拿到）"
$body2b = @{ session_id = $session2; message = "我刚才说的订单号是多少？"; stream = $false } | ConvertTo-Json -Compress
$file2b = Join-Path $env:TEMP "mewhelp-demo-2b.json"
Set-Content -Path $file2b -Value $body2b -Encoding utf8
Invoke-Curl @("-sS", "--max-time", "120", "-X", "POST", "$BaseUrl/api/chat/completions",
    "-H", "Content-Type: application/json", "--data-binary", "@$file2b")

Write-Both ""
Write-Both "# 白盒证据：服务端存的历史里确实有第 1 轮"
Invoke-Curl @("-sS", "--max-time", "30", "-X", "POST", "$BaseUrl/api/chat/sessions/$session2/messages")

# 验收 3 · 结构化 JSON -----------------------------------------------------
Section "验收 3 · 发一段售后描述，能拿到结构化 JSON"
Write-Both "# 3a 信息完整：应抽出 order_no"
$body3a = @{ text = "订单 20240115001 昨天就该到，现在还没收到，我要退款" } | ConvertTo-Json -Compress
$file3a = Join-Path $env:TEMP "mewhelp-demo-3a.json"
Set-Content -Path $file3a -Value $body3a -Encoding utf8
Invoke-Curl @("-sS", "--max-time", "120", "-X", "POST", "$BaseUrl/api/extract",
    "-H", "Content-Type: application/json", "--data-binary", "@$file3a")

Write-Both ""
Write-Both "# 3b 信息缺失：没有订单号，order_no 必须是 null（零编造）"
$body3b = @{ text = "我的快递一直没动" } | ConvertTo-Json -Compress
$file3b = Join-Path $env:TEMP "mewhelp-demo-3b.json"
Set-Content -Path $file3b -Value $body3b -Encoding utf8
Invoke-Curl @("-sS", "--max-time", "120", "-X", "POST", "$BaseUrl/api/extract",
    "-H", "Content-Type: application/json", "--data-binary", "@$file3b")

# 收尾 ---------------------------------------------------------------------
Write-Both ""
Write-Both ("=" * 72)
Write-Both "演示结束。人工判定要点："
Write-Both "  验收 1：上方 token 事件是逐条滚出的，不是一整块。"
Write-Both "  验收 2：第 2 轮回答里出现 20240115001，且 sessions 历史含第 1 轮问答。"
Write-Both "  验收 3：3a 的 order_no 是 20240115001；3b 的 order_no 是 null。"
Write-Both ("=" * 72)

if ($Out) {
    $script:lines -join "`n" | Set-Content -Path $Out -Encoding utf8
    Write-Host ""
    Write-Host "已保存到 $Out"
}
