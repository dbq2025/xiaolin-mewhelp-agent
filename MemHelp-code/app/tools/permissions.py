"""读写权限策略：写操作必须由「用户已确认」这条外部事实放行。

刻意不读 MCP Server 的 annotations —— 工具的用途声明由对方自己写，不可信；
能不能调只认本侧的 ``ToolSpec.read_only``。
"""

from __future__ import annotations

from app.tools.contract import ToolSpec


def check_permission(spec: ToolSpec, *, confirmed: bool) -> str | None:
    """返回拒绝原因；``None`` 表示放行。纯函数，不抛异常。"""
    if spec.read_only:
        return None
    if confirmed:
        return None
    return f"写操作 {spec.name} 需要用户确认后才能执行"
