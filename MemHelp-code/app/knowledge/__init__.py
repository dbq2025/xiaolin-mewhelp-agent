"""离线建库与在线检索的子系统。

分文件而不是塞进 ``app/db/repositories.py``：这里除了 SQL 还有嵌入模型、向量库、
Markdown 解析与 LLM 抽取，它们各自有独立的失败模式与测试方式。
"""
