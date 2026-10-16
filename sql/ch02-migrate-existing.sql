-- =============================================================
-- ch02 · 对**已存在的** conversations 表做幂等迁移
-- =============================================================
--
-- ## 什么时候需要这份脚本
--
-- `sql/ch02-ddl.sql` 只在新库（数据目录为空、compose 首次 up）时由 MySQL 的
-- initdb 机制执行。如果数据库**已经存在**（例如复用一个别的章节建好的库），
-- initdb 不会重跑，那份 DDL 里的 `session_id` 永远加不上，
-- 而本章的 ORM 靠它定位会话 —— 表现为真库报 `Unknown column 'session_id'`。
--
-- 所以凡是「复用既有库」的场景，都要单独跑这一份。
--
-- ## 幂等
--
-- 全部操作都先查 `information_schema` 再决定做不做，所以可以反复执行。
-- 已经迁过的库再跑一次是空操作（不会报 Duplicate column / Duplicate key）。
--
-- ## 为什么不是简单一句 ADD COLUMN
--
-- 本章的 ORM 声明 `session_id` 为 **NOT NULL + UNIQUE**，而集成测试
-- （`tests/test_db_mysql.py`）会逐列断言 ORM 与真库一致。既有库里可能有若干行
-- 老会话没有会话键，直接 `ADD COLUMN ... NOT NULL` 会因为空值失败或被 MySQL
-- 填成 ''。
--
-- 而如果用「可空 + 唯一键」，MySQL 允许**多个 NULL**，于是「同一字符串对应两行
-- 会话」这条唯一性约束在 NULL 上形同虚设。所以这里走三部曲：
--
--   1. `ADD COLUMN ... NOT NULL DEFAULT ''` —— 既有行被回填成 ''（非 NULL）
--   2. 把 '' 的既有行赋成可读的 `demo-seed-<id>` —— 顺带让它们可被接口访问
--   3. `ADD UNIQUE KEY` —— 此时没有重复值，索引建得起来
--
-- 这样迁完之后表结构与新库完全一致，且既有数据一行不丢。
--
-- ## 会改什么（重要）
--
-- - 给 `conversations` **加一列** `session_id` + 唯一键；
-- - 把 `session_id` 为空的历史行**赋值为** `demo-seed-<id>`；
-- - **不动**其它任何表、任何列、任何行（faq / messages / tickets 完全不碰）。
--
-- 用法（在有 mysql 客户端的任何环境）：
--   docker exec -i <容器名> mysql -uroot -p<pw> mewhelp < sql/ch02-migrate-existing.sql
-- 或者用 Python（不依赖本机装 mysql 客户端）：
--   uv run --directory MemHelp-code python scripts/migrate_existing.py

SET NAMES utf8mb4;

-- ---------------------------------------------------------------------------
-- 1. 加列（不存在才加）
-- ---------------------------------------------------------------------------
SET @has_col := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'conversations'
    AND COLUMN_NAME = 'session_id'
);

SET @sql := IF(
  @has_col = 0,
  'ALTER TABLE conversations ADD COLUMN session_id VARCHAR(64) NOT NULL DEFAULT '''' COMMENT ''对外会话标识,聊天页与接口用它'' AFTER id',
  'SELECT ''skip: conversations.session_id 已存在'' AS note'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ---------------------------------------------------------------------------
-- 2. 回填既有行的会话键
--
--    赋成 ``demo-seed-<id>`` 而不是留空串：空串虽然也能建唯一键，
--    但那会让所有历史会话共享一个「谁都查不到」的键；
--    赋成可读值之后，这些会话也能被 /api/chat/sessions/<id>/messages 访问。
-- ---------------------------------------------------------------------------
UPDATE conversations
SET session_id = CONCAT('demo-seed-', id)
WHERE session_id IS NULL OR session_id = '';

-- ---------------------------------------------------------------------------
-- 3. 加唯一键（不存在才加）
--
--    放在回填之后：既有行此时已有唯一值，索引一定建得起来。
--    若先建索引再回填，且历史行里有重复值，这一步会失败。
-- ---------------------------------------------------------------------------
SET @has_idx := (
  SELECT COUNT(*) FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'conversations'
    AND INDEX_NAME = 'uk_session_id'
);

SET @sql := IF(
  @has_idx = 0,
  'ALTER TABLE conversations ADD UNIQUE KEY uk_session_id (session_id)',
  'SELECT ''skip: uk_session_id 已存在'' AS note'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ---------------------------------------------------------------------------
-- 4. messages.tool_traces —— 存**给界面看的**工具轨迹
--
-- 为什么不把耗时/成败塞进已有的 ``tool_calls`` JSON：那一列会被原样塞回
-- ``AIMessage(tool_calls=...)`` 作为**模型历史**，而 LangChain 对 tool_call
-- 的字段是严格的 —— 实测多一个 ``duration_ms`` 会直接抛：
--
--     TypeError: tool_call() got an unexpected keyword argument 'duration_ms'
--
-- 所以模型历史必须保持干净的 ``{name, args, id}``；界面信息另开一列。
-- 这一列的语义是「展示用」，与模型无关，因此格式由我们自己定。
-- ---------------------------------------------------------------------------
SET @has_traces := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'messages'
    AND COLUMN_NAME = 'tool_traces'
);

SET @sql := IF(
  @has_traces = 0,
  'ALTER TABLE messages ADD COLUMN tool_traces JSON NULL COMMENT ''给界面看的工具轨迹(含耗时/成败)''',
  'SELECT ''skip: messages.tool_traces 已存在'' AS note'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ---------------------------------------------------------------------------
-- 5. 自检
-- ---------------------------------------------------------------------------
SELECT
  (SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'conversations'
      AND COLUMN_NAME = 'session_id' AND IS_NULLABLE = 'NO')          AS col_not_null_ok,
  (SELECT COUNT(*) FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'conversations'
      AND INDEX_NAME = 'uk_session_id' AND NON_UNIQUE = 0)           AS unique_key_ok,
  (SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'messages'
      AND COLUMN_NAME = 'tool_traces')                               AS tool_traces_ok,
  (SELECT COUNT(*) FROM conversations
    WHERE session_id IS NULL OR session_id = '')                     AS blank_session_ids,
  (SELECT COUNT(*) FROM conversations)                               AS total_conversations;
