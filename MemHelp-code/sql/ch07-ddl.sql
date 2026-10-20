-- 第七章：上下文管理 —— 两个锚点列 + 会话摘要表。
--
-- 三层结构靠**两个整数锚点**划边界，不靠搬数据：
--
--   messages.id <= summary_upto_msg_id            → 层 3（已被梗概覆盖，不进上下文）
--   summary_upto_msg_id < id < layer1_from_msg_id → 层 2（规则截短）
--   messages.id >= layer1_from_msg_id             → 层 1（原文，一个字不压）
--
-- 为什么用 messages.id 而不是「第几条」或时间戳：
--   * id 是单调自增的，天然有全序；
--   * 时间戳在容器里可能同秒、可能回拨，做边界会「同一秒里切在中间」；
--   * 「第几条」要每次重算，一条消息被删就整体错位。
--
-- 为什么摘要单开一张表、不塞进 conversations 一列：
--   梗概是**一段段攒**的（每次后台任务追加一段），不是「当前最新的一坨」。
--   塞一列就只剩最后一次结果，旧段被覆盖 —— 而旧段恰恰是「这个事实已经被
--   有损压过一次」的证据。逐段留档才能做到「只追加、不回头重写」。
--   也绝不塞进 messages 表：那样 ``history_as_messages`` 会把梗概当成一条
--   真实对话消息回灌给模型。
--
-- 幂等执行：MySQL 不支持 ADD COLUMN IF NOT EXISTS（8.0 也不支持），
-- 所以先查 information_schema。重复执行本脚本是安全的。
-- 注意：**不 DROP 任何东西**。真库里可能还留着旧方案的 ``conversations.summary``
-- 列，留着不碍事；DROP 掉反而会让「回滚到上一版」变成不可能。

SET NAMES utf8mb4;

-- ---------------------------------------------------------------------------
-- 1. conversations.summary_upto_msg_id：梗概覆盖到哪条消息
-- ---------------------------------------------------------------------------

SET @col_exists := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'conversations'
    AND COLUMN_NAME = 'summary_upto_msg_id'
);

SET @ddl := IF(
  @col_exists = 0,
  'ALTER TABLE conversations ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL AFTER status',
  'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ---------------------------------------------------------------------------
-- 2. conversations.layer1_from_msg_id：层 1（原文区）从哪条消息起
-- ---------------------------------------------------------------------------

SET @col_exists := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'conversations'
    AND COLUMN_NAME = 'layer1_from_msg_id'
);

SET @ddl := IF(
  @col_exists = 0,
  'ALTER TABLE conversations ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL AFTER summary_upto_msg_id',
  'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ---------------------------------------------------------------------------
-- 3. conversation_summaries：一段梗概一行
-- ---------------------------------------------------------------------------

SET @tbl_exists := (
  SELECT COUNT(*) FROM information_schema.TABLES
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'conversation_summaries'
);

SET @ddl := IF(
  @tbl_exists = 0,
  'CREATE TABLE conversation_summaries (
     id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
     conversation_id BIGINT UNSIGNED NOT NULL,
     segment_no INT NOT NULL,
     from_msg_id BIGINT UNSIGNED NOT NULL,
     upto_msg_id BIGINT UNSIGNED NOT NULL,
     content TEXT NOT NULL,
     msg_count INT NOT NULL DEFAULT 0,
     source_tokens INT NOT NULL DEFAULT 0,
     model VARCHAR(64) NULL,
     elapsed_ms INT NULL,
     created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
     PRIMARY KEY (id),
     UNIQUE KEY uq_summary_segment (conversation_id, segment_no),
     KEY idx_summary_conversation_id (conversation_id, segment_no),
     CONSTRAINT fk_summary_conversation FOREIGN KEY (conversation_id)
       REFERENCES conversations (id)
   ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci',
  'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 兜底：如果表是早前手工建过的、少了唯一键，补上。
-- 唯一键是 Review Focus 2 的最后一道防线 —— 多进程同时压同一段时，
-- 应用侧的 in-flight 集合各自为政，只有数据库能真正收口。
SET @idx_exists := (
  SELECT COUNT(*) FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'conversation_summaries'
    AND INDEX_NAME = 'uq_summary_segment'
);

SET @ddl := IF(
  @idx_exists = 0,
  'ALTER TABLE conversation_summaries
     ADD UNIQUE KEY uq_summary_segment (conversation_id, segment_no)',
  'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
