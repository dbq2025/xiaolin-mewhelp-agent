-- 第五章：给 messages 加「建议动作」快照列。
--
-- 形状：[{"action": "transfer_human", "ticket_type": "投诉"},
--        {"action": "create_ticket", "ticket_type": "投诉"}]
--
-- 为什么存动作而不是「按钮是否点过」：点击状态是浏览器侧的一次性 UX，
-- 不该冒充业务事实；工单是否真的创建，以 tickets 表为准。
--
-- 幂等执行：先查列是否存在。MySQL 不支持 ADD COLUMN IF NOT EXISTS
-- （8.0 也不支持），所以脚本用 information_schema 判断。

SET NAMES utf8mb4;

SET @col_exists := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'messages'
    AND COLUMN_NAME = 'actions'
);

SET @ddl := IF(
  @col_exists = 0,
  'ALTER TABLE messages ADD COLUMN actions JSON NULL AFTER citations',
  'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 同一份迁移的第二部分：tickets 的建单幂等键。
-- 前端每次「建工单」按钮生成一个 key；重复提交返回同一张工单，
-- 防止「服务端已成功、客户端超时后重试」建出两张。

SET @ticket_col_exists := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'tickets'
    AND COLUMN_NAME = 'idempotency_key'
);

SET @ddl := IF(
  @ticket_col_exists = 0,
  'ALTER TABLE tickets ADD COLUMN idempotency_key VARCHAR(64) NULL AFTER status',
  'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @ticket_idx_exists := (
  SELECT COUNT(*) FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'tickets'
    AND INDEX_NAME = 'uq_tickets_idempotency_key'
);

SET @ddl := IF(
  @ticket_idx_exists = 0,
  'ALTER TABLE tickets ADD UNIQUE KEY uq_tickets_idempotency_key (idempotency_key)',
  'SELECT 1'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
