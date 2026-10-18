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
