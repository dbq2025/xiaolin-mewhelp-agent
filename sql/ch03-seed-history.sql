-- ch03 演示用历史客服会话（对话挖知识任务的输入）。
-- 可重复执行：先清掉 demo-ch03-* 再灌。
SET NAMES utf8mb4;

DELETE FROM messages WHERE conversation_id IN (
  SELECT id FROM (SELECT id FROM conversations WHERE session_id LIKE 'demo-ch03-%') AS t
);
DELETE FROM conversations WHERE session_id LIKE 'demo-ch03-%';

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-001', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-001' LIMIT 1), 'user', '邮费是多少');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-001' LIMIT 1), 'assistant', '单笔订单满 99 元包邮，未满收取 10 元运费，偏远地区另计。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-002', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-002' LIMIT 1), 'user', '运费怎么算的');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-002' LIMIT 1), 'assistant', '单笔订单满 99 元包邮，未满收取 10 元运费，偏远地区运费另计。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-003', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-003' LIMIT 1), 'user', '寄回去要我出钱吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-003' LIMIT 1), 'assistant', '质量问题或错发漏发由平台承担退回运费；七天无理由退货由买家承担。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-004', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-004' LIMIT 1), 'user', '退货运费谁出');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-004' LIMIT 1), 'assistant', '质量问题由平台承担，非质量原因由买家承担。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-005', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-005' LIMIT 1), 'user', '买了七天还能退吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-005' LIMIT 1), 'assistant', '签收后 7 天内可以申请无理由退货，商品需保持完好。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-006', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-006' LIMIT 1), 'user', '超过七天能退吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-006' LIMIT 1), 'assistant', '超过 7 天不再受理无理由退货；质量问题仍可申请售后。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-007', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-007' LIMIT 1), 'user', '发票怎么开');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-007' LIMIT 1), 'assistant', '在订单页点击申请开票，填写抬头与税号，电子发票 3 个工作日内发到邮箱。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-008', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-008' LIMIT 1), 'user', '能开发票吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-008' LIMIT 1), 'assistant', '可以，电子发票在申请后 3 个工作日内发送到预留邮箱。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-009', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-009' LIMIT 1), 'user', '几天能到');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-009' LIMIT 1), 'assistant', '现货商品付款后 48 小时内发货，到货时间以物流信息为准。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-010', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-010' LIMIT 1), 'user', '多久发货');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-010' LIMIT 1), 'assistant', '现货 48 小时内发货，预售商品以详情页标注时间为准。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-011', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-011' LIMIT 1), 'user', '可以换货吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-011' LIMIT 1), 'assistant', '收到商品 7 天内可申请换货，寄回核验后补发新品。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-012', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-012' LIMIT 1), 'user', '换货运费谁出');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-012' LIMIT 1), 'assistant', '换货为非质量原因时退回运费由买家承担，质量问题由平台承担。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-013', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-013' LIMIT 1), 'user', '东西破了怎么赔');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-013' LIMIT 1), 'assistant', '运输破损按订单实付金额赔付，单笔最高 500 元。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-014', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-014' LIMIT 1), 'user', '质量问题赔多少');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-014' LIMIT 1), 'assistant', '按实付金额赔付，单笔上限 500 元，超出部分需人工审核。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-015', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-015' LIMIT 1), 'user', '邮费能便宜点吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-015' LIMIT 1), 'assistant', '运费按标准价收取，满 99 元即可包邮；活动价以活动页为准。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-016', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-016' LIMIT 1), 'user', '退回去的运费怎么算');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-016' LIMIT 1), 'assistant', '非质量原因由买家承担，质量问题由平台承担。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-017', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-017' LIMIT 1), 'user', '七天无理由是什么意思');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-017' LIMIT 1), 'assistant', '签收后 7 天内、商品不影响二次销售即可无理由退货。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-018', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-018' LIMIT 1), 'user', '发票多久能开好');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-018' LIMIT 1), 'assistant', '电子发票 3 个工作日内发送到预留邮箱。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-019', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-019' LIMIT 1), 'user', '发什么快递');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-019' LIMIT 1), 'assistant', '默认发合作快递，具体承运商以订单物流信息为准。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-020', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-020' LIMIT 1), 'user', '预售多久发');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-020' LIMIT 1), 'assistant', '预售商品以详情页标注的发货时间为准。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-021', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-021' LIMIT 1), 'user', '换货要多久');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-021' LIMIT 1), 'assistant', '寄回核验无误后 48 小时内补发。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-022', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-022' LIMIT 1), 'user', '退款多久到账');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-022' LIMIT 1), 'assistant', '仓库验收通过后 3 个工作日内发起退款，到账时间以银行为准。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-023', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-023' LIMIT 1), 'user', '偏远地区包邮吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-023' LIMIT 1), 'assistant', '偏远地区不参与满额包邮，运费按对照表另计。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-024', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-024' LIMIT 1), 'user', '发票抬头写错了怎么办');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-024' LIMIT 1), 'assistant', '联系客服作废重开，需要提供正确的抬头与税号。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-025', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-025' LIMIT 1), 'user', '收到货不满意能退吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-025' LIMIT 1), 'assistant', '7 天内且商品完好可以申请无理由退货。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-026', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-026' LIMIT 1), 'user', '商品有瑕疵怎么处理');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-026' LIMIT 1), 'assistant', '可申请换货或退货，退回运费由平台承担。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-027', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-027' LIMIT 1), 'user', '订单没发货能取消吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-027' LIMIT 1), 'assistant', '未发货订单可在订单页直接取消，款项原路退回。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-028', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-028' LIMIT 1), 'user', '退货运单号要填吗');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-028' LIMIT 1), 'assistant', '需要，寄回后在售后页填写运单号，便于仓库查收。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-029', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-029' LIMIT 1), 'user', '包邮门槛是多少');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-029' LIMIT 1), 'assistant', '单笔订单满 99 元包邮，偏远地区除外。');

INSERT INTO conversations (session_id, user_id, status) VALUES ('demo-ch03-030', 'anonymous', '已结束');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-030' LIMIT 1), 'user', '售后多久处理');
INSERT INTO messages (conversation_id, role, content) VALUES ((SELECT id FROM conversations WHERE session_id='demo-ch03-030' LIMIT 1), 'assistant', '提交后 48 小时内首次响应。');
