# ch06 多轮指代/意图/扩写评估

- 模型：`deepseek-flash`
- 轮数：**22**
- 意图 accuracy：**1.000**（阈值 0.85）
- 指代通过率：**1.000**（阈值 0.85）
- other 落点：**3/3 = 1.000**（阈值 0.8）
- JSON 解析失败：**0**（要求 0）
- 扩写通过：**8/8**（阈值 0.85）

## 逐轮明细

| 用例 | 轮 | 用户 | 补全后 | 期望 | 实际 | 置信度 | 指代 | 意图 |
|---|---|---|---|---|---|---|---|---|
| M01 | 1 | 订单 20240115001 的物流到哪了 | 订单 20240115001 的物流到哪了 | logistics | logistics | 0.97 | PASS | PASS |
| M01 | 2 | 那它能退吗 | 订单 20240115001 能退吗 | refund_return | refund_return | 0.90 | PASS | PASS |
| M01 | 3 | 那现在到哪了 | 订单 20240115001 现在到哪了 | logistics | logistics | 0.96 | PASS | PASS |
| M02 | 1 | XH-3000A 续航多久 | XH-3000A 续航多久 | product_consult | product_consult | 0.95 | PASS | PASS |
| M02 | 2 | 这个能退吗 | XH-3000A 能退吗 | refund_return | refund_return | 0.90 | PASS | PASS |
| M02 | 3 | 那换货呢 | XH-3000A 能换货吗 | after_sales | after_sales | 0.88 | PASS | PASS |
| M03 | 1 | 收到的杯子是碎的，我要投诉并退款 | 收到的杯子是碎的，我要投诉并退款 | complaint | complaint | 0.97 | PASS | PASS |
| M03 | 2 | 那能赔多少 | 杯子碎了这种情况能赔多少 | after_sales | after_sales | 0.72 | PASS | PASS |
| M04 | 1 | 你们公司股票能买吗 | 你们公司股票能买吗 | other | other | 0.95 | PASS | PASS |
| M04 | 2 | 今天天气怎么样 | 今天天气怎么样 | chitchat | chitchat | 0.90 | PASS | PASS |
| M04 | 3 | 帮我看看那个 | 帮我看看那个 | other | other | 0.80 | PASS | PASS |
| M04 | 4 | 这个怎么弄 | 这个怎么弄 | other | other | 0.55 | PASS | PASS |
| M05 | 1 | 你好 | 你好 | chitchat | chitchat | 1.00 | PASS | PASS |
| M05 | 2 | 谢谢 | 谢谢 | chitchat | chitchat | 1.00 | PASS | PASS |
| M06 | 1 | 订单 20240115001 多少钱 | 订单 20240115001 多少钱 | order | order | 0.95 | PASS | PASS |
| M06 | 2 | 那退款呢 | 订单 20240115001 的退款怎么处理 | refund_return | refund_return | 0.92 | PASS | PASS |
| M07 | 1 | 发票怎么开 | 发票怎么开 | after_sales | after_sales | 0.90 | PASS | PASS |
| M07 | 2 | 那保修呢 | 保修怎么办理 | after_sales | after_sales | 0.90 | PASS | PASS |
| M08 | 1 | 这个能退吗 | 这个能退吗 | refund_return | refund_return | 0.85 | PASS | PASS |
| M08 | 2 | 订单 20240115001 的物流 | 订单 20240115001 的物流 | logistics | logistics | 0.95 | PASS | PASS |
| M08 | 3 | 那它到哪了 | 订单 20240115001 的物流到哪了 | logistics | logistics | 0.97 | PASS | PASS |
| M08 | 4 | 我要投诉 | 我要投诉 | complaint | complaint | 0.95 | PASS | PASS |

## 失败明细

无。

> 口径：本评估只测 coref/意图/扩写，不连 Milvus；订单上下文固定使用 `20240115001`。少数用例与 Prompt few-shot 语义相近，不能当作完全 held-out。
