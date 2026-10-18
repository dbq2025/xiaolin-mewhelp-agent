# ch05 意图分类评估

- 题数：**56**
- 整体 accuracy：**1.000**（阈值 0.85）
- 出口 route accuracy：**1.000**（阈值 0.85）
- 平均延迟：757 ms
- 来源计数：`{'llm': 51, 'rule': 5}`

> 口径说明：`INTENT_PROMPT` 的边界示例与部分标注用例（L01/O01/C01/C03/R01/H01）
> 高度重叠，56/56 里含一定的提示记忆；正式版应补 held-out 集。

## 每类 recall

| 意图 | recall |
|---|---|
| `logistics` | 1.000 |
| `order` | 1.000 |
| `product_consult` | 1.000 |
| `refund_return` | 1.000 |
| `after_sales` | 1.000 |
| `complaint` | 1.000 |
| `chitchat` | 1.000 |

## 混淆矩阵

| 期望 \ 预测 | logistics | order | product_consult | refund_return | after_sales | complaint | chitchat |
|---|---|---|---|---|---|---|---|
| `logistics` | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| `order` | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| `product_consult` | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| `refund_return` | 0 | 0 | 0 | 8 | 0 | 0 | 0 |
| `after_sales` | 0 | 0 | 0 | 0 | 8 | 0 | 0 |
| `complaint` | 0 | 0 | 0 | 0 | 0 | 8 | 0 |
| `chitchat` | 0 | 0 | 0 | 0 | 0 | 0 | 8 |

## 失败明细

无。
