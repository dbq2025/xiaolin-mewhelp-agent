# ch09 证据置信度校准报告

- 评估集：`F:\DeepSeek-Harness\memhelp-code-myself\MemHelp-code\evals\ch04_cases.jsonl`
- 样本数：48（正类 40）
- 阈值：0.95
- balanced accuracy（留一法）：0.9000
- 是否发布 artifact：是

## 混淆矩阵

| TP | FP | TN | FN |
| --- | --- | --- | --- |
| 32 | 0 | 8 | 8 |

## 分桶

| 桶 | 条数 | 判对 | 准确率 |
| --- | --- | --- | --- |
| A_policy | 12 | 11 | 91.67% |
| B_model | 12 | 12 | 100.00% |
| C_colloquial | 12 | 6 | 50.00% |
| E_multi | 12 | 11 | 91.67% |

## 逐条预测

| 用例 | 桶 | 期望 | 置信度 | 判定 |
| --- | --- | --- | --- | --- |
| A01 | A_policy | answer | 1.000 | ✓ |
| A02 | A_policy | answer | 0.997 | ✓ |
| A03 | A_policy | answer | 0.999 | ✓ |
| A04 | A_policy | answer | 0.991 | ✓ |
| A05 | A_policy | answer | 0.999 | ✓ |
| A06 | A_policy | answer | 0.998 | ✓ |
| A07 | A_policy | answer | 0.946 | ✗ |
| A08 | A_policy | answer | 0.970 | ✓ |
| A09 | A_policy | answer | 0.994 | ✓ |
| A10 | A_policy | answer | 0.993 | ✓ |
| A11 | A_policy | refuse | 0.239 | ✓ |
| A12 | A_policy | refuse | 0.194 | ✓ |
| B01 | B_model | answer | 0.999 | ✓ |
| B02 | B_model | answer | 0.999 | ✓ |
| B03 | B_model | answer | 0.999 | ✓ |
| B04 | B_model | answer | 0.994 | ✓ |
| B05 | B_model | answer | 0.999 | ✓ |
| B06 | B_model | answer | 0.996 | ✓ |
| B07 | B_model | answer | 0.995 | ✓ |
| B08 | B_model | answer | 1.000 | ✓ |
| B09 | B_model | answer | 0.990 | ✓ |
| B10 | B_model | answer | 0.997 | ✓ |
| B11 | B_model | refuse | 0.540 | ✓ |
| B12 | B_model | refuse | 0.756 | ✓ |
| C01 | C_colloquial | answer | 0.990 | ✓ |
| C02 | C_colloquial | answer | 0.993 | ✓ |
| C03 | C_colloquial | answer | 0.991 | ✓ |
| C04 | C_colloquial | answer | 0.998 | ✓ |
| C05 | C_colloquial | answer | 0.101 | ✗ |
| C06 | C_colloquial | answer | 0.876 | ✗ |
| C07 | C_colloquial | answer | 0.350 | ✗ |
| C08 | C_colloquial | answer | 0.630 | ✗ |
| C09 | C_colloquial | answer | 0.415 | ✗ |
| C10 | C_colloquial | answer | 0.815 | ✗ |
| C11 | C_colloquial | refuse | 0.205 | ✓ |
| C12 | C_colloquial | refuse | 0.194 | ✓ |
| E01 | E_multi | answer | 0.989 | ✓ |
| E02 | E_multi | answer | 0.566 | ✗ |
| E03 | E_multi | answer | 0.990 | ✓ |
| E04 | E_multi | answer | 0.993 | ✓ |
| E05 | E_multi | answer | 0.998 | ✓ |
| E06 | E_multi | answer | 0.996 | ✓ |
| E07 | E_multi | answer | 0.987 | ✓ |
| E08 | E_multi | answer | 0.999 | ✓ |
| E09 | E_multi | answer | 0.994 | ✓ |
| E10 | E_multi | answer | 0.999 | ✓ |
| E11 | E_multi | refuse | 0.275 | ✓ |
| E12 | E_multi | refuse | 0.929 | ✓ |

## 特征

- `top1`：精排 Top1 相关性分。
- `margin`：Top1 与 Top2 的分差；Top2 缺失按 0。
- `effective_count`：分数 ≥ max(0.1, top1×0.5) 的证据条数。
